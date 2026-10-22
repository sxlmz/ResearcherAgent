# -*- coding: utf-8 -*-
"""
fulltext.py —— 轻量深度阅读：从结构化全文源取论文正文（不下 PDF、不 OCR）

调研结论（成熟做法）：
  - arXiv  → ar5iv（LaTeXML XHTML）：https://ar5iv.labs.arxiv.org/html/<id>
             按 h1-h6 切 section，跳过 script/style/math（取 math 的 alttext 作 LaTeX）
  - PMC/Europe PMC → JATS XML：https://www.ebi.ac.uk/europepmc/webservices/rest/<PMCID>/fullTextXML
             免 API key；404 = 非开放获取

对比 MCP 的 read_*_paper：那条路是"下载 PDF + 提取文本"，重且慢；本模块直连结构化源。

设计边界：**只读结构化 HTML/XML 全文，不碰 PDF blob、不做 OCR**。
"""
import re
import urllib.request
import urllib.error
import urllib.parse
import threading
import time as _time

# CORE v3 全局限速：个人档 5 requests / 10s（实测并发下载 6 篇会打爆 429）。
# fetch_core_fulltext 可能被 read_sections 并发调用 → 用锁把 CORE 请求串行化，
# 强制 ≥2.2s 间隔（10s 最多 4 次，留余量）。
_CORE_LOCK = threading.Lock()
_CORE_LAST_REQ = [0.0]
_CORE_MIN_INTERVAL = 2.2


def _core_rate_wait() -> None:
    """CORE v3 请求前调用：串行 + 最小间隔，防 429。"""
    with _CORE_LOCK:
        now = _time.monotonic()
        wait = _CORE_MIN_INTERVAL - (now - _CORE_LAST_REQ[0])
        if wait > 0:
            _time.sleep(wait)
        _CORE_LAST_REQ[0] = _time.monotonic()


def _core_request(url: str, headers: dict, tries: int = 3, retry_5xx: bool = True,
                  retry_429: bool = True) -> tuple:
    """CORE v3 请求：限速 + 对 429/403/5xx 退避重试（works/{id} 实测间歇性 403）。

    retry_5xx=False：search 端点对部分 query 固定 500（清洗后仍可能触发），
    退避重试纯浪费时间——一次即抛，由调用方走下一个兜底（v1.2.9）。
    retry_429=False（v1.2.14 基准实测）：CORE free key 一旦 429 即进入长时间惩罚期
    （>>60s），重试 4s/8s/12s 全部白费且连锁打爆配额——429 直接放弃，由调用方
    走下一个 id / 下一篇 / 其他通路，绝不重试。
    """
    import json as _json
    for i in range(max(1, tries)):
        _core_rate_wait()
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429 and not retry_429:
                raise
            if e.code in (429, 403) or (retry_5xx and e.code in (500, 503)):
                if i + 1 < tries:
                    print(f"    [fulltext] CORE {e.code}，退避 {4 * (i + 1)}s 重试 {url[:70]}…")
                    _time.sleep(4 * (i + 1))
                    continue
            raise
    raise urllib.error.HTTPError(url, 429, "core retry exhausted", None, None)

# v1.2.9（方案 B）：弱 UA（Mozilla/5.0 (compatible; ReviewAgent/1.0)）会被 CORE 前端
# Cloudflare 反爬直接 403——统一升级为完整浏览器请求头；对 core.ac.uk 的 PDF 端点再补 Referer。
_UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}
_TIMEOUT = 20

# ── HTML 表格 → Markdown（实验指标的唯一权威来源）──
# 方案 v1.2：表格数据必须保留行列语义（如 | 模型 | ETTh1 MSE | … |——示例，领域变通），
# 否则去标签后表格退化成空格堆叠的文本，LLM 只能凭印象填数字。
_TABLE_BLOCK_RE = re.compile(r"(?is)<table[^>]*>(.*?)</table>")
_ROW_RE = re.compile(r"(?is)<tr[^>]*>(.*?)</tr>")
_CELL_RE = re.compile(r"(?is)<t[dh][^>]*>(.*?)</t[dh]>")


def _html_table_to_markdown(inner: str) -> str:
    """单个 <table> 内容 → Markdown 表格（表头行 + --- 分隔 + 数据行）。"""
    rows = []
    for row in _ROW_RE.findall(inner):
        cells = [_strip_tags(c).strip().replace("\n", " ") for c in _CELL_RE.findall(row)]
        if cells:
            rows.append(cells)
    if len(rows) < 2:
        return ""
    n = max(len(r) for r in rows)
    out = []
    for i, r in enumerate(rows):
        r = r + [""] * (n - len(r))
        out.append("| " + " | ".join(r) + " |")
        if i == 0:
            out.append("| " + " | ".join(["---"] * n) + " |")
    return "\n".join(out)


def _tables_to_markdown(html: str) -> str:
    """把 HTML 里的 <table> 全部替换为 Markdown 表格（在去标签之前调用）。"""
    return _TABLE_BLOCK_RE.sub(lambda m: "\n\n" + _html_table_to_markdown(m.group(1)) + "\n\n", html)


def _http_get(url: str) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        raw = resp.read()
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception as e:
        print(f"    [fulltext] 解码失败 {url}: {type(e).__name__}: {str(e)[:120]}")
        return ""


def _strip_tags(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    # 保留公式的 alttext（LaTeX 源）
    html = re.sub(r'(?is)<math[^>]*alttext="([^"]*)"[^>]*>.*?</math>', r" $\1$ ", html)
    html = re.sub(r"(?is)<math[^>]*>.*?</math>", " ", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    html = re.sub(r"&nbsp;", " ", html)
    html = re.sub(r"&amp;", "&", html)
    html = re.sub(r"&lt;", "<", html)
    html = re.sub(r"&gt;", ">", html)
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n\s*\n+", "\n\n", html)
    return html.strip()


_A_HREF_RE = re.compile(r"(?is)<a\s[^>]*href\s*=\s*[\"']([^\"']+)[\"'][^>]*>")


def _preserve_links(html: str) -> str:
    """HTML 转文本前保留 a[href]：<a href="URL">文本</a> → "URL 文本"。
    论文标注的开源代码链接（github.com/...）通常藏在 href 里，去标签会丢。"""
    return _A_HREF_RE.sub(lambda m: m.group(1) + " ", html)


def _extract_sections_from_html(html: str, max_chars: int | None, section_max: int = 6000) -> str:
    """从 ar5iv XHTML 里按 h1-h6 提 section。

    section_max：单节正文上限（默认 6000 字符，覆盖绝大多数论文单节长度）；
    不再把每节砍到 2000——那是"只读前 6000 字"的根因之一。

    max_chars：**全篇模式传 None**（抽取阶段不吃 LLM 上下文，切段/打分在
    graph_nodes 层做；2026-10 方案：去字符截断，仅超大文档按段落数预算）。
    """
    # 表格先行转 Markdown（保留行列语义），再保留链接、去标签
    html = _tables_to_markdown(html)
    # 保留链接后转文本（否则 github 代码链接会丢）
    html = _preserve_links(html)
    # 退化：直接去标签
    try:
        # 按标题切段
        parts = re.split(r"(?is)<h[1-6][^>]*>", html)
        sections = []
        for part in parts[1:]:  # 第一段是标题前的内容
            m = re.match(r"(?is)(.*?)</h[1-6]>", part)
            if not m:
                continue
            heading = _strip_tags(m.group(1)).strip()
            body = _strip_tags(part[m.end():])[:section_max]
            if heading and body and len(body) > 40:
                sections.append(f"## {heading}\n{body}")
        text = "\n\n".join(sections) if sections else _strip_tags(html)
    except Exception:
        text = _strip_tags(html)
    return text if max_chars is None else text[:max_chars]


def _tavily_extract(url: str) -> str:
    """Tavily extract 抓取整页/PDF 文本（绕本机对出版社源的网络限制）。
    返回剥离标签后的正文；失败返回空串。有 key 才调用。"""
    import json
    import os
    try:
        key = os.environ.get("TAVILY_API_KEY") or ""
        if not key:
            from ResearcherAgent.core.config import _tavily_key
            key = _tavily_key or ""
        if not key:
            return ""
        body = json.dumps({"api_key": key, "urls": [url],
                           "extract_depth": "advanced"}).encode()
        req = urllib.request.Request("https://api.tavily.com/extract", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            d = json.loads(resp.read().decode())
        raw = ""
        for it in (d.get("results") or []):
            if it.get("raw_content"):
                raw += it.get("raw_content", "")
        return _strip_tags(raw).strip()
    except Exception as e:
        print(f"    [fulltext] Tavily {url[:70]} 失败: {type(e).__name__}: {str(e)[:80]}")
        return ""


def tavily_search(query: str, max_results: int = 8) -> list[dict]:
    """Tavily search：网页搜索（用于「演进脉络调研」等需要综述性信息的场景）。

    返回 [{title, url, content}]，content 为摘要片段（≤800 字符）；失败返回 []。
    有 TAVILY_API_KEY 才调用；key 缺省回退 config._tavily_key。
    """
    import json
    import os
    try:
        key = os.environ.get("TAVILY_API_KEY") or ""
        if not key:
            from ResearcherAgent.core.config import _tavily_key
            key = _tavily_key or ""
        if not key:
            return []
        body = json.dumps({"api_key": key, "query": str(query)[:300],
                           "search_depth": "advanced",
                           "max_results": max(1, int(max_results)),
                           "include_answer": False}).encode()
        req = urllib.request.Request("https://api.tavily.com/search", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            d = json.loads(resp.read().decode())
        out = []
        for it in (d.get("results") or []):
            c = re.sub(r"\s+", " ", _strip_tags(it.get("content") or ""))[:800]
            out.append({"title": (it.get("title") or "")[:200],
                        "url": (it.get("url") or ""),
                        "content": c})
        return out
    except Exception as e:
        print(f"    [fulltext] Tavily search 失败: {type(e).__name__}: {str(e)[:80]}")
        return []


def _ft_cache_get(key: str) -> str | None:
    """Redis 全文缓存（v1.2.19）。返回：None=无缓存；""=缓存过失败（放弃）；
    其他=缓存全文。命中即跳过网络下载 / CORE 配额消耗。"""
    from ResearcherAgent.core import cache
    text, hit = cache.fulltext_get(key)
    return text if hit else None


def _ft_cache_set(key: str, text: str) -> None:
    from ResearcherAgent.core import cache
    cache.fulltext_set(key, text)   # 空文本也缓存（__none__ 失败标记）


def fetch_arxiv_fulltext(arxiv_id: str, max_chars: int | None = 30000, section_max: int = 6000) -> str:
    """从 ar5iv 拿 arXiv 论文的结构化全文（不下 PDF）。失败返回空串。

    max_chars=None → 全篇下载（方案 v1.2：抽取不吃上下文，去字符截断）。
    v1.2.4：ar5iv / 摘要页本机通路都失败时，用 Tavily extract 抓 arxiv 页面兜底。
    v1.2.19：Redis 全文缓存（键 arxiv:<id>，TTL 30 天）——命中直接返回，不重新下载。"""
    arxiv_id = (arxiv_id or "").strip()
    arxiv_id = re.sub(r"^arxiv:", "", arxiv_id, flags=re.I)
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)  # 去版本号
    if not arxiv_id:
        return ""
    _ckey = "arxiv:" + arxiv_id
    _cached = _ft_cache_get(_ckey)
    if _cached is not None:
        if _cached:
            print(f"    [fulltext] 缓存命中 arXiv {arxiv_id}（{len(_cached)} 字符）")
        return _cached
    reasons = []
    for url in (
        f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
        f"https://arxiv.org/abs/{arxiv_id}",   # 兜底：摘要页（至少拿摘要）
    ):
        try:
            html = _http_get(url)
            if not html:
                reasons.append(f"{url} → 空响应")
                continue
            text = _extract_sections_from_html(html, max_chars, section_max=section_max)
            if len(text) > 200:
                _ft_cache_set(_ckey, text)
                return text
            reasons.append(f"{url} → 正文过短({len(text)}字符，可能是跳转/错误页)")
        except Exception as e:
            # 不静默：区分「网络不可达」与「该论文无结构化全文」——两者对调用方含义完全不同
            reasons.append(f"{url} → {type(e).__name__}: {str(e)[:80]}")
    # v1.2.4 Tavily 兜底：本机 ar5iv/摘要页通路失败时，用 Tavily extract 抓 arxiv 页面
    t = _tavily_extract(f"https://arxiv.org/abs/{arxiv_id}")
    if t and len(t) > 200:
        t = t if max_chars is None else t[:max_chars]
        _ft_cache_set(_ckey, t)
        return t
    if t:
        reasons.append(f"tavily → 正文过短({len(t)}字符)")
    print(f"    [fulltext] arXiv {arxiv_id} 取全文失败：{' | '.join(reasons)}")
    _ft_cache_set(_ckey, "")   # 失败也缓存（30 天内不反复重试）
    return ""


def fetch_pmc_fulltext(pmcid: str, max_chars: int | None = 30000) -> str:
    """从 Europe PMC 拿开放获取论文的 JATS 全文（不下 PDF）。失败返回空串。

    max_chars=None → 全篇下载。
    v1.2.19：Redis 全文缓存（键 pmc:<PMCID>，TTL 30 天）。"""
    pmcid = (pmcid or "").strip().upper()
    if not pmcid.startswith("PMC"):
        return ""
    _ckey = "pmc:" + pmcid
    _cached = _ft_cache_get(_ckey)
    if _cached is not None:
        if _cached:
            print(f"    [fulltext] 缓存命中 Europe PMC {pmcid}（{len(_cached)} 字符）")
        return _cached
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        xml = _http_get(url)
        if not xml:
            print(f"    [fulltext] Europe PMC {pmcid} 空响应")
            _ft_cache_set(_ckey, "")
            return ""
        text = _strip_tags(xml)
        text = text if max_chars is None else text[:max_chars]
        if text:
            _ft_cache_set(_ckey, text)
        return text
    except urllib.error.HTTPError as e:
        # 404 = 该文献非开放获取（正常情况，不是故障），其余码才是异常
        tag = "非开放获取" if e.code == 404 else f"HTTP {e.code} {e.reason}"
        print(f"    [fulltext] Europe PMC {pmcid}: {tag}")
        _ft_cache_set(_ckey, "")
        return ""
    except Exception as e:
        print(f"    [fulltext] Europe PMC {pmcid} 失败: {type(e).__name__}: {str(e)[:120]}")
        _ft_cache_set(_ckey, "")
        return ""

if __name__ == "__main__":
    # 自测：用一篇知名 arXiv 论文验证 ar5iv 通路
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    aid = sys.argv[1] if len(sys.argv) > 1 else "1706.03762"  # Attention is all you need
    t = fetch_arxiv_fulltext(aid, max_chars=1500)
    print(f"arxiv {aid} 全文长度: {len(t)}")
    print(t[:800])


def _fetch_pdf_text(pdf_url: str) -> str:
    """本机直下 PDF + PyPDF2 提取正文（前 12 页）。失败返回空串。

    v1.2.9（方案 B）：请求头升级为浏览器头；对 core.ac.uk 的 PDF 端点补 Referer
    （Cloudflare 反爬对无 Referer/弱 UA 的非浏览器请求直接 403）。
    """
    try:
        import io
        from PyPDF2 import PdfReader
        _h = dict(_UA)
        if "core.ac.uk" in pdf_url:
            _h["Referer"] = "https://core.ac.uk/"
        req = urllib.request.Request(pdf_url, headers=_h)
        with urllib.request.urlopen(req, timeout=45) as resp:
            blob = resp.read()
        rd = PdfReader(io.BytesIO(blob))
        n = min(len(rd.pages), 12)
        txt = "\n".join((rd.pages[i].extract_text() or "") for i in range(n))
        return txt if len(txt) > 300 else ""
    except Exception as e:
        print(f"    [fulltext] PDF 提取失败 {pdf_url[:60]}: {type(e).__name__}: {str(e)[:60]}")
        return ""


def fetch_core_fulltext(doi: str = "", title: str = "", core_id: str = "") -> str:
    """CORE API（core.ac.uk）全文文本——对 MDPI/DOAJ 系 OA 期刊最有价值。

    v1.2.5 迁移到 CORE v3（实测 v2 端点已下线，v3 端点矩阵验证）：
      - 搜索：GET /v3/search/works?q=<query>&limit=1
        （title 检索最稳（title:"..."）；DOI 字符串直接查可能 500/命中 0，仅作兜底）
      - 全文：GET /v3/works/{id} → 响应 `fullText` 字段直接是全文文本（无需解析 PDF）
    v1.2.14（CORE 基准实测）：core_id 非空 → 直连 works/{id}（跳过 title 反查，省 1 跳/篇）；
      title 反查只取前 3 个 id（id≤3 已够命中）；429 不重试、403 重试 1 次——
      一旦 429 即放弃该 id/该篇（惩罚期 >>60s，重试白费），由其他通路补位。
    需要免费 API key（https://core.ac.uk/services/api 注册），填 .env 的 CORE_API_KEY；
    无 key 自动跳过（返回空串）。
    配额友好：结果按 DOI/标题落盘缓存（FULLTEXT_CACHE_DIR）——同一篇论文只消耗一次
    配额，重跑/多会话不重复扣 token；全局限速 _core_rate_wait 防个人档 429。
    """
    import json
    import os
    key = os.environ.get("CORE_API_KEY") or ""
    if not key:
        return ""
    ident = (core_id or "").strip()
    if not ident and (doi or "").strip():
        ident = (doi or "").strip()
    if not ident and title:
        ident = (title or "").strip()
    if not ident:
        return ""
    # v1.2.19 Redis 全文缓存（键 core:<ident>，TTL 30 天）：命中即不再烧 CORE 配额。
    # 磁盘缓存（FULLTEXT_CACHE_DIR）仍保留——Redis 是跨会话/跨进程的统一层。
    _ckey = "core:" + ident
    _cached = _ft_cache_get(_ckey)
    if _cached is not None:
        if _cached:
            print(f"    [fulltext] 缓存命中 CORE {ident[:50]}（{len(_cached)} 字符）")
        return _cached
    # 磁盘缓存：同 DOI 只调一次 CORE（个人配额 1000 tokens/天 有限）
    cache_dir = os.environ.get("FULLTEXT_CACHE_DIR") or ""
    cf = ""
    if cache_dir:
        cf = os.path.join(cache_dir, f"core_ft_{re.sub(r'[^A-Za-z0-9]', '_', ident)}.txt")
        try:
            if os.path.exists(cf):
                v = open(cf, encoding="utf-8").read()
                return "" if v == "__none__" else v
        except Exception:
            pass
    try:
        if core_id:
            # ── v1.2.14 直连：建池已带 CORE work id，跳过 title 反查 ──
            ids = [str(core_id).strip()]
        else:
            # ── v3 检索：title 优先（最稳），DOI 兜底 ──
            # ⚠️ title 里的冒号/破折号等会破坏 CORE 的 title:"..." 短语语法（实测 500），
            # 先清洗成纯字母数字空格再包引号。
            q = ""
            if (title or "").strip():
                clean = re.sub(r"[^A-Za-z0-9 ]+", " ", title).strip()
                clean = re.sub(r"\s+", " ", clean)[:120]
                q = f'title:"{clean}"'
            elif doi:
                q = doi
            if not q:
                return ""
            url = ("https://api.core.ac.uk/v3/search/works?q=" +
                   urllib.parse.quote(q) + "&limit=3")  # v1.2.14：id 上限 5→3
            _st, _body = _core_request(url, {
                "Authorization": "Bearer " + key, "User-Agent": "ReviewAgent/1.0"},
                tries=1, retry_5xx=False, retry_429=False)  # search 429/500 均不重试
            d = json.loads(_body)
            arts = (d.get("results") or []) if isinstance(d, dict) else []
            if not arts:
                print(f"    [fulltext] CORE 未收录: {ident[:60]}")
                _core_cache_write(cf, "")
                _ft_cache_set(_ckey, "")
                return ""
            ids = [str(a.get("id") or "") for a in arts if a.get("id")]
        # ── v3 全文：works/{id} 的 fullText 字段直接是全文文本 ──
        # ⚠️ works/{id} 实测存在间歇性 403（与 id 无关）→ 逐个 id 兜底尝试
        # （同一论文常有多个聚合副本记录，换 id 即可能成功）。
        last_err = ""
        for cid in ids:
            try:
                _st2, _body2 = _core_request(f"https://api.core.ac.uk/v3/works/{cid}",
                                             {"Authorization": "Bearer " + key,
                                              "User-Agent": "ReviewAgent/1.0"},
                                             tries=2, retry_429=False)  # 403 重 1 次、429 放弃
                d2 = json.loads(_body2)
                t = _strip_tags(str(d2.get("fullText") or ""))
                if len(t) >= 300:
                    _core_cache_write(cf, t)
                    _ft_cache_set(_ckey, t)
                    return t
                last_err = f"全文过短({len(t)}字符)"
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:60]}"
                continue
        print(f"    [fulltext] CORE 全文不可得 {ident[:60]}: {last_err}")
        _core_cache_write(cf, "")
        _ft_cache_set(_ckey, "")
        return ""
    except Exception as e:
        print(f"    [fulltext] CORE 失败 {ident[:60]}: {type(e).__name__}: {str(e)[:80]}")
        return ""


def _fetch_core_work_fulltext(work_id: str, max_chars: int | None = None) -> str:
    """CORE API v3 works/{id} → fullText 字段（v1.2.9 方案 A，绕前端 403）。

    core.ac.uk/download/<id>.pdf 这类**前端 PDF 直链**的数字 id 就是 CORE 的 work id——
    直接走 api.core.ac.uk/v3/works/{id} 拿全文文本（带 Bearer key），完全绕开
    Cloudflare 反爬（download 端点对非浏览器请求几乎必 403）。需要 CORE_API_KEY；
    无 key / 非 OA / 全文过短 → 返回空串（调用方继续降级）。
    """
    import os
    key = os.environ.get("CORE_API_KEY") or ""
    if not key or not str(work_id or "").strip():
        return ""
    try:
        _st, _body = _core_request(
            f"https://api.core.ac.uk/v3/works/{work_id}",
            {"Authorization": "Bearer " + key, "User-Agent": "ReviewAgent/1.0"},
            tries=2, retry_429=False)  # v1.2.14：403 间歇性重试 1 次(4s)，429 直接放弃
        if _st != 200 or not _body:
            return ""
        import json as _json
        t = (_json.loads(_body).get("fullText") or "").strip()
    except Exception as e:
        print(f"    [fulltext] CORE works/{work_id} 失败: {type(e).__name__}: {str(e)[:80]}")
        return ""
    if len(t) < 300:
        return ""
    return t if max_chars is None else t[:max_chars]


def _core_cache_write(cf: str, text: str) -> None:
    """CORE 结果落盘缓存：text 空写 "__none__"（未收录/过短），非空写全文。"""
    if not cf:
        return
    try:
        import os
        os.makedirs(os.path.dirname(cf), exist_ok=True)
        with open(cf, "w", encoding="utf-8") as f:
            f.write(text if text else "__none__")
    except Exception:
        pass


def fetch_oa_fulltext(url: str, max_chars: int | None = 30000, section_max: int = 6000) -> str:
    """开放获取论文全文（OpenAlex OA 直链 / landing page 的通用通路）。

    通路优先级（v1.2.4 实测网络环境；v1.2.8 移除 Semantic Scholar；v1.2.9 方案 A+B）：
      1) arXiv 链接 → ar5iv（结构化全文，最优先）
      2) core.ac.uk/download/<id> → CORE API works/{id} fullText（绕前端 403）
      3) 本机直下 PDF + PyPDF2 提取（浏览器化请求头 + core Referer）
      4) CORE API 全文文本（对 MDPI/DOAJ 系 OA 期刊；需 CORE_API_KEY）
      5) Tavily extract（第三方抓取服务器，绕本机网络限制）
    max_chars=None → 全篇下载（方案 v1.2）。
    失败返回空串（调用方跳过，不编造）。
    """
    url = (url or "").strip()
    if not url.startswith("http"):
        return ""
    # v1.2.19 Redis 全文缓存（键 oa:<url>）：命中即跳过整个通路链（含 CORE/PDF/Tavily）
    _ckey = "oa:" + url
    _cached = _ft_cache_get(_ckey)
    if _cached is not None:
        if _cached:
            print(f"    [fulltext] 缓存命中 OA {url[:70]}（{len(_cached)} 字符）")
        return _cached
    # 1) arXiv 链接 → ar5iv
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", url, re.I)
    if m:
        t = fetch_arxiv_fulltext(m.group(1), max_chars, section_max=section_max)
        if t:
            _ft_cache_set(_ckey, t)
            return t
    # 2) core.ac.uk/download/<id> → CORE API works/{id}（v1.2.9 方案 A：
    #    前端 PDF 端点几乎必 403，直接按 work id 取 fullText，不碰反爬）
    m = re.search(r"core\.ac\.uk/download/(\d+)", url, re.I)
    _core_only = False
    if m:
        _core_only = True
        t = _fetch_core_work_fulltext(m.group(1), max_chars)
        if t:
            return t
        # v1.2.14：core 前端 PDF 端点必 403（方案 B 已证），跳过 PDF 直下这跳；
        # 直连 works/{id} 失败则直接走 title/DOI 反查兜底（见 4)）。
    # 3) 本机直下 PDF（PyPDF2；浏览器化请求头）
    if not _core_only:
        t = _fetch_pdf_text(url)
        if t:
            return t if max_chars is None else t[:max_chars]
    # 4) CORE API 全文文本（v1.2.4：OA 期刊直接给全文，绕 PDF 反爬）
    doi = ""
    m = re.search(r"(?:doi\.org/|doi:|/doi/)(10\.\d{4,9}/[^\s\"'<>?#]+)", url, re.I)
    if m:
        doi = m.group(1).rstrip(".,;")
    if doi:
        t = fetch_core_fulltext(doi=doi)
        if t:
            return t if max_chars is None else t[:max_chars]
    # 5) Tavily extract（绕本机网络限制；对 PDF 直链返回全文文本）
    t = _tavily_extract(url)
    if t and len(t) > 300:
        t = t if max_chars is None else t[:max_chars]
        _ft_cache_set(_ckey, t)
        return t
    print(f"    [fulltext] OA 全文不可得: {url[:80]}")
    _ft_cache_set(_ckey, "")
    return ""

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

_UA = {"User-Agent": "Mozilla/5.0 (compatible; ReviewAgent/1.0)"}
_TIMEOUT = 20


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


def _extract_sections_from_html(html: str, max_chars: int, section_max: int = 6000) -> str:
    """从 ar5iv XHTML 里按 h1-h6 提 section。

    section_max：单节正文上限（默认 6000 字符，覆盖绝大多数论文单节长度）；
    不再把每节砍到 2000——那是"只读前 6000 字"的根因之一。
    """
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
    return text[:max_chars]


def fetch_arxiv_fulltext(arxiv_id: str, max_chars: int = 30000, section_max: int = 6000) -> str:
    """从 ar5iv 拿 arXiv 论文的结构化全文（不下 PDF）。失败返回空串。"""
    arxiv_id = (arxiv_id or "").strip()
    arxiv_id = re.sub(r"^arxiv:", "", arxiv_id, flags=re.I)
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)  # 去版本号
    if not arxiv_id:
        return ""
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
                return text
            reasons.append(f"{url} → 正文过短({len(text)}字符，可能是跳转/错误页)")
        except Exception as e:
            # 不静默：区分「网络不可达」与「该论文无结构化全文」——两者对调用方含义完全不同
            reasons.append(f"{url} → {type(e).__name__}: {str(e)[:80]}")
    print(f"    [fulltext] arXiv {arxiv_id} 取全文失败：{' | '.join(reasons)}")
    return ""


def fetch_pmc_fulltext(pmcid: str, max_chars: int = 30000) -> str:
    """从 Europe PMC 拿开放获取论文的 JATS 全文（不下 PDF）。失败返回空串。"""
    pmcid = (pmcid or "").strip().upper()
    if not pmcid.startswith("PMC"):
        return ""
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        xml = _http_get(url)
        if not xml:
            print(f"    [fulltext] Europe PMC {pmcid} 空响应")
            return ""
        text = _strip_tags(xml)
        return text[:max_chars]
    except urllib.error.HTTPError as e:
        # 404 = 该文献非开放获取（正常情况，不是故障），其余码才是异常
        tag = "非开放获取" if e.code == 404 else f"HTTP {e.code} {e.reason}"
        print(f"    [fulltext] Europe PMC {pmcid}: {tag}")
        return ""
    except Exception as e:
        print(f"    [fulltext] Europe PMC {pmcid} 失败: {type(e).__name__}: {str(e)[:120]}")
        return ""

if __name__ == "__main__":
    # 自测：用一篇知名 arXiv 论文验证 ar5iv 通路
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    aid = sys.argv[1] if len(sys.argv) > 1 else "1706.03762"  # Attention is all you need
    t = fetch_arxiv_fulltext(aid, max_chars=1500)
    print(f"arxiv {aid} 全文长度: {len(t)}")
    print(t[:800])


def fetch_oa_fulltext(url: str, max_chars: int = 30000, section_max: int = 6000) -> str:
    """开放获取论文全文（OpenAlex OA 直链 / landing page 的通用通路）。

    通路优先级（实测网络环境）：
      1) arXiv 链接 → ar5iv（结构化全文，最优先）
      2) 本机直下 PDF + PyPDF2 提取
      3) Tavily extract（第三方抓取服务器，**绕开本机对出版社源的网络限制**——
         实测：本机连不通 ebi.ac.uk / ojs.aaai.org / sciencedirect.com，
         Tavily 对 AAAI PDF 直链成功提取 42KB 全文）
    失败返回空串（调用方跳过，不编造）。
    """
    import json
    import os
    url = (url or "").strip()
    if not url.startswith("http"):
        return ""
    # 1) arXiv 链接 → ar5iv
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", url, re.I)
    if m:
        t = fetch_arxiv_fulltext(m.group(1), max_chars, section_max=section_max)
        if t:
            return t
    # 2) 本机直下 PDF（PyPDF2）
    try:
        import io
        from PyPDF2 import PdfReader
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=45) as resp:
            blob = resp.read()
        rd = PdfReader(io.BytesIO(blob))
        n = min(len(rd.pages), 12)
        txt = "\n".join((rd.pages[i].extract_text() or "") for i in range(n))
        if len(txt) > 300:
            return txt[:max_chars]
    except Exception:
        pass
    # 3) Tavily extract（绕本机网络限制；对 PDF 直链返回全文文本）
    try:
        key = os.environ.get("TAVILY_API_KEY") or ""
        if not key:
            from ResearcherAgent.core.config import _tavily_key
            key = _tavily_key or ""
        if key:
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
            raw = _strip_tags(raw)
            if len(raw) > 300:
                return raw[:max_chars]
    except Exception as e:
        print(f"    [fulltext] OA {url[:70]} Tavily 失败: {type(e).__name__}: {str(e)[:80]}")
    print(f"    [fulltext] OA 全文不可得: {url[:80]}")
    return ""

"""
agent_tools.py
==============
把旧图里写死的节点降级为**模型可调用的工具**，形成 agent ⟲ tools 的 ReAct 循环。

| 旧节点 / 边                         | 现在的工具                 |
|------------------------------------|---------------------------|
| plan_review                        | make_outline              |
| 检索指令 + chatbot_with_tools       | MCP 搜索工具（包装后入 paper_pool，只回摘要给模型） |
| coverage_judge + route_after_coverage | —（v1.2.7 删除：大纲定节已保证每节 ≥5 篇有全文，read_sections 内嵌全池重排） |
| targeted_search                    | —（模型直接再调搜索工具）    |
| deep_read                          | read_papers / read_sections |
| review_materials + interrupt       | —（v1.2.7 删除：全部卡片直接用，无用户审核门） |
| （新）显式计划                       | write_todos               |

设计原则（Anthropic《Building Effective Agents》）：
- 工具只提供**能力**，顺序与终止由模型判断
- 全量结果存 state（paper_pool / paper_cards），上下文只放**摘要 + 句柄**（上下文卸载）
- 工具失败以 ToolMessage(status="error") 回给模型，让模型自己换源/换词，而不是代码兜底
"""

import asyncio
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from ResearcherAgent.core.config import (
    TOOL_CONCURRENCY, DOWNLOAD_CONCURRENCY, EXTRACT_CONCURRENCY, MAX_TOOL_RESULT_CHARS, MAX_SEARCH_ROUNDS,
    PAPER_SEARCH_TOOL_WHITELIST, WEB_SEARCH_TOOL_PREFIXES,
    SECTION_CANDIDATES_K, RERANK_MIN_SCORE, RERANK_MIN_CAND_SCORE, MIN_PAPERS_PER_SECTION, CARDS_PER_SECTION,
    SRC_FAIL_STREAK, SRC_COOLDOWN_SEC, MCP_CALL_TIMEOUT_SEC, RETRY_BACKOFF_SEC,
)
from ResearcherAgent.core import cache
from ResearcherAgent.core import tracing

from ResearcherAgent.retrieval import retrieval_cache as rcache


# ==========================================
# 0. paper_pool：结构化论文池 + reducer
# ==========================================

def paper_key(p: dict) -> str:
    """去重键：doi > paper_id > 归一化标题。"""
    doi = str(p.get("doi") or "").strip().lower()
    if doi:
        return "doi:" + doi
    pid = str(p.get("paper_id") or p.get("id") or p.get("paperId") or "").strip().lower()
    if pid:
        return "id:" + pid
    title = re.sub(r"[^a-z0-9一-鿿]+", " ", str(p.get("title") or "").lower()).strip()
    return "title:" + " ".join(title.split()[:12])


def merge_paper_cards(left: list | None, right: list | None) -> list:
    """AgentState.paper_cards 的 reducer：追加并去重（模型可能并行多次 read_papers）。"""
    if right is None:
        return []
    out = list(left or [])
    seen = {c.get("id") for c in out if c.get("id")}
    for c in right or []:
        if not isinstance(c, dict) or not c.get("card"):
            continue
        k = c.get("id") or c.get("title")
        if k and k in seen:
            continue
        if k:
            seen.add(k)
        out.append(c)
    return out


def merge_innovations(left: list | None, right: list | None) -> list:
    """AgentState.innovations 的 reducer：按 id 合并去重（后者覆盖同 id 旧条目）。

    innovate 流水线中模型可能并发调用 propose/novelty 等工具，各自返回
    Command(update={"innovations": ...}) —— LastValue 通道并发写会抛
    InvalidUpdateError，故用 reducer 合并。
    """
    left = left or []
    if right is None:
        return left
    out: list[dict] = []
    idx: dict[str, int] = {}
    for item in list(left) + list(right):
        if not isinstance(item, dict):
            continue
        k = str(item.get("id") or "")
        if k and k in idx:
            out[idx[k]] = item          # 同 id：新条目覆盖旧条目
        else:
            idx[k] = len(out)
            out.append(item)
    return out


def merge_experiments(left: dict | None, right: dict | None) -> dict:
    """AgentState.experiments 的 reducer：按 innovation_id 合并（并发实现/判定时防 LastValue 冲突）。"""
    out = dict(left or {})
    if right:
        out.update(right)
    return out


def merge_papers(left: list | None, right: list | None) -> list:
    """AgentState.paper_pool 的 reducer：按 paper_key 去重合并，保留先到的记录并补空字段。

    right 为 None 时表示清空（重规划从零开始时用）。
    """
    if right is None:
        return []
    left = left or []
    seen: dict[str, dict] = {}
    out: list[dict] = []
    for p in list(left) + list(right):
        if not isinstance(p, dict) or not (p.get("title") or p.get("raw")):
            continue
        k = paper_key(p)
        if k in seen:
            # 补空字段 / 合并 section 归属
            base = seen[k]
            for fk, fv in p.items():
                if fv and not base.get(fk):
                    base[fk] = fv
            if p.get("section") and p["section"] not in (base.get("sections") or []):
                base.setdefault("sections", []).append(p["section"])
            continue
        q = dict(p)
        if q.get("section"):
            q.setdefault("sections", [q["section"]])
        seen[k] = q
        out.append(q)
    return out


def parse_papers(raw: Any, source_hint: str = "") -> list[dict]:
    """把 MCP 检索工具的返回（JSON 文本 / 文本块列表 / dict 列表）解析成论文 dict 列表。

    v1.2.5：解析出的每条论文统一附加 `fulltext_hint` 字段（arxiv:<id> / pmc:<PMCID> /
    oa:<url> / ""）——全文可得性在**检索建池阶段**就确定，供 make_outline 定节、
    覆盖度统计、read_sections 选卡直接使用，不再等到精读阶段才暴露。
    解析不了的内容退化为一条 {"title": <首行>, "raw": <全文>} blob，绝不丢信息。
    """
    items = _parse_papers_items(raw, source_hint)
    for p in items:
        if isinstance(p, dict) and not p.get("fulltext_hint"):
            p["fulltext_hint"] = _paper_id_for_fulltext(p)
    return items


def _parse_papers_items(raw: Any, source_hint: str = "") -> list[dict]:
    """parse_papers 的内部实现（只解析，不附加 fulltext_hint）。"""
    items: list = []
    if isinstance(raw, list):
        for it in raw:
            if isinstance(it, dict) and "text" in it and len(it) <= 3:   # content block
                items.extend(parse_papers(it["text"], source_hint))
            elif isinstance(it, dict):
                items.append(it)
            elif isinstance(it, str):
                items.extend(parse_papers(it, source_hint))
        return items
    if isinstance(raw, dict):
        if "papers" in raw and isinstance(raw["papers"], list):
            return parse_papers(raw["papers"], source_hint)
        # Semantic Scholar MCP（semantic-scholar-mcp）返回 {"data": [...], "total":..., ...}
        if "data" in raw and isinstance(raw["data"], list):
            return parse_papers(raw["data"], source_hint)
        return [raw]
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return parse_papers(data, source_hint)
    except Exception:
        pass
    # 非 JSON：尝试按空行分块，每块首行当标题
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) >= 2 and all(len(b) < 3000 for b in blocks):
        out = []
        for b in blocks:
            first = b.splitlines()[0].strip(" -*#").strip()
            out.append({"title": first[:200], "raw": b, "source": source_hint})
        return out
    return [{"title": text.splitlines()[0][:200], "raw": text[:4000], "source": source_hint}]


def _paper_id_for_fulltext(p: dict) -> str:
    """从论文记录里找出可用于取全文的 id。
    优先级：arXiv id（ar5iv 结构化）→ PMCID（EuropePMC JATS）→
    OA PDF/landing URL（返回 "oa:<url>"，走 fetch_oa_fulltext：本机 PDF / Tavily）。
    """
    for cand in (p.get("paper_id"), p.get("id"), p.get("url"), p.get("pdf_url"), p.get("doi")):
        s = str(cand or "")
        m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", s)
        if m and ("arxiv" in s.lower() or s.strip() == m.group(0)):
            return m.group(1)
        m = re.search(r"(PMC\d+)", s, re.I)
        if m:
            return m.group(1).upper()
    # OA 兜底：pdf_url / oa_url / landing_url 任一 http 链接 → Tavily/PDF 通路
    for cand in (p.get("pdf_url"), p.get("oa_url"), p.get("landing_url")):
        s = str(cand or "").strip()
        if re.match(r"https?://", s):
            if "arxiv.org" in s.lower():
                m = re.search(r"(\d{4}\.\d{4,5})", s)
                if m:
                    return m.group(1)
            return "oa:" + s
    return ""


_TAG_RE = re.compile(r"<[^>]{0,200}>")


def _clean_text(s) -> str:
    """清理摘要文本：去掉混入的 HTML 标签并压缩空白。

    crossref 的 abstract 常带 `<title>Abstract</title><p>...` 这类标签（实测占 56%），
    直接拼进 prompt 既浪费 400 字符的黄金位置，也带来无关噪音。
    """
    if not s:
        return ""
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", str(s))).strip()


_ABSTRACT_CAP = 400          # 与 render_paper 的截断长度一致


def _fingerprint(obj) -> str:
    """给任意可 JSON 化的对象算指纹，用于缓存键（只关心「内容变了没」）。"""
    import hashlib as _hl
    return _hl.blake2b(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"),
        digest_size=16).hexdigest()


def paper_rerank_text(p: dict) -> str:
    """构造用于重排的文本：标题 + 摘要前 400 字（与 render_paper 给模型看的内容一致）。"""
    title = str(p.get("title") or "").strip()
    ab = _clean_text(str(p.get("abstract") or p.get("raw") or ""))[:400]
    return (f"{title}. {ab}".strip())[:600]


def section_query(o: dict) -> str:
    """用大纲条目构造该节的重排查询。

    把节名（中文）、聚焦说明、以及模型给的英文检索词拼在一起——
    重排模型对中英混合查询工作良好，英文关键词能显著提升学术库的召回精度。
    """
    parts = [str(o.get("section") or "").strip()]
    if o.get("focus"):
        parts.append(str(o["focus"]).strip())
    qs = [str(q).strip() for q in (o.get("queries") or []) if str(q).strip()]
    if qs:
        parts.append("；".join(qs))
    return "。".join(p for p in parts if p)


def select_section_candidates(pool: list, outline: list, k: int = SECTION_CANDIDATES_K,
                              thread_id: str = "") -> dict:
    """对每个章节做**全池重排**，取 top-K 作为该节候选。

    v1.2.4「先筛全文再精排」：重排池 = **有全文可达线索**的论文（arXiv/PMC id 或 OA URL），
    无任何全文线索的题录不参与重排/覆盖度统计——防止精排 top 选中后取不到全文，
    导致 read_sections 后续步骤失效；它们仍在 paper_pool，写作时可作「未精读·仅摘要」引用。

    返回 {section: {"query":..., "candidates":[{idx,key,score,title,has_abstract}]}}；
    重排 API 不可用时返回 {}（调用方自动降级回旧排序）。
    """
    from ResearcherAgent.retrieval import rerank as _rerank
    pool = [p for p in (pool or []) if isinstance(p, dict)]
    pool_ft = [p for p in pool if _paper_id_for_fulltext(p)]
    if not pool_ft:
        print(f"    ℹ️ [rerank] 池内 {len(pool)} 篇均无全文可达线索（arXiv/PMC/OA），"
              f"跳过重排——请换可开放获取的检索源")
        return {}
    if not outline or not _rerank.available():
        if pool_ft and outline and not _rerank.available():
            print("    ℹ️ [rerank] 未配置 RERANK_API_*，跳过重排（降级为按素材可用性排序）")
        return {}

    # ---- L5 缓存：同样的（池 + 大纲）不重复重排 ----
    pool_fp = _fingerprint(sorted(paper_key(p) for p in pool_ft))
    outline_fp = _fingerprint([f"{o.get('section')}|{o.get('focus')}|{o.get('queries')}"
                               for o in outline])
    hit = cache.select_get(pool_fp, outline_fp)
    if hit:
        print(f"--> [Rerank] ♻️ L5 选材缓存命中（池 {len(pool_ft)} 篇 × {len(outline)} 节），"
              f"跳过 {len(outline)} 次全池重排")
        return hit

    gidx = global_index(pool_ft)
    docs = [paper_rerank_text(p) for p in pool_ft]
    out: dict = {}
    for o in outline:
        sec = str(o.get("section") or "").strip()
        if not sec:
            continue
        q = section_query(o)
        # 拿**全池**分数：top-K 用作候选，「全池过阈值的篇数」用作覆盖度判定
        res = _rerank.rerank(q, docs, thread_id=thread_id)   # 不传 top_n → 返回全部分数
        if not res:
            continue
        # 覆盖度：统计有全文线索的池里有多少篇达到阈值（这才是「这一节能精读的资料够不够」）
        n_rel = sum(1 for _, s in res if s >= RERANK_MIN_SCORE)
        cands = []
        for i, score in res[:k]:
            # v1.2.11 候选过滤：低于 RERANK_MIN_CAND_SCORE 直接丢弃——
            # 低分论文语义上与本节不匹配，精读了只会造成「章节标题与内容对不上」
            # （coverage_gap / 内容深度低）。宁可少精读，也不塞不相关论文。
            if score < RERANK_MIN_CAND_SCORE:
                continue
            if not (0 <= i < len(pool_ft)):
                continue
            p = pool_ft[i]
            cands.append({"idx": gidx.get(paper_key(p)), "key": paper_key(p),
                          "score": round(float(score), 4),
                          "title": (p.get("title") or "")[:120],
                          "has_abstract": has_abstract(p),
                          "readable_id": _paper_id_for_fulltext(p)})
        top = cands[0]["score"] if cands else 0.0
        out[sec] = {"query": q[:120], "candidates": cands,
                    "top_score": top, "n_relevant": n_rel,
                    "scored_total": len(res)}
        print(f"--> [Rerank] 「{sec[:24]}」有全文池 {len(pool_ft)} 篇打分，"
              f"≥{RERANK_MIN_SCORE} 的 {n_rel} 篇；≥{RERANK_MIN_CAND_SCORE} 过滤后取 "
              f"{len(cands)} 篇作候选（最高 {top:.3f}）")
    if out:
        cache.select_set(pool_fp, outline_fp, out)
    return out


def per_section_top_cards(cands: dict, per_section: int = CARDS_PER_SECTION,
                          prefer_readable: bool = True) -> list:
    """从每节候选中挑出要精读的论文（默认每节 top-2）。

    v1.2.4：**只挑有全文 id 的候选**（arXiv/PMC/OA）——无全文线索的不选
    （宁缺毋滥，read_sections 会对下载失败的自动补位）。
    """
    picks = []
    for sec, info in (cands or {}).items():
        cs = list(info.get("candidates") or [])
        if prefer_readable:
            # 窗口内只取有 readable_id 的：相关性优先于可读性，但**无全文线索的不选**
            window = cs[:max(per_section * 3, 5)]
            chosen = [c for c in window if c.get("readable_id")][:per_section]
        else:
            chosen = cs[:per_section]
        for c in chosen:
            picks.append({**c, "section": sec})
    return picks


def _paper_rank(p: dict) -> tuple:
    """写作素材的可用性排序键（越大越优先）。

    依据：一段摘要（写作时截 400 字）的价值远高于一条纯题录——实测按插入序渲染时，
    每节 45 篇里只有 16 篇有摘要，排序后同样 15000 字符预算可注入 23 篇（+45%）。

    ⚠️ 首键必须是「**有没有**摘要」（0/1），不能是摘要长度：render_paper 只取前 400 字符，
    超过 400 的长度差异对写作毫无意义，若拿长度当主键，会把「摘要长但不太相关」的论文
    排到「摘要短但高度相关」的前面。长度只作 400 以内的次级参考。
    """
    ab = _clean_text(p.get("abstract") or p.get("raw") or "")
    try:
        cites = int(p.get("citations") or 0)
    except (TypeError, ValueError):
        cites = 0
    return (1 if ab else 0,                       # 有摘要 > 仅题录
            min(len(ab), _ABSTRACT_CAP),          # 400 以内越长信息越多，之外不再加分
            1 if _paper_id_for_fulltext(p) else 0,  # 可被 read_papers 精读
            cites)


def has_abstract(p: dict) -> bool:
    return bool(_clean_text(p.get("abstract") or p.get("raw") or ""))


def order_pool(pool: list, section: str | None = None) -> list:
    """渲染顺序：本节归属的论文优先，组内按素材可用性降序。

    ⚠️ 必须排序。按插入序渲染时，「先检索到的」会占满字符预算，把后检索到、
    但有摘要的论文挤出去——多检索不仅无益，反而主动降低素材质量。
    """
    pool = [p for p in (pool or []) if isinstance(p, dict)]
    if not section:
        return sorted(pool, key=_paper_rank, reverse=True)
    first = [p for p in pool if section in (p.get("sections") or [])]
    first_ids = {id(p) for p in first}          # 用 id() 而非 `p not in first`（后者是 O(n²) 字典比较）
    rest = [p for p in pool if id(p) not in first_ids]
    return (sorted(first, key=_paper_rank, reverse=True)
            + sorted(rest, key=_paper_rank, reverse=True))


def global_index(pool: list) -> dict:
    """给池中每篇论文分配**稳定的全局编号**（1-based，按 order_pool 的规范顺序）。

    为什么必须全局稳定：正文里的 `[N]` 要能确定性映射回真实论文，才能程序化拼装参考文献表。
    旧行为是「每次渲染从 1 重新计数」——第 2 节的 `[3]` 和第 5 节的 `[3]` 指向不同论文，
    参考表根本无法装配。改用全局编号后，同一篇论文在任何章节的渲染里编号都一样。

    order_pool 的排序是确定性的，所以同一份 pool 多次调用得到同一套编号。
    """
    return {paper_key(p): i for i, p in enumerate(order_pool(pool), 1)}


def _fill_budget(ordered: list, max_chars: int, with_abstract: bool,
                 gidx: dict | None = None) -> tuple[str, int, int]:
    """按字符预算填充，返回 (全文, 实际注入篇数, 其中有摘要的篇数)。

    gidx 给定时用全局编号渲染（正文 [N] 可回溯）；否则退回局部序号。
    """
    parts, total, n_items, n_ab = [], 0, 0, 0
    for p in ordered:
        i = (gidx or {}).get(paper_key(p))
        s = render_paper(p, i, with_abstract)
        if total + len(s) > max_chars:
            parts.append(f"…（另有 {len(ordered) - n_items - 1} 篇未展开）")
            break
        parts.append(s)
        total += len(s) + 1
        n_items += 1
        if has_abstract(p):
            n_ab += 1
    return "\n".join(parts), n_items, n_ab


def pool_budget_stats(pool: list, section: str | None = None, max_chars: int = 15000,
                      paper_idx: list | None = None) -> dict:
    """该节的「预算可用性」——让模型知道能真正进 prompt 的有几篇。

    只看 count 会误导：count 只增不减，模型看到数字涨就以为在进步；
    实际瓶颈是字符预算 + 摘要覆盖率。

    section_total / section_with_abstract 统计**归属该节**的论文，优先用 LLM 本次判定
    给出的 paper_idx（1-based），其次退回 pool 里的 sections 归属字段。

    ⚠️ 为什么不只用 sections 字段：归属是重排/大纲**自己**的产物，第一次调用时
    pool 里还没有任何归属信息，全部会算成 0/0，导致收尾判据误报「全部章节未达标」。
    """
    pool = [p for p in (pool or []) if isinstance(p, dict)]
    if paper_idx:
        own = [pool[i - 1] for i in paper_idx if isinstance(i, int) and 0 < i <= len(pool)]
    elif section:
        own = [p for p in pool if section in (p.get("sections") or [])]
    else:
        own = pool
    _, n_items, n_ab = _fill_budget(order_pool(pool, section), max_chars, True,
                                    gidx=global_index(pool))
    return {"injectable": n_items,
            "injectable_with_abstract": n_ab,
            "section_total": len(own),
            "section_with_abstract": sum(1 for p in own if has_abstract(p))}


REF_HEADING = "## 参考文献"


_AUTHOR_CAP = {          # 按引用体例规定的作者数上限（超出则用省略标记）
    "GB/T 7714": (3, "，等"),      # 国标：>3 位取前 3 + 等
    "IEEE": (6, ", et al."),       # IEEE：>6 位取前 6 + et al.
    "APA": (19, ", … "),           # APA 7th：>20 位取前 19 + … + 末位
    "MLA": (2, ", et al."),
}


def format_authors(authors: str, style: str = "GB/T 7714") -> str:
    """按引用体例截断作者列表。

    为什么必须截：检索源（openalex/crossref）返回的是**完整**作者列表，实测池内
    214 篇里 88 篇的作者字段超过 60 字符（最长一条光作者就 300+ 字符，20 个作者全列）。
    实测参考段 8,285 字里，光作者名就占大头——按 GB/T 7714 截断后可省 78%。
    这既是体积问题，**更是规范问题**：GB/T 7714 规定作者超过 3 位只列前 3 位加「等」。
    """
    a = str(authors or "").strip().rstrip(".")
    if not a:
        return ""
    parts = [x.strip() for x in re.split(r"[;；]\s*|\s*,\s*(?=[A-Z一-鿿])", a) if x.strip()]
    if len(parts) <= 1:
        return a
    cap, mark = _AUTHOR_CAP.get(style, _AUTHOR_CAP["GB/T 7714"])
    if len(parts) <= cap:
        return a
    if style == "APA":                      # APA：前 19 + … + 末位
        return ", ".join(parts[:cap]) + mark + parts[-1]
    return "; ".join(parts[:cap]) + mark    # 其余：前 N + 省略标记


def format_reference(n: int, p: dict, style: str = "GB/T 7714") -> str:
    """用池内**真实元数据**格式化一条参考文献。缺哪个字段就省略哪个，绝不编造。"""
    bits = []
    authors = format_authors(p.get("authors"), style)
    year = str(p.get("published_date") or p.get("updated_date") or "")[:4]
    title = str(p.get("title") or "").strip().rstrip(".")
    if authors:
        bits.append(f"{authors}.")
    if year.isdigit():
        bits.append(f"({year}).")
    if title:
        bits.append(title if title.endswith((".", "?", "!")) else f"{title}.")
    doi = str(p.get("doi") or "").strip()
    url = str(p.get("url") or "").strip()
    src = str(p.get("source") or "").strip()
    if doi:
        bits.append(f"DOI: {doi}")
    elif url:
        bits.append(url)
    elif src:
        bits.append(f"来源: {src}")
    return f"[{n}] " + " ".join(bits) if bits else f"[{n}] （元数据缺失）"


def _strip_model_references(draft: str) -> str:
    """删掉草稿里模型自己写的参考文献段——我们会在末尾追加权威版本，避免重复。

    ⚠️ 判定依据是**内容**不是位置。曾用「标题落在文末 40% 内」这个启发式，结果：
    参考段一多（实测占全文 45%），标题就落到 55% 处、被判在范围外 → 没剥掉 →
    同一份参考表被追加两遍。现在改成「标题后面确实跟着 `[N]` 条目」才认定是参考段，
    这样无论参考段占多大比例都能正确识别，而且正文里偶然提到「参考文献」不会误伤。
    """
    for m in reversed(list(re.finditer(r"(?m)^\s*#{0,4}\s*(?:参考文献|References)\b", draft))):
        if re.search(r"(?m)^\s*\[\d{1,3}\]", draft[m.end():]):
            return draft[:m.start()].rstrip()
    return draft


def build_references(draft: str, pool: list, style: str = "GB/T 7714") -> tuple[str, dict]:
    """扫描正文的 `[N]` 标记，用 paper_pool 的真实元数据拼装参考文献表。

    为什么不交给 LLM 写：这是本项目最大的幻觉来源——实测正文有 148 处 `[N]` 引用、
    参考表 0 条，所有引用都无法查证。池里已有真实标题/作者/年份/DOI，程序拼装
    既准确又免费，而且**结构上不可能编造**。

    返回 (追加了参考表的完整稿件, 校验报告)。报告里 `unmatched` 是正文引用了、
    但池子里找不到对应论文的编号——这些正是**疑似模型凭空编的引用**，必须暴露出来。
    """
    pool = [p for p in (pool or []) if isinstance(p, dict)]
    body = _strip_model_references(draft or "")
    if not body.strip():
        return draft or "", {"refs": 0, "unmatched": [], "pool": len(pool)}

    gidx = global_index(pool)                       # paper_key -> 全局编号
    by_num = {v: k for k, v in gidx.items()}        # 全局编号 -> paper_key
    key2paper = {paper_key(p): p for p in pool}

    cited = sorted({int(x) for x in re.findall(r"\[(\d{1,3})\]", body)})
    lines, unmatched = [], []
    for n in cited:
        p = key2paper.get(by_num.get(n, ""))
        if p is None:
            # 正文引用了池里不存在的编号 → 疑似幻觉引用，列出来而不是默默丢弃
            unmatched.append(n)
            continue
        lines.append(format_reference(n, p, style))

    if not lines:
        return body, {"refs": 0, "unmatched": unmatched, "pool": len(pool)}

    full = body + f"\n\n---\n\n{REF_HEADING}\n\n" + "\n".join(lines) + "\n"
    if unmatched:
        full += (f"\n> 说明：正文中的 {len(unmatched)} 个引用编号"
                 f"（{', '.join(f'[{n}]' for n in unmatched[:10])}）"
                 f"在所检索到的文献中没有对应记录，未列入上表。\n")
    report = {"refs": len(lines), "unmatched": unmatched, "pool": len(pool),
              "cited": len(cited), "style": style,
              "ref_chars": sum(len(x) + 1 for x in lines),
              "distinct_unmatched": len(unmatched)}
    print(f"--> [Refs] 参考文献已按真实元数据拼装：正文引用 {len(cited)} 个编号 → "
          f"{len(lines)} 条真实条目"
          + (f"，⚠️ 另有 {len(unmatched)} 个编号在池中无对应记录 {unmatched[:8]}" if unmatched else ""))
    return full, report


def render_paper(p: dict, idx: int | None = None, with_abstract: bool = True) -> str:
    head = f"[{idx}] " if idx is not None else ""
    title = p.get("title") or "(无标题)"
    meta = []
    if p.get("authors"):
        meta.append(str(p["authors"])[:80])
    if p.get("published_date"):
        meta.append(str(p["published_date"])[:4])
    if p.get("source"):
        meta.append(str(p["source"]))
    pid = p.get("paper_id") or p.get("doi") or ""
    if pid:
        meta.append(f"id={pid}")
    line = f"{head}{title}" + (f"（{' · '.join(meta)}）" if meta else "")
    if with_abstract:
        ab = _clean_text(p.get("abstract") or p.get("raw") or "")
        if ab:
            line += f"\n    摘要：{ab[:400]}"
    return line


def _norm_card_title(t: str) -> str:
    """标题归一化（用于「卡片 ↔ 论文池」匹配）：小写 + 去空白/标点。"""
    import unicodedata
    s = unicodedata.normalize("NFKC", t or "").lower()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s)


def _render_unread_line(p: dict, idx: int | None = None) -> str:
    """未精读论文的写作材料行：作者 + 标题 + 方法提示（摘要首句）。

    设计（2026-10 用户决策）：未精读论文统一按「XX等人提出了XXX，通过XXX方法
    解决XXX」一句话提及，**不展开实验数据/数字结论**——所以这里刻意不渲染摘要
    中的实验数据，只留方法提示，杜绝 writer 引用未读论文的数字。

    保留编号 `[N]`：综述引用池内未精读论文是正常学术实践（摘要级引用）；
    评审侧把这类引用标记为**摘要级弱证据**进低置信清单交用户审核，而不是砍编号。
    """
    head = f"[{idx}] " if idx is not None else ""
    title = p.get("title") or "(无标题)"
    meta = []
    if p.get("authors"):
        meta.append(str(p["authors"])[:60])
    if p.get("published_date"):
        meta.append(str(p["published_date"])[:4])
    ab = _clean_text(p.get("abstract") or p.get("raw") or "")
    first = ""
    if ab:
        first = re.split(r"[。．.!?；;]", ab)[0].strip()
    line = f"{head}{title}" + (f"（{' · '.join(meta)}）" if meta else "")
    line += "【未精读·仅摘要】"
    if first:
        line += f"\n    方法提示（摘要首句）：{first[:150]}"
    return line


def render_section_candidates(pool: list, cands: dict, section: str,
                              max_chars: int = 15000, card_only: bool = False,
                              cards: list | None = None) -> str:
    """用**该节的重排候选**渲染写作材料——按相关性分数排序，而非按素材可用性。

    这是重排真正产生价值的地方：旧做法是「本节优先 + 其余按摘要长度追加」，
    本节之外的补充材料与本节无关；现在补充材料也是**针对本节重排出来的高分论文**。

    ⚠️ 编号必须用 global_index —— 正文的 `[N]` 要能映射回真实论文才能拼参考表。

    card_only=True（证据约束，2026-10 升级为**分级材料**）：
      · **精读卡论文**：卡片全文一并注入 → 可编号引用、可扩展（方法/实验数据/局限）
      · **未精读论文**（该节候选里没有卡片）：仅作者+方法提示行 → 只能按
        「XX等人提出了XXX，通过XXX解决XXX」一句话提及，禁止展开实验数据
      · 该节无候选（引言/结论等 section="" 场景）→ 退化为「全部卡片论文」，
        **不再回退到全池摘要渲染**（旧实现会放行无卡片论文 → writer 敢引 → 评审大砍）
    """
    pool = [p for p in (pool or []) if isinstance(p, dict)]
    info = ((cands or {}).get("sections") or {}).get(section) or {}
    cs = info.get("candidates") or []

    by_key = {paper_key(p): p for p in pool}
    gidx = global_index(pool)
    card_by_title = {_norm_card_title(c.get("title", "")): c for c in (cards or [])}

    if card_only:
        ordered = [by_key[c["key"]] for c in cs if c.get("key") in by_key]
        if not ordered:
            # 无该节候选（引言/结论 section=""）→ 用全部「有卡片」论文，不分节
            ordered = [p for p in pool
                       if _norm_card_title(p.get("title", "")) in card_by_title]
        if not ordered:
            return ("（本节暂无精读卡片：请仅基于已有知识综述背景，"
                    "正文**不要**给出任何引用编号；待补充精读后再写。）")
        read_parts, unread_parts, total = [], [], 0
        for p in ordered:
            _k = _norm_card_title(p.get("title", ""))
            c = card_by_title.get(_k)
            if c:
                s = render_paper(p, gidx.get(paper_key(p)), True)
                s += f"\n【精读卡片】{c.get('card', '')}"
                bucket = read_parts
            else:
                s = _render_unread_line(p, gidx.get(paper_key(p)))
                bucket = unread_parts
            if total + len(s) > max_chars and read_parts:
                break
            bucket.append(s)
            total += len(s) + 1
        if not read_parts and not unread_parts:
            return ("（本节材料按证据约束为空：请基于已有知识综述背景，"
                    "正文**不要**给出任何引用编号；待补充精读后再写。）")
        head = (f"（本节材料按证据约束分级：{len(read_parts)} 篇精读"
                f"（带原文证据卡片，可展开方法/实验数据）+ {len(unread_parts)} 篇未精读"
                f"（仅摘要级：可编号引用，但只能『XX等人提出了XXX，通过XXX解决XXX』"
                f"一句话提及，禁止展开实验数据））\n")
        return head + "\n".join(read_parts + unread_parts)

    # ── 非 card_only：按相关性渲染（摘要全量）──
    if not cs:
        return render_pool(pool, section=section, max_chars=max_chars)
    ordered = [by_key[c["key"]] for c in cs if c.get("key") in by_key]
    if not ordered:
        return render_pool(pool, section=section, max_chars=max_chars)
    parts, total, n_items = [], 0, 0
    for p in ordered:
        s = render_paper(p, gidx.get(paper_key(p)), True)
        if total + len(s) > max_chars:
            break
        parts.append(s)
        total += len(s) + 1
        n_items += 1
    if not parts:
        return render_pool(pool, section=section, max_chars=max_chars)
    head = (f"（本节材料按 Cross-Encoder 相关性重排，取自全池 {len(pool)} 篇的 top-{len(ordered)}）\n")
    return head + "\n".join(parts)


def render_pool(pool: list, section: str | None = None, max_chars: int = 15000,
                with_abstract: bool = True) -> str:
    """把 paper_pool 渲染成写作 / 评委可用的「可引用材料」文本。

    section 给定时优先列该节归属的论文，其余追加在后；组内按素材可用性排序（见 order_pool）。
    """
    text, _, _ = _fill_budget(order_pool(pool, section), max_chars, with_abstract,
                              gidx=global_index(pool))
    return text


# ==========================================
# 1. MCP 工具包装：限流 + 入池 + 只回摘要
# ==========================================

_sem_lock = threading.Lock()
_semaphores: dict[int, asyncio.Semaphore] = {}


def _get_semaphore() -> asyncio.Semaphore:
    """每个事件循环一个 Semaphore（asyncio 原语不能跨 loop 复用）。"""
    loop = asyncio.get_running_loop()
    with _sem_lock:
        sem = _semaphores.get(id(loop))
        if sem is None:
            sem = asyncio.Semaphore(TOOL_CONCURRENCY)
            _semaphores[id(loop)] = sem
    return sem


def is_paper_search_tool(name: str) -> bool:
    return name in PAPER_SEARCH_TOOL_WHITELIST


def is_web_search_tool(name: str) -> bool:
    return any(name.startswith(pfx) for pfx in WEB_SEARCH_TOOL_PREFIXES)


def _compact_paper_summary(new: list[dict], dup: int, tool_name: str, query: str,
                           alts: list[str] | None = None) -> str:
    lines = [f"[{tool_name}] 查询「{query[:60]}」：新增 {len(new)} 篇（去重 {dup} 篇），已存入 paper_pool。"]
    for p in new[:12]:
        lines.append("  - " + render_paper(p, with_abstract=False)[:160])
    if len(new) > 12:
        lines.append(f"  …另 {len(new) - 12} 篇")
    if not new:
        # 换源建议用**运行期健康状态**算出来的，不写死源名
        # （写死过一次「换 search_arxiv/search_europepmc」，结果 arxiv 自己就是最不稳的那个）
        hint = f"可换源：{'、'.join(alts)}" if alts else "可换检索词/改变表述"
        lines.append(f"  （无新结果——可换检索词，或{hint}）")
    return "\n".join(lines)


# ==========================================
# 检索源健康度（per-session 运行期状态）
# ==========================================
# 动因（实测）：某源连续 9 次返回 0 条，模型仍反复重试它——
#   因为 MCP 的**各平台客户端自己把 requests 异常吞了**
#   （crossref.py 就是 `except requests.RequestException: return []`），
#   于是 CallToolResult 恒为 `isError=False, structuredContent={'result': []}`，
#   **「源挂了」和「真的没结果」在协议层完全同形**（_probe_mcp_shapes.py 实测确认）。
# 参考成熟做法：research-mcp「主源被限流 → 自动降级备用源」；
#   paper-chaser-mcp「payload["degraded"]=True」——降级必须显式标记，不能静默返回空。
#
# 这里记录**运行时信号**（不是写死的知识）：某源在当前会话里连续失败就熔断一段时间，
# 并明确告诉模型「哪些源还健康」，让模型自己换源——保持「结构交给模型」的原则。
# 熔断是**心跳检测**：冷却期内不再打该源，到期自动放行（下一轮若仍失败会再次熔断，
# 相当于按 SRC_COOLDOWN_SEC 周期探活），不需要额外的定时任务。
_HEALTH_LOCK = threading.Lock()
_HEALTH: dict[str, dict] = {}      # session_id → {source: {"streak", "ok", "fail", "down_until"}}
_MAX_HEALTH_SESSIONS = 200         # 防长跑进程内存无界增长（同 tracing 的做法）


def _health_entry(thread_id: str, source: str) -> dict:
    with _HEALTH_LOCK:
        if thread_id not in _HEALTH and len(_HEALTH) >= _MAX_HEALTH_SESSIONS:
            _HEALTH.pop(next(iter(_HEALTH)))       # 丢最早那个会话
        return _HEALTH.setdefault(thread_id, {}).setdefault(
            source, {"streak": 0, "ok": 0, "fail": 0, "down_until": 0.0})


def _src_is_down(thread_id: str, source: str) -> float:
    """该源是否处于熔断中；返回剩余冷却秒数（0 = 可用）。"""
    h = _health_entry(thread_id, source)
    left = h.get("down_until", 0.0) - time.time()
    return left if left > 0 else 0.0


def _src_mark(thread_id: str, source: str, ok: bool) -> bool:
    """记录一次成败。返回 True 表示**本次刚触发熔断**。"""
    h = _health_entry(thread_id, source)
    with _HEALTH_LOCK:
        if ok:
            h["streak"] = 0
            h["ok"] += 1
            return False
        h["fail"] += 1
        h["streak"] += 1
        if h["streak"] >= SRC_FAIL_STREAK and h.get("down_until", 0.0) < time.time():
            h["down_until"] = time.time() + SRC_COOLDOWN_SEC
            return True
    return False


def _healthy_sources(thread_id: str, exclude: str = "") -> list[str]:
    """当前会话里**未熔断**的论文检索源，按实测可靠性排序（白名单顺序即实测序）。

    返回的顺序有意义：它是回给模型的「换源建议」的先后次序——
    首选是实测最稳最快的源（openalex），而不是随手一个。
    """
    return [s for s in PAPER_SEARCH_TOOL_WHITELIST
            if s != exclude and _src_is_down(thread_id, s) <= 0]


def wrap_mcp_tool(orig: BaseTool) -> BaseTool:
    """把一个 MCP 工具包成：熔断检查 → 超时+退避重试 → 论文入 paper_pool → 摘要回模型。

    相比旧版加了四件事（都参考成熟项目，阈值由本机实测确定）：
      1. 熔断/心跳——源挂了就不必再打，直接告诉模型换哪个源（冷却到期自动放行 = 周期探活）
      2. 超时+退避重试——单次 45s 上限（砍掉实测见过的那条 70s 长尾），
         空结果/异常时按 2s/6s 退避重试（协议层无法区分空与失败，故一律重试）
      3. 降级显式标记——重试后仍空，明确标注「重试了几次、源是否已熔断」，不静默返回空
      4. 结果缓存（L3）——命中即返回，跨会话有效
    """
    name = orig.name
    is_paper = is_paper_search_tool(name)

    def _payload_of(raw) -> int:
        """结果条数（论文源按解析后的论文数算，其它按原始长度）。"""
        if is_paper:
            return len(parse_papers(raw, source_hint=name.replace("search_", "")))
        return len(raw or [])

    async def _run(tool_call_id: str, state: dict, **kwargs) -> Command:
        thread_id = (state or {}).get("session_id", "")
        query = str(kwargs.get("query") or kwargs.get("q") or "")

        # ---- ① 熔断检查：源已挂就别浪费调用 ----
        if is_paper:
            left = _src_is_down(thread_id, name)
            if left > 0:
                alts = _healthy_sources(thread_id, exclude=name)
                return Command(update={"messages": [ToolMessage(
                    content=(f"[{name}] ⏸️ 该源已熔断（连续 {SRC_FAIL_STREAK} 次无结果），"
                             f"{left:.0f}s 后自动恢复。\n"
                             f"👉 请改用可用源：{'、'.join(alts) or '（全部熔断，请稍后重试）'}"),
                    tool_call_id=tool_call_id)]})

        sem = _get_semaphore()
        # ---- L3 检索缓存 ----
        # ⚠️ key **不含 stamp**：外部检索源（arxiv/crossref/…）的结果只依赖查询词，
        #    与我们的数据库无关。旧版把 stamp 掺进 key，导致两个荒谬后果：
        #      · 跨会话永不命中（实测命中率 0%）
        #      · 删除任意一个文档就全部失效
        #    改为只靠 TTL 失效（源挂了时也才有旧结果可兜底）。
        max_n = kwargs.get("max_results") or kwargs.get("top_k") or kwargs.get("limit") or 0
        ckey = (name, query, max_n)
        cached_raw = rcache.search_get(*ckey) if query else None
        # 兜底初始化：缓存命中时下面那个分支不会跑，而降级文案要引用这两个变量
        attempts, last_err = 0, ""
        if cached_raw is not None:
            raw = cached_raw
            with tracing.span("tool", name, thread_id, query=query[:80], cache_hit=True):
                pass
        else:
            # ---- ② 单次超时 + 退避重试 ----
            raw = None
            async with sem:
                with tracing.span("tool", name, thread_id, query=query[:80]) as sp:
                    for i in range(len(RETRY_BACKOFF_SEC) + 1):
                        attempts = i + 1
                        try:
                            # asyncio.wait_for 砍长尾：实测 arxiv 单次可达 70s，
                            # 会白占一个并发位（TOOL_CONCURRENCY=3）拖慢整轮检索。
                            # 已实测「在途取消 MCP 调用后会话仍可用」，取消是安全的。
                            candidate = await asyncio.wait_for(
                                orig.ainvoke(kwargs), timeout=MCP_CALL_TIMEOUT_SEC)
                        except asyncio.TimeoutError:
                            candidate, last_err = None, f"超时(>{MCP_CALL_TIMEOUT_SEC:.0f}s)"
                        except Exception as e:
                            candidate, last_err = None, f"{type(e).__name__}: {str(e)[:120]}"
                        if candidate is not None and _payload_of(candidate) > 0:
                            raw = candidate
                            break
                        if i < len(RETRY_BACKOFF_SEC):
                            await asyncio.sleep(RETRY_BACKOFF_SEC[i])
                    sp["retries"] = attempts - 1
                    if last_err:
                        sp["error"] = last_err
            # ---- ③ 健康度记账（心跳）----
            if is_paper:
                just_tripped = _src_mark(thread_id, name, raw is not None)
                if just_tripped:
                    print(f"--> [Health] ⛔ {name} 连续 {SRC_FAIL_STREAK} 次无结果 → "
                          f"熔断 {SRC_COOLDOWN_SEC}s")
            if raw is not None and query:
                try:
                    rcache.search_set(*ckey, raw if isinstance(raw, str)
                                      else json.dumps(raw, ensure_ascii=False, default=str))
                except Exception:
                    pass

        if is_paper:
            papers = parse_papers(raw, source_hint=name.replace("search_", "")) if raw else []
            existing = {paper_key(p) for p in (state or {}).get("paper_pool", []) or []}
            new = [p for p in papers if paper_key(p) not in existing]
            dup = len(papers) - len(new)
            msg = _compact_paper_summary(new, dup, name, query,
                                         alts=_healthy_sources(thread_id, exclude=name))
            # ---- ④ 降级显式标记（学 paper-chaser-mcp 的 payload["degraded"]）----
            if not papers:
                alts = _healthy_sources(thread_id, exclude=name)
                left = _src_is_down(thread_id, name)
                why = f"（{last_err}）" if last_err else ""
                msg += (f"\n  ⚠️ [{name}] 已重试 {attempts} 次仍无结果{why}"
                        + (f"，该源已熔断 {left:.0f}s" if left > 0 else "，该源本次可能不可达")
                        + f"。\n  👉 可用源：{'、'.join(alts) or '（全部熔断，稍后重试）'}")
            return Command(update={
                "paper_pool": papers,
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
            })

        # web / 其它：截断，保留 URL
        text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        if len(text) > MAX_TOOL_RESULT_CHARS * 4:
            urls = re.findall(r"https?://[^\s\"'<>\]\)]+", text)[:15]
            text = text[:MAX_TOOL_RESULT_CHARS * 4] + "\n…（已截断）\n来源 URL：\n" + "\n".join(urls)
        return Command(update={"messages": [ToolMessage(content=text, tool_call_id=tool_call_id)]})

    # 保留原工具的参数 schema，只追加注入参数
    schema = orig.args_schema
    fields: dict = {}
    if schema is not None:
        try:
            from pydantic import create_model
            base_fields = getattr(schema, "model_fields", None)
            if base_fields:
                for fname, finfo in base_fields.items():
                    fields[fname] = (finfo.annotation, finfo)
            else:  # JSON-schema dict
                props = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
                req = set((schema.get("required") or []) if isinstance(schema, dict) else [])
                for fname, prop in props.items():
                    typ = {"string": str, "integer": int, "number": float, "boolean": bool,
                           "array": list, "object": dict}.get(prop.get("type"), Any)
                    default = ... if fname in req else prop.get("default", None)
                    fields[fname] = (typ if fname in req else typ | None,
                                     Field(default, description=prop.get("description", "")))
            fields["tool_call_id"] = (Annotated[str, InjectedToolCallId], Field(default=""))
            fields["state"] = (Annotated[dict, InjectedState], Field(default=None))
            new_schema = create_model(f"{name}_args", **fields)
        except Exception:
            new_schema = None
    else:
        new_schema = None

    desc = (orig.description or "").strip()
    if is_paper:
        desc += ("\n（结果自动存入 paper_pool 并去重；返回给你的只是新增论文的标题摘要。"
                 "建议 max_results=15；同一主题多换几个关键词/来源能提升覆盖度。）")
    return StructuredTool.from_function(
        coroutine=_run, name=name, description=desc[:1024],
        args_schema=new_schema,
    )


# ==========================================
# 2. 本地工具（返回 Command 直接更新 state）
# ==========================================

class OutlineSection(BaseModel):
    section: str = Field(description="章节名（中文）")
    focus: str = Field(default="", description="该节聚焦什么，一句话")
    # ⚠️ queries 必须带**场景/领域限定词**，只写泛方法词会检索失败。
    # 实测（主题「风电功率预测中的时序预测方法」，同一套检索+重排，只改检索词）：
    #   只搜 "time series forecasting"      → 严格 P@10 = 0.00（召回的全是泛时序论文，无一讲风电）
    #   带场景词 "wind power forecasting …" → 严格 P@10 = 1.00（10/10 命中目标领域）
    #   用用户原话字面检索                    → 严格 P@10 = 0.30
    queries: list[str] = Field(
        default_factory=list,
        description=(
            "2-3 个**英文**检索关键词。**每个关键词都必须带上用户所指的场景/领域限定词**，"
            "不能只写泛化的方法名——实测只写泛方法词会让窄领域检索命中率降为 0。\n"
            "✗ 反例（太泛，只写领域最泛的上位方法词，会召回大量无关论文）："
            "如时序领域的 \"time series forecasting\"（示例，领域变通）\n"
            "✓ 正例（场景/领域限定词 + 方法）：如时序领域的 \"wind power forecasting\" / "
            "\"wind speed prediction LSTM\"（示例，领域变通）\n"
            "若用户没指定具体场景，则用主题里最有区分度的领域词，不要退化成最泛的上位词。"
        ))


def make_survey_evolution(llm):
    """演进脉络调研工具工厂：Tavily 搜主题演进 → LLM 梳理 3-4 个演进方向 → 存 state。

    综述流程第一步（v1.2.6）：先摸清「该方向是怎么演进的」，再按演进方向定向检索有全文论文，
    避免大纲凭空列世代、列了没全文支撑的空节。
    """
    _SYS = (
        "你是科研综述策划专家。下面是用 Tavily 网页搜索「{topic}」领域研究演进脉络得到的片段。\n"
        "请梳理该领域的**演进脉络**，列出 3-4 个演进方向/技术世代（由远及近）。每个方向：\n"
        "1) name：方向名。示例（**仅示意「世代」的含义，具体领域必须变通、严禁照抄**）：\n"
        "   时序=经典统计方法→深度学习初期（RNN/LSTM）→Transformer 时代→后 Transformer 改进；\n"
        "   视觉=手工特征/传统方法→CNN→注意力/Transformer→基础模型/扩散等；\n"
        "   医药=传统统计/评分系统→传统机器学习→深度学习→LLM/多模态等。\n"
        "2) era：大致时间段（年份区间或描述）。\n"
        "3) methods：代表性方法 2-4 个。\n"
        "4) search_queries：2-3 个英文检索词（**必须带领域限定词**，用于定向检索该方向论文，"
        "不要只写泛方法词：如时序领域写 wind power forecasting 而不是 time series forecasting）。\n"
        "5) description：一句话说明该方向解决什么问题、有什么局限（供大纲 focus 使用）。\n"
        "只输出 JSON 数组，不要输出其他任何文字。Tavily 片段不足时基于领域常识补全，"
        "并在对应 description 末尾标注『（基于常识，待文献验证）』。"
    )

    @tool
    async def survey_evolution(
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
    ) -> Command:
        """演进脉络调研（综述第一步）：Tavily 搜索该主题领域「如何演进」，梳理出 3-4 个演进方向
        （技术世代）存入 state.evolution_directions。

        ★ **本工具只负责探索演进方向，不要在调用后立即检索论文**（v1.2.13 流程修订）：
        正确顺序 = survey_evolution（本步）→ make_outline（按方向写大纲，方法论默认 3 节、
        最多 4 节）→ **按大纲方法节定向检索**（每节 1 个 query，arXiv/EuropePMC/CORE 三源并发，
        一轮、禁止补检）→ read_sections（精排 + 全文制卡，有全文 <5 篇的节自动删除或并入相邻节）。
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        from ResearcherAgent.retrieval.fulltext import tavily_search
        from ResearcherAgent.innovate.innovation_tools import _extract_json_block

        topic = state.get("topic") or state.get("user_request") or ""
        if not topic:
            return Command(update={"messages": [ToolMessage(
                content="缺少主题：请先提供研究方向（topic）。", tool_call_id=tool_call_id,
                status="error")]})
        # 幂等：已调研过则直接复用，避免重复扣 Tavily 配额
        if state.get("evolution_directions"):
            dirs = state["evolution_directions"]
            names = "、".join(d.get("name", "?") for d in dirs)
            return Command(update={"messages": [ToolMessage(
                content=f"演进方向已存在（{names}），无需重复调研；"
                        "直接调用 make_outline 按这些方向写大纲，再按方法节定向检索。",
                tool_call_id=tool_call_id)]})
        hits = tavily_search(f"{topic} research evolution history timeline survey", max_results=8)
        if not hits:
            return Command(update={"messages": [ToolMessage(
                content="Tavily 演进调研失败（网络或配额）。可重试一次；若持续失败，"
                        "可跳过本步直接 make_outline（按领域惯例写演进式大纲），"
                        "再按方法节定向检索有全文论文。",
                tool_call_id=tool_call_id, status="error")]})
        evidence = "\n".join(
            f"- {h.get('title', '')}: {h.get('content', '')[:300]}" for h in hits[:6])
        try:
            resp = await llm.ainvoke([SystemMessage(content=_SYS.format(topic=topic)),
                                      HumanMessage(content=evidence)])
            dirs = _extract_json_block(str(resp.content))
        except Exception as e:
            return Command(update={"messages": [ToolMessage(
                content=f"演进方向梳理失败：{type(e).__name__}: {str(e)[:80]}。可重试或跳过本步。",
                tool_call_id=tool_call_id, status="error")]})
        if not isinstance(dirs, list) or not dirs:
            return Command(update={"messages": [ToolMessage(
                content="LLM 未返回合法的演进方向 JSON 数组（需 3-4 个对象，每个含 "
                        "name/era/methods/search_queries/description），请重试。",
                tool_call_id=tool_call_id, status="error")]})
        dirs = [d for d in dirs if isinstance(d, dict) and d.get("name")][:4]
        lines = [f"{i+1}. {d.get('name')}（{d.get('era') or '?'}）："
                 f"{d.get('description') or ''}｜检索词："
                 f"{'、'.join(d.get('search_queries') or []) or '（未给检索词）'}"
                 for i, d in enumerate(dirs)]
        msg = (f"演进脉络调研完成，共 {len(dirs)} 个演进方向（由远及近）：\n" + "\n".join(lines) +
               "\n\n下一步：**直接调用 make_outline** 按这些演进方向写大纲（方法论默认 3 节、"
               "最多 4 节；大纲无需预先检索建池，某方法节检索后有全文论文 <5 篇会在精读时"
               "自动删除或并入相邻节）。大纲确定后**按每个方法节定向检索一次**"
               "（每节 1 个 query，arXiv/EuropePMC/CORE 三源并发，一轮、禁止补检），"
               "然后调用 read_sections 精读制卡。")
        return Command(update={
            "evolution_directions": dirs,
            "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
        })

    return survey_evolution


@tool
def make_outline(
    sections: list[OutlineSection],
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """设定/更新综述大纲（5-6 节）。大纲同时是检索计划和写作结构。

    ★ 结构契约（确定性后校验，不满足直接报错让模型重给）：
    - **节数 5-6**：引言 1 节 + 技术迭代 3-4 节 + 展望总结 1 节；
      **方法论最好只列三个，除非实在很多，允许有四节大纲**；
    - **唯一引言**：含「引言」的节只能 1 个，禁止「1 引言」与「1 引言与问题定义」并存；
    - **唯一展望**：含「展望/结论/总结/未来」的节只能 1 个，且必须有；
    - **节名去重**：不允许两个同名节。

    ★ 技术迭代节必须按**该领域的技术世代由浅入深**组织（演进式，而非主题罗列）。
    世代序列**由 LLM 依当前主题领域自适应**，严禁照抄示例：
    - few-shot 示例（仅示意「世代」的含义，具体领域必须变通）：
      时序 = 经典统计 → 深度学习初期（RNN/LSTM）→ Transformer → 后 Transformer 改进；
      视觉 = 手工特征/传统方法 → CNN → 注意力/Transformer → 基础模型/扩散等；
      医药 = 传统统计/评分系统 → 传统 ML → 深度学习 → LLM/多模态等。
    - 判断准则：写清「承接上一代解决了什么 + 本代**方法族**与待回答问题
      （供检索/定向抽取定位）+ 局限引出下一代」；近 2-3 年的最新世代节是
      「热门研究方向」的证据源，检索时优先保证该节有近 2-3 年文献。
    - **若 state 里已有 evolution_directions（survey_evolution 的产物），
      方法节应尽量与其对应**（可合并相近方向，不要无依据另起世代）；
      若池内某方向有全文论文 <5 篇，删除该节或并入相邻节。

    ★ **大纲基于 survey_evolution 的演进方向定节，不要求预先检索建池（v1.2.13 流程修订）**：
    论文检索发生在大纲确定之后（按方法节定向检索一轮）。因此：
    - 大纲直接按演进方向/技术世代组织（由远及近），**不必先攒 paper_pool 再定节**；
    - 若某方法节检索后有全文论文 <5 篇，read_sections 会自动删除该节或并入相邻节——
      大纲里可以列出，但**禁止为凑章节预先大规模检索建池**；
    - 严禁按领域惯例硬列章节——世代划分由 LLM 依当前主题领域自适应（few-shot 引导，变通）。

    首次规划时给出完整大纲；用户要求修改时传入**修改后的完整大纲**（保留未提及的章节，
    只增删/调整用户要求的部分）。

    ⚠️ **每节的 queries 必须带场景/领域限定词**。这一步决定后续所有检索的质量：
    只写泛方法词（如 time series forecasting）会召回一大堆无关领域的论文，
    实测窄领域命中率直接降为 0；带上场景词（如 wind power forecasting）才能命中目标文献。
    """

    outline = [s.model_dump() if hasattr(s, "model_dump") else dict(s) for s in sections]
    outline = [o for o in outline if o.get("section")]
    if not outline:
        return Command(update={"messages": [ToolMessage(
            content="大纲为空，请至少给出 1 节。", tool_call_id=tool_call_id, status="error")]})
    # ---- 确定性后校验：演进式结构契约（双引言/缺展望/节数越界/重名 → 报错重给） ----
    names = [o["section"] for o in outline]
    intro_hits = [n for n in names if "引言" in n]
    outlook_hits = [n for n in names if any(k in n for k in ("展望", "结论", "总结", "未来"))]
    dup = {n for n in names if names.count(n) > 1}
    errs = []
    if len(intro_hits) != 1:
        errs.append(f"引言节必须且只能有 1 个（当前 {len(intro_hits)} 个：{intro_hits or '无'}）——"
                    f"不要同时输出「1 引言」和「1 引言与问题定义」之类的重复节")
    if len(outlook_hits) != 1:
        errs.append(f"展望/结论总结节必须有且只有 1 个（当前 {len(outlook_hits)} 个）")
    if not (5 <= len(names) <= 6):
        errs.append(f"总节数必须为 5-6 节（引言 1 + 技术迭代 3-4 + 展望 1，当前 {len(names)} 节）——"
                    "**方法论最好只列三个，除非实在很多，允许有四节大纲**")
    if dup:
        errs.append(f"节名重复：{sorted(dup)}——每个技术世代只能对应一节，请合并")
    if errs:
        return Command(update={"messages": [ToolMessage(
            content="大纲不满足演进式结构契约：\n" + "\n".join(f"- {e}" for e in errs)
                    + "\n\n请按以下结构重新规划（共 5-6 节）：\n"
                    "1) 引言（1 节，含问题定义/范围/路线图）\n"
                    "2) 技术迭代 3-4 节（**最好只列三个，除非实在很多，允许有四节大纲**）："
                    "按**当前主题领域的技术世代由浅入深**——"
                    "世代序列由你依据该领域自行划分（few-shot：时序=经典统计→RNN/LSTM→Transformer→"
                    "后Transformer；视觉=手工特征→CNN→注意力→基础模型——仅示例，必须按当前领域变通，"
                    "严禁照抄）；每节 focus 写清「承接上一代 + 代表方法与机制 + 局限引出下一代」\n"
                    "3) 展望总结（1 节）",
            tool_call_id=tool_call_id, status="error")]})
    print(f"--> [Tool:make_outline] {len(outline)} 节: {names}")
    # v1.2.4/v1.2.5：返回时给模型"有全文素材分布"提示，让模型审视大纲是否硬列了空节。
    # v1.2.5 起池内论文在检索建池阶段已过滤为「有全文线索」（fulltext_hint 非空），
    # 这里统计口径 = 池内全部（fulltext_hint 兜底 _paper_id_for_fulltext 以兼容旧池）。
    pool = state.get("paper_pool", []) or []
    ft = [p for p in pool if p.get("fulltext_hint") or _paper_id_for_fulltext(p)]
    if ft:
        years = sorted({str(p.get("year") or "") for p in ft if p.get("year")})
        msg = (f"大纲已更新（" + "、".join(names) + f"）。池内论文 {len(pool)} 篇，"
               f"其中可获取全文（arXiv/PMC/OA）的 {len(ft)} 篇（年份覆盖 {'–'.join(years[:1] + years[-1:])}）。"
               f"请确认每个技术迭代节都有 **≥5 篇有全文**论文可支撑（不足 5 篇 → 删除该节或并入相邻节）；"
               f"若某方向（如经典统计时代）无全文素材，read_sections 会因候选不足自动删除该节"
               f"或并入相邻节，无需再次调整大纲。\n"
               f"⚠️ 方法论最好只列三个，除非实在很多，允许有四节大纲。\n"
               f"⚠️ **每个大纲只检索一轮，禁止任何补检**（补检浪费时间且返回的多是只有摘要的题录）："
               f"不要调用 search_* 二次检索；某节全文候选不足 5 篇 → read_sections 自动删除/合并该节。"
               f"直接进入 read_sections 精读（每节 top-10，全池确定性重排，无需 check_coverage）。")
    else:
        msg = (f"大纲已更新（" + "、".join(names) + f"）。⚠️ 池内 {len(pool)} 篇论文均无全文可达线索，"
               f"read_sections 将无法精读制卡——请先换有开放获取的源检索（如 arXiv/Europe PMC），"
               f"再调用 make_outline 按可精读论文分布重定大纲。")
    return Command(update={
        "review_outline": outline,
        "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
    })


# ==========================================
# 2b. 子主题种子检索（LLM 拆子方向 → 代表性论文种子 → 引文网络扩展）
#     主检索路径：从种子出发做引文扩展，把扩展论文并入 paper_pool。
# ==========================================

class TodoItem(BaseModel):
    task: str = Field(description="任务描述")
    status: str = Field(default="pending", description="pending | in_progress | done")


@tool
def write_todos(
    todos: list[TodoItem],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """维护你的任务清单（可变计划）。开始复杂任务前先写计划，每完成一步就更新状态。

    传入**完整**清单（覆盖式更新）。状态：pending / in_progress / done。
    """
    items = [t.model_dump() if hasattr(t, "model_dump") else dict(t) for t in todos]
    icon = {"done": "✅", "in_progress": "🔄", "pending": "⬜"}
    rendered = "\n".join(f"{icon.get(t.get('status', 'pending'), '⬜')} {t.get('task', '')}" for t in items)
    print(f"--> [Tool:write_todos] {sum(1 for t in items if t.get('status') == 'done')}/{len(items)} done")
    return Command(update={"todos": items,
                           "messages": [ToolMessage(content="任务清单已更新：\n" + rendered,
                                                    tool_call_id=tool_call_id)]})


def fetch_and_card(items: list, llm, thread_id: str = "", on_card=None,
                   on_paper=None) -> tuple[list, list]:
    """并发取结构化全文（**全篇下载**，方案 v1.2）→ **流式**分字段 Map 成卡片。
    items: [{"id","title","section"}, ...]

    v1.2.10 流式化：下载用 as_completed——每篇全文取到**立即**单篇制卡（embedding
    快），并回调 on_card(card)。read_sections 借此把卡片**边制边送 MQ**（JSON 块
    提取与后续下载并行，不再等全部制完才启动提取）。

    v1.2.15 MQ1 制卡流水线：on_paper 非空时启用——下载完成**只回调 on_paper(block)**
    （read_sections 送 MQ1 队列），不再主线程制卡（制卡由 MQ1 消费者并发做，
    关键词失败转 MQ2 LLM 兜底，制卡成功自动入 MQ3）；返回 (cards=[], failed)。
    on_paper 为空 → 旧路径（主线程制卡 + on_card 流式送 MQ3）。

    抽成独立函数，让 read_papers（模型指定 id）与 read_sections（按节自动选）
    共用同一套「取全文 + 提炼卡片」逻辑，避免两份实现漂移。
    返回 (cards, failed_ids)。
    """
    from ResearcherAgent.core.graph_nodes import _title_relevant, _map_to_cards
    from ResearcherAgent.retrieval.fulltext import (fetch_arxiv_fulltext, fetch_pmc_fulltext,
                                      fetch_oa_fulltext, fetch_core_fulltext)

    def _fetch(it: dict) -> dict | None:
        pid, title, section = it.get("id", ""), it.get("title", ""), it.get("section", "")
        with tracing.span("tool", "fulltext", thread_id, id=pid):
            if re.match(r"^PMC\d+", pid, re.I):
                t = fetch_pmc_fulltext(pid, None)
            elif pid.lower().startswith("arxiv:"):
                t = fetch_arxiv_fulltext(pid.split(":", 1)[1], None)
            elif re.match(r"^\d{4}\.\d{4,5}", pid):
                t = fetch_arxiv_fulltext(pid, None)
            elif pid.lower().startswith("oa:"):
                url = pid[3:]
                t = fetch_oa_fulltext(url, None)
                # v1.2.9：core.ac.uk/download/<id> 是**前端 PDF 端点**，Cloudflare 对服务器
                # 直连/浏览器头/Tavily 全部拦截（403）；且 download 的 id ≠ CORE work id
                # （works/{id} 实测 404）。真正有效通路是 CORE API 按论文 DOI/标题取全文
                # （E2E 已证 works/{id} 稳定返回 fullText）——用记录里的 doi/title 补一次。
                if not t and "core.ac.uk/download" in url:
                    t = fetch_core_fulltext(doi=(it.get("doi") or ""), title=(it.get("title") or ""))
            else:
                return None
        if not t or len(t) < 300:
            print(f"    ✗ {title[:40] or pid}：全文不可得")
            return None
        if title and not _title_relevant(title, t):
            print(f"    ✗ {title[:40]}：全文与标题不匹配，丢弃")
            return None
        return {"title": title or pid, "id": pid, "text": t, "section": section,
                **({"num": it.get("num")} if it.get("num") is not None else {})}

    items = [it for it in (items or []) if it.get("id")]
    if not items:
        return [], []
    # 下载并发（网络 IO 为主，DOWNLOAD_CONCURRENCY=6，保留退避）+ 流式：
    # on_paper 模式（v1.2.15 MQ1）——下载完成立即回调（送 MQ1 制卡队列），不制卡；
    # 旧路径——每篇全文就绪立即制卡（制卡确定性、embedding 快），并触发 on_card（送 MQ3）。
    cards: list = []
    blocks: list = []
    import concurrent.futures as _cf
    with ThreadPoolExecutor(max_workers=DOWNLOAD_CONCURRENCY) as ex:
        futs = {ex.submit(_fetch, it): it for it in items}
        for fut in _cf.as_completed(futs):
            b = fut.result()
            if not b:
                continue
            blocks.append(b)
            if on_paper is not None:
                try:
                    on_paper(b)
                except Exception as _e:
                    print(f"    [fulltext] on_paper 回调失败 {type(_e).__name__}: {str(_e)[:80]}")
                continue
            try:
                cs = _map_to_cards([b], llm)
            except Exception:
                cs = []
            for c in cs:
                c["section"] = b.get("section", "")
                cards.append(c)
                if on_card:
                    try:
                        on_card(c)
                    except Exception as _e:
                        print(f"    [fulltext] on_card 回调失败 {type(_e).__name__}: {str(_e)[:80]}")
    failed = [it["id"] for it in items if it["id"] not in {b["id"] for b in blocks}]
    return cards, failed


def make_read_sections(llm):
    """read_sections：**按章节**精读——每节从重排候选中取 top-N 生成卡片。

    与 read_papers 的区别：read_papers 由模型指定全局 id 列表（容易集中在少数几节），
    read_sections 按节分配名额，保证**每一节都有专属素材**。
    """

    @tool
    def read_sections(
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        sections: list[str] | None = None,
        per_section: int = CARDS_PER_SECTION,
    ) -> Command:
        """按**章节**精读：每节对池内论文做 Cross-Encoder 重排（确定性，无需 LLM），
        取前 N 篇取结构化全文并提炼成卡片（问题/方法/结果/局限），卡片带章节归属。

        sections：要精读的章节名列表；留空则读全部章节。
        per_section：每节读几篇（默认 2）。**方法节请传 10**（方案 v1.2：方法节 top-10，
        共 30-40 篇，引言/展望不单独选文）；全文通路不足的会自动跳过，不编造。
        """
        outline = state.get("review_outline", []) or []
        pool = state.get("paper_pool", []) or []
        if not outline:
            return Command(update={"messages": [ToolMessage(
                content="尚无大纲，请先调用 make_outline。",
                tool_call_id=tool_call_id, status="error")]})
        # v1.2.7：不再依赖 check_coverage（该步已从链路删除）——这里直接对池内论文
        # 做确定性重排选候选（select_section_candidates 为 Cross-Encoder 全池重排，无 LLM）。
        cands_map = select_section_candidates(pool, outline, thread_id=state.get("session_id", ""))
        cands = cands_map or {}
        # v1.2.11 节级护栏①（候选不足 → 删节或并入相邻节，禁止补检）：
        # 可精读候选 < MIN_PAPERS_PER_SECTION → 优先**并入大纲中它前面最近的方法节**
        # （候选合并到该节列表，制卡时卡片 section 直接用合并后节名 → write 归节自然映射
        # 到新大纲节，即"合并=json 大纲对应到新大纲章节"）；前面没有可并的方法节才删除。
        # **不再做第二轮补检**——补检浪费时间、返回的又多是只有摘要的题录。
        def _is_method_sec(name: str) -> bool:
            return not any(k in name for k in (
                "引言", "展望", "结论", "总结", "未来", "背景",
                "Introduction", "Conclusion", "Outlook", "Future", "Background",
                "introduction", "conclusion", "outlook", "future", "background"))

        # v1.2.18：精读节**强制以 state.review_outline 节名为准**（防节名漂移）。
        # 长 agent 循环中模型传参可能复用旧大纲节名 → 护栏①③删不到新大纲节、
        # write 归节失败、补写无素材节（忠实度受损）。只精读大纲中现存的方法节；
        # 模型传入的旧节名丢弃（打印提示），未传节名则全部方法节。
        outline_names = [o.get("section", "") for o in outline]
        _valid_method = [s for s in outline_names
                         if s in cands and _is_method_sec(s)]
        if sections:
            want = [s for s in (sections or []) if s in _valid_method]
            if want:
                print(f"--> [Tool:read_sections] ℹ️ 精读节以大纲为准"
                      f"（忽略传入的旧节名: "
                      f"{[s for s in (sections or []) if s not in want] or '∅'}）")
            else:
                want = list(_valid_method)   # 传入全为旧名 → 回退全部方法节
        else:
            want = list(_valid_method)
        # 每节候选（有全文 id 的，已按 RERANK_MIN_CAND_SCORE 过滤）拉全列表，
        # 前 per_section 篇下载失败 → 自动从该节剩余候选中补位，直到凑满名额或候选耗尽。
        sec_lists = {}
        for s in want:
            cs = [c for c in (cands.get(s) or {}).get("candidates") or []
                  if c.get("readable_id")]
            sec_lists[s] = cs

        dropped: list = []
        merged_map: dict[str, str] = {}   # 被并节 → 并入的目标方法节
        for s in list(sec_lists.keys()):
            if len(sec_lists[s]) >= MIN_PAPERS_PER_SECTION:
                continue
            prev = None
            for m in outline_names:        # 大纲顺序中 s 之前最近的、仍保留的方法节
                if m == s:
                    break
                if m in sec_lists and _is_method_sec(m):
                    prev = m
            if prev is not None:
                sec_lists[prev] = sec_lists[prev] + sec_lists[s]   # 候选合并到前一方法节
                merged_map[s] = prev
            else:
                dropped.append(s)
            del sec_lists[s]
        outline_kept = [o for o in outline
                        if o.get("section", "") not in dropped
                        and o.get("section", "") not in merged_map]
        if dropped or merged_map:
            print(f"--> [Tool:read_sections] ⚠️ 候选 < {MIN_PAPERS_PER_SECTION} 篇（不再补检）："
                  f"并入相邻方法节 {merged_map or '∅'}，删除 {dropped or '∅'}")

        # v1.2.11 节级护栏②（已精读节跳过，幂等）：
        # 模型重复调用 read_sections 时，该节已有足量卡片 → 直接复用，不再重新下载精读。
        existing_by_sec: dict = {}
        for _c in state.get("paper_cards", []) or []:
            existing_by_sec.setdefault(_c.get("section", ""), []).append(_c)
        skip_existing = [s for s in sec_lists
                         if len(existing_by_sec.get(s, [])) >= min(per_section, MIN_PAPERS_PER_SECTION)]
        for s in skip_existing:
            del sec_lists[s]
        if skip_existing:
            print(f"--> [Tool:read_sections] ℹ️ 以下节已精读足量卡片，跳过不重读："
                  f"{[s[:24] for s in skip_existing]}")

        if not any(sec_lists.values()):
            return Command(update={"messages": [ToolMessage(
                content="各节候选里都没有可获取全文的 id（arXiv/PMC/OA）。"
                        "按规则**不再补检**：候选不足的节已删除/合并，若所有方法节都被删除，"
                        "请直接结束研究进入 write（由引言/展望素材概括成稿），不要再次调用 search_*。",
                tool_call_id=tool_call_id)]})

        thread_id = state.get("session_id", "")
        pool = state.get("paper_pool", []) or []
        by_key = {paper_key(p): p for p in pool}

        # v1.2.10 流式 MQ：卡片边制边送（每出一张卡立即入队，JSON 块提取与后续
        # 下载/补位并行，不等全部制完；write 节点用 json_build_session_id 收集）。
        session_id = state.get("json_build_session_id") or ""

        def _on_card(c):
            nonlocal session_id
            try:
                if not session_id:
                    from ResearcherAgent.async_pipeline import start_json_session
                    session_id = start_json_session([c], llm)
                else:
                    from ResearcherAgent.async_pipeline import extend_json_session
                    extend_json_session(session_id, [c])
            except Exception as _e:
                print(f"    [card_to_json] 流式送 MQ3 失败 {type(_e).__name__}: {str(_e)[:80]}"
                      f" → session_id 保持 {'空' if not session_id else session_id}，"
                      f"write 节点 submit_cards_for_json 降级")

        # v1.2.15 MQ1 制卡流水线：先启动制卡会话（MQ1）+ JSON 会话（MQ3 空启动，
        # 让制卡消费者边制卡边送 MQ3）；下载完成即送 MQ1（并发制卡，关键词失败
        # 转 MQ2 LLM 兜底），collect 等全部制卡完成后做护栏③。MQ 不可用 →
        # 降级同步制卡（旧路径：主线程 _map_to_cards + _on_card 流式送 MQ3）。
        cb_sid = ""
        try:
            from ResearcherAgent.async_pipeline import start_card_build_session
            cb_sid = start_card_build_session([], llm)
        except Exception as _e:
            print(f"    [card_build] MQ1 会话启动失败 {type(_e).__name__}: {str(_e)[:80]}"
                  f" → 降级同步制卡")
            cb_sid = ""

        cards: list = []
        failed_global: list = []
        got_by_sec: dict = {}
        tried_keys: set = set()
        print(f"--> [Tool:read_sections] {len(want)} 节 × 每节 {per_section} 篇"
              f"（下载失败自动补位 1 轮，候选池 {sum(len(v) for v in sec_lists.values())} 篇）"
              + (f" | MQ1 制卡流水线开启" if cb_sid else " | MQ 不可用 → 同步制卡降级"))
        # v1.2.11 补位轮数 3 → 1：一轮下载失败即止，不足 MIN_PAPERS_PER_SECTION 的节
        # 由护栏③在下载后复核删除——不反复重试下载浪费时间（用户决策：不再第二轮补检）。
        for _round in range(1):  # 仅 1 轮补位（候选耗尽即止）
            items = []
            for s in want:
                if s not in sec_lists:
                    continue  # 该节已被护栏①删除
                need = per_section - len(got_by_sec.get(s, []))
                if need <= 0:
                    continue
                picked = [c for c in sec_lists[s] if c.get("key") not in tried_keys][:need]
                for c in picked:
                    tried_keys.add(c.get("key"))
                    p = by_key.get(c.get("key") or "")
                    pid = c.get("readable_id") or (p and _paper_id_for_fulltext(p)) or ""
                    if pid:
                        items.append({"id": pid, "title": c.get("title", ""),
                                      "section": s})
            if not items:
                break
            if cb_sid:
                jsid = session_id or ""
                if not jsid:
                    try:
                        from ResearcherAgent.async_pipeline import start_json_session
                        jsid = start_json_session([], llm)   # MQ3 空启动，消费者边制卡边送
                        session_id = jsid                    # 写回外层，尾部 update state
                    except Exception:
                        jsid = ""

                def _on_paper(b, _jsid=jsid):
                    try:
                        from ResearcherAgent.async_pipeline import extend_card_build_session
                        extend_card_build_session(cb_sid,
                                                  [{**b, "json_session_id": _jsid}])
                    except Exception as _e:
                        print(f"    [card_build] 流式送 MQ1 失败 {type(_e).__name__}: "
                              f"{str(_e)[:80]}（该篇走同步制卡兜底）")

                _, failed_r = fetch_and_card(items, llm, thread_id, on_paper=_on_paper)
                try:
                    from ResearcherAgent.async_pipeline import collect_card_build_session
                    replies = collect_card_build_session(cb_sid)
                    cards_r = [r["data"]["card"] for r in replies.values()
                               if r and r.get("ok") and (r.get("data") or {}).get("card")]
                except Exception as e:
                    print(f"    [card_build] MQ1 collect 失败 {str(e)[:80]} → 本节卡片为空"
                          f"（护栏③将删节/合并）")
                    cards_r = []
            else:
                cards_r, failed_r = fetch_and_card(items, llm, thread_id, on_card=_on_card)
            failed_global += failed_r
            for c in cards_r:
                got_by_sec.setdefault(c.get("section", ""), []).append(c.get("title", ""))
            cards += cards_r

        # v1.2.11 句级引用锚点：制卡阶段分配池内全局编号 [N]（写入卡片 num 字段），
        # LLM 提取块时直接在论断句后标注 [N]；组装期只校验纠正，不再重映射。
        # v1.2.17：gidx 提前算——MQ1 消息带 num（关键词制卡/MQ2 直接提取共用同一编号）。
        try:
            gidx = global_index(pool)
        except Exception:
            gidx = {}
        for it in items:
            n = None
            pid = it.get("id", "")
            for p in pool:
                if pid and pid in {str(p.get("id", "")), str(p.get("paper_id", "")),
                                   str(p.get("arxiv_id", "")), str(p.get("pmcid", "") or "")}:
                    n = gidx.get(paper_key(p))
                    break
            if n is None:
                tn = _norm_card_title(it.get("title", ""))
                if tn:
                    for p in pool:
                        if _norm_card_title(p.get("title", "")) == tn:
                            n = gidx.get(paper_key(p))
                            break
            it["num"] = n  # None → 组装期兜底补编号

        for c in cards:
            n = None
            pid = c.get("id", "")
            for p in pool:
                if pid and pid in {str(p.get("id", "")), str(p.get("paper_id", "")),
                                   str(p.get("arxiv_id", "")), str(p.get("pmcid", "") or "")}:
                    n = gidx.get(paper_key(p))
                    break
            if n is None:
                tn = _norm_card_title(c.get("title", ""))
                if tn:
                    for p in pool:
                        if _norm_card_title(p.get("title", "")) == tn:
                            n = gidx.get(paper_key(p))
                            break
            c["num"] = n  # None → 组装期兜底补编号

        # v1.2.11 节级护栏③（下载后复核）：实际精读卡片 < MIN_PAPERS_PER_SECTION 的节，
        # 优先**并入大纲中它前面最近的方法节**（该节已制卡片的 section 改写为合并后节名 →
        # write 归节与 MQ 块自然映射到新大纲节）；前面没有可并的方法节才删除。
        # 全程不再补检——下载失败不反复重试（用户决策：不再第二轮补检）。
        by_sec_real: dict = {}
        for c in cards:
            by_sec_real.setdefault(c.get("section", ""), []).append(c)
        kept_names = [o.get("section", "") for o in outline_kept]
        dropped2: list = []
        rename2: dict[str, str] = {}
        for s in sec_lists:
            if len(by_sec_real.get(s, [])) >= MIN_PAPERS_PER_SECTION:
                continue
            prev = None
            for m in kept_names:             # 大纲（护栏①后）中 s 之前最近的方法节
                if m == s:
                    break
                if _is_method_sec(m):
                    prev = m
            if prev is not None:
                rename2[s] = prev
            else:
                dropped2.append(s)
        for c in cards:
            if c.get("section", "") in rename2:
                c["section"] = rename2[c["section"]]
        outline_final = [o for o in outline_kept
                         if o.get("section", "") not in dropped2
                         and o.get("section", "") not in rename2]
        if dropped2 or rename2:
            print(f"--> [Tool:read_sections] ⚠️ 实际精读 < {MIN_PAPERS_PER_SECTION} 篇（不再补检）："
                  f"并入相邻方法节 {rename2 or '∅'}，删除 {dropped2 or '∅'}")

        lines = [f"分节精读完成：{len(cards)} 张卡片（每节最多 {per_section} 篇，已带章节归属；"
                 f"下载失败已从同节候选自动补位）。"]
        by_sec = {}
        for c in cards:
            by_sec.setdefault(c.get("section", ""), []).append(c.get("title", ""))
        for s in want:
            if s in merged_map or s in rename2:
                lines.append(f"  ➤ {s[:34]}：已并入相邻方法节（不再补检）")
                continue
            got = by_sec.get(s, [])
            lines.append(f"  {'✓' if got else '✗'} {s[:34]}：{len(got)} 张")
            for t in got:
                lines.append(f"      · {t[:60]}")
        if failed_global:
            lines.append(f"  ✗ 取不到全文（已补位仍失败）：{', '.join(failed_global[:6])}")
        if dropped:
            lines.append(f"  ✗ 候选不足 {MIN_PAPERS_PER_SECTION} 篇已删除大纲节（不再补检）：{dropped}")
        if merged_map:
            lines.append(f"  ➤ 候选不足 {MIN_PAPERS_PER_SECTION} 篇已并入相邻方法节（不再补检）："
                         f"{merged_map}")
        if dropped2:
            lines.append(f"  ✗ 实际精读不足 {MIN_PAPERS_PER_SECTION} 篇已删除大纲节：{dropped2}")
        if rename2:
            lines.append(f"  ➤ 实际精读不足 {MIN_PAPERS_PER_SECTION} 篇已并入相邻方法节"
                         f"（卡片归属已改写）：{rename2}")
        # 写回重排结果供「一轮检索后证据不足拒答门」（_pool_relevant_ratio_mean）使用：
        # 该门读 state.section_candidates.sections[*].n_relevant/scored_total。
        update = {
            "paper_cards": cards,
            "section_candidates": {"_pool_n": len(pool), "sections": cands_map},
            "messages": [ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)],
        }
        if outline_final != outline:
            update["review_outline"] = outline_final
        if session_id:
            update["json_build_session_id"] = session_id
            print(f"--> [Tool:read_sections] 流式 MQ 会话 {session_id}"
                  f"（累计 {len(cards)} 张卡片本次增量入队）")
        return Command(update=update)

    return read_sections


def make_read_papers(llm):
    @tool
    def read_papers(
        ids: list[str],
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
    ) -> Command:
        """对指定论文取**结构化全文**（arXiv→ar5iv / PMC→Europe PMC，不下载 PDF），
        并提炼成 ~400 字结构化卡片（问题/方法/结果/局限）存入 paper_cards。

        ids：arXiv id（如 2301.12345）或 PMCID（如 PMC1234567），一次最多 6 个，**并发获取**。
        取不到全文的会跳过（不编造）。
        """
        from ResearcherAgent.core.graph_nodes import _title_relevant, _map_to_cards
        from ResearcherAgent.retrieval.fulltext import fetch_arxiv_fulltext, fetch_pmc_fulltext

        thread_id = state.get("session_id", "")
        pool = state.get("paper_pool", []) or []
        title_by_id = {}
        section_by_id = {}
        for p in pool:
            pid = _paper_id_for_fulltext(p)
            if pid:
                title_by_id[pid] = p.get("title", "")
                section_by_id[pid] = (p.get("sections") or [""])[0]

        ids = [str(i).strip() for i in ids if str(i).strip()][:6]
        if not ids:
            return Command(update={"messages": [ToolMessage(
                content="未给出 id。", tool_call_id=tool_call_id, status="error")]})

        def _fetch(pid: str) -> dict | None:
            with tracing.span("tool", "fulltext", thread_id, id=pid):
                if re.match(r"^PMC\d+", pid, re.I):
                    t = fetch_pmc_fulltext(pid, None)
                elif pid.lower().startswith("arxiv:"):
                    pid = pid.split(":", 1)[1]
                    t = fetch_arxiv_fulltext(pid, None)
                elif re.match(r"^\d{4}\.\d{4,5}", pid):
                    t = fetch_arxiv_fulltext(pid, None)
                else:
                    return None
            title = title_by_id.get(pid, "")
            if not t or len(t) < 300:
                print(f"    ✗ {title[:40] or pid}：全文不可得")
                return None
            if title and not _title_relevant(title, t):
                print(f"    ✗ {title[:40]}：全文与标题不匹配，丢弃")
                return None
            return {"title": title or pid, "id": pid, "text": t, "section": section_by_id.get(pid, "")}

        with ThreadPoolExecutor(max_workers=DOWNLOAD_CONCURRENCY) as ex:
            blocks = [b for b in ex.map(_fetch, ids) if b]

        cards = []
        if blocks:
            with tracing.span("llm", "map_to_cards", thread_id):
                n = max(1, min(EXTRACT_CONCURRENCY, len(blocks)))
                chunks = [blocks[i::n] for i in range(n)]
                with ThreadPoolExecutor(max_workers=n) as ex2:
                    card_lists = list(ex2.map(lambda ch: _map_to_cards(ch, llm), chunks))
                cards = [c for cl in card_lists for c in cl]
            sec_by_id = {b["id"]: b.get("section", "") for b in blocks}
            for c in cards:
                c["section"] = sec_by_id.get(c.get("id", ""), "")
        print(f"--> [Tool:read_papers] 请求 {len(ids)} 篇 → 取到 {len(blocks)} 篇全文 → {len(cards)} 张卡片")

        failed = [i for i in ids if i not in {b["id"] for b in blocks}]
        lines = [f"精读完成：{len(cards)} 张卡片已存入 paper_cards（写作时按章节注入）。"]
        for c in cards:
            lines.append(f"  ✓ {c.get('title', '')[:60]}" + (f"（{c['section']}）" if c.get("section") else ""))
        if failed:
            lines.append(f"  ✗ 取不到全文（网络/非开放获取）：{', '.join(failed)}")
        if not cards and blocks:
            lines.append("  ⚠️ 全文已取到但卡片提炼为空。")
        if not blocks and ids:
            lines.append("  💡 全文通道可能不可达：可改用 search_europepmc 找有 PMC 全文的论文，"
                         "或直接用检索到的标题+摘要写作，不必反复重试。")
        return Command(update={
            "paper_cards": cards,   # 只回新增卡片，AgentState 用 merge_paper_cards 追加去重
            "messages": [ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)],
        })

    return read_papers


_CARD_FIELD_ORDER = ("background", "method", "experiment", "conclusion")


# 3. 渐进式工具加载：按 skills 选工具子集
# ==========================================

def build_local_tools(llm) -> dict[str, BaseTool]:
    out = {
        "survey_evolution": make_survey_evolution(llm),
        "make_outline": make_outline,
        "write_todos": write_todos,
        "read_papers": make_read_papers(llm),
        "read_sections": make_read_sections(llm),
    }
    # —— 代码执行工具组（skill "code_exec" 启用；照搬 HKUDS/AI-Researcher terminal_tools）——
    from ResearcherAgent.sandbox.code_executor import (
        execute_command, read_file, create_file, write_file,
        list_files, create_directory, gen_code_tree_structure, run_python,
    )
    out.update({
        "execute_command": execute_command,
        "read_file": read_file,
        "create_file": create_file,
        "write_file": write_file,
        "list_files": list_files,
        "create_directory": create_directory,
        "gen_code_tree_structure": gen_code_tree_structure,
        "run_python": run_python,
    })
    # —— 创新点验证工具组（skill "innovate" 启用：检索精读(复用) → 创新点 → 实验 → 修复重跑 → 判定）——
    from ResearcherAgent.innovate.innovation_tools import (
        make_deep_survey, make_propose_innovations, make_implement_innovation,
        make_judge_innovation, make_fix_and_rerun, run_baseline,
        make_novelty_check, make_code_survey, make_prepare_reference_repos,
    )
    out.update({
        "deep_survey": make_deep_survey(llm),
        "propose_innovations": make_propose_innovations(llm),
        "prepare_reference_repos": make_prepare_reference_repos(llm),
        "run_baseline": run_baseline,
        "implement_innovation": make_implement_innovation(llm),
        "fix_and_rerun": make_fix_and_rerun(llm),
        "judge_innovation": make_judge_innovation(llm),
        "novelty_check": make_novelty_check(llm),
        "code_survey": make_code_survey(llm),
    })
    # —— CC 风格项目记忆（Claude Code Auto Memory 模式 v2）——
    # remember：写前过滤+查重（LLM 三分类 ADD/UPDATE/NOOP）；consolidate：Dreams 式整理；
    # recall_memory：按索引定位 topic 文件读具体说明（索引全量注入 + 细节按需读）。
    from ResearcherAgent.memory.cc_memory import (make_remember_memory,
                                                  make_consolidate_memory,
                                                  recall_memory)
    out.update({
        "remember_memory": make_remember_memory(llm),
        "consolidate_memory": make_consolidate_memory(llm),
        "recall_memory": recall_memory,
    })
    return out


def select_tools(skills: list[str], mcp_tools: dict[str, BaseTool],
                 local_tools: dict[str, BaseTool]) -> list[BaseTool]:
    """62 个 MCP 工具一次性 bind → 按本次 skills 只暴露相关子集（提升选择准确率、缩小 prompt）。"""
    out: list[BaseTool] = []
    if "search_papers" in skills:
        out += [t for n, t in mcp_tools.items() if is_paper_search_tool(n)]
    if "write_review" in skills:
        out += [local_tools[k] for k in ("write_todos", "survey_evolution",
                                         "make_outline", "read_papers", "read_sections")]
        # L3 按需加载：SKILL.md 正文指向 skills/<name>/references/，模型需要细节时用 read_file 读取
        out += [local_tools[k] for k in ("read_file",) if k in local_tools]
    if "code_exec" in skills:
        # 代码执行工具组：模型可写文件、跑命令、跑训练脚本（后端由 LANGGRAPH_EXEC_BACKEND 控制）
        from ResearcherAgent.sandbox.code_executor import CODE_EXEC_TOOLS
        out += [local_tools[k] for k in CODE_EXEC_TOOLS if k in local_tools]
        # CC 风格项目记忆（跨会话沉淀，与代码执行配套）：remember/consolidate/recall
        out += [local_tools[k] for k in ("remember_memory", "consolidate_memory", "recall_memory")
                if k in local_tools]
    if "innovate" in skills:
        # 创新点验证流水线：检索 + 精读制卡（复用）+ 创新工具 + 代码执行
        from ResearcherAgent.innovate.innovation_tools import INNOVATE_TOOLS
        from ResearcherAgent.sandbox.code_executor import CODE_EXEC_TOOLS
        out += [t for n, t in mcp_tools.items() if is_paper_search_tool(n)]
        out += [local_tools[k] for k in ("read_papers", "read_sections")
                if k in local_tools]
        out += [local_tools[k] for k in INNOVATE_TOOLS if k in local_tools]
        out += [local_tools[k] for k in CODE_EXEC_TOOLS if k in local_tools]
        out += [local_tools[k] for k in ("remember_memory", "consolidate_memory", "recall_memory")
                if k in local_tools]
    # 去重保序
    seen, uniq = set(), []
    for t in out:
        if t.name not in seen:
            seen.add(t.name)
            uniq.append(t)
    return uniq

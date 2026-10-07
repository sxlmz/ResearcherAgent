"""
agent_tools.py
==============
把旧图里写死的节点降级为**模型可调用的工具**，形成 agent ⟲ tools 的 ReAct 循环。

| 旧节点 / 边                         | 现在的工具                 |
|------------------------------------|---------------------------|
| plan_review                        | make_outline              |
| 检索指令 + chatbot_with_tools       | MCP 搜索工具（包装后入 paper_pool，只回摘要给模型） |
| coverage_judge + route_after_coverage | check_coverage（是否补搜由模型决定） |
| targeted_search                    | —（模型直接再调搜索工具）    |
| deep_read                          | read_papers               |
| review_materials + interrupt       | ask_user                  |
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
    TOOL_CONCURRENCY, MAX_TOOL_RESULT_CHARS, MAX_SEARCH_ROUNDS,
    PAPER_SEARCH_TOOL_WHITELIST, WEB_SEARCH_TOOL_PREFIXES,
    SECTION_CANDIDATES_K, RERANK_MIN_SCORE, CARDS_PER_SECTION,
    SRC_FAIL_STREAK, SRC_COOLDOWN_SEC, MCP_CALL_TIMEOUT_SEC, RETRY_BACKOFF_SEC,
)
from ResearcherAgent.core import cache
from ResearcherAgent.core import tracing

from ResearcherAgent.research import llm_seeds
from ResearcherAgent.research import must_read as mr
from ResearcherAgent.research import discover_directions as dd
from ResearcherAgent.retrieval import retrieval_cache as rcache


# ==========================================
# 0. paper_pool：结构化论文池 + reducer
# ==========================================

def paper_key(p: dict) -> str:
    """去重键：doi > paper_id > 归一化标题。"""
    doi = (p.get("doi") or "").strip().lower()
    if doi:
        return "doi:" + doi
    pid = (p.get("paper_id") or p.get("id") or "").strip().lower()
    if pid:
        return "id:" + pid
    title = re.sub(r"[^a-z0-9一-鿿]+", " ", (p.get("title") or "").lower()).strip()
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

    解析不了的内容退化为一条 {"title": <首行>, "raw": <全文>} blob，绝不丢信息。
    """
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
    title = (p.get("title") or "").strip()
    ab = _clean_text(p.get("abstract") or p.get("raw") or "")[:400]
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

    这是「宽召回 → 精确重排 → 少量进 prompt」里的第二级。相比旧做法（按「有没有摘要」排序），
    重排直接回答「这篇和这一节有多相关」，且视野是**全池**而非 26 篇。

    返回 {section: {"query":..., "candidates":[{idx,key,score,title,has_abstract}]}}；
    重排 API 不可用时返回 {}（调用方自动降级回旧排序）。
    """
    from ResearcherAgent.retrieval import rerank as _rerank
    pool = [p for p in (pool or []) if isinstance(p, dict)]
    if not pool or not outline or not _rerank.available():
        if pool and outline and not _rerank.available():
            print("    ℹ️ [rerank] 未配置 RERANK_API_*，跳过重排（降级为按素材可用性排序）")
        return {}

    # ---- L5 缓存：同样的（池 + 大纲）不重复重排 ----
    pool_fp = _fingerprint(sorted(paper_key(p) for p in pool))
    outline_fp = _fingerprint([f"{o.get('section')}|{o.get('focus')}|{o.get('queries')}"
                               for o in outline])
    hit = cache.select_get(pool_fp, outline_fp)
    if hit:
        print(f"--> [Rerank] ♻️ L5 选材缓存命中（池 {len(pool)} 篇 × {len(outline)} 节），"
              f"跳过 {len(outline)} 次全池重排")
        return hit

    gidx = global_index(pool)
    docs = [paper_rerank_text(p) for p in pool]
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
        # 覆盖度：统计全池里有多少篇达到阈值（这才是「这一节资料够不够」）
        n_rel = sum(1 for _, s in res if s >= RERANK_MIN_SCORE)
        cands = []
        for i, score in res[:k]:
            if not (0 <= i < len(pool)):
                continue
            p = pool[i]
            cands.append({"idx": gidx.get(paper_key(p)), "key": paper_key(p),
                          "score": round(float(score), 4),
                          "title": (p.get("title") or "")[:120],
                          "has_abstract": has_abstract(p),
                          "readable_id": _paper_id_for_fulltext(p)})
        top = cands[0]["score"] if cands else 0.0
        out[sec] = {"query": q[:120], "candidates": cands,
                    "top_score": top, "n_relevant": n_rel,
                    "scored_total": len(res)}
        print(f"--> [Rerank] 「{sec[:24]}」全池 {len(res)} 篇打分，"
              f"≥{RERANK_MIN_SCORE} 的 {n_rel} 篇；取 top-{len(cands)} 作候选（最高 {top:.3f}）")
    if out:
        cache.select_set(pool_fp, outline_fp, out)
    return out


def per_section_top_cards(cands: dict, per_section: int = CARDS_PER_SECTION,
                          prefer_readable: bool = True) -> list:
    """从每节候选中挑出要精读的论文（默认每节 top-2）。

    优先挑**能取到结构化全文**的（有 arXiv/PMC id）——否则 read_papers 拿不到正文。
    如果没有可读 id 的，退回按分数取，让 read_papers 去尝试。
    """
    picks = []
    for sec, info in (cands or {}).items():
        cs = list(info.get("candidates") or [])
        if prefer_readable:
            # ⚠️ 只在**靠前的候选**里挑可读的：若为了「有 arXiv id」一路往下捞到第 12 名，
            # 就把一篇相关性很低的论文塞进了精读——相关性优先于可读性，读不到就跳过。
            window = cs[:max(per_section * 3, 5)]
            readable = [c for c in window if c.get("readable_id")]
            chosen = readable[:per_section] if readable else window[:per_section]
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

    ⚠️ 为什么不只用 sections 字段：归属是 check_coverage **自己**的产物，第一次调用时
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


def render_section_candidates(pool: list, cands: dict, section: str,
                              max_chars: int = 15000, card_only: bool = False,
                              cards: list | None = None) -> str:
    """用**该节的重排候选**渲染写作材料——按相关性分数排序，而非按素材可用性。

    这是重排真正产生价值的地方：旧做法是「本节优先 + 其余按摘要长度追加」，
    本节之外的补充材料与本节无关；现在补充材料也是**针对本节重排出来的高分论文**。

    ⚠️ 编号必须用 global_index —— 正文的 `[N]` 要能映射回真实论文才能拼参考表。
    候选不足或重排不可用时退回 render_pool。

    card_only=True（证据约束）：只渲染**有精读卡片**的论文，卡片全文一并注入，
    杜绝写作引用「只见过摘要/没读过」的论文（忠实度升级）。
    """
    pool = [p for p in (pool or []) if isinstance(p, dict)]
    info = ((cands or {}).get("sections") or {}).get(section) or {}
    cs = info.get("candidates") or []
    if not cs:
        return render_pool(pool, section=section, max_chars=max_chars)

    by_key = {paper_key(p): p for p in pool}
    gidx = global_index(pool)
    ordered = [by_key[c["key"]] for c in cs if c.get("key") in by_key]
    if not ordered:
        return render_pool(pool, section=section, max_chars=max_chars)

    if card_only:
        # 证据约束：只保留有卡片论文；卡片全文追加到该论文条目后
        card_by_title = {_norm_card_title(c.get("title", "")): c for c in (cards or [])}
        _kept = []
        for p in ordered:
            _k = _norm_card_title(p.get("title", ""))
            if _k in card_by_title:
                _kept.append((p, card_by_title[_k]))
        if not _kept:
            return ("（本节暂无精读卡片：请仅基于已有知识综述背景，"
                    "正文**不要**给出任何引用编号；待补充精读后再写。"
                    f"本节候选 {len(ordered)} 篇均无卡片）")
        parts, total = [], 0
        for p, c in _kept:
            s = render_paper(p, gidx.get(paper_key(p)), True)
            s += f"\n【精读卡片】{c.get('card', '')}"
            if total + len(s) > max_chars:
                break
            parts.append(s)
            total += len(s) + 1
        head = (f"（本节材料按证据约束：仅 {len(_kept)} 篇已精读论文（含卡片全文））\n")
        return head + "\n".join(parts) if parts else ""

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
            "✗ 反例（太泛，会召回大量无关领域论文）：time series forecasting / "
            "deep learning recommendation\n"
            "✓ 正例（含场景限定词）：wind power forecasting / wind speed prediction LSTM / "
            "SCADA wind turbine time series\n"
            "若用户没指定具体场景，则用主题里最有区分度的领域词，不要退化成最泛的上位词。"
        ))


@tool
def make_outline(
    sections: list[OutlineSection],
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """设定/更新综述大纲（3-6 节）。大纲同时是检索计划和写作结构。

    首次规划时给出完整大纲；用户要求修改时传入**修改后的完整大纲**（保留未提及的章节，
    只增删/调整用户要求的部分）。调用后覆盖度报告会被清空，需要重新 check_coverage。

    ⚠️ **每节的 queries 必须带场景/领域限定词**。这一步决定后续所有检索的质量：
    只写泛方法词（如 time series forecasting）会召回一大堆无关领域的论文，
    实测窄领域命中率直接降为 0；带上场景词（如 wind power forecasting）才能命中目标文献。
    """

    outline = [s.model_dump() if hasattr(s, "model_dump") else dict(s) for s in sections]
    outline = [o for o in outline if o.get("section")]
    if not outline:
        return Command(update={"messages": [ToolMessage(
            content="大纲为空，请至少给出 1 节。", tool_call_id=tool_call_id, status="error")]})
    names = [o["section"] for o in outline]
    print(f"--> [Tool:make_outline] {len(outline)} 节: {names}")
    msg = "大纲已更新（" + "、".join(names) + "）。下一步：按各节 queries 调用检索工具，然后 check_coverage。"
    return Command(update={
        "review_outline": outline, "coverage_report": {}, "gaps": [], "selected_papers": [],
        "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
    })


# ==========================================
# 2b. 子主题种子检索（LLM 拆子方向 → 代表性论文种子 → 引文网络扩展）
#     主检索路径：从种子出发做引文扩展，把扩展论文并入 paper_pool。
# ==========================================

@tool
def make_subtopics(
    topic: str = "",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[dict, InjectedState] = None,
) -> Command:
    """把研究方向拆成 **最多 5 个** 子主题，并为每个子主题给出 1-2 篇代表性论文（种子）。

    子主题覆盖：经典基础方法、主流深度学习方法、近两年热点方向、相邻领域的交叉方法。
    生成结果存入 state.subtopics，是 expand_seeds 引文扩展的种子来源。
    拆完后调用 expand_seeds 从这些种子做引文网络扩展，把扩展出的论文并入 paper_pool；
    无需再按大纲逐节文本检索（search_* 仅作补充最新/偏门文献用）。
    """
    topic_ = (topic or "").strip() or (state.get("topic") or state.get("user_request") or "")
    if not topic_:
        return Command(update={"messages": [ToolMessage(
            content="缺少研究方向：请传入 topic 参数，或先确认任务主题。",
            tool_call_id=tool_call_id, status="error")]})
    print(f"--> [Tool:make_subtopics] 主题: {topic_[:40]}")
    try:
        subtopics = llm_seeds.generate(topic_)
    except Exception as e:
        return Command(update={"messages": [ToolMessage(
            content=f"LLM 拆解子主题失败：{str(e)[:120]}。可稍后重试，或直接用 search_* 检索。",
            tool_call_id=tool_call_id, status="error")]})
    if not subtopics:
        return Command(update={"messages": [ToolMessage(
            content="LLM 未返回有效子主题，请重试或改用 search_* 检索。",
            tool_call_id=tool_call_id, status="error")]})
    seeds = llm_seeds.flatten(subtopics)
    lines = [f"{i + 1}. {it['subtopic']} → " +
             "；".join(f"{p['title'][:46]} ({p.get('year')})" for p in it['papers'])
             for i, it in enumerate(subtopics)]
    msg = ("已拆解子主题并给出代表性论文（种子）：\n" + "\n".join(lines) +
           f"\n\n共 {len(subtopics)} 个子主题、{len(seeds)} 篇种子。"
           "下一步调用 expand_seeds 做引文网络扩展（如需调整种子，可传 seed_titles 覆盖）。")
    return Command(update={
        "subtopics": subtopics, "seed_papers": seeds,
        "coverage_report": {}, "gaps": [], "selected_papers": [],
        "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
    })


def _oa_abstract(w: dict) -> str:
    """OpenAlex abstract_inverted_index（倒排词表）→ 可读摘要文本。"""
    inv = (w.get("abstract_inverted_index") or {})
    if not inv:
        return ""
    pos: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))[:1500]


def _oa_work_to_pool(w: dict, seed_links: int = 0, is_seed: bool = False) -> dict:
    """OpenAlex 工作记录 → paper_pool 条目（与 MCP 检索结果同构，供去重/渲染/重排复用）。"""
    src = ((w.get("primary_location") or {}).get("source") or {})
    return {
        "title": (w.get("display_name") or "").strip(),
        "year": w.get("publication_year"),
        "citations": w.get("cited_by_count") or 0,
        "abstract": _oa_abstract(w),
        "url": w.get("id"),
        "paper_id": mr._wid(w.get("id") or ""),
        "doi": w.get("doi"),
        "venue": src.get("display_name") or "",
        "source": "openalex_expand",
        "seed_links": seed_links,
        "is_seed": is_seed,
        # ★ OA 全文字段（原实现丢弃 → 全文通路永远缺失；实测 Tavily 可提取）
        "is_oa": bool((w.get("open_access") or {}).get("is_oa")),
        "oa_url": (w.get("open_access") or {}).get("oa_url") or "",
        "pdf_url": ((w.get("best_oa_location") or {}).get("pdf_url")
                    or next((l.get("pdf_url") for l in (w.get("locations") or [])
                             if l.get("pdf_url")), "")),
        "landing_url": ((w.get("best_oa_location") or {}).get("landing_page_url") or ""),
    }


@tool
def expand_seeds(
    seed_titles: list[str] | None = None,
    top_n: int = 100,
    max_candidates: int = 1500,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[dict, InjectedState] = None,
) -> Command:
    """从代表性论文（种子）出发做**引文网络双向扩展**，把扩展出的论文并入 paper_pool。

    种子默认取 make_subtopics 生成的结果；可用 seed_titles 传入自定义标题（与已有种子合并）。
    扩展：OpenAlex 引文网络（种子引用了谁 + 谁引用了种子），自动去重、按
    「被几篇种子引用 + 年均被引 + venue 影响力」打分，取 top_n 篇入池（种子本身也入池）。
    ⚠️ 首次调用可能耗时 30-90 秒（引文网络请求 + 打分）；结果有磁盘缓存，重复调用更快。
    若某些种子解析置信度低，返回里会列出，请核实标题后重试或换用 search_* 补检索。
    """
    top_n = max(int(top_n or 100), 10)
    max_candidates = max(int(max_candidates or 1500), 200)
    base_seeds: list[dict] = [dict(s) for s in (state.get("seed_papers") or [])]
    for tt in (seed_titles or []):
        s = str(tt or "").strip()
        if s and not any(str(x.get("title", "")).strip().lower() == s.lower()
                         for x in base_seeds):
            base_seeds.append({"title": s})
    if not base_seeds:
        return Command(update={"messages": [ToolMessage(
            content="没有可用的种子：请先调用 make_subtopics 生成子主题与种子，"
                    "或用 seed_titles 传入论文标题。",
            tool_call_id=tool_call_id, status="error")]})
    print(f"--> [Tool:expand_seeds] 种子 {len(base_seeds)} 篇, top_n={top_n}, max_candidates={max_candidates}")

    # ① 解析种子（OpenAlex 标题/DOI/id → 工作记录）
    try:
        sr = mr.resolve_seeds(base_seeds)
    except Exception as e:
        return Command(update={"messages": [ToolMessage(
            content=f"种子解析失败：{str(e)[:120]}（OpenAlex 可能限流）。"
                    "可稍后重试，或改用 search_* 直接检索。",
            tool_call_id=tool_call_id, status="error")]})
    resolved = [x for x in sr if x]
    low_conf = [x for x in resolved if (x.get("match_score") or 0) < 0.6]
    seed_works = [x["work"] for x in resolved if (x.get("match_score") or 0) >= 0.6]
    if not seed_works:
        return Command(update={"messages": [ToolMessage(
            content="所有种子解析置信度都过低（可能标题不准确），无法扩展。"
                    "请核实标题后重试。",
            tool_call_id=tool_call_id, status="error")]})
    n_fail = len(base_seeds) - len(resolved)
    n_low = len(resolved) - len(seed_works)
    print(f"  解析成功 {len(resolved)} / {len(base_seeds)}（低置信 {n_low}，失败 {n_fail}）")

    # ② 引文网络双向扩展 + 去重 + 打分
    seed_titles_norm = {mr.norm_title(w.get("display_name", "")) for w in seed_works}
    cand = mr.expand(seed_works, direction="both", max_candidates=max_candidates)
    items = mr.dedupe(cand, seed_titles_norm)
    stats = mr.fetch_source_stats(items)
    items = mr.score(items, time.gmtime().tm_year, stats)
    ranked = [it for it in items if not it.get("suspect")]
    ranked.sort(key=lambda x: -x["total"])
    print(f"  候选 {len(cand)} → 去重 {len(items)} → 打分完成")

    # ③ 种子 + Top-n 扩展 → pool 条目
    pool_items: list[dict] = [_oa_work_to_pool(w, seed_links=len(seed_works), is_seed=True)
                              for w in seed_works]
    pool_items += [_oa_work_to_pool(it["work"], seed_links=len(it.get("seed_links") or []))
                   for it in ranked[:top_n]]

    # ④ 去重并入池（reducer 会再做一次全局去重；这里先算新增数）
    existing = {paper_key(p) for p in (state or {}).get("paper_pool", []) or []}
    new = [p for p in pool_items if paper_key(p) not in existing]
    dup = len(pool_items) - len(new)

    # ⑤ 回给模型：新增论文标题 + 统计（截断，全量已在 pool）
    heads = "\n".join(
        f"  · [{i + 1}] {p.get('title', '')[:60]} ({p.get('year')}, 被引{p.get('citations')})"
        for i, p in enumerate(new[:12]))
    note = ""
    if n_fail:
        note += f"\n⚠️ 有 {n_fail} 篇种子未解析到（标题可能不准确）：" +                 "、".join(str(x.get("title"))[:40] for x in base_seeds if x not in resolved)
    if n_low:
        note += f"\n⚠️ 有 {n_low} 篇种子解析置信度低（未参与扩展）：" +                 "、".join(f"{x['input'][:36]}→{x['work'].get('display_name','')[:36]}"
                          for x in low_conf)
    msg = (f"引文扩展完成：种子 {len(seed_works)} 篇 + 扩展 Top-{top_n} 篇 → "
           f"本次新增入池 {len(new)} 篇（去重 {dup} 篇）；paper_pool 现有 "
           f"{len(existing) + len(new)} 篇。\n新增论文（前 {len(heads.splitlines())} 条）：\n{heads}{note}")
    if len(msg) > 1500:
        msg = msg[:1500] + "\n…（已截断，全量已在 paper_pool）"
    return Command(update={
        "paper_pool": pool_items,
        "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
    })




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


def make_check_coverage(llm):
    """check_coverage：**用 Cross-Encoder 全池重排**做逐节相关性判断。

    为什么不再靠 LLM 判相关性：旧实现把 render_pool(max_chars=16000) 给 LLM，
    而 16000 字符只够约 26 篇——池子 500 篇时**只有 5% 被真正判过**。
    重排模型直接对「章节查询 × 全池论文」打分，视野 100%，而且分数可比较、可复现。

    LLM 仍保留一个作用：从候选中挑出**真正值得精读**的（重排分高 ≠ 内容有价值，
    可能是综述性文章）。但候选池从「26 篇随机」变成了「每节重排 top-K」。
    """

    @tool
    def check_coverage(
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
    ) -> Command:
        """评估 paper_pool 对大纲各节的覆盖度，并给出每节的**重排候选**。

        做法：对每一节构造查询，把**全池**论文交给 Cross-Encoder 重排，
        保留 top-K 作为该节候选。报告里给出每节的候选数、相关性分数分布。
        候选会存入 section_candidates，供 read_sections 精读使用。

        需先 make_outline 并做过检索。看候选数偏少的章节时，由你决定是否换词再检索。
        """
        outline = state.get("review_outline", []) or []
        pool = state.get("paper_pool", []) or []
        sections = [o["section"] for o in outline]
        thread_id = state.get("session_id", "")
        if not sections:
            return Command(update={"messages": [ToolMessage(
                content="尚无大纲，请先调用 make_outline。", tool_call_id=tool_call_id, status="error")]})
        if not pool:
            report = {s: {"count": 0, "status": "missing"} for s in sections}
            return Command(update={"coverage_report": report, "gaps": sections, "selected_papers": [],
                                   "messages": [ToolMessage(content="paper_pool 为空，所有章节 missing，请先检索。",
                                                            tool_call_id=tool_call_id)]})

        # ---- ① 全池重排：每节 top-K 候选（核心新增）----
        # 池子没变（篇数相同）就复用上次结果，避免每调一次覆盖度就全量重排 6 次
        prev = state.get("section_candidates") or {}
        cached_pool_n = (prev.get("_pool_n") if isinstance(prev, dict) else None)
        cands = {}
        if cached_pool_n == len(pool) and prev.get("sections"):
            cands = prev["sections"]
            print(f"--> [Rerank] 池内篇数未变（{len(pool)}），复用上轮重排结果")
        else:
            print(f"--> [Rerank] 全池重排 {len(pool)} 篇 × {len(sections)} 节 ...")
            cands = select_section_candidates(pool, outline, thread_id=thread_id)
            if cands:
                cands = {"_pool_n": len(pool), "sections": cands}

        # 重排不可用时降级：退回旧的 LLM 判断
        if not cands.get("sections"):
            return _check_coverage_llm_fallback(llm, state, tool_call_id, outline, pool, sections)

        sec_cands = cands["sections"]

        # ---- ② 覆盖度：从重排结果确定性推导 ----
        report, pool_updates, sel_out = {}, [], []
        _key2p = {paper_key(pp): pp for pp in pool if isinstance(pp, dict)}
        for s in sections:
            info = sec_cands.get(s) or {}
            cs = info.get("candidates") or []
            n_rel = int(info.get("n_relevant") or 0)
            status = "missing" if n_rel == 0 else ("thin" if n_rel < 4 else "covered")
            report[s] = {"count": n_rel, "status": status,
                         "top_score": round(float(info.get("top_score") or 0), 3),
                         "candidates": len(cs)}
            # 回写只为了给论文补上「归属章节」——**必须携带原记录的标识字段**。
            # ⚠️ 曾只回写 {"title": <候选里被截断到 120 字符的标题>, "section": s}，
            # 而 merge_papers 按 paper_key 去重：原记录有 doi（键 doi:xxx），回写只有标题
            # （键 title:xxx）→ 键不同 → 被当成新论文追加，池里出现「同标题，一条有摘要一条没有」。
            for c in cs:
                if c.get("key") and c.get("score", 0) >= RERANK_MIN_SCORE:
                    src = _key2p.get(c["key"])
                    if src:
                        pool_updates.append({
                            **{k: src.get(k) for k in ("title", "doi", "paper_id")},
                            "section": s})

        gaps = [s for s, v in report.items() if v["status"] == "missing"]
        thin = [s for s, v in report.items() if v["status"] == "thin"]

        # ---- ③ 每节挑 2 篇作精读候选（优先能取到全文的）----
        picks = per_section_top_cards(sec_cands, per_section=CARDS_PER_SECTION)
        sel_out = [{"title": c.get("title", ""), "id": c.get("readable_id", ""),
                    "source": "", "section": c.get("section", ""), "score": c.get("score")}
                   for c in picks]

        icon = {"covered": "✅", "thin": "⚠️", "missing": "❌"}
        lines = [f"覆盖度报告（口径：对**全池 {len(pool)} 篇**做 Cross-Encoder 重排后，"
                 f"每节分数 ≥{RERANK_MIN_SCORE} 的论文数）："]
        for s, v in report.items():
            lines.append(f"  {icon.get(v['status'], '•')} {s}：相关 {v['count']} 篇（{v['status']}）"
                         f"｜候选 {v['candidates']} 篇，最高分 {v['top_score']}")
        if gaps:
            lines.append(f"缺口（missing）：{'、'.join(gaps)}")
        if thin:
            lines.append(f"偏少（thin）：{'、'.join(thin)}")
        lines.append(f"\n📊 每节已选 {CARDS_PER_SECTION} 篇"
                     f"最有代表性且能取到全文的论文作为精读候选（共 {len(sel_out)} 篇）。"
                     f"调 read_sections 可一次性精读各节候选，生成分节卡片。")

        print(f"--> [Tool:check_coverage] {report} | gaps={gaps}")
        return Command(update={
            "coverage_report": report, "gaps": gaps, "selected_papers": sel_out,
            "section_candidates": cands,
            "paper_pool": pool_updates,
            "messages": [ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)],
        })

    return check_coverage


def _check_coverage_llm_fallback(llm, state, tool_call_id, outline, pool, sections):
    """重排不可用时的降级路径：退回「LLM 看渲染池判相关性」（旧行为）。"""
    from langchain_core.prompts import ChatPromptTemplate
    thread_id = state.get("session_id", "")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是资料覆盖度评审。给定综述大纲的章节列表和论文列表，判断每个章节有多少篇**相关**论文。\n"
         "输出 JSON（只输出 JSON）：\n"
         '{{"coverage":[{{"section":"章节名","count":<相关论文数>,'
         '"status":"covered|thin|missing"}}]}}\n'
         "status：covered=≥4篇、thin=1-3篇、missing=0篇。"),
        ("user", "章节列表：{sections}\n\n论文列表：\n{data}"),
    ])
    coverage = []
    with tracing.span("llm", "check_coverage_fallback", thread_id) as sp:
        try:
            resp = (prompt | llm).invoke({"sections": "、".join(sections),
                                          "data": render_pool(pool, max_chars=16000)})
            um = getattr(resp, "usage_metadata", None) or {}
            sp["tokens"] = int(um.get("total_tokens", 0) or 0)
            raw = resp.content if hasattr(resp, "content") else str(resp)
            s, e = raw.find("{"), raw.rfind("}")
            coverage = json.loads(raw[s:e + 1]).get("coverage", []) if s != -1 and e != -1 else []
        except Exception as ex:
            print(f"--> [Tool:check_coverage] 降级路径解析失败: {ex}")

    report = {}
    for s in sections:
        hit = next((c for c in coverage if c.get("section") == s), None)
        report[s] = {"count": int((hit or {}).get("count") or 0),
                     "status": (hit or {}).get("status", "missing")}
    gaps = [s for s, v in report.items() if v["status"] == "missing"]
    lines = ["覆盖度报告（⚠️ 重排不可用，已降级为 LLM 判断）："]
    icon = {"covered": "✅", "thin": "⚠️", "missing": "❌"}
    for s, v in report.items():
        lines.append(f"  {icon.get(v['status'], '•')} {s}：{v['count']} 篇（{v['status']}）")
    return Command(update={"coverage_report": report, "gaps": gaps,
                           "messages": [ToolMessage(content="\n".join(lines),
                                                    tool_call_id=tool_call_id)]})

def fetch_and_card(items: list, llm, thread_id: str = "") -> tuple[list, list]:
    """并发取结构化全文 → Map 成卡片。items: [{"id","title","section"}, ...]

    抽成独立函数，让 read_papers（模型指定 id）与 read_sections（按节自动选）
    共用同一套「取全文 + 提炼卡片」逻辑，避免两份实现漂移。
    返回 (cards, failed_ids)。
    """
    from ResearcherAgent.core.graph_nodes import _title_relevant, _map_to_cards
    from ResearcherAgent.retrieval.fulltext import (fetch_arxiv_fulltext, fetch_pmc_fulltext,
                                      fetch_oa_fulltext)

    def _fetch(it: dict) -> dict | None:
        pid, title, section = it.get("id", ""), it.get("title", ""), it.get("section", "")
        with tracing.span("tool", "fulltext", thread_id, id=pid):
            if re.match(r"^PMC\d+", pid, re.I):
                t = fetch_pmc_fulltext(pid, 30000)
            elif pid.lower().startswith("arxiv:"):
                t = fetch_arxiv_fulltext(pid.split(":", 1)[1], 30000)
            elif re.match(r"^\d{4}\.\d{4,5}", pid):
                t = fetch_arxiv_fulltext(pid, 30000)
            elif pid.lower().startswith("oa:"):
                t = fetch_oa_fulltext(pid[3:], 30000)
            else:
                return None
        if not t or len(t) < 300:
            print(f"    ✗ {title[:40] or pid}：全文不可得")
            return None
        if title and not _title_relevant(title, t):
            print(f"    ✗ {title[:40]}：全文与标题不匹配，丢弃")
            return None
        return {"title": title or pid, "id": pid, "text": t, "section": section}

    items = [it for it in (items or []) if it.get("id")]
    if not items:
        return [], []
    with ThreadPoolExecutor(max_workers=TOOL_CONCURRENCY) as ex:
        blocks = [b for b in ex.map(_fetch, items) if b]

    cards = []
    if blocks:
        with tracing.span("llm", "map_to_cards", thread_id):
            cards = _map_to_cards(blocks, llm)
        sec_by_id = {b["id"]: b.get("section", "") for b in blocks}
        for c in cards:
            c["section"] = sec_by_id.get(c.get("id", ""), "")
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
        """按**章节**精读：每节从 check_coverage 产出的重排候选中取前 N 篇，
        取结构化全文并提炼成卡片（问题/方法/结果/局限），卡片带章节归属。

        sections：要精读的章节名列表；留空则读全部章节。
        per_section：每节读几篇（默认 2）。
        建议在 check_coverage 之后调用，这样每节都有专属素材，避免某些节一张卡片都没有。
        取不到全文的会跳过（不编造）。
        """
        cands = (state.get("section_candidates") or {}).get("sections") or {}
        if not cands:
            return Command(update={"messages": [ToolMessage(
                content="尚无分节候选，请先调用 check_coverage（它会做全池重排）。",
                tool_call_id=tool_call_id, status="error")]})

        want = [s for s in (sections or []) if s in cands] or list(cands.keys())
        picks = per_section_top_cards({s: cands[s] for s in want}, per_section=per_section)
        if not picks:
            return Command(update={"messages": [ToolMessage(
                content="候选为空，无可精读论文。", tool_call_id=tool_call_id, status="error")]})

        thread_id = state.get("session_id", "")
        pool = state.get("paper_pool", []) or []
        by_key = {paper_key(p): p for p in pool}
        items = []
        for c in picks:
            p = by_key.get(c.get("key") or "")
            pid = c.get("readable_id") or (p and _paper_id_for_fulltext(p)) or ""
            if pid:
                items.append({"id": pid, "title": c.get("title", ""),
                              "section": c.get("section", "")})
        if not items:
            return Command(update={"messages": [ToolMessage(
                content="各节候选里都没有可获取全文的 id（arXiv/PMC）。"
                        f"可改用有全文通路的源补检索（如 {_healthy_sources(thread_id)}），"
                        "或直接 ask_user。",
                tool_call_id=tool_call_id)]})

        print(f"--> [Tool:read_sections] {len(want)} 节 × 每节 {per_section} 篇 → 待读 {len(items)} 篇")
        cards, failed = fetch_and_card(items, llm, thread_id)

        lines = [f"分节精读完成：{len(cards)} 张卡片（每节最多 {per_section} 篇，已带章节归属）。"]
        by_sec = {}
        for c in cards:
            by_sec.setdefault(c.get("section", ""), []).append(c.get("title", ""))
        for s in want:
            got = by_sec.get(s, [])
            lines.append(f"  {'✓' if got else '✗'} {s[:34]}：{len(got)} 张")
            for t in got:
                lines.append(f"      · {t[:60]}")
        if failed:
            lines.append(f"  ✗ 取不到全文：{', '.join(failed[:6])}")
        return Command(update={
            "paper_cards": cards,
            "messages": [ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)],
        })

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
        selected = state.get("selected_papers", []) or []
        title_by_id = {}
        section_by_id = {}
        for s in selected:
            if s.get("id"):
                title_by_id[s["id"]] = s.get("title", "")
                section_by_id[s["id"]] = s.get("section", "")
        for p in pool:
            pid = _paper_id_for_fulltext(p)
            if pid and pid not in title_by_id:
                title_by_id[pid] = p.get("title", "")
                section_by_id[pid] = (p.get("sections") or [""])[0]

        ids = [str(i).strip() for i in ids if str(i).strip()][:6]
        if not ids:
            return Command(update={"messages": [ToolMessage(
                content="未给出 id。", tool_call_id=tool_call_id, status="error")]})

        def _fetch(pid: str) -> dict | None:
            with tracing.span("tool", "fulltext", thread_id, id=pid):
                if re.match(r"^PMC\d+", pid, re.I):
                    t = fetch_pmc_fulltext(pid, 30000)
                elif pid.lower().startswith("arxiv:"):
                    pid = pid.split(":", 1)[1]
                    t = fetch_arxiv_fulltext(pid, 30000)
                elif re.match(r"^\d{4}\.\d{4,5}", pid):
                    t = fetch_arxiv_fulltext(pid, 30000)
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

        with ThreadPoolExecutor(max_workers=TOOL_CONCURRENCY) as ex:
            blocks = [b for b in ex.map(_fetch, ids) if b]

        cards = []
        if blocks:
            with tracing.span("llm", "map_to_cards", thread_id):
                cards = _map_to_cards(blocks, llm)
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
                         "或直接用检索到的标题+摘要写作（ask_user 后进入写作），不必反复重试。")
        return Command(update={
            "paper_cards": cards,   # 只回新增卡片，AgentState 用 merge_paper_cards 追加去重
            "messages": [ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)],
        })

    return read_papers


def render_review_view(state: dict) -> str:
    """确定性渲染人机审核视图：大纲 + 每节覆盖度 + 按节列出的论文（不再用 LLM 重排版）。"""
    outline = state.get("review_outline", []) or []
    coverage = state.get("coverage_report", {}) or {}
    gaps = state.get("gaps", []) or []
    pool = state.get("paper_pool", []) or []
    cards = state.get("paper_cards", []) or []
    todos = state.get("todos", []) or []

    lines = ["## 📋 综述大纲与资料覆盖情况", ""]
    icon = {"covered": "✅", "thin": "⚠️", "missing": "❌"}
    if outline:
        for o in outline:
            sec = o.get("section", "")
            c = coverage.get(sec, {})
            status = c.get("status", "?")
            budget = ""
            if c.get("section_with_abstract") is not None:
                budget = (f"｜带摘要 {c['section_with_abstract']}/{c.get('section_total', '?')} 篇"
                          f"｜写作可注入 {c.get('injectable', '?')} 篇")
            lines.append(f"- {icon.get(status, '•')} **{sec}**：{c.get('count', '?')} 篇"
                         + budget
                         + (f" — {o.get('focus')}" if o.get("focus") else "")
                         + ("（建议补充）" if status == "thin" else "")
                         + ("（未找到，建议补充或删除此节）" if status == "missing" else ""))
        if gaps:
            lines.append("\n> ⚠️ 缺口方向：" + "、".join(gaps))
    else:
        lines.append("（尚未规划大纲）")
    lines.append(f"\n> 共检索到 {len(pool)} 篇去重论文，已精读 {len(cards)} 篇。")
    lines.append("> 请审核**范围**（分类是否合理、要不要增删方向），再决定是否继续。\n")
    if todos:
        ti = {"done": "✅", "in_progress": "🔄", "pending": "⬜"}
        lines.append("**Agent 任务清单：**")
        lines += [f"- {ti.get(t.get('status'), '⬜')} {t.get('task', '')}" for t in todos]
        lines.append("")
    lines.append("---\n")

    # 按节列论文
    listed = set()
    for o in outline:
        sec = o.get("section", "")
        sec_papers = [p for p in pool if sec in (p.get("sections") or [])]
        if not sec_papers:
            continue
        lines.append(f"### {sec}")
        for p in sec_papers[:15]:
            listed.add(paper_key(p))
            lines.append("- " + render_paper(p, with_abstract=False))
            ab = (p.get("abstract") or "").strip().replace("\n", " ")
            if ab:
                lines.append(f"  摘要：{ab[:120]}")
        if len(sec_papers) > 15:
            lines.append(f"  …另 {len(sec_papers) - 15} 篇")
        lines.append("")
    others = [p for p in pool if paper_key(p) not in listed]
    if others:
        lines.append("### 其它检索结果（未归属章节）")
        for p in others[:20]:
            lines.append("- " + render_paper(p, with_abstract=False))
        if len(others) > 20:
            lines.append(f"  …另 {len(others) - 20} 篇")
    return "\n".join(lines)


@tool
def ask_user(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    note: str = "",
) -> Command:
    """暂停并把当前大纲 + 覆盖度 + 资料清单交给用户审核（人机检查点）。

    用户会回复 accept（继续写作）/ skip（跳过审核直接写）/ re_search（附修改要求：增删方向、换关键词等）。
    ⚠️ 必须**单独调用**（不要与其它工具并列）；写综述前必须至少调用一次。
    note：可选，你想对用户补充说明的一句话（如哪些方向检索困难）。
    """
    search_round = int(state.get("search_round", 0) or 0)
    if search_round >= MAX_SEARCH_ROUNDS:
        return Command(update={"messages": [ToolMessage(
            content=f"已达最大审核/检索轮次（{MAX_SEARCH_ROUNDS}），不再询问用户，请直接收尾进入写作。",
            tool_call_id=tool_call_id)]})

    view = render_review_view(state)
    if note:
        view = f"> 💬 Agent 说明：{note}\n\n" + view
    print(f"--> [Tool:ask_user] 第 {search_round + 1} 轮，等待用户审核...")
    decision = interrupt({"type": "review_materials", "materials": view, "search_round": search_round + 1})

    fb = decision if isinstance(decision, dict) else {"action": str(decision), "feedback": ""}
    action = fb.get("action", "accept")
    feedback = (fb.get("feedback") or "").strip()
    print(f"--> [Tool:ask_user] 用户决策: {action} {feedback[:80]!r}")
    if action == "re_search":
        msg = (f"用户要求重新检索/调整范围：「{feedback or '（未说明）'}」。\n"
               "请据此用 make_outline 增量修改大纲（保留未提及章节），针对新增/薄弱方向再检索，"
               "check_coverage 后再次 ask_user。")
    elif action == "skip":
        msg = "用户选择跳过审核，请立即停止调用工具，进入写作。"
    else:
        msg = "用户已接受当前范围与资料，请立即停止调用工具，进入写作。"
    return Command(update={
        "human_feedback": json.dumps({"action": action, "feedback": feedback}, ensure_ascii=False),
        "search_round": search_round + 1,
        "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
    })


# ==========================================


@tool
def discover_directions(
    min_year: int = 2024,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[dict, InjectedState] = None,
) -> Command:
    """从论文池中**发现新兴研究方向**：LLM 从近两年文献提取候选方向 → 代码统计验证高频/增长
    → LLM 对照大纲决策（必须引用具体论文）→ 新方向并入子主题。

    用法：在 expand_seeds 扩展出论文池之后调用（池需 ≥20 篇且含近两年文献）。
    若新方向需要补充文献，可再用 expand_seeds(seed_titles=新方向的代表性论文标题) 扩展，
    然后重新 check_coverage。
    """
    pool = state.get("paper_pool") or []
    if len(pool) < 20:
        return Command(update={"messages": [ToolMessage(
            content=f"论文池仅 {len(pool)} 篇（需 ≥20 且含近两年文献），"
                    "先 expand_seeds 扩展后再发现新方向。",
            tool_call_id=tool_call_id, status="error")]})
    topic = state.get("topic") or state.get("user_request") or ""
    subs = state.get("subtopics") or []
    try:
        res = dd.run_discovery(pool, topic, subs, min_year=min_year)
    except Exception as e:
        return Command(update={"messages": [ToolMessage(
            content=f"新方向发现失败：{str(e)[:120]}。可稍后重试或继续现有流程。",
            tool_call_id=tool_call_id, status="error")]})
    accepted = [d for d in res.get("decisions", []) if d.get("verdict") == "accept"]
    if not accepted:
        msg = ("已核对候选方向与统计证据：未发现值得新增的研究方向。"
               "可继续 check_coverage 或直接进入写作。")
        return Command(update={"messages": [ToolMessage(
            content=msg, tool_call_id=tool_call_id)]})
    lines = [f"+ {d['concept']}（{d['reason'][:60]}）" for d in accepted]
    msg = ("发现并新增以下研究方向：\n" + "\n".join(lines) +
           f"\n\n新方向已并入子主题（{len(res.get('new_subtopics', []))} 个）。"
           "建议：对新方向再调用 expand_seeds 补一轮引文扩展，然后重新 check_coverage。")
    return Command(update={
        "subtopics": res.get("updated_subtopics", subs),
        "coverage_report": {}, "gaps": [], "selected_papers": [],
        "discovery_report": res.get("decisions", []),
        "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
    })



# 3. 渐进式工具加载：按 skills 选工具子集
# ==========================================

def build_local_tools(llm) -> dict[str, BaseTool]:
    out = {
        "make_subtopics": make_subtopics,
        "expand_seeds": expand_seeds,
        "make_outline": make_outline,
        "write_todos": write_todos,
        "check_coverage": make_check_coverage(llm),
        "discover_directions": discover_directions,
        "read_papers": make_read_papers(llm),
        "read_sections": make_read_sections(llm),
        "ask_user": ask_user,
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
    # —— CC 风格项目记忆（Claude Code Auto Memory 模式）：remember 记录 / consolidate 整理 ——
    from ResearcherAgent.memory.cc_memory import remember_memory, make_consolidate_memory
    out.update({
        "remember_memory": remember_memory,
        "consolidate_memory": make_consolidate_memory(llm),
    })
    return out


def select_tools(skills: list[str], mcp_tools: dict[str, BaseTool],
                 local_tools: dict[str, BaseTool]) -> list[BaseTool]:
    """62 个 MCP 工具一次性 bind → 按本次 skills 只暴露相关子集（提升选择准确率、缩小 prompt）。"""
    out: list[BaseTool] = []
    if "search_papers" in skills:
        out += [t for n, t in mcp_tools.items() if is_paper_search_tool(n)]
    if "search_web" in skills:
        out += [t for n, t in mcp_tools.items() if is_web_search_tool(n)]
    if "write_review" in skills:
        out += [local_tools[k] for k in ("write_todos", "make_subtopics", "expand_seeds",
                                         "make_outline", "check_coverage",
                                         "read_papers", "read_sections",
                                         "discover_directions")]
    if "human_review" in skills:
        out.append(local_tools["ask_user"])
    if "code_exec" in skills:
        # 代码执行工具组：模型可写文件、跑命令、跑训练脚本（后端由 LANGGRAPH_EXEC_BACKEND 控制）
        from ResearcherAgent.sandbox.code_executor import CODE_EXEC_TOOLS
        out += [local_tools[k] for k in CODE_EXEC_TOOLS if k in local_tools]
        # CC 风格项目记忆（跨会话沉淀，与代码执行配套）
        out += [local_tools[k] for k in ("remember_memory", "consolidate_memory")
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
        out += [local_tools[k] for k in ("remember_memory", "consolidate_memory")
                if k in local_tools]
    # 去重保序
    seen, uniq = set(), []
    for t in out:
        if t.name not in seen:
            seen.add(t.name)
            uniq.append(t)
    return uniq

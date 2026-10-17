#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""innovate · novelty_check 新颖性查重的确定性实现（独立可复现版本）。

运行时的 novelty_check 工具（ResearcherAgent/innovate/innovation_tools.py）用 LLM 判定
novel / partially_explored / duplicate；本脚本提供**纯确定性**的可复现判定：
  1. 从创新点描述中抽取方法关键词（Title / Keywords / Method 字段 + 常见算法词表）；
  2. 并行查询 OpenAlex / arXiv / Crossref（优先 OpenAlex，失败自动降级）；
  3. 用标题/摘要 token 重叠率 + 关键词命中数给每条命中打分，取 top-k；
  4. 最高分超过阈值 → duplicate；次高分超半阈值 → partially_explored；否则 novel。
输出 JSON Lines：{"id","name","verdict","confidence","evidence"}

用法：
    python novelty_check.py '{"id":"inn_01","name":"分解-多尺度频率增强PatchTST","method":"...","novelty":"..."}' \
        [--topk 3] [--threshold 0.35] [--timeout 20]

依赖：requests（沙箱/本地均预装）。Semantic Scholar 已整体移除（429 限流不稳定），
本脚本默认走 OpenAlex / arXiv（无需 key），与运行时检索白名单（OpenAlex/EuropePMC/arXiv）一致。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

try:
    import requests
except ImportError:
    sys.stderr.write("缺少 requests：pip install requests\n")
    sys.exit(1)

# 通用算法/方法词表（领域无关，仅用于给候选关键词排序；
# 其中含少量时序示例词，其他领域可在检索式生成阶段由 LLM 补充领域词）
_ALGO_TERMS = [
    "transformer", "attention", "patch", "token", "embedding", "frequency",
    "decomposition", "low-rank", "gate", "channel", "multi-scale", "state space",
    "mamba", "lstm", "gru", "cnn", "convolution", "adapter",
    "prompt", "contrastive", "graph", "diffusion", "flow",
]


def _keywords(innovation: dict) -> list[str]:
    """从创新点描述中抽取检索关键词（英文优先；领域词从描述本身提取，不写死）。"""
    src = " ".join(
        str(innovation.get(k, "")) for k in ("title", "name", "method", "novelty", "description")
    )
    src = re.sub(r"[^\w\s\-]", " ", src)
    tokens = [t for t in src.split() if len(t) >= 3]
    tokens = [t for t in tokens if not re.fullmatch(r"\d+", t)]
    # 优先保留与算法词表重合的词，再补高频原词
    ranked = [t for t in tokens if t.lower() in _ALGO_TERMS] + tokens
    seen, out = set(), []
    for t in ranked:
        tl = t.lower()
        if tl not in seen and len(out) < 6:
            seen.add(tl)
            out.append(t)
    if out:
        return out
    # 兜底：从名称/方法里取第一个词，避免空检索
    base = re.sub(r"[^\w\s\-]", " ", str(innovation.get("name") or innovation.get("method") or ""))
    base = [t for t in base.split() if len(t) >= 3]
    return (base[:1] or [""])


def _query_openalex(kw: str, limit: int = 8, timeout: int = 20) -> list[dict]:
    url = ("https://api.openalex.org/works?"
           + urllib.parse.urlencode({"search": kw, "per-page": limit,
                                     "sort": "relevance_score:desc"}))
    r = requests.get(url, timeout=timeout,
                     headers={"User-Agent": "ResearcherAgent-novelty-check/1.0"})
    r.raise_for_status()
    data = r.json()
    out = []
    for w in data.get("results", []):
        title = (w.get("title") or "").strip()
        if not title:
            continue
        abstract = w.get("abstract_inverted_index") or {}
        if abstract:  # 还原摘要词序
            pos = {}
            for word, idxs in abstract.items():
                for i in idxs:
                    pos[i] = word
            abstract = " ".join(pos[i] for i in sorted(pos))[:600]
        else:
            abstract = ""
        out.append({"title": title, "abstract": abstract,
                    "year": w.get("publication_year"),
                    "doi": (w.get("doi") or "").replace("https://doi.org/", "")})
    return out


def _query_arxiv(kw: str, limit: int = 5, timeout: int = 20) -> list[dict]:
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f'all:"{kw}"', "max_results": limit})
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    import xml.etree.ElementTree as ET
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    out = []
    for e in root.findall("a:entry", ns):
        title = " ".join((e.findtext("a:title", default="", namespaces=ns) or "").split())
        summary = " ".join((e.findtext("a:summary", default="", namespaces=ns) or "").split())[:600]
        out.append({"title": title, "abstract": summary,
                    "year": (e.findtext("a:published", default="", namespaces=ns) or "")[:4],
                    "doi": ""})
    return out


def _overlap_score(paper: dict, kw_set: set[str]) -> float:
    """标题 + 摘要的 token 重叠率 + 关键词命中数（确定性近似）。"""
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    tokens = set(re.findall(r"[a-z0-9\-]{3,}", text))
    if not tokens:
        return 0.0
    hit = len(tokens & kw_set)
    return min(1.0, hit / max(1, len(kw_set)))


def novelty_check(innovation: dict, topk: int = 3, threshold: float = 0.35,
                  timeout: int = 20) -> dict:
    kws = _keywords(innovation)
    kw_set = {k.lower() for k in kws}
    hits: list[dict] = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = []
        for kw in kws[:3]:
            futs.append(ex.submit(_query_openalex, kw, 8, timeout))
            futs.append(ex.submit(_query_arxiv, kw, 5, timeout))
        for f in futs:
            try:
                hits.extend(f.result())
            except Exception:
                continue  # 单源失败降级，不阻塞
    scored = sorted(
        ({**p, "score": _overlap_score(p, kw_set)} for p in hits),
        key=lambda p: p["score"], reverse=True)
    top = scored[:topk]
    best = top[0]["score"] if top else 0.0
    if best >= threshold:
        verdict, conf = "duplicate", f"high ({best:.2f})"
    elif best >= threshold / 2 and len(top) >= 2:
        verdict, conf = "partially_explored", f"medium ({best:.2f})"
    else:
        verdict, conf = "novel", f"low ({best:.2f})"
    evidence = " | ".join(
        f"{p.get('title', '')[:80]} ({p.get('year') or '?'}, score={p['score']:.2f})" for p in top)
    return {"id": innovation.get("id", ""), "name": innovation.get("name", ""),
            "verdict": verdict, "confidence": conf, "evidence": evidence}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="novelty_check 确定性可复现实现")
    ap.add_argument("innovation_json", help='创新点 JSON，如 \'{"id":"inn_01","name":"...","method":"..."}\'')
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--timeout", type=int, default=20)
    args = ap.parse_args(argv)
    try:
        innovation = json.loads(args.innovation_json)
    except json.JSONDecodeError as exc:
        print(f"innovation_json 不是合法 JSON：{exc}", file=sys.stderr)
        return 1
    result = novelty_check(innovation, args.topk, args.threshold, args.timeout)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

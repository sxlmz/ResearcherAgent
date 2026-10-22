# -*- coding: utf-8 -*-
"""
rerank.py —— Cross-Encoder 重排（相关性精排）

为什么需要它：
  检索是「宽召回」——search_crossref 返回什么就收什么，池子里 500+ 篇，没有相关性判断。
  旧做法是拿「有没有摘要」当排序代理，筛出来的不是「最相关的 N 篇」，而是「摘要最全的 N 篇」。
  重排是成熟 RAG 管线的标准第二级：宽召回 → 精确重排 → 少量进 prompt。

  实测（SiliconFlow Qwen3-Reranker-0.6B）：全池 520 篇一次调用 5.9s、72,800 tokens，
  判别力很好（相关论文 0.9998 / 0.9848，无关论文 0.0003）。所以可以整池重排，不用抽样。

接口（POST {BASE}/rerank）：
    {"model": ..., "query": "...", "documents": [...], "top_n": N}
  → {"results": [{"index": <原下标>, "relevance_score": <0-1>}], "meta": {"tokens": {...}}}

设计原则：**失败绝不让主流程崩**——API 不可用时返回 None，调用方降级回旧的排序逻辑。
"""
import json
import os
import time
import urllib.error
import urllib.request

from ResearcherAgent.core import cache
from ResearcherAgent.core import tracing

_TIMEOUT = 90          # 全池重排实测 5.9s；留足余量
_MAX_RETRY = 2
# 单次请求的文档上限（分批用）。实测 520 篇可一次提交，这里留更大余量。
_BATCH = 512


def _cfg() -> tuple[str, str, str]:
    """**每次调用时**读环境变量，不在导入时固化。

    动因：本模块可能在 load_dotenv() 之前被导入（模块级常量会读到空值），
    导致「配置明明写了却判定为不可用」。惰性读取消除这个时序依赖。
    """
    return (os.environ.get("RERANK_API_BASE", "").rstrip("/"),
            os.environ.get("RERANK_API_KEY", ""),
            os.environ.get("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B"))


def available() -> bool:
    b, k, m = _cfg()
    return bool(b and k and m)


def model_name() -> str:
    return _cfg()[2]


def _post(query: str, documents: list[str], top_n: int | None = None) -> dict | None:
    base, key, model = _cfg()
    body = {"model": model, "query": query, "documents": documents}
    if top_n:
        body["top_n"] = top_n
    req = urllib.request.Request(
        f"{base}/rerank",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _call_api(query: str, documents: list[str], sp: dict | None = None) -> list[float] | None:
    """调一次重排 API，返回与 documents 对齐的分数列表；失败返回 None。"""
    last_err = None
    for attempt in range(_MAX_RETRY + 1):
        try:
            data = _post(query, documents)
            tk = ((data.get("meta") or {}).get("tokens") or {}).get("input_tokens", 0)
            if sp is not None:
                sp["tokens"] = sp.get("tokens", 0) + int(tk or 0)
            scores = [0.0] * len(documents)
            for r in (data.get("results") or []):
                i = int(r["index"])
                if 0 <= i < len(scores):
                    scores[i] = float(r.get("relevance_score") or 0.0)
            return scores
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:150]}"
            if 400 <= e.code < 500 and e.code != 429:   # 请求本身有问题，重试无用
                break
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:150]}"
        if attempt < _MAX_RETRY:
            time.sleep(1.5 * (attempt + 1))
    print(f"    ⚠️ [rerank] 调用失败：{last_err}")
    return None


def rerank(query: str, documents: list[str], top_n: int | None = None,
           thread_id: str = "") -> list[tuple[int, float]] | None:
    """对 documents 按与 query 的相关性重排。

    **L4 缓存**：按「(model, query, 单篇文档 md5) → 分数」逐条缓存。
    重排是整条链路最贵的一级（6 节 × 500 篇 = 3000 次打分），而
    「同一节的查询 × 同一篇论文」在多次 check_coverage 之间大量重复，
    逐条缓存能让重复部分零成本命中（批量 mget，一次往返）。

    返回 [(原始下标, 分数)] 已按分数降序；失败返回 None（调用方应降级）。
    """
    documents = list(documents or [])
    if not documents or not available():
        return None
    if len(documents) > _BATCH:
        documents = documents[:_BATCH]
    _base, _key, model = _cfg()

    with tracing.span("tool", "rerank", thread_id, n_docs=len(documents),
                      query=query[:60]) as sp:
        # ---- v1.2.20 重排分数不再单独缓存 ----
        # 理由：选材缓存（sel:，粒度为「池+大纲 → 每节 top-K 候选」）已整体覆盖
        # 重排结果——同池同大纲重跑时 select 命中直接跳过全部重排；换池/换大纲时
        # 各节 query 随大纲 focus 变化，细粒度重排分数基本也 miss，缓存价值趋近于零。
        # 重排分数由选材缓存放行，本函数只做实时打分（打分是确定性计算，无外部配额）。
        got = _call_api(query, documents, sp)
        if got is None:
            sp["error"] = "api_failed"
            return None
        scores = got

        out = [(i, float(s)) for i, s in enumerate(scores) if s is not None]
        if not out:
            return None
        out.sort(key=lambda x: -x[1])
        # ⚠️ 默认返回**全部分数**，不在这里截断 top_n。
        # 原因：调用方需要「全池里有多少篇过阈值」这个统计量；若在这里就只留 top-K，
        # 再拿这 K 篇去比阈值，等于拿最优的 K 个过线——阈值永远命中，判定失效。
        return out[:top_n] if top_n else out

"""
tracing.py
==========
轻量可观测性：节点 / 工具 / LLM 的耗时、token、失败率。

- 每个 span 追加一行 JSON 到 logs/trace.jsonl（不可变事件日志，便于事后分析）
- 内存里按 thread_id 汇总 {steps, tool_calls, tool_failures, tokens, ms}，
  供 run_stream 的 result 事件和 GET /api/trace 使用

不引入 Langfuse 等外部依赖——需要时把 `_emit` 换成 SDK 调用即可。
"""

import json
import time
import threading
from contextlib import contextmanager
from pathlib import Path

TRACE_PATH = Path("logs") / "trace.jsonl"
_lock = threading.Lock()
_summary: dict[str, dict] = {}      # thread_id → 汇总
_MAX_THREADS = 500                   # 内存汇总上限，防止长时间运行无限增长


def _emit(rec: dict) -> None:
    try:
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock, TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _bump(thread_id: str, kind: str, ms: float, ok: bool, tokens: int) -> None:
    if not thread_id:
        return
    with _lock:
        if thread_id not in _summary and len(_summary) >= _MAX_THREADS:
            _summary.pop(next(iter(_summary)))
        s = _summary.setdefault(thread_id, {
            "steps": 0, "tool_calls": 0, "tool_failures": 0,
            "llm_calls": 0, "tokens": 0, "ms": 0.0,
        })
        s["ms"] += ms
        if kind == "node":
            s["steps"] += 1
        elif kind == "tool":
            s["tool_calls"] += 1
            if not ok:
                s["tool_failures"] += 1
        elif kind == "llm":
            s["llm_calls"] += 1
        s["tokens"] += tokens


@contextmanager
def span(kind: str, name: str, thread_id: str = "", **extra):
    """用法：
        with span("tool", "search_crossref", thread_id) as sp:
            sp["tokens"] = 1234      # 可选
            sp["cache_hit"] = 42     # 可选，任意附加字段都会被记录
    kind ∈ {"node", "tool", "llm"}。异常会被记录为 ok=False 后继续抛出。

    ⚠️ 运行期往 `sp` 里塞的**任意字段**都会被导出。曾只导出 `tokens`，
    导致 `rerank` 里写的 `sp["cache_hit"]` 被静默丢弃——埋了却查不到，
    无法统计缓存到底省了多少时间。
    """
    t0 = time.perf_counter()
    sp: dict = {"tokens": 0}
    ok = True
    try:
        yield sp
    except BaseException:
        ok = False
        raise
    finally:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        rec = {"ts": time.time(), "thread_id": thread_id, "kind": kind, "name": name,
               "ms": ms, "ok": ok, **extra}
        for k, v in sp.items():            # ← 带上 sp 里运行期设置的字段
            rec.setdefault(k, v)
        rec.setdefault("tokens", 0)
        _emit(rec)
        _bump(thread_id, kind, ms, ok, int(sp.get("tokens", 0) or 0))


def record(kind: str, name: str, thread_id: str = "", ms: float = 0.0,
           ok: bool = True, tokens: int = 0, **extra) -> None:
    """非上下文管理器版本（已知耗时时直接记一条）。"""
    _emit({"ts": time.time(), "thread_id": thread_id, "kind": kind, "name": name,
           "ms": ms, "ok": ok, "tokens": tokens, **extra})
    _bump(thread_id, kind, ms, ok, tokens)


def summary(thread_id: str) -> dict:
    with _lock:
        return dict(_summary.get(thread_id, {}))


def reset(thread_id: str) -> None:
    with _lock:
        _summary.pop(thread_id, None)


def recent_spans(thread_id: str, limit: int = 200) -> list[dict]:
    """从 trace.jsonl 尾部读取该 thread 的最近 spans（供 /api/trace）。"""
    if not TRACE_PATH.exists():
        return []
    try:
        lines = TRACE_PATH.read_text(encoding="utf-8").splitlines()[-5000:]
    except Exception:
        return []
    out = []
    for ln in reversed(lines):
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        if rec.get("thread_id") == thread_id:
            out.append(rec)
            if len(out) >= limit:
                break
    return list(reversed(out))

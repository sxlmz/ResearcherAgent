# -*- coding: utf-8 -*-
"""
retrieval_cache.py —— 检索结果多级缓存（内存 → SQLite 持久化 → Redis 共享）

为什么加这一层：
  MCP 检索（arxiv/crossref）的 L3 缓存挂在 Redis 上，TTL 24h 且 Redis 不可用时
  降级为进程内内存（进程结束即丢）。检索是整条管线里最耗时的一级（实测单源单查询
  数十秒），重复 query 反复打外部源浪费严重。这里补一个**本地 SQLite 持久层**，
  组成三级：L0 进程内存（最快）→ L1 SQLite 数据库（跨会话/跨进程持久，TTL 7 天）
  → L2 Redis（现有 cache.search_*，跨项目共享，TTL 24h）。

读取顺序：内存 → SQLite → Redis（Redis 命中时回填 SQLite+内存）。
写入：三层都写。任何一层失败都吞掉（缓存是加速器，不是依赖）。
键：source + 规范化 query + max_results（与 cache.search_* 的 L3 键一致口径）。
"""
import hashlib
import sqlite3
import threading
import time
from pathlib import Path

from agent.core.paths import REPO_ROOT

DB_PATH = REPO_ROOT / "logs" / "retrieval_cache.db"
TTL = 7 * 24 * 3600          # SQLite 层 7 天（比 Redis 24h 长，作持久兜底）
_mem: dict = {}              # L0: {(source, qkey): (expires, payload)}
_mem_lock = threading.Lock()
_db_lock = threading.Lock()


def _qkey(source: str, query, max_results) -> str:
    q = " ".join(str(query).split()).lower()
    return hashlib.md5(f"{source}|{q}|{max_results}".encode("utf-8")).hexdigest()


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS retrieval_cache ("
        "source TEXT, qkey TEXT, payload TEXT, created REAL, expires REAL,"
        "PRIMARY KEY(source, qkey))")
    return conn


def _mem_get(source: str, key: str):
    with _mem_lock:
        v = _mem.get((source, key))
        if v is None:
            return None
        exp, payload = v
        if exp > time.time():
            return payload
        _mem.pop((source, key), None)
    return None


def _mem_set(source: str, key: str, payload, expires: float):
    with _mem_lock:
        _mem[(source, key)] = (expires, payload)


def _db_get(source: str, key: str):
    try:
        with _db_lock:
            conn = _conn()
            try:
                row = conn.execute(
                    "SELECT payload, expires FROM retrieval_cache "
                    "WHERE source=? AND qkey=?", (source, key)).fetchone()
            finally:
                conn.close()
        if row and row[1] > time.time():
            return row[0], row[1]
    except Exception:
        pass
    return None


def _db_set(source: str, key: str, payload, expires: float):
    try:
        with _db_lock:
            conn = _conn()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO retrieval_cache (source, qkey, payload, created, expires) "
                    "VALUES (?,?,?,?,?)",
                    (source, key, payload, time.time(), expires))
                conn.commit()
            finally:
                conn.close()
    except Exception:
        pass


def search_get(source: str, query, max_results=0):
    """多级读：内存 → SQLite → Redis。返回 payload（字符串）或 None。"""
    source = str(source or "")
    key = _qkey(source, query, max_results)
    v = _mem_get(source, key)
    if v is not None:
        return v
    row = _db_get(source, key)
    if row is not None:
        _mem_set(source, key, row[0], row[1])
        return row[0]
    try:
        from agent.core import cache
        v = cache.search_get(source, query, max_results)
        if v is not None:
            exp = time.time() + TTL
            _mem_set(source, key, v, exp)
            _db_set(source, key, v, exp)
            return v
    except Exception:
        pass
    return None


def search_set(source: str, query, max_results=0, payload=None):
    """多级写：内存 + SQLite + Redis。payload 须为字符串（与 cache.search_set 口径一致）。"""
    if payload is None:
        return
    source = str(source or "")
    key = _qkey(source, query, max_results)
    exp = time.time() + TTL
    _mem_set(source, key, payload, exp)
    _db_set(source, key, payload, exp)
    try:
        from agent.core import cache
        cache.search_set(source, query, max_results, payload)
    except Exception:
        pass


def stats() -> dict:
    """命中/存储统计（调试用）。"""
    try:
        with _db_lock:
            conn = _conn()
            try:
                total = conn.execute("SELECT COUNT(*) FROM retrieval_cache").fetchone()[0]
                live = conn.execute(
                    "SELECT COUNT(*) FROM retrieval_cache WHERE expires > ?",
                    (time.time(),)).fetchone()[0]
            finally:
                conn.close()
        return {"db_rows": total, "db_live": live, "mem_entries": len(_mem)}
    except Exception:
        return {"db_rows": -1, "db_live": -1, "mem_entries": len(_mem)}

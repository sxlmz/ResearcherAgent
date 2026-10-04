# -*- coding: utf-8 -*-
"""
cache.py —— Redis 五层缓存（精确匹配，参考 RAGFlow / RAG-test 的缓存层）

为什么需要：
  现在的研究阶段已经是一条 RAG 管线（宽召回 → 重排 → 选材 → 生成），
  每一级都有「同样输入反复算」的情况，而重排是其中最贵的一级：
  6 节 × 500 篇 = 3000 次打分，每补检索一次池子就变，若不做缓存会反复全额付费。

五层（全部精确匹配，一字之差即 miss——不做相似缓存，避免"看着像"的污染）：

  L1 LLM       (model + system + user)                → 文本         TTL 24h
  L2 Embedding (model + text)                         → 向量         TTL 24h
  L3 检索      (source + query + max_results)         → 论文 JSON    TTL 24h
  L4 重排分数  (model + query + 文档 md5)              → 分数         TTL 24h   ← 最值钱
  L5 选材      (池指纹 + 大纲指纹)                     → 每节候选      TTL 24h

隔离与降级（重要）：
  - 这台机器上的 Redis（mall-redis）是**多项目共享**的：DB0 是商城、DB3 是 RAG-test。
    本模块固定用 **DB1 + `agn:` 前缀**，flush 只删前缀内的键，绝不影响他人。
  - Redis 不可用时**自动退回进程内内存**，主流程不因缓存挂掉而失败。
  - 任何一层读写异常都被吞掉（缓存是加速器，不是依赖）。

失效（**只有 TTL**，没有索引代次 —— 这是刻意设计，见下）：
  - L1/L2 的输入是「模型 + 文本」，L3 的输入是「源 + 查询词」，L4 是「查询 × 单篇文档」，
    L5 是「论文池指纹 + 大纲指纹」——每一层的键都**已经含全了决定它的输入**。
  - 早期版本给 L3 的键掺了一个全局 `stamp`（删文档/会话时自增），实测产生两个荒谬后果：
      · 外部检索源（arxiv/crossref/…）的结果与我们自己的数据库毫无关系，
        却被"删了一个文档"整体作废；跨会话更是永不命中（命中率实测 0%）。
      · 因为一旦失效就查不到任何旧值，**stale fallback 无从实现**（源挂了没有旧结果可兜底）。
    所以 stamp 机制已整体移除。真正与本地库有关的 L5 用**内容指纹**（池/大纲的哈希），
    池或大纲一变指纹就变，比全局代次更精确、也不会被无关操作误伤。
  - `flush()` 只清本项目前缀。
"""
import hashlib
import json
import os
import threading
import time

try:
    import xxhash
    def _h(*parts) -> str:
        h = xxhash.xxh64()
        for p in parts:
            h.update(str(p).encode("utf-8", "replace"))
            h.update(b"\x1f")          # 分隔符，避免 ("ab","c") 与 ("a","bc") 撞键
        return h.hexdigest()
except Exception:                       # xxhash 缺失时退回 blake2b（够快且无依赖）
    def _h(*parts) -> str:
        h = hashlib.blake2b(digest_size=16)
        for p in parts:
            h.update(str(p).encode("utf-8", "replace"))
            h.update(b"\x1f")
        return h.hexdigest()


PREFIX = "agn:"
DB = int(os.environ.get("CACHE_REDIS_DB", "1"))     # 默认 DB1，与同一 Redis 实例上的其他应用隔离
TTL = int(os.environ.get("CACHE_TTL", str(24 * 3600)))
ENABLED = os.environ.get("CACHE_ENABLED", "1") not in ("0", "false", "False")

_lock = threading.Lock()


class _MemoryBackend:
    """Redis 不可用时的进程内兜底（带 TTL，避免长跑进程内存无界增长）。"""

    def __init__(self):
        self._d: dict = {}

    def get(self, k):
        v = self._d.get(k)
        if v is None:
            return None
        exp, val = v
        if exp and exp < time.time():
            self._d.pop(k, None)
            return None
        return val

    def set(self, k, v, ttl=None):
        self._d[k] = (time.time() + ttl if ttl else 0, v)

    def mget(self, keys):
        return [self.get(k) for k in keys]

    def delete_prefix(self, prefix):
        with _lock:
            ks = [k for k in self._d if k.startswith(prefix)]
            for k in ks:
                self._d.pop(k, None)
        return len(ks)


class _Cache:
    def __init__(self):
        self._backend = None
        self._name = "off"
        self.stats = {k: {"hit": 0, "miss": 0}
                      for k in ("llm", "embed", "search", "rerank", "select")}
        self._connect()

    def _connect(self):
        if not ENABLED:
            return
        try:
            import redis as redis_py
            c = redis_py.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", "6379")),
                db=DB, password=os.environ.get("REDIS_PASSWORD") or None,
                socket_connect_timeout=2, socket_timeout=3, decode_responses=True)
            c.ping()
            self._backend, self._name = c, "redis"
            print(f"✅ 缓存后端 = redis (db={DB}, prefix={PREFIX})")
        except Exception as e:
            self._backend = _MemoryBackend()
            self._name = "memory"
            print(f"ℹ️ Redis 不可用（{type(e).__name__}），缓存降级为进程内内存")

    # ── 低层 ─────────────────────────────────────────────
    def _get(self, key):
        if not self._backend:
            return None
        try:
            return self._backend.get(PREFIX + key)
        except Exception:
            return None

    def _set(self, key, val, ttl=TTL):
        if not self._backend or val is None:
            return
        try:
            if self._name == "redis":
                self._backend.set(PREFIX + key, val, ex=ttl)
            else:
                self._backend.set(PREFIX + key, val, ttl=ttl)
        except Exception:
            pass

    def _mget(self, keys):
        if not self._backend:
            return [None] * len(keys)
        try:
            return self._backend.mget([PREFIX + k for k in keys])
        except Exception:
            return [None] * len(keys)

    def _bump(self, layer, hit):
        self.stats[layer]["hit" if hit else "miss"] += 1

    # ── L1 · LLM ─────────────────────────────────────────
    def llm_get(self, model, system, user, extra=""):
        if not ENABLED:
            return None
        v = self._get("llm:" + _h(model, system, user, extra))
        self._bump("llm", bool(v))
        return v

    def llm_set(self, model, system, user, value, extra=""):
        self._set("llm:" + _h(model, system, user, extra), value)

    # ── L2 · Embedding ───────────────────────────────────
    def embed_get(self, model, text):
        if not ENABLED:
            return None
        v = self._get("emb:" + _h(model, text))
        self._bump("embed", bool(v))
        try:
            return json.loads(v) if v else None
        except Exception:
            return None

    def embed_set(self, model, text, vec):
        self._set("emb:" + _h(model, text), json.dumps(vec))

    # ── L3 · 检索结果 ────────────────────────────────────
    def search_get(self, source, query, max_results):
        """⚠️ key 里**没有** stamp：外部检索源的结果只由「源 + 查询词 + 条数」决定。"""
        if not ENABLED:
            return None
        v = self._get("src:" + _h(source, " ".join(str(query).split()).lower(),
                                  max_results))
        self._bump("search", bool(v))
        return v

    def search_set(self, source, query, max_results, payload):
        self._set("src:" + _h(source, " ".join(str(query).split()).lower(),
                              max_results), payload)

    # ── L4 · 重排分数（按「查询 × 单篇文档」缓存，跨请求复用）────
    def rerank_mget(self, model, query, docs):
        """批量取分，一次往返。返回 [float|None]，None 表示未命中。"""
        if not ENABLED:
            return [None] * len(docs)
        q = " ".join(str(query).split()).lower()
        keys = ["rrk:" + _h(model, q, hashlib.md5(str(d).encode("utf-8")).hexdigest())
                for d in docs]
        vals = self._mget(keys)
        out = []
        for v in vals:
            if v is None:
                self._bump("rerank", False)
                out.append(None)
                continue
            try:
                out.append(float(v))
                self._bump("rerank", True)
            except Exception:
                self._bump("rerank", False)
                out.append(None)
        return out

    def rerank_mset(self, model, query, docs, scores):
        q = " ".join(str(query).split()).lower()
        for d, s in zip(docs, scores):
            if s is None:
                continue
            self._set("rrk:" + _h(model, q, hashlib.md5(str(d).encode("utf-8")).hexdigest()),
                      str(float(s)))

    # ── L5 · 选材结果（每节候选）────────────────────────
    def select_get(self, pool_fp, outline_fp):
        if not ENABLED:
            return None
        v = self._get("sel:" + _h(pool_fp, outline_fp))
        self._bump("select", bool(v))
        try:
            return json.loads(v) if v else None
        except Exception:
            return None

    def select_set(self, pool_fp, outline_fp, payload):
        self._set("sel:" + _h(pool_fp, outline_fp),
                  json.dumps(payload, ensure_ascii=False))

    # ── 运维 ─────────────────────────────────────────────
    def summary(self) -> dict:
        out = {"backend": self._name, "db": DB if self._name == "redis" else None}
        tot_h = tot_m = 0
        for layer, st in self.stats.items():
            h, m = st["hit"], st["miss"]
            tot_h += h
            tot_m += m
            out[layer] = {"hit": h, "miss": m,
                          "rate": round(h / (h + m), 3) if (h + m) else 0.0}
        out["total"] = {"hit": tot_h, "miss": tot_m,
                        "rate": round(tot_h / (tot_h + tot_m), 3) if (tot_h + tot_m) else 0.0}
        return out

    def reset_stats(self):
        for k in self.stats:
            self.stats[k] = {"hit": 0, "miss": 0}

    def flush(self):
        """只删本项目前缀的键——Redis 是共享的，绝不能 flushdb。"""
        if self._name == "redis":
            n = 0
            for k in self._backend.scan_iter(match=PREFIX + "*", count=500):
                self._backend.delete(k)
                n += 1
            return n
        return self._backend.delete_prefix(PREFIX)


cache = _Cache()

# ── 模块级委托 ──
# 让调用点写 `cache.llm_get(...)` 而不是 `cache.cache.llm_get(...)`——
# 否则 `import cache` 拿到的是模块、方法却在单例上，很容易写成 cache.rerank_mget 而报错。
llm_get = cache.llm_get
llm_set = cache.llm_set
embed_get = cache.embed_get
embed_set = cache.embed_set
search_get = cache.search_get
search_set = cache.search_set
rerank_mget = cache.rerank_mget
rerank_mset = cache.rerank_mset
select_get = cache.select_get
select_set = cache.select_set
summary = cache.summary
reset_stats = cache.reset_stats
flush = cache.flush
backend_name = lambda: cache._name

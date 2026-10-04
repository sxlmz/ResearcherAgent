# -*- coding: utf-8 -*-
"""
memory_orchestrator.py —— 记忆检索编排器

职责：一次请求内按「路由 + 短路 + 懒加载」决定每层记忆查不查，避免逐层全查。

流程：
  ① L0 工作记忆（Redis）—— 最快，直接读
  ② 记忆路由 —— 确定性判断「这个问题要不要碰用户文档库」（should_use_docs）
      闲聊/无关 → 短路，不查重型存储
  ③ L2 语义记忆（Infinity mem 表）+ 对话历史检索（Infinity conv 表）
  ④ 文档检索（Infinity rag 表）—— 仅当确定性路由命中
  ⑤ 懒加载 —— 仅当命中分超过 cite 阈值才取原文

运行环境：Python 3.11。依赖 memory_stack。
"""
import os

# 阈值（可配置，见记忆系统升级方案 §10）
DOC_TRIGGER_THRESHOLD = float(os.environ.get("DOC_TRIGGER_THRESHOLD", "0.35"))  # 路由触发检索
DOC_INJECT_THRESHOLD = float(os.environ.get("DOC_INJECT_THRESHOLD", "0.4"))     # 注入 prompt 最低分
DOC_CITE_THRESHOLD = float(os.environ.get("DOC_CITE_THRESHOLD", "0.6"))         # 懒加载原文最低分


class MemoryOrchestrator:
    def __init__(self, stack, embed_fn=None):
        self.stack = stack          # MemoryStack
        self.infinity = stack.infinity
        self.pg = stack.pg
        self.redis = stack.redis
        self.embed_fn = embed_fn    # str -> list[float]

    def _degraded(self) -> bool:
        """embedding 是否降级（远程不可用 → 确定性伪向量）。"""
        return bool(getattr(self.embed_fn, "degraded", False))

    # ── 确定性文档路由 ──
    def should_use_docs(self, query, user_id) -> bool:
        """对用户文档库做一次廉价检索，判断是否需要 RAG（确定性，不靠 LLM）。

        优先 cosine 相似度 ≥ 阈值；embedding 降级或低余弦时，
        用全文 BM25 关键词命中兜底（关键字重叠即视为相关）。
        """
        if not self.embed_fn:
            return False
        try:
            vec = self.embed_fn(query)
            cosine = self.infinity.search_docs(user_id, query, vec, topk=1,
                                               threshold=DOC_TRIGGER_THRESHOLD,
                                               search_method="cosine")
            if cosine and cosine[0]["score"] >= DOC_TRIGGER_THRESHOLD:
                return True
            # 全文兜底：BM25 关键词命中
            ft = self.infinity.search_docs(user_id, query, vec, topk=1,
                                           threshold=0.0, search_method="bm25")
            return bool(ft)
        except Exception as e:
            print(f"--> [Orchestrator] 文档路由失败: {e}")
            return False

    # ── 记忆检索（核心） ──
    def retrieve(self, query, user_id, session_id, skills=None) -> dict:
        """按需检索记忆，返回融合后的上下文 dict。

        任何存储层的故障都被隔离（try/except 降级为空），绝不让单库异常拖垮 agent。
        """
        result = {
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "use_docs": False,
            "working_memory": [],
            "semantic_memories": [],
            "conversation_hits": [],
            "documents": [],
            "lazy_raw": None,          # 懒加载：高分文档的完整原文
            "profiles": [],            # L1 常驻档案（来自 Redis 缓存或 Postgres）
        }

        # ① L0 工作记忆（Redis，最快）
        try:
            result["working_memory"] = self.redis.get_working(session_id) or []
        except Exception:
            result["working_memory"] = []

        # ② 记忆路由：chitchat / 无意图 → 短路
        intent = self._classify_intent(query)
        if intent in ("chitchat", "greeting"):
            return result   # 短路，不碰任何重型存储

        if not self.embed_fn:
            return result

        try:
            vec = self.embed_fn(query)
        except Exception:
            return result

        # ③ L1 常驻档案（Redis 缓存优先，未命中查 Postgres）
        try:
            result["profiles"] = self._load_profiles(user_id)
        except Exception:
            result["profiles"] = []

        # ③ L2 语义记忆 + 对话历史（会话隔离：只查当前 session，不跨会话共享）
        try:
            result["semantic_memories"] = self.infinity.search_memories(
                user_id, vec, topk=5, session_id=session_id)
        except Exception:
            result["semantic_memories"] = []
        try:
            result["conversation_hits"] = self.infinity.search_conversation(
                user_id, query, vec, topk=3, threshold=0.1, session_id=session_id)
        except Exception:
            result["conversation_hits"] = []

        # ④ 文档检索 —— 仅当确定性路由命中
        degraded = self._degraded()
        if intent == "doc" or self.should_use_docs(query, user_id):
            try:
                # 混合检索（稠密 + 全文 BM25）；embedding 降级时稠密失效，BM25 仍召回
                docs = self.infinity.search_docs(
                    user_id, query, vec, topk=10, threshold=0.0, search_method="hybrid")
                docs = self._rerank(query, docs)[:5]
                # 注入前相关性门槛：正常 embedding 下按 cosine 过滤；降级时信任 BM25 召回
                if not degraded:
                    docs = [d for d in docs if d["score"] >= DOC_INJECT_THRESHOLD]
                result["documents"] = docs
                result["use_docs"] = bool(docs)
                # ⑤ 懒加载：仅最高分文档 ≥ cite 阈值才取原文（降级时取 top 文档）
                if docs and (degraded or docs[0]["score"] >= DOC_CITE_THRESHOLD):
                    result["lazy_raw"] = docs[0].get("content")
                    result["lazy_doc_id"] = docs[0].get("doc_id")
            except Exception:
                result["documents"] = []

        return result

    # ── 组装注入 prompt 的文本（供 graph 节点使用） ──
    # 注意：不含 profiles（L1 常驻档案由 _with_memory 统一注入，避免重复）
    def format_for_prompt(self, r: dict, max_chars: int = 3000) -> str:
        parts = []
        if r["working_memory"]:
            parts.append("【会话上下文】\n" + "\n".join(
                f"- [{c.get('role')}] {c.get('content', '')[:150]}" for c in r["working_memory"][-4:]))
        if r["semantic_memories"]:
            parts.append("【历史记忆】\n" + "\n".join(
                f"- ({m['mem_type']}) {m['summary'][:200]}" for m in r["semantic_memories"]))
        if r["conversation_hits"]:
            parts.append("【相关对话片段】\n" + "\n".join(
                f"- [{c.get('role')}] {c.get('content', '')[:200]}" for c in r["conversation_hits"]))
        if r["documents"]:
            parts.append("【文档检索结果】\n" + "\n".join(
                f"- [{d.get('docnm','')}] {d.get('content', '')[:400]}" for d in r["documents"]))
        text = "\n\n".join(parts)
        return text[:max_chars]

    # ── 内部辅助 ──
    def _load_profiles(self, user_id):
        cached = self.redis.get_profile_cache(user_id)
        if cached is not None:
            return cached
        profiles = self.pg.get_profiles(user_id)
        if profiles:
            self.redis.set_profile_cache(user_id, profiles)
        return profiles

    @staticmethod
    def _classify_intent(query):
        q = (query or "").strip()
        if len(q) <= 4:
            return "chitchat"
        greeting = ("你好", "在吗", "嗨", "hello", "hi", "谢谢", "再见")
        if any(g in q.lower() for g in greeting):
            return "chitchat"
        return "semantic"

    def _rerank(self, query, docs):
        """轻量重排：按综合分降序（cross-encoder 的替代，可后续替换）。"""
        scored = [d for d in docs if d.get("score") is not None]
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored

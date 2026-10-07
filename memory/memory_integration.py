# -*- coding: utf-8 -*-
"""
memory_integration.py —— 把记忆系统 v2 接入主 agent 的写入钩子

在原有存储（JSONL / MD / Chroma）之外，把记忆镜像写入新存储：
  - 对话原文 → Infinity conv_{uid}（用户级，跨会话可检索）
  - L1 常驻档案 / L2 语义记忆 → 由 memory_classifier.classify_and_store 写入

全部 best-effort：任何异常都被吞掉，绝不因记忆系统故障影响主 agent 运行。
用环境变量 MEMORY_STACK_ENABLED=1（默认）开启。
"""
import os

_ENABLED = os.environ.get("MEMORY_STACK_ENABLED", "1") == "1"

_embed = None
_infinity = None
_initialized = False


def _ensure():
    """惰性初始化（首次调用才连接，避免 import 开销/失败影响主 agent）。"""
    global _embed, _infinity, _initialized
    if _initialized:
        return _infinity is not None
    _initialized = True
    if not _ENABLED:
        return False
    try:
        from ResearcherAgent.memory.memory_classifier import make_embed_fn
        from ResearcherAgent.memory.memory_stack import InfinityStore
        _embed = make_embed_fn()
        _infinity = InfinityStore()
        return True
    except Exception as e:
        print(f"--> [MemStack] 初始化失败（不影响主 agent）: {e}")
        return False


WORKING_MEMORY_LIMIT = int(os.environ.get("WORKING_MEMORY_LIMIT", "20"))


def on_conversation_turn(user_id, session_id, role, content):
    """每轮对话的写入钩子（best-effort）：
      ① L0 工作记忆 → Redis（本会话最近 N 轮，TTL 自动过期）
      ② 对话原文   → Infinity conv_{uid}（用户级，会话可检索）
    """
    # ① 工作记忆（Redis，本地快，不依赖 embedding）
    try:
        from ResearcherAgent.memory.memory_stack import RedisWorkingMemory
        r = RedisWorkingMemory()
        cur = r.get_working(session_id) or []
        cur.append({"role": role, "content": str(content)[:500]})
        r.set_working(session_id, cur[-WORKING_MEMORY_LIMIT:])
    except Exception as e:
        print(f"--> [MemStack] 工作记忆写入 Redis 失败: {e}")

    # ② 对话原文 → Infinity
    if not _ensure():
        return
    try:
        _infinity.save_conversation_turn(user_id, session_id, role, content, _embed)
    except Exception as e:
        print(f"--> [MemStack] 对话写入 Infinity 失败: {e}")


# ---- 记忆检索编排器单例（供 graph 节点 memory_node / simple_chat_node 调用）----
_orch = None
_orch_stack = None
_orch_embed = None


def get_orchestrator():
    """惰性构建并返回全局 MemoryOrchestrator（真实 embedding，带降级兜底）。"""
    global _orch, _orch_stack, _orch_embed
    if _orch is None:
        from ResearcherAgent.memory.memory_classifier import make_embed_fn
        from ResearcherAgent.memory.memory_stack import MemoryStack
        from ResearcherAgent.memory.memory_orchestrator import MemoryOrchestrator
        _orch_embed = make_embed_fn()
        _orch_stack = MemoryStack(embed_fn=_orch_embed)
        _orch = MemoryOrchestrator(_orch_stack, embed_fn=_orch_embed)
    return _orch

def store_profile(user_id, profile_type, content, session_id=""):
    """写一条 L1 常驻档案到 PostgreSQL user_profiles（best-effort）。"""
    try:
        orch = get_orchestrator()
        return orch.stack.pg.upsert_profile(user_id, profile_type, content, session_id)
    except Exception as e:
        print(f"--> [MemStack] 档案写入 Postgres 失败: {e}")
        return None

def delete_session_data(user_id, session_id):
    """删除某会话在记忆系统 v2 中的数据：
    对话原文（Infinity conv）+ 该会话的语义记忆（Infinity mem）+ 工作记忆（Redis）。
    注意：L1 档案（Postgres）与文档库（rag）属跨会话，不随单条对话删除。
    """
    try:
        orch = get_orchestrator()
        orch.stack.infinity.clear_conversation(user_id, session_id)
        orch.stack.infinity.clear_session_memories(user_id, session_id)
        orch.stack.redis.r.delete(f"working:{session_id}")
        # 缓存无需失效：L3（检索）只依赖源+查询词（外部数据），L5（选材）的键是
        # 「论文池指纹 + 大纲指纹」，池或大纲一变指纹就变——比全局代次更精确。
    except Exception as e:
        print(f"--> [MemStack] 会话数据删除降级: {e}")


def delete_document(user_id, doc_id):
    """删除一篇文档：Infinity rag chunk + Postgres 元数据 + 磁盘上传文件。"""
    from pathlib import Path
    try:
        orch = get_orchestrator()
        orch.stack.infinity.clear_doc_chunks(user_id, doc_id)
        cur = orch.stack.pg._cur()
        cur.execute("SELECT source_key FROM documents WHERE doc_id=%s AND user_id=%s",
                    (doc_id, int(user_id)))
        row = cur.fetchone()
        cur.execute("UPDATE documents SET status='deleted' WHERE doc_id=%s AND user_id=%s",
                    (doc_id, int(user_id)))
        if row and row[0]:
            p = Path(row[0])
            if p.exists():
                p.unlink()
        print(f"--> [MemStack] 已删除文档 {doc_id}（缓存无需失效，见 delete_session_data）")
    except Exception as e:
        print(f"--> [MemStack] 文档删除降级: {e}")

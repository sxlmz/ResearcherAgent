# -*- coding: utf-8 -*-
"""
memory_stack.py —— 记忆系统存储栈（v2，Infinity 版）

三个存储后端，均为用户级隔离：
  Infinity    → 文档 chunk + 对话原文 + L2 语义记忆（向量 + 全文混合检索）
  PostgreSQL  → L1 常驻档案（user_profiles）+ L3 事实/元数据（documents / conversations）
  Redis       → L0 工作记忆（会话热上下文 + 档案缓存）

运行环境：**Python 3.11**。
infinity-sdk 0.7.2 的原生绑定在 Python 3.13 下段错误，因此本模块必须跑在 Python 3.11
（Python 3.11 及以上可用，3.13 需验证 infinity-sdk 绑定）

数据库隔离：
  - Infinity：使用独立库 agent_memory（INFINITY_DB），不与其他项目共用。
  - PostgreSQL：agent_memory 库（docker-compose 初始化建表）。
  - Redis：DB 0，键均带 user/session 前缀。

用法（供 memory_orchestrator / memory_classifier 调用）：
    from ResearcherAgent.memory.memory_stack import MemoryStack
    stack = MemoryStack(embed_fn=embed)   # embed_fn: str -> list[float]
    stack.infinity.save_conversation_turn(900001, "session_12345", "user", "你好", embed)
    stack.pg.upsert_profile(900001, "USER", "偏好：APA 引用", "session_1")
    stack.redis.set_working("session_12345", [...])
"""
import os
import re
import json
import time
import uuid

import infinity
from infinity.common import NetworkAddress, ConflictType, SortType
from infinity.index import IndexInfo, IndexType

import psycopg2
import psycopg2.extras

import redis as redis_py


# ============================================================
# 配置（.env 或环境变量）
# ============================================================
INFINITY_HOST = os.environ.get("INFINITY_HOST", "localhost")
INFINITY_PORT = int(os.environ.get("INFINITY_PORT", "23817"))
INFINITY_DB = os.environ.get("INFINITY_DB", "agent_memory")

PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_USER = os.environ.get("PG_USER", "agent")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "agent_pass")
PG_DB = os.environ.get("PG_DB", "agent_memory")

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "") or None

# text-embedding-3-large 的维度（测试时会实测校验）
EMBED_DIM = int(os.environ.get("EMBED_DIM", "3072"))

_WORKING_MEMORY_TTL = int(os.environ.get("WORKING_MEMORY_TTL", "3600"))
_PROFILE_CACHE_TTL = int(os.environ.get("PROFILE_CACHE_TTL", "3600"))


def _sanitize(name: str) -> str:
    """Infinity 表名只允许字母/数字/下划线"""
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(name))


# ============================================================
# 1. Infinity 存储（文档 chunk + 对话原文 + L2 语义记忆）
# ============================================================
class InfinityStore:
    """用户级 Infinity 存储。每用户三张表（在独立库 agent_memory 内）：
        rag_{uid}   — 文档 chunk（doc_id / docnm / content / position / page / available）
        conv_{uid}  — 对话原文（session_id / role / content / seq / create_time）
        mem_{uid}   — L2 语义记忆（mem_type / summary / anchor_json / session_id）
    每张表带 HNSW cosine 向量索引 + content 全文索引（BM25 由引擎侧完成）。
    """

    def __init__(self, host=None, port=None, db_name=None):
        self.host = host or INFINITY_HOST
        self.port = port or INFINITY_PORT
        self.db_name = db_name or INFINITY_DB
        self._conn = None
        self._db = None

    # ---------- 连接 ----------
    def _connect(self):
        if self._conn is None:
            self._conn = infinity.connect(NetworkAddress(self.host, self.port))
        return self._conn

    def _db_handle(self):
        if self._db is None:
            self._db = self._connect().create_database(self.db_name, ConflictType.Ignore)
        return self._db

    def list_databases(self):
        """列出 Infinity 中全部库（含本库）。"""
        try:
            resp = self._connect().list_databases()
            # 兼容不同 SDK 版本：可能是 list，也可能是带 db_names 的响应对象
            if isinstance(resp, (list, tuple)):
                return list(resp)
            if hasattr(resp, "db_names"):
                return list(resp.db_names)
            return []
        except Exception as e:
            return [f"<list_databases 不可用: {e}>"]

    def list_other_databases(self):
        """列出 Infinity 中现有的库（排除本库），用于确认隔离。"""
        return [d for d in self.list_databases() if d != self.db_name]

    def disconnect(self):
        try:
            if self._conn is not None:
                self._conn.disconnect()
        except Exception:
            pass
        self._conn = None
        self._db = None

    # ---------- 建表 ----------
    @staticmethod
    def _vec_col():
        return f"q_{EMBED_DIM}_vec"

    @staticmethod
    def _safe_embed(embed_fn, text):
        """embed 失败降级为 0 向量——保证写入不因 embedding 超时/故障而丢失。
        0 向量不影响全文(BM25)检索；稠密检索在 embed 可用时自动恢复。"""
        if embed_fn is None:
            return [0.0] * EMBED_DIM
        try:
            return embed_fn(text)
        except Exception:
            return [0.0] * EMBED_DIM

    def _ensure_table(self, name, cols: dict):
        schema = {k: {"type": v, "default": ""} for k, v in cols.items()}
        schema[self._vec_col()] = {"type": f"vector,{EMBED_DIM},float"}
        table = self._db_handle().create_table(name, schema, ConflictType.Ignore)
        # 向量索引
        try:
            table.create_index(
                "q_vec_idx",
                IndexInfo(self._vec_col(), IndexType.Hnsw,
                          {"M": "16", "ef_construction": "50", "metric": "cosine"}),
                ConflictType.Ignore,
            )
        except Exception:
            pass
        # content 全文索引（BM25 引擎侧）
        if "content" in cols:
            for analyzer in ("rag-coarse", "rag-fine"):
                try:
                    table.create_index(
                        f"ft_content_{analyzer.replace('-', '_')}",
                        IndexInfo("content", IndexType.FullText, {"ANALYZER": analyzer}),
                        ConflictType.Ignore,
                    )
                except Exception:
                    pass
        return table

    def _table(self, name):
        return self._db_handle().get_table(name)

    @staticmethod
    def _ft_content():
        return "content@ft_content_rag_coarse"

    # ---------- 文档 chunk（用户级，跨对话） ----------
    _RAG_COLS = {
        "id": "varchar", "user_id": "varchar", "doc_id": "varchar",
        "docnm": "varchar", "content": "varchar",
        "position_int": "varchar", "page_num_int": "varchar", "available_int": "integer",
    }

    def index_document_chunks(self, user_id, doc_id, docnm, chunks, embed_fn):
        """把一篇文档的 chunks 写入 rag_{uid}。chunks: list[str] 或 list[(text, page)]。"""
        tname = f"rag_{_sanitize(user_id)}"
        tbl = self._ensure_table(tname, self._RAG_COLS)
        rows = []
        for i, c in enumerate(chunks):
            if isinstance(c, (tuple, list)):
                text, page = c
            else:
                text, page = c, 0
            text = str(text)
            rows.append({
                "id": str(uuid.uuid4()),
                "user_id": str(user_id),
                "doc_id": str(doc_id),
                "docnm": str(docnm),
                "content": text,
                "position_int": json.dumps([i]),
                "page_num_int": json.dumps([int(page)]),
                "available_int": 1,
                self._vec_col(): self._safe_embed(embed_fn, text),
            })
        tbl.insert(rows)
        return len(rows)

    def search_docs(self, user_id, query_text, query_vec, topk=8,
                    threshold=0.0, search_method="hybrid"):
        """混合检索用户文档库（用户级，跨会话共享知识库）。返回 [{id,doc_id,docnm,content,score}]"""
        tname = f"rag_{_sanitize(user_id)}"
        try:
            tbl = self._table(tname)
        except Exception:
            return []
        return self._hybrid_search(tbl, ["id", "doc_id", "docnm", "content", "position_int", "page_num_int"],
                                   query_text, query_vec, topk, threshold, search_method)

    def clear_doc_chunks(self, user_id, doc_id):
        """删除某用户某文档的全部 chunk（重传覆盖时调用）。"""
        tname = f"rag_{_sanitize(user_id)}"
        try:
            tbl = self._table(tname)
            tbl.delete(f"doc_id = '{_sanitize(str(doc_id))}'")
        except Exception:
            pass

    def doc_exists(self, user_id, doc_id) -> bool:
        """该用户的该文档是否已索引（跨轮次复用判断）。"""
        tname = f"rag_{_sanitize(user_id)}"
        try:
            tbl = self._table(tname)
            df, _ = tbl.output(["id"]).filter(f"doc_id = '{_sanitize(str(doc_id))}'").limit(1).to_df()
            return not df.empty
        except Exception:
            return False

    # ---------- 对话原文（用户级，可跨会话检索） ----------
    _CONV_COLS = {
        "id": "varchar", "user_id": "varchar", "session_id": "varchar",
        "role": "varchar", "content": "varchar",
        "seq_int": "integer", "create_time": "varchar",
    }

    def save_conversation_turn(self, user_id, session_id, role, content, embed_fn):
        """写入一条对话消息到 conv_{uid}（用户级，跨会话可检索）。"""
        tname = f"conv_{_sanitize(user_id)}"
        tbl = self._ensure_table(tname, self._CONV_COLS)
        seq = self._next_seq(tbl, "seq_int")
        content = str(content)
        tbl.insert([{
            "id": str(uuid.uuid4()),
            "user_id": str(user_id),
            "session_id": str(session_id),
            "role": role,
            "content": content,
            "seq_int": seq,
            "create_time": str(int(time.time())),
            self._vec_col(): self._safe_embed(embed_fn, content),
        }])
        return seq

    def get_conversation(self, user_id, session_id, n=50):
        """按会话取最近 n 条消息（精确查询，按 session_id 过滤）。"""
        tname = f"conv_{_sanitize(user_id)}"
        try:
            tbl = self._table(tname)
            df, _ = tbl.output(["role", "content", "seq_int"]) \
                .filter(f"session_id = '{_sanitize(session_id)}'") \
                .sort([("seq_int", SortType.Desc)]) \
                .limit(n).to_df()
        except Exception:
            return []
        rows = [{"role": r, "content": c} for r, c in zip(df["role"], df["content"])]
        return list(reversed(rows))

    def clear_conversation(self, user_id, session_id):
        """删除某会话的对话消息（测试/清理/删对话用）。"""
        tname = f"conv_{_sanitize(user_id)}"
        try:
            tbl = self._table(tname)
            tbl.delete(f"session_id = '{_sanitize(session_id)}'")
        except Exception:
            pass

    def clear_session_memories(self, user_id, session_id):
        """删除某会话产生的 L2 语义记忆（删对话时清理，防残留）。"""
        tname = f"mem_{_sanitize(user_id)}"
        try:
            tbl = self._table(tname)
            tbl.delete(f"session_id = '{_sanitize(session_id)}'")
        except Exception:
            pass

    def search_conversation(self, user_id, query_text, query_vec, topk=5,
                            threshold=0.0, search_method="hybrid", session_id=None):
        """检索用户对话历史。

        session_id 非空时**只在本会话内检索**（会话隔离）；为空则跨会话（默认不跨）。
        """
        tname = f"conv_{_sanitize(user_id)}"
        try:
            tbl = self._table(tname)
        except Exception:
            return []
        where = f"session_id = '{_sanitize(session_id)}'" if session_id else None
        return self._hybrid_search(tbl, ["id", "session_id", "role", "content", "create_time"],
                                   query_text, query_vec, topk, threshold, search_method, where=where)

    # ---------- L2 语义记忆（跨会话用户记忆，带锚点回源） ----------
    _MEM_COLS = {
        "id": "varchar", "user_id": "varchar", "mem_type": "varchar",
        "summary": "varchar", "anchor_json": "varchar", "session_id": "varchar",
    }

    def store_memory(self, user_id, mem_type, summary, anchor=None, session_id="", embed_fn=None):
        """写入一条 L2 语义记忆到 mem_{uid}（带锚点指向对话原文）。"""
        tname = f"mem_{_sanitize(user_id)}"
        tbl = self._ensure_table(tname, self._MEM_COLS)
        vec = self._safe_embed(embed_fn, summary)
        tbl.insert([{
            "id": str(uuid.uuid4()),
            "user_id": str(user_id),
            "mem_type": mem_type,
            "summary": str(summary),
            "anchor_json": json.dumps(anchor or {}, ensure_ascii=False),
            "session_id": str(session_id),
            self._vec_col(): vec,
        }])

    def list_memories(self, user_id, n=20):
        """列出用户最近 n 条语义记忆（不依赖 query 向量，用于管理/验证）。"""
        tname = f"mem_{_sanitize(user_id)}"
        try:
            tbl = self._table(tname)
            df, _ = tbl.output(["mem_type", "summary", "anchor_json", "session_id"]) \
                .limit(n).to_df()
        except Exception:
            return []
        results = []
        for _, row in df.iterrows():
            anchor = None
            try:
                anchor = json.loads(row.get("anchor_json") or "{}")
            except Exception:
                anchor = None
            results.append({
                "mem_type": row.get("mem_type", ""),
                "summary": row.get("summary", ""),
                "anchor": anchor,
                "session_id": row.get("session_id", ""),
            })
        return results

    def search_memories(self, user_id, query_vec, topk=5, mem_types=None, session_id=None):
        """按向量检索 L2 语义记忆（带摘要 + 锚点）。

        session_id 非空时**只在本会话内检索**（会话隔离）；为空则跨会话。
        """
        tname = f"mem_{_sanitize(user_id)}"
        try:
            tbl = self._table(tname)
        except Exception:
            return []
        builder = tbl.output(["mem_type", "summary", "anchor_json", "session_id", "similarity()"])
        if session_id:
            builder = builder.filter(f"session_id = '{_sanitize(session_id)}'")
        df, _ = builder.match_dense(self._vec_col(), query_vec, "float", "cosine", topk,
                                    {"threshold": "0.1"}) \
            .limit(topk).to_df()
        results = []
        for _, row in df.iterrows():
            mt = row.get("mem_type")
            if mem_types and mt not in mem_types:
                continue
            anchor = None
            try:
                anchor = json.loads(row.get("anchor_json") or "{}")
            except Exception:
                anchor = None
            results.append({
                "mem_type": mt, "summary": row.get("summary", ""),
                "anchor": anchor, "session_id": row.get("session_id", ""),
                "score": float(row.get("SIMILARITY") or 0),
            })
        return results

    # ---------- 通用混合检索 ----------
    def _hybrid_search(self, tbl, cols, query_text, query_vec, topk,
                       threshold, search_method, where=None):
        output = list(cols)
        if search_method in ("hybrid", "bm25"):
            output.append("score()")
        elif search_method == "cosine":
            output.append("similarity()")
        builder = tbl.output(output)
        if where:
            builder = builder.filter(where)
        if search_method in ("hybrid", "bm25"):
            builder = builder.match_text(self._ft_content(), str(query_text), topk)
        if search_method in ("hybrid", "cosine"):
            builder = builder.match_dense(self._vec_col(), query_vec, "float", "cosine",
                                          topk, {"threshold": str(threshold)})
        if search_method == "hybrid":
            builder = builder.fusion("weighted_sum", topk, {"weights": "0.001,1", "normalize": "atan"})
        try:
            df, _ = builder.offset(0).limit(topk).to_df()
        except Exception as e:
            print(f"--> [Infinity] hybrid 检索失败: {e}")
            return []
        if df.empty:
            return []
        results = []
        for _, row in df.iterrows():
            item = {c: row.get(c) for c in cols}
            item["score"] = float(row.get("SCORE") or 0) if "SCORE" in row.index else float(row.get("SIMILARITY") or 0)
            results.append(item)
        return results

    @staticmethod
    def _next_seq(tbl, col="seq_int"):
        try:
            df, _ = tbl.output([col]).sort([(col, SortType.Desc)]).limit(1).to_df()
            if df.empty or df[col].isna().all():
                return 0
            return int(df[col].iloc[0]) + 1
        except Exception:
            return 0


# ============================================================
# 2. PostgreSQL 存储（L1 常驻档案 + L3 事实/元数据）
# ============================================================
class PostgresStore:
    def __init__(self, dsn=None):
        self.dsn = dsn or (
            f"host={PG_HOST} port={PG_PORT} user={PG_USER} password={PG_PASSWORD} dbname={PG_DB}"
        )
        self._conn = None

    def _cur(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = True
        return self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def close(self):
        try:
            if self._conn is not None and not self._conn.closed:
                self._conn.close()
        except Exception:
            pass
        self._conn = None

    # ---- L1 常驻档案 ----
    def get_profiles(self, user_id):
        cur = self._cur()
        cur.execute(
            "SELECT profile_type, content, version FROM user_profiles "
            "WHERE user_id=%s ORDER BY created_at", (int(user_id),))
        return [dict(r) for r in cur.fetchall()]

    def upsert_profile(self, user_id, profile_type, content, source_session=""):
        """写 L1 档案。同名去重：内容相同则不动，内容更新则 version+1。"""
        cur = self._cur()
        cur.execute(
            "SELECT id FROM user_profiles WHERE user_id=%s AND profile_type=%s AND md5(content)=md5(%s)",
            (int(user_id), profile_type, content))
        if cur.fetchone():
            return "dup"
        cur.execute(
            "UPDATE user_profiles SET version=version+1, content=%s, source_session=%s "
            "WHERE user_id=%s AND profile_type=%s AND id = ("
            "  SELECT id FROM user_profiles WHERE user_id=%s AND profile_type=%s "
            "  ORDER BY id DESC LIMIT 1)"
            " RETURNING id",
            (content, source_session, int(user_id), profile_type, int(user_id), profile_type))
        row = cur.fetchone()
        if row:
            return "updated"
        cur.execute(
            "INSERT INTO user_profiles (user_id, profile_type, content, source_session) "
            "VALUES (%s, %s, %s, %s)",
            (int(user_id), profile_type, content, source_session))
        return "inserted"

    # ---- L3 文档元数据 ----
    def upsert_document(self, doc_id, user_id, filename, source_key="",
                        content_hash="", chunk_count=0, version=None):
        cur = self._cur()
        if version is None:
            cur.execute("SELECT version FROM documents WHERE doc_id=%s AND user_id=%s",
                        (doc_id, int(user_id)))
            row = cur.fetchone()
            version = (row["version"] + 1) if row else 1
        cur.execute(
            "INSERT INTO documents (doc_id, user_id, filename, source_key, content_hash, chunk_count, version) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (doc_id) DO UPDATE SET filename=EXCLUDED.filename, "
            "content_hash=EXCLUDED.content_hash, chunk_count=EXCLUDED.chunk_count, "
            "version=EXCLUDED.version, updated_at=now(), status='active'",
            (doc_id, int(user_id), filename, source_key, content_hash, chunk_count, version))
        return version

    def list_documents(self, user_id):
        cur = self._cur()
        cur.execute(
            "SELECT doc_id, filename, chunk_count, version, status, created_at "
            "FROM documents WHERE user_id=%s AND status='active' ORDER BY created_at",
            (int(user_id),))
        return [dict(r) for r in cur.fetchall()]

    def delete_document(self, doc_id, user_id):
        cur = self._cur()
        cur.execute("UPDATE documents SET status='deleted' WHERE doc_id=%s AND user_id=%s",
                    (doc_id, int(user_id)))

    # ---- 对话索引 ----
    def register_conversation(self, thread_id, user_id, title="新对话"):
        cur = self._cur()
        cur.execute(
            "INSERT INTO conversations (thread_id, user_id, title) VALUES (%s,%s,%s) "
            "ON CONFLICT (thread_id) DO NOTHING",
            (thread_id, int(user_id), title))

    def set_title(self, thread_id, title):
        cur = self._cur()
        cur.execute("UPDATE conversations SET title=%s WHERE thread_id=%s AND title='新对话'",
                    (title, thread_id))

    def list_conversations(self, user_id):
        cur = self._cur()
        cur.execute(
            "SELECT thread_id, title, created_at FROM conversations "
            "WHERE user_id=%s ORDER BY created_at DESC", (int(user_id),))
        return [dict(r) for r in cur.fetchall()]


# ============================================================
# 3. Redis 存储（L0 工作记忆）
# ============================================================
class RedisWorkingMemory:
    """L0 工作记忆：会话热上下文 + L1 档案缓存。键均带 user/session 前缀隔离。"""

    def __init__(self, host=None, port=None, password=None):
        self.r = redis_py.Redis(
            host=host or REDIS_HOST, port=port or REDIS_PORT,
            password=REDIS_PASSWORD, decode_responses=True,
            socket_connect_timeout=5,
        )

    def set_working(self, session_id, data, ttl=_WORKING_MEMORY_TTL):
        self.r.set(f"working:{session_id}", json.dumps(data, ensure_ascii=False), ex=ttl)

    def get_working(self, session_id):
        raw = self.r.get(f"working:{session_id}")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def set_profile_cache(self, user_id, profiles, ttl=_PROFILE_CACHE_TTL):
        self.r.set(f"profile:{user_id}", json.dumps(profiles, ensure_ascii=False), ex=ttl)

    def get_profile_cache(self, user_id):
        raw = self.r.get(f"profile:{user_id}")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None


# ============================================================
# 汇总容器
# ============================================================
class MemoryStack:
    """三个存储后端的统一容器（供 orchestrator / classifier 使用）。"""

    def __init__(self, embed_fn=None):
        self.infinity = InfinityStore()
        self.pg = PostgresStore()
        self.redis = RedisWorkingMemory()
        self.embed_fn = embed_fn

    def close(self):
        self.infinity.disconnect()
        self.pg.close()

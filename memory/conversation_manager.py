"""
conversation_manager.py
=======================
Conversation storage and lifecycle management: JSONL per-session history,
conversation index (thread_id -> metadata), session-scoped deletion.

Extracted from agent_core.py — all logic preserved exactly as-is.

⚠️ 用户隔离：所有对话操作通过 memory_manager._current_user_id (ContextVar)
获取当前用户 ID，确保不同用户的对话数据完全隔离。
- 注册对话时存储 user_id 到索引
- 列表查询时只返回当前用户的对话
- 删除/读取时校验所有权
- ContextVar 未设置时回退到旧行为（兼容 CLI 模式）
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional

from agent.memory.memory_manager import _current_user_id


# ==========================================
# 0. 用户上下文（从 ContextVar 读取，server.py 在请求进入时设置）
# ==========================================
def _get_user_id() -> int | None:
    """获取当前请求的用户 ID（从 ContextVar 读取）。

    server.py 在每个请求处理开始时通过 _current_user_id.set(uid) 设置，
    同一异步上下文内的所有函数调用都可以读取到。
    返回 None 表示未设置（CLI 模式或测试），此时回退到无过滤的旧行为。
    """
    return _current_user_id.get()


def _check_ownership(thread_id: str, operation: str = "访问") -> bool:
    """校验当前用户是否拥有该对话。

    - 若索引中没有该对话 → 返回 True（让调用方报 404，不泄漏存在性）
    - 若对话没有 user_id 字段（旧数据）→ 返回 True（兼容）
    - 若当前 ContextVar 未设置 → 返回 True（兼容 CLI 模式）
    - 若 user_id 不匹配 → 返回 False
    """
    index = _load_conv_index()
    entry = index.get(thread_id)
    if entry is None:
        return True  # 让调用方返回 404

    owner_id = entry.get("user_id")
    if owner_id is None:
        return True  # 旧数据无 user_id，兼容

    current_uid = _get_user_id()
    if current_uid is None:
        return True  # ContextVar 未设置，兼容 CLI

    if owner_id != current_uid:
        print(f"--> [Conv] 🚫 用户 {current_uid} 试图{operation}用户 {owner_id} 的对话 {thread_id}")
        return False

    return True


# ==========================================
# 1. JSONL 对话存储（按 session）
# ==========================================
CONVERSATIONS_DIR = Path("./conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)


def _session_path(session_id: str) -> Path:
    """返回指定 session 的 JSONL 文件路径"""
    # 路径遍历防护
    safe = hashlib.sha256(session_id.encode()).hexdigest()[:16]
    return CONVERSATIONS_DIR / f"{safe}.jsonl"


def save_conversation_turn(session_id: str, role: str, content: str,
                           metadata: Optional[dict] = None) -> None:
    """向 session 的 JSONL 文件追加一条对话记录"""
    # 轻量所有权校验（仅在有用户上下文时检查，CLI 模式跳过）
    if not _check_ownership(session_id, "写入"):
        return
    entry = {
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "content": content[:8000],  # 截断过长内容，避免单行爆炸
        "metadata": metadata or {},
    }
    with open(_session_path(session_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ---- 记忆系统 v2：对话原文镜像写入 Infinity（用户级，跨会话可检索）----
    try:
        from agent.memory.memory_integration import on_conversation_turn
        uid = _get_user_id()
        if uid is not None:
            on_conversation_turn(uid, session_id, role, content)
    except Exception:
        pass


def load_conversation_history(session_id: str, last_n: int = 20) -> str:
    """读取指定 session 最近 N 条对话记录，返回可读文本"""
    if not _check_ownership(session_id, "读取历史"):
        return ""
    path = _session_path(session_id)
    if not path.exists():
        return ""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                lines.append(f"[{entry.get('role', '?')} @ {entry.get('timestamp', '?')[:19]}]: "
                             f"{entry.get('content', '')[:500]}")
            except json.JSONDecodeError:
                continue
    return "\n".join(lines[-last_n:])


def load_conversation_messages(session_id: str, last_n: int = 100) -> list:
    """读取指定 session 的对话记录，返回结构化列表（供前端重建聊天区）。

    只返回面向用户展示的轮次：user 消息 + assistant 回复；system（工具日志）跳过。

    ⚠️ 用户隔离：校验当前用户是否拥有该对话，拒绝跨用户读取。
    """
    if not _check_ownership(session_id, "读取消息"):
        return []
    path = _session_path(session_id)
    if not path.exists():
        return []
    msgs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = entry.get("role", "")
            if role not in ("user", "assistant"):
                continue
            msgs.append({"role": role, "content": entry.get("content", "")})
    return msgs[-last_n:]


# ==========================================
# 对话索引 —— thread_id → {标题, 时间, 关联文件}
# ==========================================
CONV_INDEX_PATH = CONVERSATIONS_DIR / "index.json"


def _load_conv_index() -> dict:
    """读取对话索引（thread_id → 元数据）。文件不存在或损坏时返回空 dict。"""
    if not CONV_INDEX_PATH.exists():
        return {}
    try:
        return json.loads(CONV_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_conv_index(index: dict) -> None:
    CONV_INDEX_PATH.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def register_conversation(thread_id: str) -> None:
    """确保对话在索引中存在（首次出现时登记，标题稍后由首条消息生成）。

    自动从 ContextVar 读取当前 user_id 并存储，实现用户隔离。
    """
    if not thread_id:
        return
    index = _load_conv_index()
    if thread_id not in index:
        uid = _get_user_id()
        index[thread_id] = {
            "title": "新对话",
            "created_at": datetime.now().isoformat(),
            "file_paths": [],
            "user_id": uid,  # None 表示 CLI 模式（兼容旧数据）
        }
        _save_conv_index(index)
        print(f"--> [Conv] 注册对话 thread={thread_id} user={uid}")


def set_conversation_title(thread_id: str, title: str) -> None:
    """设置对话标题（仅当当前仍为默认标题时写入，避免覆盖已生成的标题）。"""
    if not thread_id or not title:
        return
    if not _check_ownership(thread_id, "修改标题"):
        return
    index = _load_conv_index()
    entry = index.get(thread_id)
    if entry is None:
        uid = _get_user_id()
        entry = {"title": "新对话", "created_at": datetime.now().isoformat(),
                 "file_paths": [], "user_id": uid}
        index[thread_id] = entry
    if entry.get("title", "新对话") in ("", "新对话"):
        entry["title"] = title[:30]
        _save_conv_index(index)


def add_conversation_file(thread_id: str, file_path: str) -> None:
    """把上传文件路径登记到对应对话（删除对话时据此清理磁盘文件）。"""
    if not thread_id or not file_path:
        return
    if not _check_ownership(thread_id, "添加文件"):
        return
    index = _load_conv_index()
    uid = _get_user_id()
    entry = index.setdefault(
        thread_id,
        {"title": "新对话", "created_at": datetime.now().isoformat(),
         "file_paths": [], "user_id": uid},
    )
    if file_path not in entry.setdefault("file_paths", []):
        entry["file_paths"].append(file_path)
        _save_conv_index(index)


def list_conversations() -> list:
    """返回按创建时间倒序的对话列表：[{thread_id, title, created_at}, ...]

    ⚠️ 用户隔离：只返回当前用户的对话。ContextVar 未设置时返回全部（兼容 CLI）。
    """
    current_uid = _get_user_id()
    index = _load_conv_index()
    items = []
    for tid, meta in index.items():
        owner_id = meta.get("user_id")
        # 有 user_id 的条目：严格按当前用户过滤
        if owner_id is not None and current_uid is not None:
            if owner_id != current_uid:
                continue  # 跳过其他用户的对话
        # owner_id 为 None（旧数据）且 current_uid 不为 None：
        # 旧数据不属于任何用户，在前端登录场景下跳过（避免用户 A 看到旧的无主对话）
        # 在 CLI 场景下（current_uid 为 None）返回全部
        if owner_id is None and current_uid is not None:
            continue
        items.append({
            "thread_id": tid,
            "title": meta.get("title", "新对话"),
            "created_at": meta.get("created_at", ""),
        })
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


def delete_conversation(thread_id: str) -> dict:
    """彻底删除一条对话的全部痕迹（**不动 L1 常驻档案 / 文档库，它们是跨对话的**）：

    1. 对话 JSONL 文件
    2. 记忆系统 v2：Infinity 对话原文 + Redis 工作记忆（用户级记忆/文档不删）
    3. 该对话关联的上传文件（磁盘）
    4. PostgreSQL conversations 索引行

    ⚠️ 用户隔离：校验当前用户是否拥有该对话，拒绝跨用户删除。
    """
    if not thread_id:
        return {"ok": False, "reason": "empty thread_id"}

    if not _check_ownership(thread_id, "删除"):
        return {"ok": False, "reason": "无权删除该对话（不属于当前用户）"}

    index = _load_conv_index()
    meta = index.get(thread_id, {})
    report = {"ok": True, "thread_id": thread_id}

    # 1. JSONL
    jsonl = _session_path(thread_id)
    try:
        if jsonl.exists():
            jsonl.unlink()
        report["jsonl_deleted"] = True
    except Exception as e:
        report["jsonl_deleted"] = f"failed: {e}"

    # 2. 记忆系统 v2：Infinity 对话原文 + Redis 工作记忆（best-effort）
    try:
        from agent.memory.memory_integration import delete_session_data
        delete_session_data(_get_user_id(), thread_id)
        report["memories_deleted"] = True
    except Exception as e:
        report["memories_deleted"] = f"skipped: {e}"

    # 2b. PostgreSQL conversations 索引行
    try:
        from agent.memory.memory_stack import PostgresStore
        cur = PostgresStore()._cur()
        cur.execute("DELETE FROM conversations WHERE thread_id=%s", (thread_id,))
    except Exception:
        pass

    # 3. 上传文件（磁盘）
    removed_files = 0
    for fp in meta.get("file_paths", []):
        try:
            p = Path(fp)
            if p.exists():
                p.unlink()
                removed_files += 1
        except Exception:
            pass
    report["files_deleted"] = removed_files

    # 4. 索引项（文档 chunk 属用户级，跨对话保留，不随单条对话删除）
    if thread_id in index:
        del index[thread_id]
        _save_conv_index(index)

    print(f"--> [Conv] 已删除对话 {thread_id}: {report}")
    return report

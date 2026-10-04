"""
server.py
=========
文献综述 Agent 的 FastAPI 后端 —— 前端直连。

Java 后端仅负责：用户注册/登录（JWT 签发）。
Python 后端负责：Agent 图执行、对话历史（JSONL）、记忆系统（Infinity + PostgreSQL + Redis）、
文件上传、对话列表管理、追踪。

用户识别：前端登录后已知 user_id，通过 X-User-Id 请求头传给 Python。
Python 不做 JWT 校验——认证已在 Java 层完成，Python 在内网只被前端/Vite proxy 访问。

接口：
- POST /api/chat                  : SSE 流式 Agent
- POST /api/review-action         : 人机审核门恢复
- POST /api/upload                : 文件上传
- GET  /api/conversations         : 对话列表
- GET  /api/history               : 对话历史
- GET  /api/trace                 : 追踪汇总 + spans
- DELETE /api/conversations/:id   : 删除对话
- DELETE /api/documents/:doc_id   : 删除文档

运行：
    uvicorn server:app --host 0.0.0.0 --port 8000
"""
# --- repo-root bootstrap：保证 `python <本文件>` 在任意 cwd 下都能 import 到包 ---
import sys
from pathlib import Path
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "agent" / "__init__.py").exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from contextlib import asynccontextmanager
import json
import uuid
import os

# Windows 控制台默认 GBK，强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from fastapi import FastAPI, Request, Query, Body, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agent.core.agent_core import (
    runtime,
    run_stream,
    load_conversation_messages,
    delete_conversation,
)
from agent.memory.memory_manager import _current_user_id
from agent.memory.conversation_manager import (
    register_conversation,
    add_conversation_file,
    list_conversations,
    _load_conv_index,
)
from agent.retrieval.rag_pipeline import save_upload, extract_text, UPLOAD_DIR
from pathlib import Path as _Path


# ==========================================
# 0. 用户识别（前端已验证 JWT，通过 X-User-Id 头传 user_id）
# ==========================================
def get_user_id(request: Request) -> int:
    """从 X-User-Id 请求头读取当前用户 ID。

    前端在登录后已知 user_id（Java 返回），通过 axios 拦截器自动附加到所有请求。
    Python 不做 JWT 校验——认证已在 Java 层完成。
    """
    user_id_str = request.headers.get("X-User-Id", "")
    if not user_id_str:
        print("--> [Auth] 401 缺少 X-User-Id 请求头")
        raise HTTPException(status_code=401, detail="缺少 X-User-Id 请求头")
    try:
        user_id = int(user_id_str)
        return user_id
    except ValueError:
        print(f"--> [Auth] 401 无效的 X-User-Id: {user_id_str!r}")
        raise HTTPException(status_code=401, detail=f"无效的 X-User-Id: {user_id_str}")


# ==========================================
# 1. 应用生命周期
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化 Agent，关闭时清理资源"""
    print("⏳ 正在初始化文献综述 Agent（含智能路由 + 记忆 MD + RAG + Harness 评委）...")
    await runtime.init()
    print("✅ Agent 就绪，开始接收请求。")
    yield
    print("🛑 服务正在关闭...")
    await runtime.shutdown()


app = FastAPI(title="文献综述 Agent · API", lifespan=lifespan)

# 确保上传目录存在
UPLOAD_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. 数据模型
# ==========================================
class ChatRequest(BaseModel):
    """前端 POST /api/chat 的 JSON 请求体。

    user_id 不再需要——从 JWT 中提取。
    file_path 不再需要——由 file_id 在服务端解析。
    """
    topic: str = ""
    thread_id: str = ""
    file_id: str = ""


# ==========================================
# 3. 文件上传
# ==========================================
# 内存文件注册表：file_id → {"file_path": ..., "filename": ..., "thread_id": ..., "user_id": ...}
# ⚠️ 用户隔离：注册和查询时均校验 user_id，防止跨用户文件访问
_file_registry: dict[str, dict] = {}


@app.post("/api/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    thread_id: str = Query(..., description="关联的对话 ID"),
):
    """上传文件用于文档问答/RAG。

    前端直连此接口（替代旧 Java FileController）。
    文件保存到 ./uploads/，元数据注册到内存注册表。
    返回 file_id 供后续 /api/chat 引用。
    """
    user_id = get_user_id(request)
    _current_user_id.set(user_id)

    # 验证文件
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名为空")

    # 检查扩展名
    allowed = {".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
               ".pdf", ".docx", ".html", ".css", ".csv", ".xml", ".log",
               ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    # 生成唯一 file_id 并保存
    file_id = uuid.uuid4().hex
    content = await file.read()
    dest = save_upload(content, file.filename, user_id=user_id)

    # 提取文本预览
    try:
        text = extract_text(dest)
    except Exception:
        text = f"[无法提取文本: {file.filename}]"
    preview = text[:200] if len(text) > 200 else text

    # 注册（记录 user_id 用于文件访问隔离）
    _file_registry[file_id] = {
        "file_path": str(dest),
        "filename": file.filename,
        "thread_id": thread_id,
        "user_id": user_id,
    }
    register_conversation(thread_id)
    add_conversation_file(thread_id, str(dest))

    # 记忆系统 v2：登记文档元数据（doc_id 与 _process_file_impl 计算一致）
    doc_id = ""
    try:
        import hashlib
        from agent.memory.memory_stack import PostgresStore
        doc_id = hashlib.sha256(dest.resolve().as_posix().encode()).hexdigest()[:16]
        PostgresStore().upsert_document(doc_id, user_id, file.filename,
                                        source_key=str(dest), version=1)
    except Exception:
        pass

    print(f"--> [Upload] file_id={file_id} filename={file.filename} "
          f"size={len(content)} chars={len(text)} user={user_id} doc={doc_id}")

    return JSONResponse({
        "file_id": file_id,
        "filename": file.filename,
        "preview": preview,
        "size": len(content),
        "text_length": len(text),
        "doc_id": doc_id,
    })


# ==========================================
# 4. 核心 API
# ==========================================
@app.post("/api/chat")
async def chat_post(request: Request, body: ChatRequest = Body(...)):
    """SSE 流式 Agent —— 前端直连。

    前端 POST JSON body:
        { "topic": "...", "thread_id": "...", "file_id": "..." }

    返回 SSE 流：title / status / review_materials / result / error / done 事件。
    """
    user_id = get_user_id(request)
    _current_user_id.set(user_id)

    thread_id = body.thread_id
    if not thread_id:
        thread_id = f"web_{uuid.uuid4().hex[:12]}"

    user_request = body.topic.strip()
    if not user_request:
        return EventSourceResponse(_error_stream("请求内容不能为空"))

    # 解析 file_id → file_path（校验 user_id + thread_id 双重所有权）
    file_path = ""
    if body.file_id:
        info = _file_registry.get(body.file_id)
        if info and info["thread_id"] == thread_id and info.get("user_id") == user_id:
            file_path = info["file_path"]
        else:
            print(f"--> [Chat] file_id={body.file_id} 未找到或不属于该用户/对话 "
                  f"(user={user_id} thread={thread_id})")
    else:
        # 后续轮次无 file_id：从对话索引中查找已上传文件（跨轮次复用文档索引）
        index = _load_conv_index()
        entry = index.get(thread_id, {})
        for fp in entry.get("file_paths", []):
            if _Path(fp).exists():
                file_path = fp
                print(f"--> [Chat] 从对话索引解析到已上传文件: {fp}")
                break

    print(f"--> [Chat] user={user_id} thread={thread_id} "
          f"topic={user_request[:50]!r} file_path={file_path!r}")

    async def event_generator():
        async for event in run_stream(user_request, thread_id, file_path=file_path):
            if await request.is_disconnected():
                break
            yield {"data": json.dumps(event, ensure_ascii=False)}
        yield {"data": json.dumps({"type": "done"}, ensure_ascii=False)}

    return EventSourceResponse(event_generator())


@app.post("/api/review-action")
async def review_action(request: Request):
    """人机审核操作：用户对检索结果审核后的决策。

    前端直连此接口，从 LangGraph interrupt 断点恢复图执行。

    请求体 JSON:
        thread_id: 会话 ID（必填）
        action: "accept" | "skip" | "re_search"
        feedback: 补充说明（re_search 时用于调整检索方向）

    返回 SSE 流。
    """
    user_id = get_user_id(request)
    _current_user_id.set(user_id)

    body = await request.json()
    thread_id = body.get("thread_id", "")
    action = body.get("action", "accept")
    feedback = body.get("feedback", "")

    if not thread_id:
        return JSONResponse({"ok": False, "reason": "缺少 thread_id"}, status_code=400)

    if action not in ("accept", "skip", "re_search"):
        return JSONResponse({"ok": False, "reason": f"无效 action: {action}"}, status_code=400)

    resume_value = {"action": action, "feedback": feedback}

    async def event_generator():
        async for event in run_stream(
            thread_id=thread_id,
            resume_value=resume_value,
        ):
            if await request.is_disconnected():
                break
            yield {"data": json.dumps(event, ensure_ascii=False)}
        yield {"data": json.dumps({"type": "done"}, ensure_ascii=False)}

    return EventSourceResponse(event_generator())


# ==========================================
# 5. 对话列表 & 历史
# ==========================================
@app.get("/api/conversations")
async def conversation_list(request: Request):
    """获取对话列表（按创建时间倒序）。

    前端直连此接口（替代旧 Java ConversationController）。
    数据源：Python ./conversations/index.json。
    ⚠️ 用户隔离：只返回当前用户的对话。
    """
    user_id = get_user_id(request)
    _current_user_id.set(user_id)
    convs = list_conversations()
    return JSONResponse({"conversations": convs})


@app.get("/api/history")
async def conversation_history(
    request: Request,
    thread_id: str = Query(..., description="会话 ID"),
):
    """查询指定 session 的对话历史（结构化，供前端重建聊天区）。

    前端直连此接口（替代旧 Java ConversationController 代理）。
    ⚠️ 用户隔离：校验对话所有权。
    """
    user_id = get_user_id(request)
    _current_user_id.set(user_id)
    messages = load_conversation_messages(thread_id, last_n=100)
    return JSONResponse({
        "thread_id": thread_id,
        "messages": messages,
        "has_history": bool(messages),
    })


@app.delete("/api/conversations/{thread_id}")
async def conversations_delete(request: Request, thread_id: str):
    """彻底删除一条对话：JSONL + 向量记忆 + 上传文件 + 文档 chunk + 索引项。

    前端直连此接口（替代旧 Java ConversationController 代理）。
    不删除 SOUL/USER/MEMORY.md（MD 记忆为跨对话持久化数据）。
    ⚠️ 用户隔离：校验对话所有权，拒绝跨用户删除。
    """
    user_id = get_user_id(request)
    _current_user_id.set(user_id)
    report = delete_conversation(thread_id)
    return JSONResponse(report)


@app.delete("/api/documents/{doc_id}")
async def document_delete(request: Request, doc_id: str):
    """删除一篇文档：Infinity rag chunk + Postgres 元数据 + 磁盘上传文件。

    ⚠️ 用户隔离：按当前用户删除，文档属用户级（跨对话），删后不可恢复。
    """
    user_id = get_user_id(request)
    _current_user_id.set(user_id)
    from agent.memory.memory_integration import delete_document
    delete_document(user_id, doc_id)
    return JSONResponse({"ok": True, "doc_id": doc_id})


@app.get("/api/trace")
async def trace_get(
    request: Request,
    thread_id: str = Query(..., description="会话 ID"),
    limit: int = Query(200, ge=1, le=2000),
):
    """可观测性：返回该 thread 的汇总（steps/tool_calls/tokens/失败数）与最近 spans（logs/trace.jsonl）。"""
    get_user_id(request)
    from agent.core import tracing
    return JSONResponse({
        "thread_id": thread_id,
        "summary": tracing.summary(thread_id),
        "spans": tracing.recent_spans(thread_id, limit),
    })


# ==========================================
# 6. 辅助函数
# ==========================================
async def _error_stream(message: str):
    """生成一条错误事件的异步 SSE 流"""
    yield {"data": json.dumps({"type": "error", "message": message}, ensure_ascii=False)}
    yield {"data": json.dumps({"type": "done"}, ensure_ascii=False)}


# ==========================================
# 7. 启动入口
# ==========================================
# 启动请用仓库根的薄壳：`uvicorn server:app --host 0.0.0.0 --port 8000`
# （根 server.py → 本模块的 app）。
#
# ⚠️ 这里**故意不放** `if __name__ == "__main__"` 自启动块：
# 直接 `python agent/interfaces/server.py` 会让本模块同时以 __main__ 和
# agent.interfaces.server 两个名字各导入一次，产生两个 AgentRuntime 单例、两个 FastAPI app。

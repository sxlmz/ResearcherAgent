"""
agent_core.py
=============
Harness-style AI Agent — 体现 harness 编程思想：模型自行判断、纠错、反思。

本文件是模块导入枢纽，从各子模块汇集并重导出外部消费者需要的公共符号。
核心逻辑分布在：config, conversation_manager, graph_nodes, agent_runtime。
"""

# ── 外部消费者需要的符号（server.py / arxiv_agent.py）──
from ResearcherAgent.memory.conversation_manager import (
    _session_path, save_conversation_turn,
    load_conversation_history, load_conversation_messages,
    register_conversation, set_conversation_title,
    add_conversation_file, delete_conversation,
)
from ResearcherAgent.memory.memory_manager import _current_user_id
from ResearcherAgent.core.agent_runtime import runtime, NODE_LABELS
from ResearcherAgent.core import tracing

import asyncio
from typing import AsyncGenerator


def _cache_summary() -> dict:
    """五层缓存的命中率汇总（失败不影响主流程）。"""
    try:
        from ResearcherAgent.core import cache
        return cache.summary()
    except Exception:
        return {}


# ==========================================
# 标题生成（给侧边栏用）
# ==========================================
async def _generate_title(user_request: str, llm) -> str:
    """用 LLM 把首条消息总结成 <= 15 字的对话标题；失败则回退到截断首句。"""
    fallback = user_request.strip().replace("\n", " ")[:20] or "新对话"
    if llm is None:
        return fallback
    try:
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", "把用户的第一句话概括成一个简短的对话标题，不超过15个字，"
                       "只输出标题本身，不要引号、不要标点结尾、不要任何多余文字。"),
            ("user", "{req}"),
        ])
        resp = (prompt | llm).invoke({"req": user_request[:500]})
        title = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        title = title.strip('"“”\'` \n')[:20]
        return title or fallback
    except Exception:
        return fallback


# ==========================================
# SSE 流式接口
# ==========================================
async def run_stream(
    user_request: str = "",
    thread_id: str = "",
    file_path: str = "",
    resume_value: dict = None,
    extra_state: dict = None,
) -> AsyncGenerator[dict, None]:
    """对用户请求运行 Agent，逐节点 yield 进度事件（供 FastAPI SSE 使用）

    Args:
        user_request: 用户消息（resume 模式下可为空）
        thread_id: 会话 ID
        file_path: 可选，上传文件的服务器端路径
        resume_value: 可选，LangGraph interrupt 的恢复值（Command(resume=...) 传入）
    """
    if runtime.app is None:
        yield {"type": "error", "message": "Agent 尚未初始化完成，请稍后重试。"}
        return

    config = {"configurable": {"thread_id": thread_id}}

    # ── 恢复模式：从 interrupt 断点继续 ──
    if resume_value is not None:
        from langgraph.types import Command
        inputs = Command(resume=resume_value)
        print(f"--> [run_stream] 恢复模式 thread={thread_id}")

    # ── 正常模式：初始化新请求 ──
    else:
        is_first_turn = not _session_path(thread_id).exists()
        register_conversation(thread_id)
        if file_path:
            add_conversation_file(thread_id, file_path)

        # 先记一条用户消息到 JSONL
        save_conversation_turn(thread_id, "user", user_request)

        # 首条消息 → 生成标题写入索引
        if is_first_turn:
            try:
                title = await _generate_title(user_request, runtime.llm_of("router"))
                set_conversation_title(thread_id, title)
                yield {"type": "title", "thread_id": thread_id, "title": title}
            except Exception as e:
                print(f"--> [Conv] 标题生成跳过: {e}")

        inputs = {
            "session_id": thread_id,
            "user_request": user_request,
            "file_path": file_path,
        }
        if extra_state:
            inputs.update(extra_state)
            print(f"--> [run_stream] 注入评测初始 state: {list(extra_state.keys())}")
        print(f"--> [run_stream] thread={thread_id} file_path={file_path!r}")

    try:
        import time as _time
        if resume_value is None:
            tracing.reset(thread_id)
        t_node = _time.perf_counter()
        async for output in runtime.app.astream(inputs, config=config):
            for node_name, _value in output.items():
                now = _time.perf_counter()
                tracing.record("node", node_name, thread_id, ms=round((now - t_node) * 1000, 1))
                t_node = now
                label = NODE_LABELS.get(node_name, f"正在执行节点 [{node_name}]...")
                # agent 节点：把模型这一步选了哪些工具带给前端（可观测）
                if node_name == "agent" and isinstance(_value, dict):
                    msgs = _value.get("messages") or []
                    calls = [c.get("name") for m in msgs for c in (getattr(m, "tool_calls", None) or [])]
                    if calls:
                        label = f"🤔 Agent 决定调用：{', '.join(calls)}"
                    elif _value.get("stop_reason"):
                        label = f"🤔 Agent 结束研究阶段（{_value['stop_reason']}）"
                yield {"type": "status", "node": node_name, "message": label}

        final_state = runtime.app.get_state(config)
        values = final_state.values if final_state else {}
        final_review = values.get("final_review", "")
        review_score = values.get("review_score", {})
        save_path = values.get("save_path", "")

        # ---- 记忆提取：LLM-as-Judge 分类存储（L1 Postgres + L2 Infinity） ----
        if final_review and runtime.llm:
            try:
                from ResearcherAgent.memory.memory_classifier import classify_and_store as _classify_and_store
                from ResearcherAgent.memory.memory_integration import get_orchestrator
                uid = _current_user_id.get()
                if uid is not None:
                    orch = get_orchestrator()
                    extract_result = await asyncio.to_thread(
                        _classify_and_store,
                        user_request or "",
                        final_review,
                        uid,
                        thread_id,
                        orch.stack,   # 复用编排器的存储连接
                    )
                    if extract_result.get("stored"):
                        print(f"--> [MemoryClassify] 已存储: {extract_result['stored']}")
            except Exception as e:
                print(f"--> [MemoryClassify] 提取跳过: {e}")

        # ---- CC 式记忆自动整理（Auto Dream 双门限）：距上次 >24h 且新增 ≥3 条才触发 ----
        if runtime.llm:
            try:
                from ResearcherAgent.memory.cc_memory import maybe_auto_consolidate
                _cons_res = await asyncio.to_thread(maybe_auto_consolidate, runtime.llm)
                if _cons_res:
                    print(f"--> [MemoryAutoConsolidate] {_cons_res}")
            except Exception as e:
                print(f"--> [MemoryAutoConsolidate] 跳过: {e}")

        if not final_review:
            yield {"type": "error", "message": "未能生成内容，请检查后端日志。"}
        else:
            review_report = values.get("review_report", {}) or {}
            yield {
                "type": "result",
                "content": final_review,
                "score": {
                    "total": review_score.get("total_score", "N/A"),
                    "passed": review_score.get("passed", None),
                    "dimensions": {k: v for k, v in review_score.items()
                                   if k not in ("passed", "total_score", "feedback")},
                    "feedback": review_score.get("feedback", ""),
                },
                # 评审报告（单次评审不重写）：低置信内容清单 + 原因，交用户审核
                "review_report": {
                    "total_score": review_report.get("total_score", "N/A"),
                    "passed": review_report.get("passed", None),
                    "threshold": review_report.get("threshold", None),
                    "dimensions": review_report.get("dimensions", {}),
                    "citation_rate": review_report.get("citation_rate", None),
                    "needs_research": review_report.get("needs_research", False),
                    "issues": review_report.get("issues", []),
                    "low_confidence": review_report.get("low_confidence", []),
                    "generated_at": review_report.get("generated_at", ""),
                },
                "save_path": save_path,
                "rewrite_count": values.get("rewrite_count", 0),
                # 可观测性：本次请求的护栏/消耗汇总（前端可选展示）
                "usage": {
                    "steps": values.get("step_count", 0),
                    "tokens": values.get("tokens_used", 0),
                    "stop_reason": values.get("stop_reason", ""),
                    "papers": len(values.get("paper_pool") or []),
                    "cards": len(values.get("paper_cards") or []),
                    **tracing.summary(thread_id),
                    # 五层缓存命中率（前端可忽略未知字段）
                    "cache": _cache_summary(),
                },
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {"type": "error", "message": f"运行出错: {e}"}

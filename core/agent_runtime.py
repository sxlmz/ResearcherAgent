"""
agent_runtime.py
================
AgentRuntime — 全局 Agent 持有者，服务启动时初始化一次，后续请求复用。

拓扑（研究阶段 = 模型驱动的 ReAct 循环；外层只保留质量门 / 护栏）：
    parse ─┬→ simple_chat ───────────────────────────────→ save
           ├→ process_file → file_qa ─┬→ write → judge …
           │                          └→ save
           └→ agent ⟲ tools
                 └→(模型不再调工具)─┬→ write → judge → save / content_safety（单次评审直接交付，不重写不补检）
                                   └→ answer ─┬→ save
                                              └→ content_safety → increment_content_safety ─┬→ write
                                                                                           └→ safety_feedback → agent
Checkpointer：PostgresSaver（跨重启恢复 HITL 断点），连接失败回退 MemorySaver。
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from ResearcherAgent.core.config import MCP_CONFIG, CHECKPOINTER, model_mode
from ResearcherAgent.tools.agent_tools import wrap_mcp_tool, build_local_tools, is_paper_search_tool, is_web_search_tool
from ResearcherAgent.core.graph_nodes import (
    AgentState,
    parse_intent_node,
    simple_chat_node,
    make_agent_node,
    answer_node,
    content_safety_feedback_node,
    write_review_node,
    judge_review_node,
    async_judge_node,
    content_safety_node,
    increment_rewrite_node,
    increment_content_safety_node,
    save_node,
    reject_node,
    _process_file_impl,
    file_qa_node,
    route_after_parse,
    route_after_agent,
    route_after_file_qa,
    route_after_judge,
    route_after_content_safety,
    route_after_increment_content_safety,
    _route_after_answer,
)


class ThreadedPostgresSaver:
    """把同步 PostgresSaver 包成可在 async 图里用的 checkpointer。

    为什么不用 AsyncPostgresSaver：MCP stdio 子进程依赖 Windows 默认的 ProactorEventLoop，
    而 psycopg 的 async 模式只支持 SelectorEventLoop——二者不能共存。
    同步 saver 走线程池，在两种 loop 下都能用。
    """

    def __new__(cls, pool):
        from langgraph.checkpoint.postgres import PostgresSaver

        class _Saver(PostgresSaver):
            async def aget_tuple(self, config):
                return await asyncio.to_thread(self.get_tuple, config)

            async def alist(self, config, *, filter=None, before=None, limit=None):
                items = await asyncio.to_thread(
                    lambda: list(self.list(config, filter=filter, before=before, limit=limit)))
                for it in items:
                    yield it

            async def aput(self, config, checkpoint, metadata, new_versions):
                return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

            async def aput_writes(self, config, writes, task_id, task_path=""):
                return await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

            async def adelete_thread(self, thread_id):
                return await asyncio.to_thread(self.delete_thread, thread_id)

        return _Saver(pool)


# ── 节点级重试：只重试**瞬时故障**，不掩盖代码 bug ──
# 动因：一次 LLM 请求超时（APITimeoutError）会直接干掉整轮运行——实测在第 13 步、
# 已烧掉 12 万 token 时整轮作废，所有检索与规划成果丢失。
_TRANSIENT_EXC_NAMES = {
    "APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError",
    "TimeoutError", "Timeout", "ConnectTimeout", "ReadTimeout", "RemoteProtocolError",
    "ConnectionResetError", "ConnectionError", "ServiceUnavailableError",
}
_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def _is_transient(exc: BaseException) -> bool:
    """瞬时故障判定：网络类异常名，或 HTTP 状态码属于「稍后重试可能成功」。

    代码 bug（KeyError/TypeError/JSONDecodeError…）不在其列——重试只会白白拖慢报错。
    """
    if type(exc).__name__ in _TRANSIENT_EXC_NAMES:
        return True
    # 部分 SDK 把状态码挂在 __cause__ 上（如 openai 的 APIStatusError 包装）
    for e in (exc, getattr(exc, "__cause__", None)):
        if getattr(e, "status_code", None) in _TRANSIENT_STATUS:
            return True
    return False


# 3 次尝试、指数退避 2s→4s、带抖动（避免多个并发节点同时重试撞在一起）
LLM_RETRY = RetryPolicy(max_attempts=3, initial_interval=2.0, backoff_factor=2.0,
                        max_interval=30.0, jitter=True, retry_on=_is_transient)


def _build_checkpointer():
    """postgres（默认）→ 失败回退 memory。返回 (saver, closer)。"""
    if CHECKPOINTER != "postgres":
        print("ℹ️ checkpointer=memory（进程重启后 HITL 断点不可恢复）")
        return MemorySaver(), None
    try:
        from psycopg_pool import ConnectionPool
        from ResearcherAgent.memory.memory_stack import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DB
        dsn = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
        pool = ConnectionPool(dsn, min_size=1, max_size=5, open=True, timeout=5,
                              kwargs={"autocommit": True, "prepare_threshold": 0})
        saver = ThreadedPostgresSaver(pool)
        saver.setup()
        print(f"✅ checkpointer=postgres（{PG_HOST}:{PG_PORT}/{PG_DB}，跨重启可恢复审核门断点）")
        return saver, pool.close
    except Exception as e:
        print(f"⚠️ Postgres checkpointer 不可用（{type(e).__name__}: {str(e)[:120]}）→ 回退 MemorySaver")
        return MemorySaver(), None


# ── 按节点角色的模型路由 ──
# 为什么要分角色：不同节点的任务性质、调用频率、token 量差别极大——
#   agent 循环占总量 ~60%（15-25 步，每步重发整个历史），写作占 ~20%，制卡批量调用，
#   而标题生成/记忆分类是用户可感知的串行等待。一个模型包打天下必然是"要么贵要么慢"。
#
# 环境变量：<ROLE>_MODEL，留空则回落 MAIN_MODEL。
ROLE_ENV = {
    "router":  "ROUTER_MODEL",    # parse_intent / 标题生成 / 记忆分类 —— 轻量结构化
    "agent":   "AGENT_MODEL",     # agent ⟲ tools 规划循环（含 make_outline）—— 最需要推理
    "extract": "EXTRACT_MODEL",   # 精读制卡 6000→400 字，批量调用 —— 纯抽取压缩
    "writer":  "WRITER_MODEL",    # 分节写作 + 章节压缩 —— 成品质量决定成败
    "judge":   "JUDGE_MODEL",     # 评委打分 —— 必须与 writer 异族
    "safety":  "SAFETY_MODEL",    # 内容安全四维审核 —— 规则性较强
}


def _mk_llm(model: str, base: str | None = None, key: str | None = None):
    """统一按 OpenAI 兼容接口建模型。

    ⚠️ **必须显式指定 `model_provider="openai"`**。否则 `init_chat_model("deepseek-v4-flash")`
    会按模型名把 provider 推断成 `deepseek` → 走 DeepSeek **官方**端点，而 multi 模式下
    我们要它走代理（端点由 base/key 决定，不由模型名决定）。

    base/key 显式传入时用传入值；否则由 ChatOpenAI 读 OPENAI_API_BASE / OPENAI_API_KEY。
    """
    kw = {"model_provider": "openai", "request_timeout": 90}
    if base:
        kw["base_url"] = base
    if key:
        kw["api_key"] = key
    return init_chat_model(model, **kw)


class AgentRuntime:
    """全局 Agent 持有者——服务启动时初始化一次，后续请求复用"""

    def __init__(self):
        self.app = None
        self.llms: dict = {}     # role → LLM 实例（router/agent/extract/writer/judge/safety）
        self.llm = None          # 向后兼容：= llms["agent"]
        self.judge_llm = None    # 向后兼容：= llms["judge"]
        self.tools = None        # 本次进程可用的全部工具（包装后的 MCP 子集 + 本地工具）
        self.mcp_tools: dict = {}     # name → 包装后的 MCP 工具（仅白名单）
        self.local_tools: dict = {}   # name → 本地工具
        self._mcp_client = None
        self._mcp_session_contexts: dict = {}  # server_name → (async_ctx_mgr, session)
        self._close_checkpointer = None

    def llm_of(self, role: str):
        """取某个角色的模型；角色不存在时回落 agent（保证任何情况下都有模型可用）。"""
        return self.llms.get(role) or self.llms.get("agent")

    def _build_llms(self):
        """按模式建模型表。任一角色初始化失败 → 回落基准模型，绝不让启动失败。

        · single 模式：所有角色共用**同一个**模型实例（省钱/兜底）。
        · multi  模式：按 ROLE_ENV 各角色分别路由。
        两种模式共用同一套 role → llm 的接口（llm_of），所以切换不影响任何节点代码。
        """
        mode = model_mode()
        default = os.environ.get("MAIN_MODEL", "deepseek-v4-flash")

        # ── single：一个模型打天下 ──
        if mode == "single":
            model = (os.environ.get("SINGLE_MODEL") or default).strip()
            base = (os.environ.get("SINGLE_BASE") or "").strip() or None
            key = (os.environ.get("SINGLE_KEY") or "").strip() or None
            try:
                llm = _mk_llm(model, base, key)
            except Exception as e:
                raise RuntimeError(f"single 模式模型 {model} 初始化失败，无法启动：{e}") from e
            who = f"{model} @ {base or os.environ.get('OPENAI_API_BASE', '(默认)')}"
            for role in ROLE_ENV:
                print(f"   {role:8s} ← {who}")
            llms = {role: llm for role in ROLE_ENV}
            # ── 异族评审：single 模式下 judge 角色独立走 QWEN_EVAL_*（qwen3.7-flash@maas）──
            # 目的：评审与写作必须异族/异源，避免同族偏置（模型看不出同类稿子的问题）。
            # 条件：配置了 QWEN_EVAL_MODEL 且 DASHSCOPE_API_KEY 已填写；否则回落同源 SINGLE。
            jm = (os.environ.get("QWEN_EVAL_MODEL") or "").strip()
            jk = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
            if jm and jk:
                jb = (os.environ.get("QWEN_EVAL_BASE") or "").strip() or None
                try:
                    llms["judge"] = _mk_llm(jm, jb, jk)
                    print(f"   judge    ← {jm} @ {jb or '(默认)'}（异族评审：与 writer 不同源）")
                except Exception as e:
                    print(f"   judge    ← {who}（{jm} 初始化失败回落：{str(e)[:80]}）")
            return llms

        # ── multi：按角色路由 ──
        try:
            fallback = _mk_llm(default)
        except Exception as e:
            raise RuntimeError(f"主模型 {default} 初始化失败，无法启动：{e}") from e
        out = {}
        for role, env_key in ROLE_ENV.items():
            m = (os.environ.get(env_key) or "").strip()
            if not m:                       # 未配置 → 回落主模型（含 agent）
                out[role] = fallback
                print(f"   {role:8s} ← {default}（{env_key} 未配置，回落 MAIN_MODEL）")
                continue
            try:
                out[role] = fallback if m == default else _mk_llm(m)
                print(f"   {role:8s} ← {m}")
            except Exception as e:
                out[role] = fallback
                print(f"   {role:8s} ← {default}（{m} 初始化失败：{str(e)[:80]}）")
        return out

    async def init(self):
        """初始化所有 LLM 实例、MCP 工具、编译 LangGraph"""
        load_dotenv(Path(".") / ".env")

        print("🔧 模型路由：")
        self.llms = self._build_llms()
        self.llm = self.llms["agent"]          # 向后兼容
        self.judge_llm = self.llms["judge"]
        # 评审独立性只在 multi 模式下才有意义；single 模式本来就是同一个模型，不该误报
        if model_mode() != "single" and self.llms["judge"] is self.llms["writer"]:
            print("   ⚠️ 评委与写作者是同一个模型实例——"
                  "「独立评委」只是形式独立，同族偏置会让它看不出同类稿子的问题")

        # ── MCP 工具：持久 session（子进程只启一次），只保留白名单检索工具并包装 ──
        self._mcp_client = MultiServerMCPClient(MCP_CONFIG)
        raw_tools = []
        try:
            for name in MCP_CONFIG:
                ctx = self._mcp_client.session(name)
                session = await ctx.__aenter__()
                self._mcp_session_contexts[name] = (ctx, session)
                try:
                    tools = await load_mcp_tools(session)
                    raw_tools.extend(tools)
                    print(f"✅ 已连接 MCP [{name}]: {len(tools)} 个工具")
                except Exception:
                    await ctx.__aexit__(None, None, None)
                    del self._mcp_session_contexts[name]
                    raise
        except Exception as e:
            print(f"❌ MCP 工具加载失败！ server={list(MCP_CONFIG.keys())} {type(e).__name__}: {e}")
            for name, cfg in MCP_CONFIG.items():
                print(f"     • {name}: {cfg['command']} {' '.join(cfg.get('args', []))}")
            raise

        # 渐进式工具加载：62 个 → 只暴露白名单检索源 + web 搜索（永不暴露 download_*/read_* PDF 工具）
        self.mcp_tools = {t.name: wrap_mcp_tool(t) for t in raw_tools
                          if is_paper_search_tool(t.name) or is_web_search_tool(t.name)}
        # 本地工具的 llm 主要用于 read_sections/read_papers 里的「制卡」→ extract 角色
        self.local_tools = build_local_tools(self.llm_of("extract"))
        self.tools = list(self.mcp_tools.values()) + list(self.local_tools.values())
        print(f"✅ 工具集: MCP {sorted(self.mcp_tools)} + 本地 {sorted(self.local_tools)}"
              f"（原始 {len(raw_tools)} 个 MCP 工具，按 skills 再动态选子集）")

        # ---- 编译 LangGraph ----
        workflow = StateGraph(AgentState)

        # 各节点按角色取模型 —— 节点签名统一是 (state, llm)，所以换模型只是换实参
        async def _parse(state):       return await parse_intent_node(state, self.llm_of("router"))
        async def _simple_chat(state): return await simple_chat_node(state, self.llm_of("router"))
        def _write(state):             return write_review_node(state, self.llm_of("writer"))
        def _judge(state):             return async_judge_node(state, self.llm_of("judge"))
        def _content_safety(state):    return content_safety_node(state, self.llm_of("safety"))
        async def _file_qa(state):     return await file_qa_node(state, self.llm_of("agent"))
        def _proc_file(state):         return _process_file_impl(state, self.llm_of("router"))

        # 会调 LLM 的节点挂重试；tools（含 interrupt，重跑有副作用）、save（写文件）、
        # 纯状态累加节点不挂——重试它们只会重复副作用。
        workflow.add_node("parse", _parse, retry_policy=LLM_RETRY)
        workflow.add_node("simple_chat", _simple_chat, retry_policy=LLM_RETRY)
        workflow.add_node("agent",
                          make_agent_node(self.llm_of("agent"), self.mcp_tools, self.local_tools),
                          retry_policy=LLM_RETRY)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("answer", answer_node)
        workflow.add_node("safety_feedback", content_safety_feedback_node)
        workflow.add_node("write", _write, retry_policy=LLM_RETRY)
        workflow.add_node("judge", _judge, retry_policy=LLM_RETRY)
        workflow.add_node("content_safety", _content_safety, retry_policy=LLM_RETRY)
        workflow.add_node("increment_rewrite", increment_rewrite_node)
        workflow.add_node("increment_content_safety", increment_content_safety_node)
        workflow.add_node("reject", reject_node)
        workflow.add_node("save", save_node)
        workflow.add_node("process_file", _proc_file, retry_policy=LLM_RETRY)
        workflow.add_node("file_qa", _file_qa, retry_policy=LLM_RETRY)

        workflow.set_entry_point("parse")
        workflow.add_conditional_edges("parse", route_after_parse, {
            "simple_chat": "simple_chat",
            "agent": "agent",
            "process_file": "process_file",
            "reject": "reject",
        })

        # 文件路径
        workflow.add_edge("process_file", "file_qa")
        workflow.add_conditional_edges("file_qa", route_after_file_qa, {"write": "write", "save": "save"})
        workflow.add_edge("simple_chat", "save")

        # ★ 模型驱动的 ReAct 循环：agent ⟲ tools；模型不再调用工具即终止研究阶段
        workflow.add_conditional_edges("agent", route_after_agent, {
            "tools": "tools",
            "write": "write",
            "answer": "answer",
            "reject": "reject",
        })
        workflow.add_edge("tools", "agent")

        # answer → content_safety（如有）或 save
        workflow.add_conditional_edges("answer", _route_after_answer, {
            "content_safety": "content_safety", "save": "save",
        })

        # Harness 评委-质量门（学术质量，单次评审直接交付：不重写、不补检索）
        workflow.add_edge("write", "judge")
        workflow.add_conditional_edges("judge", route_after_judge, {
            "content_safety": "content_safety", "save": "save",
        })
        workflow.add_edge("increment_rewrite", "write")

        # 内容安全审核循环
        workflow.add_conditional_edges("content_safety", route_after_content_safety, {
            "save": "save", "increment_content_safety": "increment_content_safety",
        })
        workflow.add_conditional_edges("increment_content_safety", route_after_increment_content_safety, {
            "write": "write", "safety_feedback": "safety_feedback",
        })
        workflow.add_edge("safety_feedback", "agent")

        workflow.add_edge("reject", "save")
        workflow.add_edge("save", END)

        saver, self._close_checkpointer = _build_checkpointer()
        self.app = workflow.compile(checkpointer=saver)
        print("✅ LangGraph 已编译完成（agent ⟲ tools 循环 + HITL 门 + 评委/安全质量门 + 护栏）")

    async def shutdown(self):
        """关闭所有 MCP 持久 session 与 checkpointer 连接池。"""
        for name, (ctx, session) in list(self._mcp_session_contexts.items()):
            try:
                await ctx.__aexit__(None, None, None)
                print(f"🛑 MCP session 已关闭 [{name}]")
            except BaseException as e:  # 捕获 CancelledError 等（py311 asyncio 下 session 关闭可能抛）
                print(f"⚠️ MCP session 关闭异常 [{name}]: {e}")
        self._mcp_session_contexts.clear()
        if self._close_checkpointer:
            try:
                self._close_checkpointer()
            except Exception:
                pass


# 全局单例
runtime = AgentRuntime()

# ---- 节点 → 前端友好文案 ----
NODE_LABELS = {
    "parse":               "🧭 正在理解意图并选择技能...",
    "simple_chat":         "💬 正在回答...",
    "agent":               "🤔 Agent 正在思考下一步...",
    "tools":               "🔧 正在执行工具（检索 / 规划 / 精读）...",
    "answer":              "🤖 正在整理回答...",
    "safety_feedback":     "🔄 [safety] 根据安全反馈重答...",
    "write":               "✍️ 正在撰写综述...",
    "judge":               "⚖️ [harness] 评委正在独立评审...",
    "increment_rewrite":   "🔄 [harness] 准备修改...",
    "reject":              "🚫 未通过命题校验，已拒绝生成",
    "save":                "💾 正在保存...",
    "content_safety":      "🛡️ [safety] 正在审核内容安全性...",
    "increment_content_safety": "🔄 [safety] 准备安全重写...",
    "process_file":        "📄 正在索引文件...",
    "file_qa":             "🤖 基于文件内容生成回答...",
}

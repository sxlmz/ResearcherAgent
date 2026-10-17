"""
graph_nodes.py
==============
LangGraph 节点函数 + AgentState。

拓扑（研究阶段是模型驱动的 ReAct 循环，外层保留薄的质量门）：
    parse ─┬→ simple_chat ───────────────────────────────→ save
           ├→ process_file → file_qa ─┬→ write → judge …
           │                          └→ save
           └→ agent ⟲ tools   ← 模型选工具（make_outline / search_* / read_papers /
                 │               read_sections / write_todos），不再调用工具即终止
                 └→ write → judge → (content_safety) → save     [write_review]
                 └→ answer → (content_safety) → save            [其它]
工具实现见 agent_tools.py；护栏常量见 config.py。
"""

import json
import re
import threading
from pathlib import Path
from typing import TypedDict, Annotated
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages

from ResearcherAgent.core.skill_manager import (build_skill_index, validate_skills,
    needs_search_papers, needs_tools,
    needs_index_file, needs_write_review,
    needs_content_safety, build_skill_instructions)
from ResearcherAgent.memory.memory_manager import load_all_for_prompt
from ResearcherAgent.core.config import (MAX_REWRITE_ROUNDS, JUDGE_PASS_THRESHOLD,
    MAX_AGENT_STEPS, TOKEN_BUDGET, KEEP_RECENT_TOOL_MESSAGES,
    WRITE_SECTION_BUDGET, MAX_REVIEW_CHARS, MAX_REVIEW_CHARS_HARD,
    SECTION_OVERFLOW_TOLERANCE, REVIEW_READ_MAX, LEAD_BLOCK_INPUT_BUDGET,
    MIN_PAPERS_PER_SECTION,
    _TOOL_FAILURE_PHRASES, _TOOL_FAILURE_HEURISTIC_MAX_LEN,
    _CONTENT_SAFETY_DIMS, _CONTENT_SAFETY_THRESHOLD, _BASE_SAFETY_INSTRUCTION,
    REJECT_MALICIOUS_THRESHOLD, REJECT_RESEARCHABLE_THRESHOLD, REJECT_RATIO_THRESHOLD,
    FULLTEXT_PARAGRAPH_BUDGET, FIELD_TOP_K)
from ResearcherAgent.tools.agent_tools import (merge_papers, merge_paper_cards, render_pool, select_tools,
                                     merge_innovations, merge_experiments,
                         build_references, render_section_candidates)

import time
from ResearcherAgent.core import cache
from ResearcherAgent.core import tracing


# ==========================================
# AgentState —— 状态定义
# ==========================================
class AgentState(TypedDict):
    session_id: str
    user_request: str
    topic: str                      # 中文综述主题
    search_query: str               # 英文检索关键词
    requirements: str               # 写作约束（由 user_constraints 渲染；用户没提则为空串）
    user_constraints: dict          # 用户**明确说出**的结构化约束（经接地核验，用户没提则为 {}）
    historical_knowledge: str       # 记忆编排器检索到的历史记忆
    draft_review: str               # 当前版本的综述
    final_review: str               # 最终输出
    review_score: dict              # {"total_score": float, "passed": bool, "feedback": str, ...}
    rewrite_count: int              # 当前已重写次数（学术质量评委）
    content_safety_count: int       # 内容安全审核的重写次数
    content_safety_score: dict      # 内容安全审核的评分/反馈
    content_safety_feedback: str    # 内容安全审核的累积反馈
    tool_failed: bool               # 研究阶段是否一无所获（由 route_after_agent 判定）
    best_review: str                # keep-best：历轮中评分最高的草稿
    best_score: dict                # keep-best：对应的评分详情
    feedback_history: str           # 累积的历轮评委反馈（供重写时逐条对照）
    # ── 迭代评测（自我纠错闭环）──
    force_rounds: int                # 评测注入：即使通过也强制重写 N 轮（eval_baseline）
    draft_history: list              # 每轮草稿快照 [{"round","draft","revision_plan"}]
    revision_plan: str               # 本轮独立修订说明（write 生成，judge 快照记录）
    judge_issues: list               # claim 级缺陷清单（judge 输出）
    needs_research: bool             # 证据类缺陷占比标记（只写入评审报告，不再触发补检索路由）
    evidence_report: dict            # 逐节确定性证据检查明细 {节名: [findings]}（judge 写，write 读）
    review_report: dict              # 评审报告（总分/维度/引用率/低置信清单，随综述一并交付）
    iteration_log: list              # 每轮 judge 快照（五指标评测）
    discovery_report: dict           # discover_directions 方向发现结果
    memory_ab_delta: str             # 记忆注入前后对照（评测用，可空）
    skills: list[str]               # LLM 选择的原子 skill 列表（组合在一起决定图路径）
    save_path: str                  # 用户指定的保存路径
    file_path: str                  # 用户上传的文件路径（用于 RAG 问答）
    rag_context: str                # RAG 检索到的相关文本片段
    review_structure: str           # 用户指定的综述模块/结构（为空则用默认或 MD 偏好）
    # ── 意图门与拒答（入口意图识别 + 一轮检索后证据不足拒答）──
    intent_gate: dict               # parse 写入：{"intent","researchable","malicious","reason"}
    rejected: bool                  # 是否已拒答
    reject_reason: str              # 拒答原因（用户可见）
    # ── 综述研究产物（由工具写入，模型决定顺序）──
    review_outline: list            # 综述大纲 [{"section","focus","queries"}]（make_outline）
    paper_pool: Annotated[list, merge_papers]   # 去重论文池（检索工具入池；上下文只放摘要+句柄）
    paper_cards: Annotated[list, merge_paper_cards]  # Map 卡片（read_papers，可并行多次追加去重）
    # ── 创新点验证（skill "innovate"）──
    innovations: Annotated[list, merge_innovations]  # propose/novelty 输出（并发合并，按 id 去重）
    experiments: Annotated[dict, merge_experiments]  # implement/judge 记录（并发合并）
    citation_report: dict           # 引用真实性校验结果（judge）
    reference_report: dict          # 参考文献拼装结果 {refs, cited, unmatched, pool}（write）
    section_candidates: dict        # 每节的重排候选 {_pool_n, sections:{节名:{candidates,...}}}（read_sections 内嵌重排写回，供证据不足拒答门）
    json_build_session_id: str      # 流式 MQ 会话 id（read_sections 边制卡边送，write 节点收集）
    todos: list                     # 模型维护的任务清单（write_todos）
    # ── 护栏 ──
    step_count: int                 # agent↔tools 往返次数
    tokens_used: int                # 本次请求累计 token（usage_metadata）
    stop_reason: str                # ""|"model_done"|"max_steps"|"token_budget"|"user_accept"|...
    messages: Annotated[list, add_messages]


# ==========================================
# 内存注入工具
# ==========================================

# ---- 记忆注入工具 ----
def _with_memory(system_text: str, user_request: str = "") -> str:
    """将 SOUL/USER/MEMORY.md + 向量库跨会话记忆注入 system prompt。

    L1（MD 文件）始终注入；L2（向量记忆）按 user_request 检索注入。
    """
    mem = load_all_for_prompt(user_request)
    if mem.strip():
        return f"## 你的身份与记忆\n\n{mem}\n\n---\n\n## 当前任务\n{system_text}"
    return system_text


def _with_safety(system_text: str, user_request: str = "") -> str:
    """注入记忆 + 安全底线。"""
    return _with_memory(system_text, user_request) + _BASE_SAFETY_INSTRUCTION


# ==========================================
# 图节点定义
# ==========================================


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", str(s or ""))


def _grounded(needle: str, haystack: str) -> bool:
    """needle 是否确实出现在用户原话里（去空白后子串匹配）。"""
    n = _squash(needle)
    return len(n) >= 2 and n in _squash(haystack)


def normalize_user_constraints(raw, user_request: str) -> dict:
    """规范化 parse 节点产出的用户约束，并做**确定性接地核验**。

    动因（实测事故）：用户只输入 29 字「请帮我写一篇关于 X 的综述」，
    parse 的 LLM 却产出 851 字「写作约束」，凭空塞进「篇幅目标：约2500-4000字」
    「提供摘要与引言」「参考文献要给 DOI 链接」——用户从未提过其中任何一条。
    下游全盘当真（写作照着写、评委照着扣分），形成幻觉放大链。

    这里逐条核验：**用户原话里找不到依据的约束一律丢弃**。
    这是对 LLM 输出的确定性核验（constraint grounding），不是语义启发式——
    只做「原话里有没有这个字符串」的事实检查。
    代价：用户用同义词表达的要求可能被误丢。宁可少约束，也不能凭空造约束——
    多一条假约束会让写作追一个不存在的目标、让评委按假标准扣分。
    """
    if not isinstance(raw, dict):
        return {}
    req = user_request or ""
    out: dict = {}
    dropped: list[str] = []

    # 篇幅：用户原话里必须同时有数字和「字/词」单位
    wb = raw.get("word_budget")
    if wb:
        has_num = bool(re.search(r"\d", req))
        has_unit = bool(re.search(r"字|词|words?", req, re.I))
        ok = False
        if isinstance(wb, (list, tuple)) and len(wb) == 2 and has_num and has_unit:
            try:
                lo, hi = int(wb[0]), int(wb[1])
                if 100 <= lo <= hi <= 100000:
                    out["word_budget"] = [lo, hi]
                    ok = True
            except (TypeError, ValueError):
                pass
        if not ok:
            dropped.append(f"word_budget={wb!r}")

    # 列表型约束：逐条必须是用户原话的子串
    for key in ("required_parts", "sections", "other"):
        vals = raw.get(key)
        if not isinstance(vals, (list, tuple)):
            continue
        kept = [str(v).strip() for v in vals
                if str(v).strip() and _grounded(str(v), req)]
        for v in vals:
            if str(v).strip() and not _grounded(str(v), req):
                dropped.append(f"{key}::{str(v)[:30]!r}")
        if kept:
            out[key] = kept

    # 引用体例：用户原话里得真的在说「引用体例」。
    # ⚠️ 不能拿「文献」当关键词——「文献综述」是文体名，几乎每句用户请求里都有，
    # 会让人人都不提引用格式的用户被判定为「提了引用要求」。
    cs = raw.get("citation_style")
    if cs:
        if re.search(r"引用|著录|APA|MLA|GB/?T|编号|格式", req, re.I):
            # 用户在原话里点名了具体标准时，以用户点名的为准（LLM 可能填错体例）
            named = None
            for pat, name in ((r"APA", "APA"), (r"MLA", "MLA"),
                              (r"GB/?T\s*7714", "GB/T 7714"), (r"IEEE", "IEEE"),
                              (r"数字编号|编号引用|\[1\]", "numeric"),
                              (r"作者[-—]?年份", "author-year")):
                if re.search(pat, req, re.I):
                    named = name
                    break
            out["citation_style"] = named or str(cs)
        else:
            dropped.append(f"citation_style={cs!r}")

    # 语言：用户原话里得提到语言
    lg = raw.get("language")
    if lg:
        if re.search(r"中文|英文|汉语|英语|english|chinese", req, re.I):
            out["language"] = str(lg)
        else:
            dropped.append(f"language={lg!r}")

    if dropped:
        print(f"--> [Parse] ⚠️ 丢弃了 {len(dropped)} 条无依据的「用户约束」"
              f"（用户原话里找不到）：{'、'.join(dropped[:6])}")
    return out


def render_user_constraints(uc: dict) -> str:
    """把结构化用户约束渲染成写作/评审可读的文本。**用户没提就返回空串**。"""
    if not uc:
        return ""
    parts = []
    wb = uc.get("word_budget")
    if wb:
        parts.append(f"篇幅：{wb[0]}-{wb[1]} 字")
    if uc.get("required_parts"):
        parts.append("必须包含：" + "、".join(uc["required_parts"]))
    if uc.get("sections"):
        parts.append("必须覆盖的方向：" + "、".join(uc["sections"]))
    if uc.get("citation_style"):
        parts.append(f"引用体例：{uc['citation_style']}")
    if uc.get("language"):
        parts.append(f"语言：{uc['language']}")
    if uc.get("other"):
        parts.append("其它：" + "；".join(uc["other"]))
    return "\n".join(f"- {p}" for p in parts)


def effective_review_budget(uc: dict) -> tuple[int, str]:
    """本次写作的总字数预算。

    用户给了篇幅要求就**以用户为准**（用户约束优先于系统默认），否则用 MAX_REVIEW_CHARS。
    返回 (预算, 依据说明)。
    """
    wb = (uc or {}).get("word_budget")
    if wb and len(wb) == 2:
        lo, hi = int(wb[0]), int(wb[1])
        return hi, f"用户要求 {lo}-{hi} 字"
    return MAX_REVIEW_CHARS, f"系统默认 {MAX_REVIEW_CHARS} 字"


def _judge_intent(llm, user_request: str) -> dict:
    """意图门判定：intent 类型 / 是否值得科研的命题 / 是否恶意测试（Choice+Noul 风格，带置信度）。

    确定性兜底：LLM 输出非法或调用失败时返回保守默认（researchable=0.5, malicious=0）
    —— 不因判定器故障误拒正常请求。
    """
    _sys = (
        "你是科研选题评审。对给定的用户请求做三类原子判断，只输出 JSON：\n"
        '{"intent": "review|chat|retrieval|other", '
        '"researchable": <0-1 该请求是否是一个值得科研的命题（有明确研究对象、'
        '可通过文献检索获得证据支持）的概率>, '
        '"malicious": <0-1 该请求是否为恶意测试/伪命题/反科学命题（如“证明人类能飞”'
        '这类物理上不成立、或故意戏弄系统、或无法检索到真实文献支撑的命题）的概率>, '
        '"reason": "<一句话判定依据，不超过 40 字>"}\n'
        "规则：两者是独立维度；普通综述请求 researchable 应高（>0.7）；"
        "“证明中国人能飞”这类反事实命题 malicious 应高。"
    )
    try:
        _resp = llm.invoke([SystemMessage(content=_sys),
                           HumanMessage(content=user_request[:1200])])
        _raw = _resp.content if hasattr(_resp, "content") else str(_resp)
        _st, _en = _raw.find("{"), _raw.rfind("}")
        if _st == -1 or _en == -1:
            raise ValueError("no json")
        _d = json.loads(_raw[_st:_en + 1])
        _gate = {
            "intent": str(_d.get("intent") or "other").strip().lower(),
            "researchable": max(0.0, min(1.0, float(_d.get("researchable") or 0.5))),
            "malicious": max(0.0, min(1.0, float(_d.get("malicious") or 0.0))),
            "reason": str(_d.get("reason") or "").strip()[:80],
        }
        return _gate
    except Exception as e:
        print(f"--> [Parse] ⚠️ 意图门判定失败（默认放行）: {str(e)[:80]}")
        return {"intent": "other", "researchable": 0.5, "malicious": 0.0, "reason": ""}


async def parse_intent_node(state: AgentState, llm):
    """【节点 0 · 意图解析】

    让 LLM 输出一个 skills 列表（原子能力的组合），而非从 5 个套餐里单选。
    调度器只做前提条件校验，不做启发式猜测。
    """
    # load_conversation_history is imported from agent_core
    from ResearcherAgent.memory.conversation_manager import load_conversation_history

    print("--> [Parse] 正在解析用户意图...")
    user_request = state["user_request"]
    file_path = state.get("file_path", "")

    # 校验 file_path 是否真实存在（防止 checkpointer 恢复的过期路径）
    has_file = bool(file_path) and Path(file_path).exists()

    skill_index = build_skill_index(has_file=has_file)

    # 最近对话上下文，支持多轮意图
    recent_history = load_conversation_history(state.get("session_id", ""), last_n=4)
    history_block = (
        f"\n\n## 最近对话（用于理解指代/多轮意图，勿据此臆造需求）：\n{recent_history}"
        if recent_history else ""
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", skill_index),
        ("user", "用户请求：{user_request}{history_block}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"user_request": user_request, "history_block": history_block})
    raw = response.content if hasattr(response, "content") else str(response)

    # ---- 解析 JSON ----
    topic = user_request
    search_query = ""
    user_constraints = {}
    requirements = ""
    save_path = ""
    review_structure = ""
    skills = ["answer_direct"]  # 兜底

    try:
        text = raw.strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(text[start:end + 1])
            topic = (data.get("topic") or "").strip() or user_request
            search_query = (data.get("search_query") or "").strip()
            user_constraints = normalize_user_constraints(data.get("user_constraints"), user_request)
            requirements = render_user_constraints(user_constraints)
            save_path = (data.get("save_path") or "").strip()
            review_structure = (data.get("review_structure") or "").strip()
            skills = data.get("skills") or ["answer_direct"]
            if isinstance(skills, str):
                skills = [skills]
            if not isinstance(skills, list) or not skills:
                skills = ["answer_direct"]
    except Exception as e:
        print(f"--> [Parse] JSON 解析失败，兜底为 answer_direct: {e}")

    # ---- 意图门判定（独立调用，Choice/Noul 风格带置信度；失败默认放行）----
    intent_gate = _judge_intent(llm, user_request)
    print(f"--> [Parse] 意图门: {intent_gate}")

    # ---- 调度器：前提条件校验 ----
    skills, reason = validate_skills(skills, has_file)
    print(f"--> [Parse] 调度结果: {reason}")

    use_tools = needs_tools(skills)
    print(f"--> [Parse] skills={skills} | topic={topic!r} | has_file={has_file} | "
          f"use_tools={use_tools} | search_query={search_query!r}")
    # 用户约束为空是**正常情况**（大多数用户只给一句话），不是解析失败
    print(f"--> [Parse] 用户约束: {user_constraints or '（用户未提任何额外要求——正常）'}")

    # 初始消息就是用户请求本身（附上提炼的英文检索词作为提示）——
    # 具体调哪个工具、调几次、按什么顺序，由 agent 循环里的模型自己决定。
    hint = f"\n\n（提炼的英文检索关键词：{search_query}）" if use_tools and search_query else ""
    init_messages = [HumanMessage(content=user_request + hint, name="user_request")]

    return {
        "topic": topic,
        "search_query": search_query,
        "requirements": requirements,
        "user_constraints": user_constraints,
        "review_structure": review_structure,
        "skills": skills,
        "save_path": save_path,
        "rewrite_count": 0,
        "step_count": 0,
        "tokens_used": 0,
        "stop_reason": "",
        "tool_failed": False,
        "historical_knowledge": "",
        "todos": [],
        "review_outline": [],
        "paper_cards": [],
        "paper_pool": None,          # reducer 收到 None → 清空上一轮的论文池
        "intent_gate": intent_gate,
        "rejected": False,
        "reject_reason": "",
        "messages": init_messages,
    }


def route_after_parse(state: AgentState) -> str:
    """条件边：根据 skills 列表中的输入类 skill 决定入口节点。"""
    skills = state.get("skills", ["answer_direct"])

    # ── 意图门：恶意测试 / 伪命题 → 直接拒答，不进任何生成路径 ──
    gate = state.get("intent_gate") or {}
    _mal = float(gate.get("malicious") or 0)
    _res = float(gate.get("researchable") or 0.5)
    if _mal >= REJECT_MALICIOUS_THRESHOLD or _res < REJECT_RESEARCHABLE_THRESHOLD:
        print(f"--> [Route] 意图门拦截（malicious={_mal:.2f} researchable={_res:.2f}）→ reject")
        return "reject"

    if needs_index_file(skills):
        print(f"--> [Route] skills={skills} → process_file")
        return "process_file"

    if needs_tools(skills):
        # 联网抓取 / 论文检索 / 写综述 → 模型驱动的 agent 循环
        print(f"--> [Route] skills={skills} → agent")
        return "agent"

    # 无输入类 skill：直接回答
    print(f"--> [Route] skills={skills} → simple_chat")
    return "simple_chat"


# ---- 简单对话路径 ----

async def simple_chat_node(state: AgentState, llm):
    """【节点 · 简单对话】无需 MCP，直接由 LLM 回答"""
    from ResearcherAgent.memory.conversation_manager import load_conversation_history, save_conversation_turn
    from ResearcherAgent.memory.memory_manager import _current_user_id
    from ResearcherAgent.memory.memory_integration import get_orchestrator

    print("--> [SimpleChat] 正在生成回答...")
    user_request = state["user_request"]
    session_id = state.get("session_id", "")
    uid = _current_user_id.get()

    # 记忆检索（新系统）：L0 工作记忆 + L1 档案 + L2 语义 + 文档（按需）
    mem = get_orchestrator().retrieve(user_request, uid, session_id, state.get("skills", []))
    historical = get_orchestrator().format_for_prompt(mem)

    # 读取当前 session 最近对话
    recent_history = load_conversation_history(session_id, last_n=6)

    # 动态内容作为变量传入，避免花括号被当成模板语法解析
    prompt = ChatPromptTemplate.from_messages([
        ("system", _with_memory(
            "你是一个有帮助的 AI 学术助手。请直接回答用户的问题。",
            user_request)
         + "{historical_context}"
         + "{recent_context}"),
        ("user", "{user_request}"),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "user_request": user_request,
        "historical_context": f"\n\n【相关历史记忆】:\n{historical}" if historical else "",
        "recent_context": f"\n\n【本会话最近对话】:\n{recent_history}" if recent_history else "",
    })
    answer = response.content if hasattr(response, "content") else str(response)

    # 记录到 JSONL
    session_id = state.get("session_id", "")
    if session_id:
        save_conversation_turn(session_id, "assistant", answer,
                               {"type": "simple_chat"})

    return {"final_review": answer, "draft_review": ""}


# ---- 文件上传 & RAG 问答路径（Infinity 文档检索） ----

def _process_file_impl(state: AgentState, llm) -> dict:
    """【节点 · 文件处理】用 Infinity 索引/复用上传文档，进行混合检索。

    首次：解析文件 → 分块 → 写入 Infinity（用户级，按 doc_id）
    后续：检测到已有索引（doc_id 已存在）→ 直接检索，跳过重复索引
    """
    import hashlib
    from pathlib import Path
    from ResearcherAgent.retrieval.rag_pipeline import extract_text, sentence_tokenize, chunk_sentences
    from ResearcherAgent.memory.memory_manager import _current_user_id
    from ResearcherAgent.memory.memory_integration import get_orchestrator

    print("--> [FileProcess] 正在处理文件（Infinity 文档检索）...")
    file_path = state.get("file_path", "")
    user_request = state["user_request"]
    user_id = _current_user_id.get()

    if not file_path:
        return {"rag_context": "", "final_review": "未找到上传文件，请先上传文件后再提问。"}
    if user_id is None:
        return {"rag_context": "", "final_review": "未识别用户，无法索引文件。"}

    fp = Path(file_path)
    if not fp.exists():
        return {"rag_context": "", "final_review": f"文件不存在: {file_path}"}

    orch = get_orchestrator()
    inf = orch.stack.infinity
    embed = orch.stack.embed_fn
    doc_id = hashlib.sha256(fp.resolve().as_posix().encode()).hexdigest()[:16]

    # 首次索引：解析 → 分块 → 写入 Infinity；已存在则复用（跨轮次）
    if not inf.doc_exists(user_id, doc_id):
        text = extract_text(fp)
        if text.startswith("[") and text.endswith("]"):
            return {"rag_context": text, "final_review": f"文件处理说明: {text}"}
        sentences = sentence_tokenize(text)
        chunks = chunk_sentences(sentences, chunk_size=8, overlap_ratio=0.2)
        if not chunks:
            return {"rag_context": "", "final_review": "文件内容为空，无法索引。"}
        inf.clear_doc_chunks(user_id, doc_id)   # 覆盖重传
        n = inf.index_document_chunks(user_id, doc_id, fp.name, chunks, embed)
        print(f"--> [FileProcess] 已索引 {n} 个 chunk 到 Infinity (doc={doc_id})")
    else:
        print(f"--> [FileProcess] 复用已有文档索引 (doc={doc_id})")

    # 混合检索（稠密 + 全文 BM25）
    try:
        vec = embed(user_request)
        results = inf.search_docs(user_id, user_request, vec, topk=5,
                                  threshold=0.0, search_method="hybrid")
    except Exception as e:
        print(f"--> [FileProcess] 检索失败: {e}")
        results = []

    rag_context = "\n\n---\n".join([
        f"[score={d['score']:.3f}] {d['content'][:1000]}" for d in results
    ]) if results else "(未检索到相关内容)"

    return {"rag_context": rag_context}


async def file_qa_node(state: AgentState, llm):
    """【节点 · 文件问答】基于 RAG 检索结果回答用户问题"""
    from ResearcherAgent.memory.conversation_manager import save_conversation_turn

    print("--> [FileQA] 正在基于文件内容生成回答...")
    rag_context = state.get("rag_context", "")
    user_request = state["user_request"]

    if not rag_context:
        return {"final_review": "文件内容为空或检索未返回结果，无法回答。"}

    if rag_context.startswith("[") and rag_context.endswith("]"):
        return {"final_review": rag_context}  # 错误消息直接透传

    # 根据 skills 列表决定系统提示：有 write_review → 格式参考模式，否则 → 普通问答模式
    skills = state.get("skills", ["answer_from_context"])

    if needs_write_review(skills):
        system_instruction = _with_memory(
            "你是一个专业的学术写作助手。用户上传了一份格式参考文件。\n"
            "## 工作方式\n"
            "1. 学习文件的格式（标题层级、段落组织、引用方式），按照同样格式撰写综述\n"
            "2. 只学格式，不抄内容\n"
            "3. 结合文件中的信息和你自己的知识\n"
            "4. 写完后在末尾标注：格式参考自上传文件",
            user_request)
    else:
        system_instruction = _with_memory(
            "你是一个专业的文档分析助手。请基于下面提供的文件内容来回答用户问题。\n"
            "规则：\n"
            "1. 优先使用文件内容回答，文件内容不足时结合自己的知识\n"
            "2. 如果文件内容完全不相关，明确告知用户并给出一般性回答\n"
            "3. 回答要清晰有条理",
            user_request)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("user",
         "【文件内容片段】:\n{rag_context}\n\n"
         "【用户请求】: {user_request}"),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "rag_context": rag_context,
        "user_request": user_request,
    })
    answer = response.content if hasattr(response, "content") else str(response)

    session_id = state.get("session_id", "")
    if session_id:
        save_conversation_turn(session_id, "assistant", answer, {"type": "file_qa"})

    # 如果是文献综述（含格式参考），用 LLM 提炼格式特征保存到 USER.md
    if needs_write_review(skills) and rag_context:
        try:
            await _save_format_to_memory(user_request, rag_context, llm)
        except Exception:
            pass

    return {
        "final_review": answer,
        "draft_review": answer,  # 始终设置，由 route_after_file_qa 判断是否走 judge
    }


async def _save_format_to_memory(user_request: str, rag_context: str, llm):
    """从上传文件中用 LLM 提炼写作格式特征，精炼后保存到 USER.md（避免原文堆砌）"""
    # 只取文件内容的前 2000 字符给 LLM 分析格式
    sample = rag_context

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个文档格式分析助手。请分析以下文档片段，提炼出它的写作格式特征。\n"
         "只关注格式，不关注内容。用 3-5 条简洁的 bullet point 描述:\n"
         "- 标题层级 (如: 一级标题用 #, 二级用 ##, 或者用数字编号等)\n"
         "- 段落组织方式 (如: 每段开头是否缩进, 段间是否空行, 是否有摘要/关键词等前置部分)\n"
         "- 引用风格 (如: 顺序编号, 作者-年份, IEEE, 脚注等)\n"
         "- 其他明显的排版习惯\n"
         "每条不超过 20 字。如果没有明显特征, 写 [无明显格式特征]。\n"
         "直接输出 bullet point, 不要任何额外文字。"),
        ("user", "文档片段:\n{sample}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"sample": sample})
    analysis = response.content if hasattr(response, "content") else str(response)

    # 精炼写入 Postgres 用户档案（USER，新系统）
    summary = analysis.strip()[:300]
    entry = f"写作格式偏好（用户: {user_request[:80]}）\n{summary}"
    from ResearcherAgent.memory.memory_manager import _current_user_id
    from ResearcherAgent.memory.memory_integration import store_profile
    uid = _current_user_id.get()
    if uid is not None:
        store_profile(uid, "USER", entry, "")


def route_after_file_qa(state: AgentState) -> str:
    """条件边：有 write_review 则走 write+judge 循环，否则直接保存"""
    skills = state.get("skills", ["answer_direct"])
    if needs_write_review(skills):
        print(f"--> [Route] skills={skills} → 撰写 + 评委审核")
        return "write"
    print(f"--> [Route] skills={skills} → 直接保存")
    return "save"


def _extract_tool_content(messages: list) -> str:
    """从消息列表中提取当前轮次所有 ToolMessage 的文本内容。

    只取「当前轮次」——从最后一条 user 消息到末尾之间的 ToolMessage。
    跨轮次的旧 tool 结果不参与提取，避免话题切换后旧数据被当作当前检索结果。

    在当轮之内，从末尾向前收集连续的 ToolMessage —— 因为 LLM 单轮可能调用多个工具，
    每个工具返回一条 ToolMessage；只取最后一条会丢失同轮其他工具的结果。
    """
    if not messages:
        return ""

    def _unpack_content(msg) -> str:
        """兼容 LangChain Message 对象和 plain dict"""
        if hasattr(msg, "content") and not isinstance(msg, dict):
            raw = msg.content
        elif isinstance(msg, dict):
            raw = msg.get("content", "")
        else:
            raw = str(msg)

        if isinstance(raw, list):
            parts = []
            for item in raw:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(raw)

    def _is_tool_message(msg) -> bool:
        """判断是否为 tool 消息"""
        if hasattr(msg, "type"):
            return msg.type == "tool"
        if isinstance(msg, dict):
            return msg.get("role") == "tool" or msg.get("type") == "tool"
        return False

    # 先截取当前轮次的消息（从最后一条 user 消息开始）
    current_turn = _trim_to_current_turn(messages)

    # 从未尾向前收集连续的 ToolMessage
    parts = []
    for msg in reversed(current_turn):
        if _is_tool_message(msg):
            parts.append(_unpack_content(msg))
        else:
            break  # 遇到非 tool 消息，停止

    if parts:
        return "\n\n---\n\n".join(reversed(parts))

    # 兜底：没有任何 ToolMessage，用最后一条消息的内容
    return _unpack_content(messages[-1])


def detect_tool_failure(messages: list, papers_data: str) -> bool:
    """统一判定 arXiv/MCP 工具调用是否失败。

    判据优先级：
    1. 主判据 —— 最后一条消息的 status == "error"（框架级、最可靠）
    2. 空结果 —— 没有任何有效返回内容
    3. 兜底启发式 —— 仅当内容很短时，才用明确的失败短语匹配。
       长正文（正常论文数据）绝不走子串匹配，从根本上杜绝
       "reconstruction error"/"error rate" 这类正常术语触发误判。
    """
    # 1. 框架级状态：最可靠
    last_msg = messages[-1] if messages else None
    if last_msg is not None and getattr(last_msg, "status", None) == "error":
        return True

    text = (papers_data or "").strip()

    # 2. 空返回
    if not text:
        return True

    # 3. 仅对短内容做失败短语启发式
    if len(text) <= _TOOL_FAILURE_HEURISTIC_MAX_LEN:
        low = text.lower()
        return any(phrase in low for phrase in _TOOL_FAILURE_PHRASES)

    return False


# ---- MCP 研究路径 ----

def _trim_orphaned_tool_calls(messages: list) -> list:
    """去掉消息列表末尾的孤儿 tool_calls（没有后续 tool 消息响应的）。

    messages 用 operator.add 累积，前一轮的 assistant(tool_calls) 可能残留在末尾。
    直接喂给 LLM 会触发 OpenAI 400：
    "An assistant message with 'tool_calls' must be followed by tool messages".
    """
    if not messages:
        return messages
    cut = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        # 处理 LangChain Message 对象
        has_obj = bool(
            getattr(msg, "tool_calls", None)
            or getattr(msg, "additional_kwargs", {}).get("tool_calls")
        )
        # 处理 plain dict（parse 节点产的）
        has_dict = isinstance(msg, dict) and (
            bool(msg.get("tool_calls")) or msg.get("role") == "assistant" and "tool_calls" in msg
        )
        is_tool = getattr(msg, "type", "") == "tool" or (isinstance(msg, dict) and msg.get("role") == "tool")
        if (has_obj or has_dict) and not is_tool:
            cut = i
        else:
            break
    return messages[:cut]


def _is_user_message(msg) -> bool:
    """判断消息是否为 user/human 角色。兼容 LangChain Message 对象和 plain dict。"""
    if hasattr(msg, "type"):
        return msg.type in ("user", "human")
    if isinstance(msg, dict):
        return msg.get("role") in ("user", "human")
    return False


def _trim_to_current_turn(messages: list) -> list:
    """只保留「当前轮次」的消息——从 parse 节点产出的用户请求（name="user_request"）开始到末尾。

    messages 跨轮次累积 + checkpointer 持久化，上一轮的 tool_calls / tool_results 会污染
    当前轮的上下文（换了话题后仍返回上一话题的检索内容）。
    循环内注入的提示消息（安全反馈等）不带该 name，不会截断当前轮。
    """
    if not messages:
        return messages
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if getattr(m, "name", None) == "user_request":
            return messages[i:]
    for i in range(len(messages) - 1, -1, -1):   # 兜底：最后一条 user 消息
        if _is_user_message(messages[i]):
            return messages[i:]
    return messages


def _mask_old_tool_messages(messages: list, keep_recent: int = KEEP_RECENT_TOOL_MESSAGES) -> list:
    """上下文卸载（视图级，不改 state）：只保留最近 N 条 ToolMessage 原文，更早的替换为一行句柄。

    全量结果已在 paper_pool / paper_cards 里，模型随时可通过 read_papers / read_sections 视图看到；
    这里不做 summary(summary) 式压缩，避免漂移。
    """
    idx = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    if len(idx) <= keep_recent:
        return messages
    to_mask = set(idx[:-keep_recent])
    out = []
    for i, m in enumerate(messages):
        if i in to_mask:
            head = (m.content if isinstance(m.content, str) else str(m.content)).splitlines()[0][:120]
            out.append(ToolMessage(content=f"[已折叠的早期工具结果] {head}", tool_call_id=m.tool_call_id,
                                   name=m.name, status=m.status))
        else:
            out.append(m)
    return out


# ==========================================
# agent 循环：模型选工具，直到它不再调用工具
# ==========================================

def _render_todos(todos: list) -> str:
    icon = {"done": "✅", "in_progress": "🔄", "pending": "⬜"}
    return "\n".join(f"{icon.get(t.get('status', 'pending'), '⬜')} {t.get('task', '')}" for t in todos or [])


def _agent_system_prompt(state: AgentState, tool_names: list[str], budget_note: str) -> str:
    skills = state.get("skills", [])
    user_request = state.get("user_request", "")
    is_review = needs_write_review(skills)

    role = ["你是一个严谨的 AI 学术研究助手，可以自主决定调用哪些工具、调用几次、按什么顺序。",
            "原则：只做必要的调用；结果不理想时换关键词/换源重试，而不是放弃；"
            "拿到足够信息后**停止调用工具**并给出最终回复。"]
    if is_review:
        role += [
            "\n## 当前任务：为用户准备一篇文献综述的研究材料（写作由后续环节完成，你不用写正文）",
            "执行步骤、检索源搭配与精读要求见下方「技能执行规范 · write_review」"
            "——严格按该技能规范中的流程执行（write_todos 规划 → 演进调研 → 按方向建池 → 大纲 → "
            "read_sections 精读（每节 top-10，全池确定性重排）→ 停止，交写作环节）。",
            "最终回复：用 3-5 句话总结检索/覆盖情况即可（不要写综述正文）。",
        ]
    else:
        role += [
            "\n## 当前任务：基于工具获取的信息回答用户问题",
            "规则：优先使用检索/抓取结果；信息不足可结合自身知识但要说明；检索失败时如实告知，不编造；"
            "自动忽略与问题无关的网页噪音。",
            "【引用格式·强制】正文引用用 [1]、[2] 标注，链接写成 `[1](https://实际URL)`；"
            "末尾列「参考链接」每条一行 `[N] 标题: URL`；结果无 URL 则写「来源未提供链接」。",
        ]
    ctx = []
    if state.get("todos"):
        ctx.append("## 你的任务清单\n" + _render_todos(state["todos"]))
    if state.get("review_outline"):
        ol = "；".join(f"{o.get('section')}（{', '.join(o.get('queries', [])[:2])}）" for o in state["review_outline"])
        ctx.append("## 当前大纲\n" + ol)
    pool_n = len(state.get("paper_pool") or [])
    cards_n = len(state.get("paper_cards") or [])
    if pool_n or cards_n:
        ctx.append(f"## 资料状态\npaper_pool {pool_n} 篇（去重）；paper_cards {cards_n} 张精读卡片。")
    if state.get("historical_knowledge"):
        ctx.append("## 相关历史记忆\n" + state["historical_knowledge"][:1500])
    # 渐进式披露 L2：选中 skills 的 SKILL.md 正文注入（流程/规则以技能规范为准，不再写死在代码里）
    _skill_block = build_skill_instructions(skills)
    if _skill_block:
        ctx.append(_skill_block)
    ctx.append("## 可用工具\n" + ", ".join(tool_names))
    ctx.append("## 护栏\n" + budget_note)
    return _with_safety("\n".join(role) + "\n\n" + "\n\n".join(ctx), user_request)


def _strip_reasoning_content(msgs):
    """剥离历史 AIMessage 中的 reasoning_content（DeepSeek 思考模式字段）。

    跨轮重放（re_search 回 agent）时，checkpointer 恢复的 assistant 消息带
    reasoning_content，但 langchain-openai 构造请求未回传 → DeepSeek 400
    "reasoning_content in the thinking mode must be passed back"。
    剥离后按普通消息重放即可（保留 id 供 LangGraph add_messages 去重）。
    """
    out = []
    for m in msgs:
        if isinstance(m, AIMessage):
            ak = dict(m.additional_kwargs or {})
            ek = dict(getattr(m, "extra_kwargs", None) or {})
            if "reasoning_content" in ak:
                ak.pop("reasoning_content", None)
            if "reasoning_content" in ek:
                ek.pop("reasoning_content", None)
            if ak != dict(m.additional_kwargs or {}) or ek != dict(getattr(m, "extra_kwargs", None) or {}):
                m = AIMessage(content=m.content, additional_kwargs=ak, extra_kwargs=ek,
                              tool_calls=getattr(m, "tool_calls", None) or None,
                              tool_call_id=getattr(m, "tool_call_id", None),
                              id=m.id)
        out.append(m)
    return out


def make_agent_node(llm, mcp_tools: dict, local_tools: dict):
    """构造 agent 节点：按 skills 绑定工具子集（渐进式加载）+ 护栏 + 记忆注入。"""

    async def agent_node(state: AgentState):
        from ResearcherAgent.memory.memory_manager import _current_user_id
        from ResearcherAgent.memory.memory_integration import get_orchestrator

        skills = state.get("skills", [])
        thread_id = state.get("session_id", "")
        step = int(state.get("step_count", 0) or 0)
        tokens = int(state.get("tokens_used", 0) or 0)
        updates: dict = {}

        # 首步：记忆检索一次（原 memory 节点的职责）
        if step == 0 and not state.get("historical_knowledge"):
            try:
                mem = get_orchestrator().retrieve(state.get("topic") or state.get("user_request", ""),
                                                  _current_user_id.get(), thread_id, skills)
                updates["historical_knowledge"] = get_orchestrator().format_for_prompt(mem)
            except Exception as e:
                print(f"--> [Agent] 记忆检索跳过: {e}")
                updates["historical_knowledge"] = ""
            # CC 风格项目记忆（Claude Code Auto Memory 模式）：MEMORY.md 全局注入
            try:
                from ResearcherAgent.memory.cc_memory import read_memory_for_inject
                _cc_ctx = read_memory_for_inject()
                if _cc_ctx:
                    _base = updates.get("historical_knowledge") or ""
                    updates["historical_knowledge"] = (_base + "\n\n" + _cc_ctx if _base else _cc_ctx)
            except Exception:
                pass
        hist_state = {**state, **updates}

        # ---- 护栏：步数 / token 预算 ----
        over_steps = step >= MAX_AGENT_STEPS
        over_tokens = tokens >= TOKEN_BUDGET
        tools = select_tools(skills, mcp_tools, local_tools)
        if over_steps or over_tokens:
            reason = "max_steps" if over_steps else "token_budget"
            print(f"--> [Agent] ⚠️ 护栏触发 {reason}（step={step}, tokens={tokens}）→ 不再提供工具，要求收尾")
            budget_note = (f"⚠️ 已达上限（{reason}），本轮**不能再调用工具**。请直接基于已有资料给出最终回复。")
            bound = llm
            updates["stop_reason"] = reason
        else:
            budget_note = (f"已用 {step}/{MAX_AGENT_STEPS} 步、约 {tokens // 1000}k/{TOKEN_BUDGET // 1000}k tokens。"
                           "接近上限时请尽快收尾。")
            bound = llm.bind_tools(tools) if tools else llm

        system = _agent_system_prompt(hist_state, [t.name for t in tools], budget_note)
        msgs = _trim_orphaned_tool_calls(state.get("messages", []))
        msgs = _trim_to_current_turn(msgs)
        msgs = _mask_old_tool_messages(msgs)
        msgs = _strip_reasoning_content(msgs)   # DeepSeek thinking 兼容（400 兜底）

        with tracing.span("llm", "agent", thread_id) as sp:
            try:
                response = await bound.ainvoke([SystemMessage(content=system)] + msgs)
            except Exception as _e1:
                # 降级兜底 1：剥离历史纯文本 assistant 消息（保留工具链）
                # 实测 DeepSeek thinking 兼容偶发 400（reasoning_content 回传要求），
                # 重放同批消息有时成功有时失败 → 失败时逐级降级重试，不中断整轮。
                print(f"--> [Agent] ⚠️ 调用失败 {type(_e1).__name__}: {str(_e1)[:80]} → 降级重试（剥离纯文本历史）")
                _fb = [m for m in msgs
                       if not (isinstance(m, AIMessage) and not getattr(m, "tool_calls", None))]
                try:
                    response = await bound.ainvoke([SystemMessage(content=system)] + _fb)
                except Exception as _e2:
                    # 降级兜底 2：只留最近 2 条 user 消息 + 其后的 tool 链
                    print(f"--> [Agent] ⚠️ 再次失败 {type(_e2).__name__}: {str(_e2)[:80]} → 二次降级（只留最近 user 消息）")
                    _u_idx = [i for i, m in enumerate(msgs)
                              if isinstance(m, (HumanMessage, SystemMessage))]
                    _cut = max(0, (_u_idx[-3] if len(_u_idx) >= 3 else 0))
                    _fb2 = msgs[_cut:]
                    response = await bound.ainvoke([SystemMessage(content=system)] + _fb2)
            um = getattr(response, "usage_metadata", None) or {}
            used = int(um.get("total_tokens", 0) or 0)
            sp["tokens"] = used

        n_calls = len(getattr(response, "tool_calls", None) or [])
        print(f"--> [Agent] step={step + 1} tool_calls={n_calls} "
              f"{[c['name'] for c in (response.tool_calls or [])] if n_calls else '(收尾)'} tokens+={used}")
        updates.update({"messages": [response], "step_count": step + 1, "tokens_used": tokens + used})
        if n_calls == 0 and not updates.get("stop_reason"):
            updates["stop_reason"] = "model_done"
        return updates

    return agent_node


def _pool_relevant_ratio_mean(state: AgentState):
    """一轮检索后全池相关比例均值（与 eval_baseline 的 relevant_ratio_mean 同口径）。

    来源：read_sections 内嵌全池重排后写回的 section_candidates.sections[*].n_relevant/scored_total
    （Cross-Encoder 全池重排，分数 ≥ RERANK_MIN_SCORE 视为相关）。
    无重排结果（还没调 read_sections）→ None → 不据此拒答（放行）。
    """
    sect = state.get("section_candidates") or {}
    ratios = []
    for _sec, _v in (sect.get("sections") or {}).items():
        _nr, _st = _v.get("n_relevant"), _v.get("scored_total")
        if _st:
            ratios.append((_nr or 0) / _st)
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


def reject_node(state: AgentState):
    """【节点 · 拒答】输出拒答说明（意图门拦截 / 一轮检索后证据不足），不生成综述。"""
    gate = state.get("intent_gate") or {}
    _mal = float(gate.get("malicious") or 0)
    _res = float(gate.get("researchable") or 0.5)
    reason = str(gate.get("reason") or "").strip()
    if _mal >= REJECT_MALICIOUS_THRESHOLD:
        head = (f"该请求未能通过科研命题校验：被判定为恶意测试或伪命题"
                f"（置信度 {_mal:.0%}），不会生成综述。")
    elif _res < REJECT_RESEARCHABLE_THRESHOLD:
        head = (f"该命题未通过可研究性校验（可研究性置信度 {_res:.0%}）："
                f"缺乏可检索的科研证据基础，无法生成有据可依的综述。")
    else:
        head = "一轮检索后未找到足够相关文献，证据不足，已停止生成综述。"
    if reason:
        head = head + " 判定依据：" + reason
    print(f"--> [Reject] {head[:120]}")
    return {"final_review": head, "draft_review": "",
            "rejected": True, "reject_reason": head}


def route_after_agent(state: AgentState) -> str:
    """模型调用了工具 → tools；否则由模型终止研究阶段 → write / answer。

    （v1.2.7 已删除人机审核门 nudge——write_review 不再要求 ask_user。）
    """
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if last is not None and getattr(last, "tool_calls", None):
        return "tools"

    skills = state.get("skills", [])
    if needs_write_review(skills):
        # 一轮检索后的确定性拒答门：全池相关比例均值 < 阈值 → 证据不足拒答
        _ratio = _pool_relevant_ratio_mean(state)
        if _ratio is not None and _ratio < REJECT_RATIO_THRESHOLD:
            print(f"--> [Route] 一轮检索后全池相关比例均值 {_ratio:.3f}"
                  f" < {REJECT_RATIO_THRESHOLD} → 证据不足拒答")
            return "reject"
        print("--> [Route] agent 完成研究 → write")
        return "write"
    print("--> [Route] agent 完成 → answer")
    return "answer"


_TITLE_STOPWORDS = {
    "graph", "neural", "network", "networks", "learning", "model", "models",
    "based", "using", "approach", "approaches", "method", "methods", "deep",
    "data", "system", "systems", "recommendation", "recommender", "collaborative",
    "filtering", "analysis", "study", "research", "paper", "novel", "efficient",
    "towards", "via", "with", "from", "over", "user", "users", "item", "items",
    "attention", "representation", "embedding", "framework", "survey", "review",
}


def _title_relevant(title: str, text: str) -> bool:
    """相关性校验：全文里应出现标题的**区分性**关键词（防止 id 对不上、读错论文）。

    过滤掉 graph/neural/learning 这类跨论文通用词，只看真正能区分论文的词
    （如 lightgcn / convolution / ngcf）。通用词全部被过滤时放行（避免误杀）。
    """
    tw = set(re.findall(r"[A-Za-z]{4,}", (title or "").lower())) - _TITLE_STOPWORDS
    if not tw:
        return True  # 标题无区分性英文词（中文标题等），放行
    low = (text or "")[:5000].lower()
    hit = sum(1 for w in tw if w in low)
    return hit >= max(1, len(tw) // 3)   # 至少命中 1/3 区分性关键词


_ev_embed_fn = None
_ev_embed_cache = {}
_ev_query_cache = {}


def _get_ev_embed_fn():
    """证据段落打分的 embedding（bge-m3，模块级单例 + 向量缓存）。"""
    global _ev_embed_fn
    if _ev_embed_fn is None:
        try:
            from ResearcherAgent.memory.memory_classifier import make_embed_fn
            _ev_embed_fn = make_embed_fn()
        except Exception:
            _ev_embed_fn = None
    return _ev_embed_fn


def _ev_cos(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    n1 = sum(a * a for a in v1) ** 0.5
    n2 = sum(b * b for b in v2) ** 0.5
    if not n1 or not n2:
        return 0.0
    return sum(a * b for a, b in zip(v1, v2)) / (n1 * n2)


# ── 节类型词表（五步流水线第 3 步：字段化抽取的确定性定位）──
# 2026-10 v1.2.2 扩充：尽量用正则/词表命中章节标题，能不用 LLM 就不用。
# 覆盖通用 + 视觉/医药/时序/工程各领域常见章节命名变体（中英）。
# 注意子串匹配（kw in H）：桶顺序 background→method→experiment→conclusion，
# 避免把会被实验桶抢走的词放进前桶（如裸 "SETUP" 会让 "EXPERIMENTAL SETUP" 误归 background）。
_SECTION_TYPE_KEYWORDS = {
    "background": (
        # 通用英文
        "INTRODUCTION", "INTRO", "BACKGROUND", "MOTIVATION", "PRELIMINARIES",
        "RELATED WORK", "PROBLEM SETUP", "PROBLEM FORMULATION", "PROBLEM STATEMENT",
        "TASK DEFINITION", "TASK SETUP", "FORMULATION", "PREREQUISITES",
        "INTRODUCTION AND BACKGROUND", "BACKGROUND AND RELATED WORK",
        "BACKGROUND AND MOTIVATION", "PAPER OVERVIEW",
        # 医学/生物
        "CLINICAL BACKGROUND", "INTRODUCTION AND OBJECTIVES",
        "BACKGROUND AND OBJECTIVES", "SIGNIFICANCE", "OBJECTIVE AND AIMS",
        # 中文
        "引言", "背景", "绪论", "相关工作", "问题定义", "研究背景",
        "研究背景与意义", "引言与相关工作", "研究动机", "问题提出", "研究问题",
    ),
    "method": (
        # 通用英文
        "PROPOSED METHOD", "METHODOLOGY", "METHOD", "APPROACH",
        "MODEL ARCHITECTURE", "ARCHITECTURE", "PROPOSED MODEL", "FRAMEWORK",
        "DESIGN", "SYSTEM", "NETWORK", "BACKBONE", "GENERATIVE MODEL",
        "THE PROPOSED METHOD", "OUR METHOD", "OUR APPROACH", "PROPOSED APPROACH",
        "PROPOSED FRAMEWORK", "SYSTEM DESIGN", "SYSTEM OVERVIEW", "MODELING",
        "THE MODEL", "NETWORK DESIGN", "LEARNING ALGORITHM", "ALGORITHM DESIGN",
        "TRAINING PROCEDURE", "TRAINING DETAILS", "TRAINING", "TECHNICAL APPROACH",
        # 医学/生物
        "TRIAL DESIGN", "INTERVENTION", "PROTOCOL",
        "METHOD AND MATERIALS", "MATERIALS AND METHODS", "STUDY DESIGN",
        # 中文
        "方法", "模型", "架构", "提出", "设计", "方法学", "模型结构",
        "研究方法", "技术路线", "系统设计", "体系结构", "网络结构",
        "算法设计", "训练细节", "训练过程", "模型设计",
    ),
    "experiment": (
        # 通用英文
        "EXPERIMENTS", "EXPERIMENTAL SETUP", "RESULTS", "EVALUATION",
        "EXPERIMENTAL RESULTS", "PERFORMANCE", "BENCHMARKS", "DATASETS",
        "ABLATION", "QUALITATIVE", "EXPERIMENTS AND RESULTS", "EXPERIMENTAL STUDY",
        "EMPIRICAL STUDY", "EMPIRICAL EVALUATION", "PERFORMANCE EVALUATION",
        "BENCHMARK RESULTS", "MAIN RESULTS", "RESULTS AND ANALYSIS",
        "QUANTITATIVE EVALUATION", "NUMERICAL EXPERIMENTS", "SIMULATION RESULTS",
        "COMPARATIVE STUDY", "CASE STUDY", "ERROR ANALYSIS", "EVALUATION AND RESULTS",
        # 医学/生物
        "COHORT", "OUTCOME", "STATISTICAL ANALYSIS", "CLINICAL OUTCOMES",
        "OUTCOMES AND ANALYSIS",
        # 中文
        "实验", "结果", "评估", "数据集", "性能", "消融", "实验设置",
        "实验与结果", "实验结果与分析", "实验研究", "实证研究", "性能评估",
        "对比实验", "仿真实验", "数值实验", "定量评估",
    ),
    "conclusion": (
        # 通用英文
        "CONCLUSION", "DISCUSSION", "LIMITATIONS", "FUTURE WORK", "SUMMARY",
        "CONCLUDING REMARKS", "CONCLUSIONS", "DISCUSSION AND CONCLUSION",
        "CONCLUSION AND DISCUSSION", "DISCUSSION AND FUTURE WORK",
        "FUTURE DIRECTIONS", "FUTURE PERSPECTIVES", "OUTLOOK",
        "LIMITATIONS AND FUTURE WORK", "LIMITATION",
        # 医学/生物
        "CLINICAL IMPLICATIONS",
        # 中文
        "结论", "讨论", "局限", "未来工作", "展望", "总结",
        "结论与展望", "讨论与结论", "结语", "未来研究方向", "研究结论", "总结与展望",
    ),
}
_FIELD_ORDER = ("background", "method", "experiment", "conclusion")
# 字段语义 query（与标题拼接后做 embedding 定向打分）
_FIELD_QUERIES = {
    "background": "研究背景、动机与问题定义",
    "method": "提出的方法、模型架构与核心机制",
    "experiment": "实验设置、数据集与结果指标",
    "conclusion": "结论、局限与未来工作",
}


def _classify_section_heading(heading: str) -> str:
    """按多领域变体词表判节类型；顺序=background→method→experiment→conclusion
    （RESULTS AND DISCUSSION 这种双命中归 experiment，先查后者之前的桶）。"""
    h = (heading or "").strip().upper()
    if not h:
        return "other"
    for ftype in _FIELD_ORDER:
        for kw in _SECTION_TYPE_KEYWORDS[ftype]:
            if kw in h:
                return ftype
    return "other"


def _split_passages(text: str, max_para: int = 900, full_text: bool = True) -> list:
    """确定性切段：按空行分段，段落首行若为数字节标题（如 3.2 / 4.1）则记录节号+节标题文字+节类型。

    证据直接是**原文段落**（不做 LLM 提炼，杜绝二次有损）。
    full_text=True：不再截断前 30000 字符（方案 v1.2：抽取阶段不吃 LLM 上下文，
    全篇切段后由 _map_to_cards 按字段桶打分，超大文档按段落数预算裁剪）。
    返回 [{sec, sec_title, sec_type, text}, ...]——sec_title/sec_type 为 2026-10 新增键。
    """
    t = text or ""
    if not full_text:
        t = t[:30000]
    # 预扫描数字节号 → 节号兜底类型（无标题关键词时：最小基节→背景，最大基节→结论）
    # 节头按**首行**判定（标题与正文同段不丢节信息，2026-10 修复）
    sec_nos = []
    for chunk in re.split(r"\n{2,}", t):
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            continue
        m = re.match(r"^(\d+(?:\.\d+)*)[\.\)、]?\s*(.*)$", lines[0])
        if m and len(m.group(2)) <= 60:
            sec_nos.append(m.group(1))
    base_nos = sorted({int(s.split(".")[0]) for s in sec_nos})
    parent_type: dict = {}
    if base_nos:
        for bn in base_nos:
            if bn == base_nos[0] and bn != base_nos[-1]:
                parent_type[bn] = "background"
            elif bn == base_nos[-1] and bn != base_nos[0]:
                parent_type[bn] = "conclusion"
            else:
                parent_type[bn] = "other"

    out, cur_sec, cur_sec_title, cur_sec_type = [], "", "", "other"
    base_ft: dict = {}   # v1.2.16 基节→类型映射（子节继承父节，消除 4.3 挂实验节下却归 method 类误伤）
    _BASE_RE = re.compile(r"^(?:Appendix\s+)?([A-Z])(?:\.\d+)?\s", re.I)

    def _base_of(sec: str, title: str) -> str:
        """节的基节号：数字节号取首段（4.3→4）；字母附录节取字母（C.1→C，Appendix C→C）。"""
        if sec and re.match(r"^\d+(?:\.\d+)*$", sec):
            return sec.split(".")[0]
        m = _BASE_RE.match(title or "")
        if m:
            return m.group(1).upper()
        return ""

    for chunk in re.split(r"\n{2,}", t):
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            continue
        c = " ".join(chunk.split())
        is_heading = False
        # ar5iv/HTML 标题常带 "## " 前缀（"## 4 Experiments"）——先剥掉再按数字节号判定
        head0 = re.sub(r"^#+\s*", "", lines[0])
        head_m = re.match(r"^(\d+(?:\.\d+)*)[\.\)、]?\s*(.*)$", head0)
        if head_m and len(head_m.group(2)) <= 60:
            cur_sec = head_m.group(1)
            cur_sec_title = head_m.group(2).strip()
            is_heading = True
            if "." in cur_sec:      # 子节（4.3）→ 父节类型优先，词表兜底
                kt = _classify_section_heading(cur_sec_title)
                pft = base_ft.get(cur_sec.split(".")[0])
                if pft and pft != "other":      # 父节已确定（非 other）→ 继承
                    cur_sec_type = pft
                else:                           # 父节 other/缺失 → 词表 → 节号兜底
                    cur_sec_type = kt if kt != "other" else parent_type.get(
                        int(cur_sec.split(".")[0]), "other")
            else:                   # 基节（4）→ 词表优先，节号兜底；并更新 base_ft
                kt = _classify_section_heading(cur_sec_title)
                cur_sec_type = kt if kt != "other" else parent_type.get(int(cur_sec), "other")
                base_ft[cur_sec] = cur_sec_type
        elif _looks_like_heading(head0) and len(lines) > 1:
            # 无编号标题行（"Introduction" / "EXPERIMENTAL SETUP" 等，词表或全大写短行）
            # → 同样开启新节，证据锚点用标题本身；v1.2.16：标题行后必须有正文（单行
            #   作者名/参考文献/页脚不再当标题）。
            cur_sec = head0[:60]
            cur_sec_title = head0
            is_heading = True
            kt = _classify_section_heading(cur_sec_title)
            bsec = cur_sec.split(".")[0] if cur_sec.split(".")[0].isdigit() else ""
            is_sub = "." in cur_sec if bsec else bool(_BASE_RE.match(cur_sec_title or ""))
            if is_sub and bsec:      # 数字子节（C.1 走标题分支时 bsec 为空，走基节分支）
                kt2 = _classify_section_heading(cur_sec_title)
                pft2 = base_ft.get(bsec)
                cur_sec_type = pft2 if (pft2 and pft2 != "other") else (
                    kt2 if kt2 != "other" else parent_type.get(int(bsec), "other"))
            else:
                base = _base_of(cur_sec, cur_sec_title)
                if base and base.isalpha():     # 字母附录子节（C.1 / Appendix C）
                    if re.match(r"^[A-Z]\.\d+", cur_sec_title or ""):
                        cur_sec_type = base_ft.get(base, kt if kt != "other" else "other")
                    else:                       # 字母基节（Appendix C）
                        cur_sec_type = kt if kt != "other" else "other"
                        base_ft[base] = cur_sec_type
                else:
                    cur_sec_type = kt if kt != "other" else parent_type.get(
                        int(cur_sec.split(".")[0]) if cur_sec.split(".")[0].isdigit() else 0, "other")
                    if bsec:
                        base_ft[bsec] = cur_sec_type
        if not is_heading:
            while len(c) > max_para:
                out.append({"sec": cur_sec, "sec_title": cur_sec_title,
                            "sec_type": cur_sec_type, "text": c[:max_para]})
                c = c[max_para:]
            if c:
                out.append({"sec": cur_sec, "sec_title": cur_sec_title,
                            "sec_type": cur_sec_type, "text": c})
            continue
        out.append({"sec": cur_sec, "sec_title": cur_sec_title,
                    "sec_type": cur_sec_type, "text": cur_sec_title[:max_para]})
        body = " ".join(lines[1:]) if len(lines) > 1 else ""
        if not body:
            continue
        c = body
        while len(c) > max_para:
            out.append({"sec": cur_sec, "sec_title": cur_sec_title,
                        "sec_type": cur_sec_type, "text": c[:max_para]})
            c = c[max_para:]
        if c:
            out.append({"sec": cur_sec, "sec_title": cur_sec_title,
                        "sec_type": cur_sec_type, "text": c})
    return out


def _noise_heading(line: str) -> bool:
    """标题行噪音排除（v1.2.16 精确标题策略）：
    表格行 / 参考文献行 / 无字母词的碎片行 即使在词表命中时也**不当标题**——
    实测噪音 17/52：| System | Dev | Test |（含 SYSTEM 命中 method）、
    In Advances in neural information processing systems（含 SYSTEM 命中 method）、
    ILI 7 7、I 等全被 _looks_like_heading 误判，污染字段桶。"""
    s = (line or "").strip()
    if not s:
        return True
    if "|" in s:                      # 表格行（Markdown 表格头/分隔行）
        return True
    if s.startswith("In "):           # 参考文献行（"In <会议/期刊>..."）
        return True
    if " : " in s:                    # 论文标题副标题形态（"BERT : Pre-training of ..."）——
        return True                   # 词表命中会把它误当 method 节且与字段 query 相似度最高
    words = [w for w in s.split() if w]
    if not words or len(words) > 12:
        return True
    alpha = [w for w in words if any(ch.isalpha() for ch in w)]
    if not alpha or sum(len(w) for w in alpha) < 4:   # 纯数字/单字母碎片（"I"）
        return True
    if s[-1].isdigit() and _classify_section_heading(s) == "other":
        return True                   # 以数字结尾且无词表词（"ILI 7 7" 类页脚碎片）
    return False


def _looks_like_heading(line: str) -> bool:
    """无编号章节标题行的确定性判定：命中节类型词表，或全大写/Title-Case 短行
    （≤60 字符、不以句号结尾——避免把正文句子误判为节头）。
    v1.2.16：先过 _noise_heading 噪音排除（表格/参考文献/碎片行不再当标题）。"""
    s = (line or "").strip()
    if not s or len(s) > 60:
        return False
    if s.endswith((".", "。", "!", "？", ";", "：", ":")):
        return False
    if _noise_heading(s):
        return False
    if _classify_section_heading(s) != "other":
        return True
    words = [w for w in s.split() if w]
    if not words or len(words) > 12:
        return False
    # 全大写短行（INTRODUCTION / EXPERIMENTAL SETUP / A. RESULTS 等）
    if all(w.isupper() or w.isdigit() or w in ".()&-/:," for w in words):
        return True
    # Title-Case 短行：每词首字符大写（含介词），如 "Related Work" / "Proposed Method"
    if all(w[:1].isupper() for w in words) and len(s) <= 40:
        return True
    return False


_ARXIV_CODE_URL_CACHE: dict = {}


def _arxiv_code_url(arxiv_id: str) -> str:
    """arXiv API 元数据提取作者标注的开源代码链接（abstract 常含 'code: github.com/...'）。
    带会话内缓存，避免重复调用。"""
    if not arxiv_id or not re.match(r"^\d{4}\.\d{4,5}", arxiv_id):
        return ""
    if arxiv_id in _ARXIV_CODE_URL_CACHE:
        return _ARXIV_CODE_URL_CACHE[arxiv_id]
    import urllib.request
    url = ""
    try:
        with urllib.request.urlopen(
                f"https://export.arxiv.org/api/query?id_list={arxiv_id}", timeout=15) as r:
            xml = r.read().decode("utf-8", errors="replace")
        url = _extract_github_url(xml)
    except Exception:
        pass
    _ARXIV_CODE_URL_CACHE[arxiv_id] = url
    return url


def _paper_code_url(blk: dict, text: str) -> str:
    """论文代码链接：arXiv API 元数据优先（作者标注，精准）→ 全文正则兜底。"""
    pid = blk.get("id", "") if isinstance(blk, dict) else ""
    u = _arxiv_code_url(pid)
    if not u:
        u = _extract_github_url(text or "")
    return u


_GITHUB_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?"
    r"(?=[/#\s.,;:)\]）】]|$)|"
    r"git@github\.com:([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?(?=[\s.,;:)\]）】]|$)",
    re.I)


def _extract_github_url(text: str) -> str:
    """从全文确定性提取论文标注的开源代码链接（GitHub）；无则返回空串。"""
    m = _GITHUB_URL_RE.search(text or "")
    if not m:
        return ""
    owner = m.group(1) or m.group(3)
    repo = m.group(2) or m.group(4)
    return f"https://github.com/{owner}/{repo}"


def _text_table_to_markdown(text: str) -> str:
    """纯文本表格启发式（PDF 提取等无 HTML 结构时的兜底）：
    连续 ≥3 行含 '|' 分隔 → 视为 markdown 表格原样保留（行列语义不丢）。"""
    lines = (text or "").splitlines()
    out, buf = [], []
    for ln in lines:
        s = ln.strip()
        if s and "|" in s:
            buf.append(s)
        else:
            if len(buf) >= 3:
                out.append("\n".join(buf))
            buf = []
    if len(buf) >= 3:
        out.append("\n".join(buf))
    return "\n\n".join(out)


def _locate_sections_with_llm(llm, paras: list, title: str) -> dict:
    """关键词/节号全部失效时的 LLM 兜底定位：返回 {field: sec, ...}。
    输入=节标题列表+各节首段（不是全文，控 token），一次 LLM 调用。
    async_pipeline 的 locate_fallback 消费者复用本纯函数。"""
    heads, seen = [], set()
    for p in paras:
        s, t = p.get("sec", ""), p.get("sec_title", "")
        if not t or s in seen:
            continue
        seen.add(s)
        first = " ".join(p["text"].split())[:200]
        heads.append(f"{s}: {t} —— {first}")
    if not heads:
        return {}
    prompt = (
        "你是学术文献结构定位器。下面是一篇论文的章节列表（节号: 标题 —— 首段摘要）。\n"
        + "\n".join(heads[:40])
        + "\n\n请把章节归类到四个字段之一：background（背景/引言）、method（方法/模型/机制）、"
          "experiment（实验/数据集/结果指标）、conclusion（结论/局限/展望）。\n"
          "注意：不同领域术语不同（如医药 Trial Design/Outcome、视觉 Backbone/Ablation），"
          "请按语义判断，不要死板匹配；无法归类的返回 null。\n"
          '只输出 JSON：{"background": "节号或null", "method": "...", '
          '"experiment": "...", "conclusion": "..."}'
    )
    try:
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            return {}
        d = json.loads(m.group(0))
        return {k: str(v).strip() for k, v in d.items()
                if v and str(v).strip().lower() not in ("null", "none", "")}
    except Exception as e:
        print(f"    [locate_fallback] LLM 兜底定位失败: {type(e).__name__}: {str(e)[:80]}")
        return {}


_NO_OTHER_FILL = {"abstract", "references", "bibliography",
                  "acknowledgements", "acknowledgments",
                  "acknowledgement", "acknowledgment"}


def extract_method_json_direct(full_text: str, title: str, section: str,
                               num, llm) -> dict:
    """MQ2 兜底（v1.2.17 用户决策）：method 关键词未命中 → LLM 直接提取句式 JSON 块。

    输入=论文全文（截 60000 预算），**一次 LLM 调用**直接输出可直接组装的 method 块
    （句式四要素「提出了XX / 用XX解决了XX / 数据集+指标」），不做原文卡片再提取
    （避免两步 LLM）；LLM 判定全文无方法论 → {"drop": true} 丢弃该篇。
    返回 {"blocks": [...], "drop": bool, "reason": str}。
    """
    text = (full_text or "").strip()
    if len(text) < 300:
        return {"drop": True, "reason": "全文不可得"}
    text = text[:60000]
    n_str = f"[{num}]" if num not in (None, "") else ""
    prompt = (
        "你是学术文献方法提取器。下面是论文全文（标题：__TITLE__，所属综述节：__SECTION__，"
        "引用编号__NUM__）。\n"
        "__TEXT__\n"
        "任务：判断这篇论文是否**提出或描述了一个具体的方法/模型/框架/算法**。\n"
        "1. 如果没有（纯数据集论文、纯综述、立场文、教程，或全文只是在讨论/比较别人的方法而"
        "未提出自己的方法）→ 输出 {\"drop\": true, \"reason\": \"一句话原因\"}。\n"
        "2. 如果有 → 用一句话概括该方法，围绕四个核心要素（措辞可灵活，核心必须保留）：\n"
        "   「XX等人提出了XX，一种XXX，通过XX解决了XXX（难题），在XX数据集上达到XX（指标）。」\n"
        "   - 提出者/名称/机制/针对难题必须来自全文，不得编造；\n"
        "   - 数据集+指标必须来自实验部分；**没有实验数据就不写指标句**（论文没给数字就不能编）；\n"
        "   - 句子末尾标引用编号__NUM__（没给编号就省略）。\n"
        "   输出 {\"blocks\": [{\"content\": \"一句话...__NUM__\", "
        "\"evidence\": [\"§节号\"], \"paper_ref\": \"论文标题\"}]}\n"
        "只输出 JSON，不要其他文字。"
    )
    prompt = (prompt.replace("__TITLE__", title or "")
              .replace("__SECTION__", section or "")
              .replace("__TEXT__", text)
              .replace("__NUM__", n_str))
    try:
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            return {"drop": True, "reason": "LLM 输出非 JSON"}
        d = json.loads(m.group(0))
        if d.get("drop") or d.get("dropped"):
            return {"drop": True, "reason": str(d.get("reason") or "LLM 判定无方法论")}
        out = []
        for b in (d.get("blocks") or []):
            c = str(b.get("content") or "").strip()
            if not c:
                continue
            out.append({
                "block_type": "method",
                "content": c,
                "paper_ref": str(b.get("paper_ref") or "") or title,
                "evidence": b.get("evidence") or [],
                "num": num,
            })
        if not out:
            return {"drop": True, "reason": "LLM 未产出方法块"}
        return {"blocks": out}
    except Exception as e:
        print(f"    [mq2_direct] LLM 提取失败 {type(e).__name__}: {str(e)[:80]}")
        return {"drop": True, "reason": "LLM 提取异常"}


def _map_to_cards(blocks: list, llm, skip_llm_fallback: bool = False,
                  prelocated: dict | None = None) -> list:
    """【分字段原文证据提取】——五步流水线第 3 步：字段化卡片。

    弃用「LLM 提炼卡片」（二次有损压缩是忠实度判不出来的根因之一），改为：
      1) 读全文（全篇，不再 30000 截断；超大文档按 FULLTEXT_PARAGRAPH_BUDGET 兜底保方法/实验节）；
      2) _split_passages 确定性切段：节号 + 节标题 + 节类型（关键词优先，节号兜底）；
      3) 按 背景/方法/实验/结论 四字段分桶，桶内 bge-m3 按「字段语义 + 论文标题」打分取 top-2；
      4) 实验字段内表格转 Markdown（fulltext 层 HTML 表格已转；纯文本启发式兜底）；
      5) 字段零命中且全文非空 → _locate_sections_with_llm 兜底定位（MQ 消费者复用同一纯函数）。

    v1.2.15（MQ 制卡流水线）：新增两参数——
      · skip_llm_fallback=True：只走关键词/embedding 确定性路径，缺失字段留空（MQ1 制卡消费者用，
        命中 extract_mode=keyword 即成功；未命中 → 转 MQ2 由 LLM 兜底）；
      · prelocated：MQ2 已返回的 {field: sec_no} 定位结果直接注入字段（不再二次调 LLM）。

    card 字段 = 四字段原文拼接（带【字段】标签，写作/评审沿用）；
    fields 字段 = 结构化字典（JSON 生成/证据锚校验用）；缺失字段标 "none"，禁止编造；
    extract_mode = keyword | llm_fallback | abstract_only。
    """
    if not blocks:
        return []
    ef = _get_ev_embed_fn()
    cards = []
    _n_hit = 0
    for blk in blocks:
        title = blk.get("title", "") if isinstance(blk, dict) else ""
        text = blk.get("text", "") if isinstance(blk, dict) else str(blk)
        sec = blk.get("section", "") if isinstance(blk, dict) else ""
        # v1.2.15 缓存键按制卡路径区分：keyword-only（MQ1 无 LLM 兜底）、prelocated（MQ2）、
        # 完整（无参数）——防止 MQ1 关键词路径把"残缺卡"写进缓存后，MQ2 LLM 兜底
        # 重制时命中缓存直接返回残缺卡（LLM 白跑）。旧缓存（无后缀）仍被完整路径命中。
        mode_sfx = "|kwo" if skip_llm_fallback else (
            "|pre" if isinstance(prelocated, dict) and any(prelocated.values()) else "")
        key = f"{sec}|{title}{mode_sfx}"
        cached = cache.llm_get("extract_passages_v6", key, text[:500])
        result = None
        if cached:
            try:
                result = json.loads(cached)
                _n_hit += 1
            except Exception:
                result = None
        if result is None:
            paras = _split_passages(text, full_text=True)
            if not paras:
                # 全文不可得 → 摘要级兜底卡片（写作只按「未精读·仅摘要」一句话提及）
                result = {"fields": {f: "none" for f in _FIELD_ORDER},
                          "card": (" ".join(text.split()))[:4000],
                          "passages": [], "extract_mode": "abstract_only"}
            else:
                # 段落数预算兜底（防 PDF 噪音文本打爆打分）：超预算只保方法/实验节
                if len(paras) > FULLTEXT_PARAGRAPH_BUDGET:
                    keep = [p for p in paras if p["sec_type"] in ("method", "experiment")]
                    bg = [p for p in paras if p["sec_type"] == "background"][:40]
                    cl = [p for p in paras if p["sec_type"] == "conclusion"][:40]
                    paras = keep + bg + cl
                buckets: dict = {}
                for p in paras:
                    buckets.setdefault(p.get("sec_type", "other"), []).append(p)
                fields: dict = {}
                used_llm = False
                for ftype in _FIELD_ORDER:
                    cands = buckets.get(ftype, [])
                    if not cands:
                        # v1.2.16 词表未覆盖的标题（如 "An Embarrassingly Simple Baseline"）
                        # → other 桶节头段落按字段 query 打分补位（bge-m3 确定性，不调 LLM；
                        #   只对节头段落打分控计算量，选中节则整节内容入字段）。
                        # 排除框架节（Abstract/References/Acknowledgements 等）：它们的内容
                        # 是全文浓缩或题录，补进任何字段都会稀释该字段的证据纯度。
                        other_heads, seen_oh = [], set()
                        for p in buckets.get("other", []):
                            st = (p.get("sec_title") or "").strip()
                            if not st or st.lower() in _NO_OTHER_FILL:
                                continue
                            if st not in seen_oh:
                                seen_oh.add(st)
                                other_heads.append(p)
                        cands = other_heads[:FIELD_TOP_K * 3]
                        if not cands:
                            continue
                        from_other = True
                    else:
                        from_other = False
                    q = _FIELD_QUERIES[ftype] + " " + title
                    qv = None
                    if ef is not None:
                        _qk = ("#q#", q)
                        if _qk in _ev_query_cache:
                            qv = _ev_query_cache[_qk]
                        else:
                            try:
                                qv = ef(q)
                                _ev_query_cache[_qk] = qv
                            except Exception:
                                qv = None
                    scored = []
                    for i, p in enumerate(cands):
                        s = 0.0
                        if ef is not None and qv is not None:
                            _pk = ("#p#", p["text"][:500])
                            pv = _ev_embed_cache.get(_pk)
                            if pv is None:
                                try:
                                    pv = ef(p["text"][:500])
                                    _ev_embed_cache[_pk] = pv
                                except Exception:
                                    pv = None
                            s = _ev_cos(pv, qv)
                        else:
                            s = 1.0 - i * 1e-6   # 无 embedding 时按原文顺序保序
                        scored.append((s, p))
                    scored.sort(key=lambda x: -x[0])
                    if from_other:
                        # other 补位：选中节头 top-2 → 整节段落入字段（节头相关=整节相关）
                        sel_titles = {p.get("sec_title") for _, p in scored[:FIELD_TOP_K]
                                      if p.get("sec_title")}
                        full = [p for p in buckets.get("other", [])
                                if p.get("sec_title") in sel_titles]
                        fields[ftype] = "\n".join(
                            f"【§{p['sec'] or '?'}】{p['text']}" for p in full)
                    else:
                        fields[ftype] = "\n".join(
                            f"【§{p['sec'] or '?'}】{p['text']}" for _, p in scored[:FIELD_TOP_K])
                # 字段零命中 → LLM 兜底定位（只对缺失字段补）
                # v1.2.15：skip_llm_fallback=True（MQ1 关键词路径）时跳过 LLM；
                # prelocated（MQ2 已定位结果）直接注入，不再二次调 LLM；
                # prelocated 非空时不设 len(paras)>3 门槛（短论文 LLM 已定位也照补）。
                missing = [f for f in _FIELD_ORDER if not fields.get(f)]
                has_pre = isinstance(prelocated, dict) and any(prelocated.values())
                if missing and not skip_llm_fallback and (len(paras) > 3 or has_pre):
                    if has_pre:
                        located, used_llm = prelocated, True
                    else:
                        located = _locate_sections_with_llm(llm, paras, title)
                        used_llm = True
                    for ftype in missing:
                        sec_no = located.get(ftype)
                        if not sec_no:
                            continue
                        picks = [p for p in paras if p["sec"] == sec_no][:FIELD_TOP_K]
                        if picks:
                            fields[ftype] = "\n".join(
                                f"【§{p['sec'] or '?'}】{p['text']}" for p in picks)
                # 实验字段表格兜底（纯文本启发式，PDF 通路无 HTML 结构）
                if fields.get("experiment"):
                    tbl = _text_table_to_markdown(fields["experiment"])
                    if tbl and tbl not in fields["experiment"]:
                        fields["experiment"] += "\n\n" + tbl
                # extract_mode
                non_none = [f for f in _FIELD_ORDER if fields.get(f)]
                if not non_none:
                    mode = "abstract_only"
                elif used_llm:
                    mode = "llm_fallback"
                else:
                    mode = "keyword"
                fields = {f: (fields.get(f) or "none") for f in _FIELD_ORDER}
                card_txt = "\n".join(
                    f"【{ftype}】{fields[ftype]}" for ftype in _FIELD_ORDER if fields[ftype] != "none")
                # passages 兼容旧消费方（{sec,text} 列表，按字段出现顺序去重）
                seen_p, passages = set(), []
                for ftype in _FIELD_ORDER:
                    for seg in re.split(r"\n(?=【§)", fields[ftype]):
                        m = re.match(r"【§([^】]+)】", seg)
                        if m and m.group(1) not in seen_p:
                            seen_p.add(m.group(1))
                            passages.append({"sec": m.group(1), "text": seg})
                result = {"fields": fields, "card": card_txt,
                          "passages": passages, "extract_mode": mode}
            try:
                cache.llm_set("extract_passages_v6", key, text[:500],
                              json.dumps(result, ensure_ascii=False))
            except Exception:
                pass
        if result.get("fields"):
            cards.append({"title": title, "id": blk.get("id", "") if isinstance(blk, dict) else "",
                          "section": sec, "card": result["card"],
                          "passages": result.get("passages", []),
                          "fields": result["fields"],
                          "extract_mode": result.get("extract_mode", "keyword"),
                          "code_url": _paper_code_url(blk, text)})
    if _n_hit:
        print(f"    ♻️ [L1] 字段化证据段落缓存命中 {_n_hit}/{len(blocks)} 篇")
    return cards


def check_citation_grounding(draft: str, source_text: str) -> dict:
    """引用真实性确定性校验：综述里的引用是否来自检索材料。

    做法：提取正文里的 [N] 引用标记 + 末尾参考文献列表，逐条用标题/关键词去材料里匹配。
    返回 {total_refs, grounded, hallucinated, rate, ungrounded:[...]}。
    """
    if not draft:
        return {"total_refs": 0, "grounded": 0, "hallucinated": 0, "rate": 1.0, "ungrounded": []}

    # 参考文献段：取**最后一次**出现的、位于行首的标题。
    # 不能取首个——正文里常有「参考文献说明」这类解释性小节，各章节也可能各带参考表；
    # 取首个会把后面 80%+ 的正文误当作参考文献段来校验（实测占比 89%）。
    ref_marks = list(re.finditer(r"(?m)^\s*#{0,3}\s*(?:参考文献|References)", draft))
    ref_section = draft[ref_marks[-1].end():] if ref_marks else ""
    body = draft[:ref_marks[-1].start()] if ref_marks else draft

    # 正文引用标记：同时支持数字体例 [1] 与作者—年份体例（Author, 2024）。
    # 只认 [N] 会让 APA 体例的稿件永远报 0，指标失去意义（写作提示词是按记忆偏好选体例的）。
    numeric_cites = re.findall(r"\[(\d{1,3})\]", body)
    author_year_cites = re.findall(
        r"[（(][^（）()]{0,120}?\b(?:19|20)\d{2}[a-z]?[^（）()]{0,40}?[）)]", body)
    body_cites = numeric_cites or author_year_cites
    ref_lines = [ln.strip() for ln in re.split(r"\n", ref_section) if ln.strip()][:60]
    src = (source_text or "").lower()
    ungrounded = []
    grounded = 0
    for ln in ref_lines:
        # 取该行最长的英文片段（大概率是标题）做匹配
        cands = re.findall(r"[A-Za-z][A-Za-z0-9 ,\-:]{12,}", ln)
        key = max(cands, key=len).strip() if cands else ""
        if not key:
            continue
        # 取标题前 8 个词做弱匹配
        probe = " ".join(key.split()[:8]).lower()
        if probe and probe in src:
            grounded += 1
        else:
            ungrounded.append(ln[:100])
    total = grounded + len(ungrounded)
    rate = round(grounded / total, 3) if total else 1.0
    style = "数字[N]" if numeric_cites else ("作者—年份" if author_year_cites else "无")
    print(f"--> [Citation] 引用校验: {grounded}/{total} 有据可查 (rate={rate}), "
          f"参考段起于第 {len(draft[:ref_marks[-1].start()].splitlines()) if ref_marks else 0} 行, "
          f"正文引用 {len(set(body_cites))} 处（体例={style}）")
    return {"total_refs": total, "grounded": grounded,
            "hallucinated": len(ungrounded), "rate": rate,
            "ungrounded": ungrounded[:10],
            "body_cites": len(set(body_cites)), "cite_style": style}


def _last_ai_text(messages: list) -> str:
    """取当前轮最后一条 AIMessage 的文本（模型收尾时的最终回复）。"""
    for m in reversed(messages or []):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            c = m.content
            if isinstance(c, list):
                c = "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
            if str(c).strip():
                return str(c)
    return ""


def answer_node(state: AgentState):
    """【节点 · 回答落点】agent 循环终止后，模型最后一条非工具消息就是答案。

    只做两件事：工具全部失败时的兜底提示（保留 detect_tool_failure 判定）+ 存 JSONL。
    （旧版是再调一次 LLM 基于 papers_data 合成回答；现在合成由 agent 循环里的模型自己完成。）
    """
    from ResearcherAgent.memory.conversation_manager import save_conversation_turn

    messages = state.get("messages", [])
    answer = _last_ai_text(messages)
    tool_text = _extract_tool_content(messages)
    tool_failed = detect_tool_failure(messages, tool_text) and not (state.get("paper_pool") or [])
    print(f"--> [Answer] 模型收尾 stop_reason={state.get('stop_reason')!r} tool_failed={tool_failed} len={len(answer)}")

    if not answer:
        answer = ("[联网/检索未成功] 工具未返回有效结果，且模型未给出回复。建议稍后重试或换个提法。"
                  if tool_failed else "（模型未给出回复）")
    session_id = state.get("session_id", "")
    if session_id:
        save_conversation_turn(session_id, "assistant", answer[:3000], {"type": "answer_from_context"})
    return {"final_review": answer, "draft_review": "", "tool_failed": tool_failed}


def content_safety_feedback_node(state: AgentState):
    """安全审核未通过 → 把反馈作为一条消息投回 agent 循环，让模型重答（不再调工具也行）。"""
    fb = state.get("content_safety_feedback", "") or state.get("content_safety_score", {}).get("feedback", "")
    return {"messages": [HumanMessage(
        content="你的上一版回复未通过内容安全审核，请根据以下反馈修改后重新给出完整回复（无需再调用工具）：\n" + fb,
        name="safety_feedback")]}


def _readable_draft(draft: str) -> str:
    """给评审/安全审核读的正文。

    正文已被 MAX_REVIEW_CHARS 约束，正常情况下**全文可读**——这才是逐维度打分的前提
    （旧代码把 6 万字截成前 1.2 万字，评审判「第 3 章缺失」不是判错，是真的看不到）。
    仅当模型严重超预算（> REVIEW_READ_MAX）时才截断，且**显式告知「这是节选、实际共 N 字」**，
    杜绝把「没看到」误判成「没写」。
    """
    n = len(draft or "")
    if n <= REVIEW_READ_MAX:
        return draft
    return (draft[:REVIEW_READ_MAX]
            + f"\n\n…（正文过长，以上为节选；本文实际共 {n} 字，其余章节因长度限制未展示）")


def _norm_title_key(s: str) -> str:
    """节名/字段匹配用归一化（小写去空白标点）。"""
    import unicodedata
    t = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", t)


_POLISH_SECTIONS_PROMPT = """综述「{section}」一节的**已组装句子块**（每个块形如
「块N [引用编号或(无引用)]：内容」）。请润色成**连贯的学术段落**：
- 合并重复表达、补充衔接词、调整语序，使其像人写的综述；
- **不得新增任何事实、数字、方法或论文**；
- **不得删除、改动、新增引用编号 [N]**（无引用的句子保持无引用）；
- 只输出润色后的正文段落，不要任何解释、不要标题。

句子块：
{blocks}"""


def section_budgets(n_sections: int, total: int = MAX_REVIEW_CHARS) -> tuple[int, list[int], int]:
    """2026-10 v1.2.3 用户硬性字数分配（覆盖原比例分配）：
      引言 ~300 / 展望 ~300（共 600）；
      方法节总 2400-3000：方法 3 节 → 每节 900（800-1000 中值）；
                         方法 4 节 → 每节 700（600-750 中值）；
      非 3/4 方法节：按 total 剩余均分兜底（每节 ≥250）。
      n_sections 为大纲总节数（含引言/展望），方法节数 = n_sections - 2
      （不足 3 时退化为全部当方法节处理）。
    """
    intro, concl = 300, 300
    method_n = max(1, n_sections - 2) if n_sections >= 3 else max(1, n_sections)
    if method_n == 3:
        per = 900
    elif method_n == 4:
        per = 700
    else:
        remaining = max(0, total - intro - concl)
        per = max(250, remaining // method_n)
    # b_secs 长度对齐大纲总节数（调用方用 b_secs[i] 按 outline 索引取），per 值按方法节数定
    return intro, [per] * max(1, n_sections), concl


def write_review_node(state: AgentState, llm):
    """【节点 · 撰写综述】根据论文和历史记忆撰写文献综述

    如果是重写（rewrite_count > 0），会携带上一轮评委的反馈意见。
    **总字数受 MAX_REVIEW_CHARS 约束**：预算在生成/组装端就下发到位——
    方法节按「引用篇数」挑选、引言/展望按 300 字合成，超预算不压缩不截断，
    只提示（评审读到全文并被告知实际字数）。
    """
    rewrite_count = state.get("rewrite_count", 0)
    previous_score = state.get("review_score", {})
    print(f"--> [Write] 正在撰写文献综述... (第 {rewrite_count + 1} 次)")

    # 构造重写提示（携带累积反馈，避免在已修正项上回退）
    rewrite_hint = ""
    if rewrite_count > 0 and previous_score:
        sc = previous_score.get("total_score", 0)
        feedback_history = state.get("feedback_history", "").strip()
        fb_block = feedback_history or (previous_score.get("feedback", "") or "")
        rewrite_hint = (
            f"\n\n## ⚠️ 第 {rewrite_count} 次修改要求\n"
            f"上一版得分 {sc}/10，未通过审核。\n\n"
            f"### 历轮评委反馈（累积，请逐条对照，切勿在已改好的点上回退）：\n{fb_block}\n\n"
            f"请针对上述**所有**未解决的问题进行修改，同时保留此前已经改好的内容。"
        )
    safety_fb = (state.get("content_safety_feedback", "") or "").strip()
    if safety_fb:
        rewrite_hint += f"\n\n## 🛡️ 内容安全审核反馈（必须整改）\n{safety_fb}"

    # 如果是文件路径过来的，把 RAG 上下文也注入
    rag_context = state.get("rag_context", "")
    file_section = ""
    if rag_context and not rag_context.startswith("["):
        file_section = f"\n【上传文件参考内容】:\n{rag_context[:4000]}\n"

    # ---- 综述结构：用户指定 > MD 偏好 > 默认 ----
    user_structure = state.get("review_structure", "").strip()
    if user_structure:
        structure_section = (
            f"## 用户指定的结构（请严格遵循）\n"
            f"{user_structure}\n\n"
            f"请按照用户指定的上述章节/模块来组织综述，不要使用默认结构。"
        )
    else:
        structure_section = (
            "## 结构要求\n"
            "按「技能执行规范 · write_review」的「写作阶段规范-综述结构」执行："
            "用户指定 > 记忆档案偏好 > 默认结构（引言 / 方法分类与对比 / 研究展望 / 参考文献）。"
        )

    # 深度阅读素材：Map 出来的结构化卡片（每篇 ≤400 字，不占满上下文）
    paper_cards = state.get("paper_cards", []) or []
    cards_text = "\n\n".join(
        f"【卡片 {i+1}】{c.get('title','')}\n{c.get('card','')}"
        for i, c in enumerate(paper_cards)) if paper_cards else ""

    # 可引用材料：paper_pool（去重论文池）渲染成文本；分节写作时该节归属的论文排在前面
    pool = state.get("paper_pool", []) or []
    # 证据约束（忠实度升级）：写作**只引用已精读（有卡片）的论文**——
    # 只见过摘要的论文一律不进材料，杜绝「没读过也敢引」。
    materials_all = render_section_candidates(pool, state.get("section_candidates") or {},
                                              "", max_chars=WRITE_SECTION_BUDGET,
                                              card_only=True, cards=paper_cards)

    # ★ 把**确定性证据明细**（哪句/引了哪篇/余弦多少 + 该论文摘要）交给 writer。
    #   此前它只存在于 judge 的局部变量里，算完即弃 → writer 从未见过，
    #   只能凭 LLM 写的 feedback 盲改，实测导致重写越多忠实度越低（judgev3c 5.2→4.7）。
    #   ⚠️ 这段必须在 `pool` 定义**之后**——否则 build_cite_map_from_pool 拿不到池。
    if rewrite_count > 0:
        try:
            from ResearcherAgent.core.evidence_check import (build_cite_map_from_pool, render_writer_evidence)
            _ev = render_writer_evidence(
                state.get("evidence_report") or {},
                build_cite_map_from_pool(state.get("draft_review") or "", pool))
            if _ev:
                rewrite_hint += "\n\n" + _ev
                print(f"--> [Write] 🔍 已注入确定性证据明细 {len(_ev)} 字（含对应论文摘要）")
            else:
                print("--> [Write] ℹ️ 无确定性证据问题可注入（或 evidence_report 为空）")
        except Exception as _e:
            print(f"--> [Write] ⚠️ 证据明细注入失败（不影响写作）: {str(_e)[:80]}")

    # 本次写作的字数预算：用户明确提过篇幅就以用户为准，否则用系统默认
    review_budget, budget_src = effective_review_budget(state.get("user_constraints") or {})
    # 用户约束：只放**用户原话里确有其事**的那些（parse 节点已做接地核验）
    uc_block = state.get("requirements", "").strip()

    # 渐进式披露 L2：写作规范（结构/规则）来自 write_review/SKILL.md，不再写死在节点里
    _wskill_block = build_skill_instructions(["write_review"])
    _wskill_text = f"{_wskill_block}\n\n" if _wskill_block else ""

    write_system = _with_safety(
        "⚠️ 上面「你的身份与记忆」中记录了用户的长期偏好（引用格式、语言、结构要求等），"
        "请在写作时严格遵守这些偏好，它们与本次约束具有同等效力。\n\n"
        "你是一个顶级学术研究助手。请撰写一篇专业、逻辑严密的文献综述。\n\n"
        + _wskill_text
        + "## 写作要求\n"
        "- 语言专业但可读性强；客观引用，避免主观断言\n"
        "- **只引用「可引用材料」中明确列出的论文，严禁凭空编造或引用未列出的文献**\n"
        "- **材料按证据约束分两级（2026-10 决策，必须区分使用）**：\n"
        "  · **精读卡论文**（条目带【精读卡片】，卡片=原文证据段落）：可编号引用，可适当扩展描述其方法与实验数据\n"
        "  · **未精读论文**（条目带【未精读·仅摘要】，仅摘要级信息）：可编号引用，但统一按「XX等人提出了XXX，"
        "通过XXX（方法）解决XXX」一句话提及；**严禁展开其实验数据、数字结论或细节**，只可用其标题/作者/方法提示里的信息\n"
        "- **展望与未来挑战章节：只能基于精读卡论文展开**，未精读论文不得作为展望依据\n"
        "- **严禁编造实验数据、结论、作者**；材料里没有的细节不要写\n"
        "- **正文引用直接用材料中给出的编号**（形如 `[12]`），不要自行重新编号、"
        "也不要用作者-年份括注——参考文献表由系统按这些编号自动生成，编号对不上就无法生成\n"
        "- **不要写参考文献列表**（也不要写「参考文献占位说明」）：系统会在全文结束后"
        "自动追加一份基于真实检索记录的参考表\n"
        f"- ⚠️ **篇幅硬约束**：全文建议 {review_budget} 字（{budget_src}），"
        f"**最大不超过 {MAX_REVIEW_CHARS_HARD} 字**。论断写精炼、删掉套话与重复论述；"
        f"有精读卡论文支撑的章节必须完整写出。\n"
        "- ⚠️ **无素材节拒绝生成（v1.2.18 硬规则）**：若某节【本节卡片】为空、且可引用材料"
        "中也没有任何该节论文（精读卡片 0 篇），**直接输出「本节因缺乏可精读文献支撑，"
        "未生成内容」**，**严禁凭自身知识或池内摘要硬写该节**——宁缺毋滥优于无依据编造"
        "（编造 = 幻觉 + 无锚引用，评审会扣分）。\n"
        "{rewrite_hint}"
    )

    def _section_chain(extra_structure: str):
        # ⚠️ system 消息必须传已构造的 SystemMessage 对象：write_system 经 _with_memory 注入的
        #    记忆文本可能含裸花括号（历史坑记录/JSON/公式片段），走 f-string 模板解析会报
        #    "Nested replacement fields are not allowed"（实测 E2E v1.2 崩溃点）。
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=write_system),
            ("user",
             "综述主题: {topic}\n用户约束: {requirements}\n\n"
             "{extra_structure}\n\n"
             "【历史知识记忆】:\n{historical_knowledge}\n\n"
             "【可引用材料】:\n{materials}\n\n"
             "【本节相关论文卡片（精读提炼）】:\n{deep_section}\n\n"
             "【本次要写的内容】:\n{task}\n\n"
             "额外信息（自行判断是否有用，无用则忽视）:\n{file_section}"),
        ])

    # ════════════════════════════════════════════════════════════════
    # ★ 五步流水线拼接路径（方案 v1.2 第 5 步）：
    #   字段化卡片 →（MQ/ThreadPool）句式 JSON 块 → 按节拼接 + 每节衔接润色 → 预算挑选。
    #   拼接成功则跳过下方逐节 LLM 写作；失败（无块/异常）自动回退现有分节写作。
    # ════════════════════════════════════════════════════════════════
    def _match_outline_section(bs: str) -> str:
        bs = (bs or "").strip()
        for o in outline:
            name = (o.get("section") or "").strip()
            if not name:
                continue
            if bs and (bs in name or name in bs or _norm_title_key(bs) == _norm_title_key(name)):
                return name
        if not bs or any(k in bs for k in ("引言", "背景")):
            for o in outline:
                if "引言" in (o.get("section") or ""):
                    return o["section"]
        if any(k in bs for k in ("展望", "结论", "总结", "未来")):
            for o in outline:
                if any(k in (o.get("section") or "") for k in ("展望", "结论", "总结", "未来")):
                    return o["section"]
        return (outline[0].get("section", "") if outline else "引言")

    def _card_num(c: dict) -> int | None:
        """卡片 → 池内全局编号（引用编号映射：JSON 用标题 → 组装期统一映射 → [N]）。"""
        try:
            from ResearcherAgent.tools.agent_tools import global_index, _norm_card_title, paper_key
            gidx = global_index(pool)
            pid = c.get("id", "")
            for p in pool:
                if pid and pid in {str(p.get("id", "")), str(p.get("paper_id", "")),
                                   str(p.get("arxiv_id", "")), str(p.get("pmcid", "") or "")}:
                    n = gidx.get(paper_key(p))
                    if n:
                        return n
            tn = _norm_card_title(c.get("title", ""))
            if tn:
                for p in pool:
                    if _norm_card_title(p.get("title", "")) == tn:
                        n = gidx.get(paper_key(p))
                        if n:
                            return n
        except Exception:
            pass
        return None

    def _collect_or_submit_json_blocks():
        """收集 JSON 块：优先 read_sections 流式会话（MQ 增量，卡片边制边送已开跑），
        补提交不在会话里的其余卡片（read_papers 补充等）；全失败返回 {}。"""
        from ResearcherAgent.async_pipeline import (
            collect_json_session, submit_cards_for_json, MqUnavailable)
        blocks_map: dict = {}
        done: set = set()
        sid = state.get("json_build_session_id") or ""
        if sid:
            try:
                blocks_map = collect_json_session(sid) or {}
                done = set(blocks_map.keys())
            except MqUnavailable as e:
                print(f"--> [Write] ⚠️ 流式会话收集失败（{str(e)[:60]}），降级重提交")
                blocks_map = {}
            except Exception as e:
                print(f"--> [Write] ⚠️ 流式会话收集异常（{str(e)[:60]}），降级重提交")
                blocks_map = {}
        rest = [c for c in paper_cards if c.get("id") not in done]
        if rest:
            try:
                more = submit_cards_for_json(rest, llm, outline=outline) or {}
                for k, v in more.items():
                    blocks_map.setdefault(k, v)
            except Exception as e:
                print(f"--> [Write] ⚠️ 补提交 JSON 块异常: {str(e)[:80]}")
        return blocks_map

    def _sec_for_block(b: dict, c: dict) -> str:
        """代码确定性归节（v1.2.10：论文按大纲节检索，归属由代码标注，不再要 LLM 选节）：
        - method 块 → 卡片所属节（该节必为大纲方法节）；不在大纲 → 择优方法节；
        - intro 块 → 大纲引言节；conclusion 块 → 大纲展望节。"""
        def _sec_type(name: str) -> str:
            if any(k in name for k in ("引言", "背景", "introduction", "intro")):
                return "引言"
            if any(k in name for k in ("展望", "结论", "总结", "未来", "outlook", "conclusion")):
                return "展望"
            return "方法"

        def _pick_method_sec() -> str:
            for o in outline:
                if (o.get("type") or _sec_type(o.get("section", ""))) == "方法":
                    return o.get("section", "")
            return ""

        bt = str(b.get("block_type") or "").strip().lower()
        secs = {o.get("section", "") for o in outline}
        if bt == "method":
            sec = (c or {}).get("section") or ""
            if sec in secs:
                return sec
            return _pick_method_sec()
        if bt == "intro":
            for o in outline:
                if any(k in o.get("section", "") for k in ("引言", "背景")):
                    return o.get("section", "")
            return (outline[0].get("section", "") if outline else "")
        if bt == "conclusion":
            for o in outline:
                if any(k in o.get("section", "") for k in ("展望", "结论", "总结", "未来")):
                    return o.get("section", "")
            return ""
        # block_type 缺失兜底：按卡片主字段推断（method 优先）
        fields = (c or {}).get("fields") or {}
        if any(str(fields.get(f) or "").strip() not in ("", "(none)")
               for f in ("method", "experiment")):
            sec = (c or {}).get("section") or ""
            if sec in secs:
                return sec
            return _pick_method_sec()
        for o in outline:
            if any(k in o.get("section", "") for k in ("展望", "结论", "总结", "未来")):
                return o.get("section", "")
        for o in outline:
            if any(k in o.get("section", "") for k in ("引言", "背景")):
                return o.get("section", "")
        return ""

    def _assemble_sections():
        """返回 (parts, budgets, labels) 或 None（无块→回退现有写作）。"""
        try:
            # 优先收集流式会话（read_sections 已边制卡边送 MQ），补提交其余卡片；
            # outline 不再注入 JSON 生成 prompt（归节已代码化，见 _sec_for_block）。
            blocks_map = _collect_or_submit_json_blocks()
        except Exception as e:
            print(f"--> [Write] ⚠️ JSON 块生成异常，回退逐节写作: {str(e)[:80]}")
            return None
        flat = [b for lst in blocks_map.values() for b in lst]
        if not flat:
            print("--> [Write] ℹ️ 无句式 JSON 块（卡片为空/生成失败）→ 回退逐节写作")
            return None
        print(f"--> [Write] ★ 拼接路径：{len(flat)} 个内容块（{len(blocks_map)} 篇卡片）")

        # 块按大纲节分组（块: (content, num)；归节代码确定性，block_type + 卡片 section）
        import re as _re
        by_sec: dict = {}
        for c in paper_cards:
            for b in blocks_map.get(c.get("id", ""), []):
                sec_name = _sec_for_block(b, c)
                if not sec_name:
                    continue
                # v1.2.11 句级锚点：content 已带 LLM 标注的 [N]（制卡分配编号）。
                # 组装期只做**校验纠正**：块内所有 [N] 纠正为该卡片编号（防串号）；
                # 无编号的块 → 块尾补 [card.num]（兜底，至少块级锚点）。
                content = str(b.get("content") or "").strip()
                cnum = _card_num(c)
                if cnum:
                    if _re.search(r"\[\d{1,3}\]", content):
                        content = _re.sub(r"\[\d{1,3}\]", f"[{cnum}]", content)
                    else:
                        content = content.rstrip("。") + f"。[{cnum}]"
                by_sec.setdefault(sec_name, []).append((content, cnum))
        if not by_sec:
            return None

        _polish = ChatPromptTemplate.from_messages([
            ("system", "你是学术编辑。只做衔接润色：合并重复、补衔接词、调整语序，"
                       "**不得新增事实/数字/方法/论文，不得改动或新增引用编号 [N]**。"
                       "直接输出润色后的正文段落，不要任何解释、不要标题。"),
            ("user", _POLISH_SECTIONS_PROMPT),
        ]) | llm
        # 引言/展望节 = 带编号键值对的 **LLM 概括**（v1.2.13）：
        # 每条素材 = 某篇论文 background/conclusion 字段的忠实转述（句尾带 [N]，
        # 即「内容↔编号」键值对）。把**内容与引用编号都交给 LLM** 概括组织——
        # 防错位的关键不是禁止改写，而是强制「编号跟随内容」：A 论文的编号
        # 不得挂在 B 论文内容的论断上。
        _synth_lead = ChatPromptTemplate.from_messages([
            ("system", "你是学术综述的章节概括者。下面是多篇论文提取出的【引言】或【展望】素材块，"
                       "每条素材句尾的 [N] 是该素材来源论文的引用编号（即「内容↔编号」键值对）。"
                       "请把它们**概括合成一段连贯的章节正文**：围绕共同主题组织脉络、提炼共性、"
                       "删除完全重复项、在条与条之间补衔接词（如「此外」「与此同时」）。\n"
                       "**硬约束（防编号错位）**：\n"
                       "① 每个论断句尾的 [N] 必须紧跟**真正支撑该论断的那条素材**——"
                       "A 论文的编号不得挂在 B 论文内容的论断上（这是错位红线，评审会逐句核对）；\n"
                       "② 概括时可以重组/合并多条素材，但合并后若论断涉及多篇论文，"
                       "各编号须分别标注在各自论断句尾，不得只保留一个编号；\n"
                       "③ 不得新增素材之外的论断/事实/论文；不得删除或改写任何素材的 [N]；\n"
                       "④ 不要给没有素材支撑的概括句硬挂编号。\n"
                       "直接输出正文段落，不要标题、不要解释。"),
            ("user", "【章节】{section}（预算 {budget} 字以内）\n【素材块】\n{blocks}"),
        ]) | llm

        def _is_lead(sec: str) -> bool:
            return any(k in sec for k in ("引言", "背景", "展望", "结论", "总结", "未来", "挑战"))

        n_secs = len(outline)
        b_intro, b_secs, b_concl = section_budgets(n_secs, total=review_budget)
        parts, budgets, labels = [], [], []
        for o in outline:
            sec = o.get("section", "")
            bdgs = b_intro if "引言" in sec else (b_concl if any(
                k in sec for k in ("展望", "结论", "总结", "未来")) else b_secs[0])
            items = by_sec.get(sec, [])
            if not items:
                continue  # 该节无块：由调用方按混合模式只补写缺节（见下方判定）
            if _is_lead(sec):
                # 引言/展望：素材块按输入预算挑选（累计 ≤LEAD_BLOCK_INPUT_BUDGET），
                # 合并后一次概括合成（输出由 budget=bdgs≈300 字约束）；不是逐块拼接润色
                picked_in, in_total = [], 0
                for content, num in items:
                    if in_total + len(content) > LEAD_BLOCK_INPUT_BUDGET:
                        break
                    in_total += len(content)
                    picked_in.append((content, num))
                if not picked_in:
                    continue
                block_text = "\n".join(
                    f"素材{i + 1}：{content}" for i, (content, num) in enumerate(picked_in))
                try:
                    resp = _synth_lead.invoke({"section": sec, "budget": bdgs, "blocks": block_text})
                    out = resp.content if hasattr(resp, "content") else str(resp)
                    out = (out or "").strip()
                except Exception as e:
                    print(f"    ✗ {sec} 概括合成失败，用拼接原文: {str(e)[:80]}")
                    out = "\n\n".join(content for content, _ in items)
                if not out:
                    continue
                parts.append(out)
                budgets.append(bdgs)
                labels.append(sec)
                continue
            # 方法节：预算挑选（v1.2.3 超预算 = 减少引用篇数，不是压缩/截断）——
            # 按该节**输出预算** bdgs 累计选择句式片段，放不下的篇目本轮就不引用
            # （引用编号映射只覆盖被选中的块，未选中篇自然不进正文与参考表）。
            picked, total = [], 0
            for content, num in items:
                if total + len(content) > bdgs:
                    break
                total += len(content)
                picked.append((content, num))
            if not picked:
                continue
            block_text = "\n".join(
                f"块{i + 1}：{content}" for i, (content, num) in enumerate(picked))
            try:
                resp = _polish.invoke({"section": sec, "blocks": block_text})
                out = resp.content if hasattr(resp, "content") else str(resp)
                out = (out or "").strip()
            except Exception as e:
                print(f"    ✗ {sec} 润色失败，用拼接原文: {str(e)[:80]}")
                out = "\n".join(content for content, _ in picked)
            if out:
                parts.append(out)
                budgets.append(b_intro if "引言" in sec else (b_concl if any(
                    k in sec for k in ("展望", "结论", "总结", "未来")) else b_secs[0]))
                labels.append(sec)
        if not parts:
            return None
        # 混合模式（2026-10 v1.2.1）：**不再全有或全无**——有块的节走拼接组装，
        # 缺块的节（missing）由调用方只对这些节逐节写作，而不是整体回退逐节写作。
        covered = set(labels)
        missing = [o.get("section", "") for o in outline if o.get("section", "") not in covered]
        if missing:
            print(f"--> [Write] ℹ️ 拼接覆盖 {len(covered)}/{len(outline)} 节；"
                  f"缺 {missing} → 仅对这些节逐节写作（其余节用拼接块）")
        # 证据锚映射：编号 → 该篇被选用块的 evidence 锚点（空列表=无锚，评审标 low_conf）
        anchor_map: dict = {}
        for c in paper_cards:
            num = _card_num(c)
            if num is None:
                continue
            evs = []
            for b in blocks_map.get(c.get("id", ""), []):
                evs.extend(b.get("evidence") or [])
            anchor_map[num] = list(dict.fromkeys(evs))
        return parts, budgets, labels, anchor_map, missing

    # ---- 分节写作（每节独立一次 LLM 调用；每节只注入**该节相关**的卡片，避免长上下文硬塞） ----
    outline = state.get("review_outline", []) or []
    topic = state.get("topic", "")
    # 用户约束为空是正常的（大多数用户只给一句话）——这时不要伪造一段"约束"塞给模型
    reqs = uc_block or "（用户未提出额外要求）"
    hist = state.get("historical_knowledge", "")

    # 卡片按「归属章节」分组（coverage_judge 的 selected 已带 section；无则全给引言/结论）
    cards_by_section: dict = {}
    for c in paper_cards:
        cards_by_section.setdefault(c.get("section", ""), []).append(c)

    def _cards_for(section: str) -> str:
        cards = cards_by_section.get(section, [])
        return "\n\n".join(f"【卡片】{c.get('title','')}\n{c.get('card','')}" for c in cards)

    def _gen(task: str, deep_section: str = "", extra_structure: str = "", section: str = "") -> str:
        chain = _section_chain(extra_structure) | llm
        resp = chain.invoke({
            "topic": topic, "requirements": reqs,
            "historical_knowledge": hist,
            "materials": (render_section_candidates(pool, state.get("section_candidates") or {},
                                                    section, max_chars=WRITE_SECTION_BUDGET,
                                                    card_only=True, cards=paper_cards)
                          if section else materials_all),
            "deep_section": deep_section or "（本节暂无专属卡片，请依据上方材料）",
            "task": task, "rewrite_hint": rewrite_hint,
            "file_section": file_section, "extra_structure": extra_structure,
        })
        return resp.content if hasattr(resp, "content") else str(resp)

    # ★ 优先走拼接路径（混合模式 v1.2.1）：有块的节直接组装；缺块的节只补写缺失节，
    #   不再"全有或全无"整体回退逐节写作。
    _assembled = None
    _anchor_map: dict = {}
    if paper_cards and outline:
        _assembled = _assemble_sections()

    # ---- 单节写作（拼接缺块节 / 全量回退 共用）----
    # v1.2.18：无素材节拒绝生成——方法节必须有该节精读卡片；引言/展望须存在任何
    # 精读卡片（素材来自卡片的背景/结论字段）。不满足 → 输出占位说明而非硬写
    # （防 LLM 凭知识编造 → 无引用/幻觉/忠实度受损）。
    def _section_has_material(sec: str) -> bool:
        if any(k in sec for k in ("引言", "背景", "展望", "结论", "总结", "未来", "挑战")):
            return bool(paper_cards)
        return bool(cards_by_section.get(sec))

    _NO_MATERIAL_PLACEHOLDER = (
        "（本节未生成：未精读到 ≥{min} 篇可读全文的支撑文献（精读卡片为 0），"
        "按规则拒绝生成以避免无依据写作。可补充可读全文的论文后重新生成。）").format(
            min=MIN_PAPERS_PER_SECTION)

    # 大纲自带的引言/展望节用对应预算，其余用正文节预算
    def _write_section(sec: str, focus: str) -> tuple:
        if "引言" in sec:
            budget, task = b_intro, ("请撰写综述的【引言】章节（研究背景、问题定义、综述范围与全文路线图）。聚焦：{focus}。"
                                     "只引用「可引用材料」和「本节卡片」中的论文，每个论断标注引用。"
                                     "**本节控制在 {budget} 字以内**。")
            deep, extra = cards_text[:9000], structure_section
        elif any(k in sec for k in ("展望", "结论", "总结", "未来")):
            budget, task = b_concl, ("请撰写综述的【展望与总结】章节（当前不足与未来方向，**只能基于精读卡论文展开**，"
                                     "未精读论文不得作为展望依据）。聚焦：{focus}。"
                                     "**本节控制在 {budget} 字以内**。")
            deep, extra = cards_text[:3000], None
        else:
            budget, task = b_secs[idx_of.get(sec, 0)], ("请撰写综述的【{sec}】章节。聚焦：{focus}。"
                                                        "只引用「可引用材料」和「本节卡片」中的论文，每个论断标注引用。"
                                                        "**本节控制在 {budget} 字以内**。")
            deep, extra = _cards_for(sec), None
        return _gen(task.format(sec=sec, focus=focus, budget=budget)
                    + (" 若本节没有任何可引用的精读卡片，直接输出「本节因缺乏可精读文献支撑，"
                       "未生成内容」，不要凭自身知识或池内摘要编造本节内容。"),
                    deep_section=deep, section=sec, extra_structure=extra or ""), budget

    if _assembled is not None:
        parts, budgets, labels, _anchor_map, missing = _assembled
        if missing:
            # 混合：只对缺块节逐节写作，再按大纲顺序合并
            idx_of = {o.get("section", ""): i for i, o in enumerate(outline)}
            n_sec = len(outline)
            b_intro, b_secs, b_concl = section_budgets(n_sec, total=review_budget)
            fill = {}
            for o in outline:
                sec = o.get("section", "")
                if sec in missing:
                    # v1.2.18：无素材节拒绝生成（输出占位说明，不补写硬编）
                    if not _section_has_material(sec):
                        print(f"--> [Write] ⛔ {sec} 无精读卡片素材 → 拒绝生成（占位说明）")
                        fill[sec] = (_NO_MATERIAL_PLACEHOLDER, 0)
                        continue
                    t, b = _write_section(sec, o.get("focus", ""))
                    fill[sec] = (t, b)
            new_parts, new_budgets, new_labels = [], [], []
            for o in outline:
                sec = o.get("section", "")
                if sec in fill:
                    new_parts.append(fill[sec][0]); new_budgets.append(fill[sec][1]); new_labels.append(sec)
                elif sec in labels:
                    i = labels.index(sec)
                    new_parts.append(parts[i]); new_budgets.append(budgets[i]); new_labels.append(sec)
            parts, budgets, labels = new_parts, new_budgets, new_labels
            print(f"--> [Write] 混合拼接完成：{len(labels)} 节（拼接 {len(labels) - len(missing)} + 补写 {len(missing)}）")
    elif outline and len(outline) >= 2:
        # 全部回退逐节写作（无任何拼接块）
        # 字数预算按节下发：超预算的节写完再单独压缩，而不是整篇截断
        names = [o.get("section", "") for o in outline]
        has_intro = any("引言" in n for n in names)
        has_outlook = any(any(k in n for k in ("展望", "结论", "总结", "未来")) for n in names)
        n_sec = len(outline)
        b_intro, b_secs, b_concl = section_budgets(n_sec, total=review_budget)
        idx_of = {o.get("section", ""): i for i, o in enumerate(outline)}
        print(f"--> [Write] 分节写作 {n_sec} 节（总预算 {review_budget} 字，{budget_src}："
              f"引言 {b_intro} + 每节 {b_secs[0]} + 结论 {b_concl}"
              f"{'' if has_intro else '；大纲未含引言，自动补引言'}") 
        budgets, parts, labels = [], [], []
        if not has_intro:
            parts.append(_gen(
                f"请为整篇综述写一段**引言**（研究背景、问题定义、综述目的）与一段**摘要**。"
                f"**本节控制在 {b_intro} 字以内**。不要写参考文献列表。",
                deep_section=cards_text[:9000], extra_structure=structure_section))
            budgets.append(b_intro)
            labels.append("引言")
        for i, o in enumerate(outline):
            sec = o.get("section", "")
            focus = o.get("focus", "")
            # v1.2.18：无素材节拒绝生成（占位说明，不补写硬编）
            if not _section_has_material(sec):
                print(f"--> [Write] ⛔ {sec} 无精读卡片素材 → 拒绝生成（占位说明）")
                parts.append(_NO_MATERIAL_PLACEHOLDER)
                budgets.append(0)
                labels.append(sec)
                continue
            t, b = _write_section(sec, focus)
            parts.append(t)
            budgets.append(b)
            labels.append(sec)
        if not has_outlook:
            parts.append(_gen(
                f"请写一段简短的**结论与展望**。**本节控制在 {b_concl} 字以内**。",
                deep_section=cards_text[:3000]))
            budgets.append(b_concl)
            labels.append("结论与展望")
    else:
        # 无大纲（或只有 1 节）：一次成文，同样受总字数约束
        print(f"--> [Write] 整篇写作（总预算 {review_budget} 字，{budget_src}）...")
        labels = ["全文"]
        budgets = [review_budget]
        parts = [_gen(
            f"请撰写一篇完整的文献综述，包含摘要、引言、方法分类对比、展望。"
            f"**全文控制在 {review_budget} 字以内**，宁可每部分写短也不能省略章节。"
            f"不要写参考文献列表。",
            deep_section=cards_text[:8000], extra_structure=structure_section)]

    # ---- 超预算处理（v1.2.3 用户硬性规定：不压缩、不截断）----
    # 组装路径：方法节已在预算挑选阶段按「引用篇数」控制（放不下的片段不引用），
    # 引言/展望由合成输出预算（300 字）约束——正常不会超。
    # 回退/补写路径（LLM 自由写作）若仍超节预算：保持原文，只提示，评审读到全文并被告知实际字数。
    over = [i for i, p in enumerate(parts)
            if budgets[i] and len(p) > budgets[i] * SECTION_OVERFLOW_TOLERANCE]
    if over:
        print(f"--> [Write] ⚠️ {len(over)} 节超预算（不压缩/不截断，保持原文，评审被告知实际字数）: "
              f"{[(labels[i], len(parts[i]), budgets[i]) for i in over]}")

    # v1.2.9：按大纲节拼稿时给每节加 `## 节标题`（labels 与 parts 按大纲顺序一一对应）——
    # ① 提升综述可读性；② judge 按一级 ## 切分才能真正按节并行评审（此前纯段落只切出 1 节，
    #    局部评审退化为一次调用且只覆盖前 2800 字）。
    if labels and len(labels) == len(parts):
        _seg = []
        for _p, _lb in zip(parts, labels):
            if not _p or not _p.strip():
                continue
            _seg.append(f"## {_lb}\n\n{_p.strip()}")
        draft = "\n\n".join(_seg)
    else:
        draft = "\n\n".join(p for p in parts if p)
    if len(draft) > review_budget:
        # 超建议预算（3000）：只提示，不截断/不压缩，评审读到全文并被告知实际字数
        print(f"--> [Write] ⚠️ 正文 {len(draft)} 字（预算 {review_budget}）——"
              f"不压缩/不截断，评审会读到全文并被告知实际字数")
    else:
        print(f"--> [Write] ✅ 正文 {len(draft)} 字 / 预算 {review_budget} 字")
    # 硬上限（4000，用户"最大允许"）：仍超则只提示（用户字数分配 3节≈3300/4节≈3400，
    # 正常不会触发）；评审读到全文并被告知实际字数，交用户裁量。
    if len(draft) > MAX_REVIEW_CHARS_HARD:
        print(f"--> [Write] 🚫 超硬上限 {MAX_REVIEW_CHARS_HARD} 字 → 保持全文 {len(draft)} 字"
              f"（不压缩/不截断），评审与用户被告知实际字数")

    # ---- 参考文献：代码强制追加，用池内真实元数据拼装（不靠 LLM，结构上不可能编造） ----
    # 引用体例：用户点名了就用用户的（已有接地核验），否则按中文学术惯例用 GB/T 7714
    _style = ((state.get("user_constraints") or {}).get("citation_style") or "GB/T 7714")
    if _style.lower() in ("numeric", "author-year"):
        _style = "GB/T 7714"      # 这两种是「编号/作者-年份」的风格描述，不是著录标准
    draft, ref_report = build_references(draft, pool, style=_style)
    if ref_report.get("refs"):
        print(f"--> [Write] ✅ 追加参考文献 {ref_report['refs']} 条"
              f"（体例 {ref_report.get('style')}，{ref_report.get('ref_chars', 0):,} 字）"
              f" → 全稿 {len(draft):,} 字")
    else:
        print(f"--> [Write] ⚠️ 未能生成参考文献（正文没有 [N] 编号引用？"
              f"池内 {ref_report.get('pool', 0)} 篇）")

    # ---- 独立修订说明（revision_plan）：重写轮次用独立调用生成可执行的修改计划 ----
    revision_plan = ""
    if rewrite_count > 0 and previous_score:
        _issues_txt = "\n".join(
            f"- [{i.get('type')}] {i.get('claim','')[:80]} → {i.get('fix','')[:120]}"
            for i in (state.get("judge_issues") or [])[:12]) or "（无结构化 issues）"
        try:
            _rp_resp = llm.invoke(
                f"你是学术编辑。上一轮综述得分 {previous_score.get('total_score')}/10。\n"
                f"评委 issues（claim 级缺陷清单）：\n{_issues_txt}\n"
                f"评委 feedback：{(feedback_history or (previous_score.get('feedback') or ''))[:800]}\n\n"
                "请输出 300-500 字的中文修订说明：逐条列出本轮要改哪些内容（具体到段落/论断），"
                "以及为什么这样改（与证据、评委意见的对应关系）。只输出修订说明本身，不要其他文字。")
            revision_plan = _rp_resp.content if hasattr(_rp_resp, "content") else str(_rp_resp)
            revision_plan = revision_plan.strip()
            print(f"--> [Write] ✅ 独立生成修订说明 {len(revision_plan)} 字")
        except Exception as e:
            print(f"--> [Write] ⚠️ 修订说明生成失败: {str(e)[:80]}")

    return {"draft_review": draft, "reference_report": ref_report,
            "revision_plan": revision_plan,
            "review_block_anchors": _anchor_map}


def judge_review_node(state: AgentState, judge_llm):
    """【节点 · 评委打分 v3】分节评审 + 确定性证据信号

    针对「单模型一次评审 10000 字 × 8 维注意力分散」的改造（调研自
    AI Agents-as-Judge / DeepSurvey / DeepReviewer 的分节与确定性校验思路）：
      · 局部评审：正文按一级节（##）切分，每节一次评审（内容深度/逻辑严谨性/
        证据忠实度/覆盖度），ThreadPool 并行，输入从 10000 字降到每节约 1500 字；
      · 全局评审：结构/规范/即时约束/记忆偏好/篇幅一次评完（只喂节标题+统计+首尾抽样）；
      · 确定性证据信号：引用精确映射 + bge-m3 余弦，先于 LLM 标出低支持/无锚句子，
        注入局部评审作为硬信号（judge 从严扣分），不把证据支持度完全交给 LLM 裁量。
    从严打分：阈值由 JUDGE_PASS_THRESHOLD（默认 8.5）决定，宁严勿宽。
    """
    print("--> [Judge] 分节评审（局部×N 并行 + 全局×1 + 确定性证据信号）...")
    draft = state.get("draft_review", "")
    # ── 写后确定性约束：正文 [N] 必须属于「池内真实论文」；池外编号=幻觉引用，删除。
    # 2026-10 语义修正：池内未精读论文（仅摘要级）是**正常综述引用**（弱证据），
    # 不再移除——由确定性证据检查标记为 weak_evidence 进低置信清单交用户审核；
    # 只有引用池外不存在的编号（幻觉）才确定性删除。
    try:
        from ResearcherAgent.tools.agent_tools import global_index, paper_key
        from ResearcherAgent.core.evidence_check import enforce_citation_domain
        _pool = state.get("paper_pool") or []
        _gidx = global_index(_pool)
        _pool_nums = {_gidx.get(paper_key(p)) for p in (_pool or []) if isinstance(p, dict)}
        _pool_nums.discard(None)
        if _pool_nums:
            draft, _removed = enforce_citation_domain(draft, _pool_nums)
            if _removed:
                print(f"--> [Judge] ✂️ 确定性引用约束：移除正文 {len(_removed)} 个**池外幻觉**编号 {_removed}"
                      f"（池内未精读引用保留，作为摘要级弱证据标记）")
    except Exception as _ex:
        print(f"--> [Judge] ⚠️ 引用约束 enforce 失败（跳过）: {str(_ex)[:80]}")
    topic = state["topic"]
    requirements = state.get("requirements", "") or "无额外约束"

    if not draft or len(draft.strip()) < 50:
        return {
            "review_score": {
                "total_score": 0.0,
                "passed": False,
                "feedback": "综述内容过短或为空，请重新撰写完整综述。",
            },
        }

    # ── 切分正文（参考文献段之前，按一级 ## 节）──
    _ref_marks = list(re.finditer(r"(?m)^\s*#{0,3}\s*(?:参考文献|References)", draft))
    body = draft[:_ref_marks[-1].start()] if _ref_marks else draft
    body_len = len(body.strip())

    _sections = []
    _cur_t, _cur = None, []
    for _ln in body.split("\n"):
        _m = re.match(r"^##\s+(.+?)\s*$", _ln)
        if _m:
            if _cur_t:
                _sections.append({"title": _cur_t.strip(), "text": "\n".join(_cur).strip()})
            _cur_t, _cur = _m.group(1).strip(), []
        else:
            _cur.append(_ln)
    if _cur_t:
        _sections.append({"title": _cur_t.strip(), "text": "\n".join(_cur).strip()})
    _sections = [s for s in _sections if s["text"]]
    if not _sections:
        _sections = [{"title": "全文", "text": body.strip()}]
    print(f"--> [Judge] 切分 {len(_sections)} 节 | 正文 {body_len} 字")

    # ── 引用精确映射 + 确定性证据检查（bge-m3 余弦 + 锚点存在性）──
    # 口径：基于 paper_pool 的稳定全局编号反查（正文 71 个编号 → 71 篇池论文全可定位），
    # 不再用精读卡片（卡片只覆盖 13/71 → 会把 58 篇非卡片引用误报为 no_anchor）。
    from ResearcherAgent.core.evidence_check import (build_cite_map_from_pool, check_section_deterministic,
                                summarize_findings)
    _pool = state.get("paper_pool") or []
    cite_map = build_cite_map_from_pool(draft, _pool)
    _cards = state.get("paper_cards") or []
    # ★ 证据卡挂到 cite_map：NLI/embedding 的判定证据从「摘要」升级为「卡片」
    #   （卡片含原文证据段落摘录 → 论断细节才能被 NLI 核对，忠实度才判得出来）
    _card_by_title = {}
    for _c in (_cards or []):
        _k = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(_c.get("title") or "").lower())
        if _k:
            _card_by_title[_k] = _c.get("card", "")
    _n_card_hit = 0
    for _n, _p in cite_map.items():
        _k = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(_p.get("title") or "").lower())
        if _k in _card_by_title and _card_by_title[_k]:
            _p["card"] = _card_by_title[_k]
            _n_card_hit += 1
    print(f"--> [Judge] 引用精确映射 {len(cite_map)} 条（pool {len(_pool)} 篇）| 证据卡片 {len(_cards)} 张"
          f" | 卡片挂接 {_n_card_hit} 篇（判定证据=卡片含原文摘录）")
    _embed_fn = None
    _cos_cache = {}
    try:
        from ResearcherAgent.memory.memory_classifier import make_embed_fn
        _embed_fn = make_embed_fn()
    except Exception as e:
        print(f"--> [Judge] ⚠️ embedding 初始化失败（确定性检查降级跳过）: {str(e)[:60]}")
    det_reports = {}
    # ★ 同时保留**原始 findings**（含逐句 sim 值与锚点编号）落进 state。
    #   为什么：`summarize_findings` 压缩后的文本只够给 judge 看；writer 要**改对**，
    #   需要知道"哪一句、引了哪个 [N]、余弦多少"。此前 det_reports 是 judge 的局部变量，
    #   算完即弃——writer 从头到尾看不到任何确定性证据，只能凭 LLM 写的 feedback 盲改，
    #   实测导致重写越多忠实度越低（judgev3c：5.2→4.7 六轮净跌）。
    det_raw: dict[str, list] = {}
    if _embed_fn is not None:
        from ResearcherAgent.core.evidence_check import build_ref_sample
        _ref = build_ref_sample(cite_map)      # 对比参照（非被引论文），用于边距判定
        _card_nums = {n for n, _p in cite_map.items() if _p.get("card")}  # 精读卡论文编号集
        for _s in _sections:
            _f = check_section_deterministic(_s["text"], cite_map, _embed_fn, _cos_cache,
                                             ref_sample=_ref, card_nums=_card_nums)
            det_raw[_s["title"]] = _f
            det_reports[_s["title"]] = summarize_findings(_f)
        _n_flag = sum(1 for r in det_reports.values() if r)
        _n_hard = sum(1 for v in det_raw.values() for f in v
                      if str(f.get("verdict", "")).startswith("no_anchor")
                      or f.get("verdict") == "no_citation")
        _n_weak = sum(1 for v in det_raw.values() for f in v
                      if f.get("verdict") == "weak_evidence")
        print(f"--> [Judge] 确定性证据检查:{_n_flag}/{len(_sections)} 节有问题"
              f"（确定性问题 {_n_hard} 处：无锚/无出处；摘要级弱证据 {_n_weak} 处标记待用户审核）")

    # ── 局部评审（每节一次，并行）──
    # 渐进式披露 L2：评审维度/确定性证据档位来自 judge_review/SKILL.md，不再写死在节点里
    # ⚠️ 必须转义花括号：正文含 JSON 输出契约 {type,claim,...}，拼进 ChatPromptTemplate
    #    会被 langchain f-string formatter 解析成非法格式说明符（Invalid format specifier）。
    #    转义后模板原样渲染回 {}，LLM 看到的文本不变。
    _judge_skill = build_skill_instructions(["judge_review"]).replace("{", "{{").replace("}", "}}")

    LOCAL_SYS = _with_memory(
        "⚠️ 上面「你的身份与记忆」中的用户长期偏好与当前约束同等效力。\n\n"
        "你是严格的学术期刊审稿人。现在只评审综述的**一个章节**，从严打分（宁严勿宽，"
        "证据不足必须扣分，不许宽容放水）。\n\n"
        + (_judge_skill or "")
        + "\n按上述技能规范执行：局部评审 4 维（内容深度/逻辑严谨性/证据忠实度/覆盖度）各 1-10 分；"
        "确定性证据检查分两档，按规范区分对待（确定性问题必须扣分，weak_match 由你判断）。\n\n"
        "**引用锚点口径（v1.2.13 硬约束，评审时不得违背）**：一个以句号（。！？）结尾的完整"
        "论断句，句尾出现 [N] 即视为**已锚定**——它是对编号 N 论文的整句引用（内部可用逗号/"
        "分号连接多个小句，如「XX提出了XX，解决了XX，指标XX[12]。」）。"
        "**同一句内以逗号/分号分隔的子句不要求各自标注 [N]**，不得据此报 no_anchor/no_citation；"
        "只有**整个句号句完全没有 [N]** 且该句像需要出处的事实陈述时，才报 no_citation。\n\n"
        '只输出 JSON：{{"内容深度": <int>, "逻辑严谨性": <int>, "证据忠实度": <int>, '
        '"coverage_ok": <bool>, '
        '"issues": [{{"type": "citation_mismatch|no_anchor|evidence_insufficient|'
        '"faithfulness|logic_gap|coverage_gap|style", "claim": "<原文片段>", '
        '"problem": "<问题>", "fix": "<修改建议>"}}]}}\n"'
        "即使没问题也给出 0-2 条改进项；有问题必须列出。"
    )

    def _local_one(s):
        sec_text = s["text"][:2800]
        # 该节用到的引用 → cite_map 条目
        _nums = sorted(set(int(x) for x in re.findall(r"\[(\d{1,3})\]", sec_text)))
        _sec_map = []
        for _n in _nums[:12]:
            _c = cite_map.get(_n)
            if _c:
                _sum = str(_c.get("abstract") or _c.get("paper") or _c.get("card")
                 or _c.get("summary") or "")[:150].strip()
                _sec_map.append(f"[{_n}] {str(_c.get('title') or '')[:100]}：{_sum}")
        _map_txt = "\n".join(_sec_map) if _sec_map else "（本节引用未能在精确映射中找到论文）"
        _det_txt = det_reports.get(s["title"], "") or "（本节确定性检查未发现问题）"
        _chain = ChatPromptTemplate.from_messages([
            ("system", LOCAL_SYS),
            ("user", "【本节标题】{title}\n\n【本节文本】\n{text}\n\n"
                     "【该节引用论文（引用精确映射，核对证据忠实度）】\n{cite_map}\n\n"
                     "【确定性证据检查（硬信号，必须优先核查）】\n{det}"),
        ])
        try:
            _resp = (_chain | judge_llm).invoke({
                "title": s["title"], "text": sec_text,
                "cite_map": _map_txt, "det": _det_txt})
            _raw = _resp.content if hasattr(_resp, "content") else str(_resp)
            _st, _en = _raw.find("{"), _raw.rfind("}")
            if _st == -1 or _en == -1:
                raise ValueError("no json")
            _d = json.loads(_raw[_st:_en + 1])
            return {"title": s["title"], "ok": True,
                    "深度": _d.get("内容深度"), "逻辑": _d.get("逻辑严谨性"),
                    "忠实": _d.get("证据忠实度"),
                    "issues": _d.get("issues") or []}
        except Exception as e:
            print(f"--> [Judge] ⚠️ 节「{s['title']}」评审失败: {str(e)[:80]}")
            return {"title": s["title"], "ok": False, "error": str(e)[:100]}

    from concurrent.futures import ThreadPoolExecutor
    _local_results = []
    with ThreadPoolExecutor(max_workers=4) as _ex:
        for _r in _ex.map(_local_one, _sections):
            _local_results.append(_r)
    _ok_local = [r for r in _local_results if r.get("ok")]
    print(f"--> [Judge] 局部评审完成 {len(_ok_local)}/{len(_sections)} 节")

    # ── 全局评审（一次，输入小：节标题 + 统计 + 首尾抽样）──
    GLOBAL_SYS = _with_memory(
        "⚠️ 上面「你的身份与记忆」中的用户长期偏好与当前约束同等效力。\n\n"
        "你是严格的学术期刊审稿人。现在评审整篇综述的**全局层面**（不逐句核对内容，"
        "内容层面的证据忠实度由逐节评审负责），从严打分（宁严勿宽）。\n\n"
        + (_judge_skill or "")
        + "\n按上述技能规范执行：全局评审 5 维（结构完整性/学术规范性/即时约束遵守度/"
        "记忆偏好遵守度/篇幅遵守度）各 1-10 分。\n\n"
        '只输出 JSON：{{"结构完整性": <int>, "学术规范性": <int>, '
        '"即时约束遵守度": <int>, "记忆偏好遵守度": <int>, "篇幅遵守度": <int>, '
        '"issues": [{{"type": "citation_mismatch|no_anchor|coverage_gap|style|logic_gap", '
        '"claim": "<位置>", "problem": "<问题>", "fix": "<修改建议>"}}]}}\n"'
        "即使没问题也给出 0-2 条改进项。"
    )
    _budget, _budget_src = effective_review_budget(state.get("user_constraints") or {})
    _limit = int(_budget * 1.25)
    _titles = "\n".join(f"- {s['title']}" for s in _sections)
    _head = body.strip()[:300]
    _tail = body.strip()[-300:]
    _g_chain = ChatPromptTemplate.from_messages([
        ("system", GLOBAL_SYS),
        ("user", "评审主题：{topic}\n用户本次约束：{requirements}\n"
                 "篇幅：预算 {budget} 字（{budget_src}），容许上限 {limit} 字，"
                 "正文实际 {body_len} 字（不含参考文献表）。\n\n"
                 "【节标题清单】\n{titles}\n\n"
                 "【开头 300 字（抽样）】\n{head}\n\n"
                 "【结尾 300 字（抽样）】\n{tail}"),
    ])
    try:
        _gresp = (_g_chain | judge_llm).invoke({
            "topic": topic, "requirements": requirements, "budget": _budget,
            "budget_src": _budget_src, "limit": _limit, "body_len": body_len,
            "titles": _titles, "head": _head, "tail": _tail})
        _graw = _gresp.content if hasattr(_gresp, "content") else str(_gresp)
        _gst, _gen = _graw.find("{"), _graw.rfind("}")
        if _gst == -1 or _gen == -1:
            raise ValueError("no json")
        _g = json.loads(_graw[_gst:_gen + 1])
    except Exception as e:
        print(f"--> [Judge] ⚠️ 全局评审失败: {str(e)[:80]} → fail-closed")
        return {
            "review_score": {
                "total_score": 0.0,
                "passed": False,
                "feedback": f"全局评审输出无法解析（{e}）。请重新组织综述。",
                "_parse_error": True,
            }
        }

    # ── 聚合 8 维 ──
    def _mean(key, src_list):
        vals = [r.get(key) for r in src_list if isinstance(r.get(key), (int, float))]
        if not vals:
            return None
        return round(sum(vals) / len(vals), 1)

    dims = ["结构完整性", "内容深度", "逻辑严谨性", "学术规范性",
            "即时约束遵守度", "记忆偏好遵守度", "篇幅遵守度", "证据忠实度"]
    agg = {
        "结构完整性": _g.get("结构完整性"),
        "内容深度": _mean("深度", _ok_local),
        "逻辑严谨性": _mean("逻辑", _ok_local),
        "学术规范性": _g.get("学术规范性"),
        "即时约束遵守度": _g.get("即时约束遵守度"),
        "记忆偏好遵守度": _g.get("记忆偏好遵守度"),
        "篇幅遵守度": _g.get("篇幅遵守度"),
        "证据忠实度": _mean("忠实", _ok_local),
    }
    # 缺失兜底：≥2 维缺失 → fail-closed
    _missing = [d for d in dims if agg.get(d) is None]
    if len(_missing) >= 2:
        print(f"--> [Judge] ⚠️ 维度缺失 {_missing} → fail-closed（触发重写）")
        return {
            "review_score": {
                "total_score": 0.0,
                "passed": False,
                "feedback": f"评审输出缺少维度（{_missing}）。请重新组织综述，确保结构清晰、内容完整。",
                "_parse_error": True,
                "_missing_dims": _missing,
            }
        }
    if _missing:
        print(f"--> [Judge] ⚠️ 维度缺失 1 个（{_missing}）→ 用其余 {len(dims)-1} 维重算")

    _vals = [agg[d] for d in dims if agg.get(d) is not None]
    total = round(sum(_vals) / len(_vals), 1) if _vals else 0.0

    # ── 引用真实性（确定性校验，fail-closed）──
    cite = check_citation_grounding(draft, render_pool(state.get("paper_pool", []) or [], max_chars=200000))
    passed = total >= JUDGE_PASS_THRESHOLD
    if cite["total_refs"] >= 3 and cite["rate"] < 0.6:
        passed = False
        print(f"--> [Judge] 引用真实性 {cite['rate']} < 0.6 → fail-closed 判不通过")

    # ── issues 汇总（全局 + 各节）──
    ISSUE_TYPES = {"citation_mismatch", "no_anchor", "evidence_insufficient",
                   "faithfulness", "logic_gap", "coverage_gap", "style"}
    issues = []
    for _r in _ok_local:
        for _i in (_r.get("issues") or []):
            if isinstance(_i, dict):
                _t = str(_i.get("type", "")).strip()
                if _t not in ISSUE_TYPES:
                    _t = "style"
                issues.append({
                    "type": _t,
                    "claim": f"[{_r['title']}] {str(_i.get('claim','') or '')[:160]}",
                    "problem": str(_i.get("problem", "") or "")[:250],
                    "fix": str(_i.get("fix", "") or "")[:250],
                })
    for _i in (_g.get("issues") or []):
        if isinstance(_i, dict):
            _t = str(_i.get("type", "")).strip()
            if _t not in ISSUE_TYPES:
                _t = "style"
            issues.append({
                "type": _t,
                "claim": str(_i.get("claim", "") or "")[:160],
                "problem": str(_i.get("problem", "") or "")[:250],
                "fix": str(_i.get("fix", "") or "")[:250],
            })
    issues = issues[:16]
    dist_now = {t: sum(1 for i in issues if i["type"] == t) for t in ISSUE_TYPES} if issues else {}
    print(f"--> [Judge] 评分: {total}/10 | {'✅ 通过' if passed else '❌ 需重写'} | "
          f"引用真实率={cite['rate']} | 各维度: {agg} | issues {len(issues)} 条: {dist_now}")

    # ── 证据类缺陷占比（只写入评审报告/低置信清单；单次评审不补检不重写）──
    n_evid = sum(1 for i in issues
                 if i["type"] in ("evidence_insufficient", "no_anchor", "faithfulness"))
    needs_research = bool(issues and n_evid >= max(2, int(len(issues) * 0.5)))
    if needs_research:
        print(f"--> [Judge] ⚠️ 证据类缺陷 {n_evid}/{len(issues)} 占主导 → 计入评审报告"
              f"（单次评审：不补检索、不重写，低置信点交用户审核）")

    # ── 迭代快照 ──
    round_no = state.get("rewrite_count", 0) + 1
    iter_log = list(state.get("iteration_log") or [])
    iter_log.append({
        "round": round_no,
        "judge_total": total,
        "judge_passed": passed,
        "dims": agg,
        "citation_rate": cite.get("rate"),
        "issues_dist": dist_now,
        "issues_n": len(issues),
        "issues": issues,
        "revision_plan": state.get("revision_plan", "") or "",
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    # ── keep-best ──
    score_data = {"total_score": total, "passed": passed,
                  "feedback": "", "issues": issues,
                  "citation_rate": cite.get("rate")}
    for _d in dims:
        score_data[_d] = agg.get(_d)
    # ── 低置信内容清单（评审报告核心：确定性证据逐句 verdict + 证据类 issues）──
    # 口径与 summarize_findings 一致：no_anchor*/no_citation/nli_contradiction 为确定性问题，
    # weak_match/no_evidence_text/weak_evidence 为辅助提示（weak_evidence=摘要级引用，2026-10）；
    # nli_neutral 不算缺陷（跨语言下偏高、非缺陷）。
    LOW_CONF_ISSUE = ("no_anchor", "no_citation", "nli_contradiction",
                      "weak_match", "no_evidence_text", "weak_evidence")
    _REASON = {
        "no_anchor": "引用了论文池中不存在的编号（无锚引用）",
        "no_citation": "陈述事实但没有引用任何论文",
        "nli_contradiction": "论断与所引论文内容矛盾（NLI 判定）",
        "weak_match": "论断与所引论文相似度偏低（可能张冠李戴，弱信号）",
        "no_evidence_text": "所引论文缺少可核对的证据文本（无摘要/无卡片）",
        "weak_evidence": "所引论文仅摘要级信息（未精读原文），论断细节未经原文核对",
    }
    low_conf = []
    for _sec, _finds in (det_raw or {}).items():
        for _f in _finds or []:
            _v = str(_f.get("verdict", ""))
            if not (_v.startswith("no_anchor") or _v in LOW_CONF_ISSUE):
                continue
            _key = "no_anchor" if _v.startswith("no_anchor") else _v
            low_conf.append({
                "section": _sec,
                "sentence": str(_f.get("sentence", ""))[:200],
                "issue": _v,
                "reason": _REASON.get(_key, _v),
                "anchors": _f.get("anchors") or [],
                "sim": _f.get("sim"),
                "margin": _f.get("margin"),
            })
    # ── 五步流水线证据锚校验（方案 v1.2 5.5）──
    # 拼接路径下，write 节点把「编号 → 该篇被选用块的 evidence 锚点」写入
    # state.review_block_anchors（空列表 = 块无证据锚）。评审对引用了这些编号的句子，
    # 若其卡片块无证据锚 → 追加 weak_evidence 低置信条目（细节未经原文核对）。
    _block_anchors = state.get("review_block_anchors") or {}
    if _block_anchors:
        _seen_keys = {(x["section"], x["sentence"]) for x in low_conf}
        for _sec, _finds in (det_raw or {}).items():
            for _f in _finds or []:
                _key = (_sec, str(_f.get("sentence", ""))[:200])
                if _key in _seen_keys:
                    continue
                _no_anchor_hit = False
                for _a in (_f.get("anchors") or []):
                    _as = str(_a).strip("[]")
                    if _as.isdigit() and int(_as) in _block_anchors \
                            and not _block_anchors[int(_as)]:
                        _no_anchor_hit = True
                        break
                if _no_anchor_hit:
                    low_conf.append({
                        "section": _sec,
                        "sentence": str(_f.get("sentence", ""))[:200],
                        "issue": "weak_evidence",
                        "reason": "句式块无证据锚（卡片字段缺失/仅摘要），细节未经原文核对（流水线锚点校验）",
                        "anchors": _f.get("anchors") or [],
                    })
                    _seen_keys.add(_key)
    low_conf = low_conf[:20]
    if low_conf:
        print(f"--> [Judge] 📋 低置信内容清单 {len(low_conf)} 条"
              f"（随评审报告交付，交用户审核）")

    review_report = {
        "total_score": total,
        "passed": passed,
        "threshold": JUDGE_PASS_THRESHOLD,
        "dimensions": agg,
        "citation_rate": cite.get("rate"),
        "needs_research": needs_research,
        "issues": issues,
        "low_confidence": low_conf,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    updates = {"review_score": score_data, "citation_report": cite,
               # enforce 后的稿子回写（非证据编号已从正文/参考表移除）
               "draft_review": draft,
               "judge_issues": issues, "needs_research": needs_research,
               # 评审报告：随综述一并交付（含低置信内容清单，交用户审核）
               "review_report": review_report,
               # 逐句确定性证据明细 → 供 write 重写时定位「哪句/引了哪篇/余弦多少」
               "evidence_report": det_raw,
               "iteration_log": iter_log}
    prev_best = state.get("best_score", {})
    prev_best_total = float(prev_best.get("total_score", -1) or -1)
    if total > prev_best_total:
        updates["best_review"] = draft
        updates["best_score"] = score_data
        print(f"--> [Judge] keep-best 更新：当前 {total} > 历史最佳 {prev_best_total}")

    return updates


def _judge_report_md(updates: dict, state: dict) -> str:
    """后台评审返回 updates → 人类可读评审报告 md（总分/各维/引用率/issues/低置信清单）。"""
    report = updates.get("review_report") or {}
    dims = report.get("dimensions") or {}
    issues = report.get("issues") or []
    low = report.get("low_confidence") or []
    lines = [
        "# 综述评审报告（后台异步生成）",
        "",
        f"- 生成时间：{report.get('generated_at', '')}",
        f"- 总分：**{report.get('total_score')}/10**（阈值 {report.get('threshold')}）"
        + (" ✅ 通过" if report.get("passed") else " ❌ 未通过（单次评审不重写，低置信点交用户审核）"),
        f"- 引用真实率：{report.get('citation_rate')}",
        "",
        "## 各维度得分",
    ]
    for k, v in dims.items():
        lines.append(f"- {k}：{v}")
    lines += ["", "## 缺陷 issues"]
    for i in issues:
        lines.append(f"- [{i.get('type')}] {str(i.get('claim', ''))[:120]}｜{str(i.get('problem', ''))[:150]}")
    if not issues:
        lines.append("- 无")
    lines += ["", "## 低置信内容清单（交用户审核）"]
    if low:
        for x in low:
            lines.append(f"- 【{x.get('section')}】{str(x.get('sentence', ''))[:160]}｜"
                         f"问题：{x.get('issue')}｜原因：{str(x.get('reason', ''))[:120]}")
    else:
        lines.append("- 无")
    return "\n".join(lines)


def async_judge_node(state: AgentState, judge_llm):
    """【节点 · 评委打分 v3 异步化】综述先行交付，评审报告后台生成。

    用户决策（2026-10）：judge 同步评审曾占总耗时 58.8%（250s+），综述交付前
    用户只能干等 → 本节点只**启动 daemon 线程**执行完整评审（judge_review_node
    主体：局部×N 并行 + 全局 + 确定性证据 + 低置信清单），**立即返回**；
    评审报告写入 logs/review_report_latest.md/.json，完成后向用户补交。
    线程持有 state 快照（dict），不引用图 state；embedding 复用模块级单例。
    """
    _snap = dict(state)
    print("--> [Judge] 🚀 评审已异步化：综述先行交付，评审报告后台生成中（约数分钟）...")

    def _run():
        try:
            updates = judge_review_node(_snap, judge_llm)
        except Exception as e:
            print(f"--> [Judge] ❌ 后台评审失败: {str(e)[:200]}")
            updates = {}
        try:
            from ResearcherAgent.core.paths import REPO_ROOT
            import json as _json
            _md = _judge_report_md(updates, _snap)
            _rep = updates.get("review_report") or {}
            _p = REPO_ROOT / "logs" / "review_report_latest.md"
            _p.parent.mkdir(parents=True, exist_ok=True)
            _p.write_text(_md, encoding="utf-8")
            _pj = REPO_ROOT / "logs" / "review_report_latest.json"
            _pj.write_text(_json.dumps(_rep, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"--> [Judge] ✅ 后台评审完成 → {_p}")
        except Exception as e:
            print(f"--> [Judge] ⚠️ 报告落盘失败: {str(e)[:120]}")

    threading.Thread(target=_run, daemon=True, name="judge_async").start()
    return {"judge_async_dispatched": True,
            "review_score": {"status": "pending", "passed": True,
                             "feedback": "评审后台进行中，综述已先行交付；报告生成后另行查看。"}}


def _route_after_answer(state: AgentState) -> str:
    """answer 节点之后：有 content_safety 则先进安全审核再保存"""
    skills = state.get("skills", [])
    if needs_content_safety(skills):
        print("--> [Route] answer → content_safety")
        return "content_safety"
    print("--> [Route] answer → save")
    return "save"


def route_after_judge(state: AgentState) -> str:
    """条件边：评审后**直接交付**（不做多轮重写）。

    用户决策（2026-10）：重写太慢（judge 曾占总耗时 58.8%），改为单次评审，
    综述随评审报告（含低置信内容清单与原因）一并交付，低置信点交用户审核。
    v1.2.13：评审已异步化（async_judge_node 启动后台线程），judge 节点秒回
    `review_score.status=pending` → 综述先行交付，报告后台生成后补交。
    因此 judge → 永远进 save（或 content_safety，当前恒 False → 实际恒 save）。
    """
    score = state.get("review_score", {})
    skills = state.get("skills", [])
    if str(score.get("status")) == "pending":
        print("--> [Reflect] 🔄 评审后台进行中 → 综述先行交付（报告生成后补交）")
    else:
        passed = bool(score.get("passed", False))
        if passed:
            print("--> [Reflect] ✅ 综述通过评审 → 保存交付")
        else:
            print("--> [Reflect] ⚠️ 评审未通过（单次评审不重写）→ 附评审报告交付保存")
    return "content_safety" if needs_content_safety(skills) else "save"


def increment_re_research_node(state: AgentState):
    """【已废弃】`judge → agent` 补检索路径已删除（2026-10：单次评审直接交付，不再补检）。
    本函数仅保留签名兼容旧引用，不再挂入图。"""
    return {"re_research_count": int(state.get("re_research_count") or 0) + 1}


def increment_rewrite_node(state: AgentState):
    """【节点 · 计数】累加重写次数 + 累积本轮评委反馈 + 快照本轮草稿，然后回到 write 节点"""
    count = state.get("rewrite_count", 0) + 1
    score = state.get("review_score", {})
    fb = (score.get("feedback", "") or "").strip()
    total = score.get("total_score", "N/A")

    history = state.get("feedback_history", "")
    if fb:
        entry = f"【第 {count} 轮 · 得分 {total}/10】\n{fb}"
        history = f"{history}\n\n{entry}".strip() if history else entry

    # ---- draft_history 快照（迭代评测：每轮草稿 + 修订说明）----
    draft_history = list(state.get("draft_history") or [])
    cur = state.get("draft_review", "")
    if cur:
        draft_history.append({
            "round": count,
            "draft": cur,
            "revision_plan": state.get("revision_plan", "") or "",
        })

    return {"rewrite_count": count, "feedback_history": history,
            "draft_history": draft_history, "revision_plan": ""}



# ==== 内容安全审核（content_safety —— 独立于学术评委的第二个 harness）====

def content_safety_node(state: AgentState, llm):
    """【节点 · 内容安全审核】对生成内容审核安全性。

    与 judge_review_node 同级但维度不同——这个只管"安不安全/合不合规"，
    不管"写得好不好/学术不学术"。
    """
    draft = state.get("draft_review", "") or state.get("final_review", "")
    count = state.get("content_safety_count", 0)

    print(f"--> [Safety] 正在审核内容安全性... (第 {count + 1} 轮)")

    if not draft or len(draft.strip()) < 50:
        return {
            "content_safety_score": {
                "total_score": 10.0, "passed": True,
                "feedback": "内容过短，跳过安全审核",
            }
        }

    # 渐进式披露 L2：审核维度/阈值/输出契约来自 content_safety/SKILL.md，不再写死在节点里
    # ⚠️ 转义花括号：正文含 JSON 输出契约 {...}，作为 ChatPromptTemplate 模板会崩（同 judge 修复）。
    _safety_skill = build_skill_instructions(["content_safety"]).replace("{", "{{").replace("}", "}}")

    system = _with_memory(
        "你是一名内容安全审核员。\n\n"
        + (_safety_skill or "")
        + "\n按上述技能规范执行：从 4 个维度（政治合规性/用语文明/信息真实性/客观中立）打分（各 1-10 分），"
        "以 JSON 格式输出（只输出 JSON，不要其他文字）：\n"
        '{{"政治合规性": <int>, "用语文明": <int>, "信息真实性": <int>, '
        '"客观中立": <int>, '
        '"total_score": <4个维度算术平均，保留1位小数>, '
        '"passed": <total_score >= 7.0 则为 true，否则 false>, '
        '"feedback": "<若不通过，必须给出具体且可操作的修改意见>"'
        '}}'
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", "【待审核文本】：\n{draft}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"draft": _readable_draft(draft)})
    raw = response.content if hasattr(response, "content") else str(response)

    # 解析 JSON
    try:
        text = raw.strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("未找到 JSON")
        score_data = json.loads(text[start:end + 1])
    except Exception as e:
        print(f"--> [Safety] JSON 解析失败: {e} → fail-closed（判不通过）")
        score_data = {"total_score": 0.0, "passed": False,
                       "feedback": f"安全审核解析失败（{e}），请重写以确保内容合规。"}

    # 重算 total（不盲信模型自报）
    dim_values = []
    for d in _CONTENT_SAFETY_DIMS:
        v = score_data.get(d)
        try:
            fv = float(v)
            if 0 <= fv <= 10:
                dim_values.append(fv)
        except (TypeError, ValueError):
            continue
    if dim_values:
        computed = round(sum(dim_values) / len(dim_values), 1)
        reported = score_data.get("total_score")
        try:
            if reported is None or abs(float(reported) - computed) > 0.5:
                score_data["total_score"] = computed
        except (TypeError, ValueError):
            score_data["total_score"] = computed

    total = float(score_data.get("total_score", 0) or 0)
    passed = total >= _CONTENT_SAFETY_THRESHOLD
    score_data["passed"] = passed

    dim_scores = {d: score_data.get(d, "?") for d in _CONTENT_SAFETY_DIMS}
    print(f"--> [Safety] 安全评分: {total}/10 | {'✅ 通过' if passed else '❌ 需重写'} | {dim_scores}")
    if not passed:
        fb = score_data.get("feedback", "")
        print(f"--> [Safety] 反馈: {fb[:200]}...")

    return {"content_safety_score": score_data}


def route_after_content_safety(state: AgentState) -> str:
    """条件边：安全审核通过 → save，未通过 → 视重写次数决定"""
    score = state.get("content_safety_score", {})
    passed = bool(score.get("passed", False))
    count = state.get("content_safety_count", 0)

    if passed:
        print("--> [SafetyRoute] ✅ 内容安全审核通过 → 保存")
        return "save"

    if count >= MAX_REWRITE_ROUNDS:
        print(f"--> [SafetyRoute] ⚠️ 已达最大重写次数 ({MAX_REWRITE_ROUNDS}) → 保存")
        return "save"

    print(f"--> [SafetyRoute] 🔄 未通过，第 {count + 1}/{MAX_REWRITE_ROUNDS} 轮重写")
    return "increment_content_safety"


def increment_content_safety_node(state: AgentState):
    """计数 + 累积安全反馈，然后回到 write（综述）或经 safety_feedback 回 agent（问答）重写"""
    count = state.get("content_safety_count", 0) + 1
    score = state.get("content_safety_score", {})
    fb = (score.get("feedback", "") or "").strip()

    history = state.get("content_safety_feedback", "")
    if fb:
        entry = f"【安全审核第 {count} 轮 · 得分 {score.get('total_score', 'N/A')}/10】\n{fb}"
        history = f"{history}\n\n{entry}".strip() if history else entry

    return {"content_safety_count": count, "content_safety_feedback": history}


def route_after_increment_content_safety(state: AgentState) -> str:
    """综述 → 回 write 重写（write 会读 content_safety_feedback）；问答 → 把反馈投回 agent 循环。"""
    if needs_write_review(state.get("skills", [])):
        return "write"
    return "safety_feedback"


def save_node(state: AgentState):
    """【节点 · 保存】保存对话记录 + 记忆嵌入 + 用户指定路径的文件。

    不再自动存到 reviews/ 目录——综述内容已在 JSONL 对话历史中保留，
    前端也提供下载按钮，磁盘冗余副本没有存在必要且删除对话时无法清理。
    """
    from ResearcherAgent.memory.conversation_manager import save_conversation_turn

    session_id = state.get("session_id", "")
    topic = state.get("topic", "unknown")
    save_path = state.get("save_path", "")
    score = state.get("review_score", {})

    # ---- keep-best：若走过评委且最终版未通过，则回退到历轮最佳草稿 ----
    current_draft = state.get("draft_review", "") or state.get("final_review", "")
    best_review = state.get("best_review", "")
    best_score = state.get("best_score", {})
    current_total = float(score.get("total_score", -1) or -1)
    best_total = float(best_score.get("total_score", -1) or -1)

    if best_review and best_total > current_total:
        print(f"--> [Save] keep-best 生效：采用历史最佳草稿 {best_total} 替代当前 {current_total}")
        final_output = best_review
        score = best_score
    else:
        final_output = current_draft

    print("--> [Save] 正在保存...")

    # ---- 1. 写入 JSONL 对话记录 ----
    skills = state.get("skills", ["answer_direct"])
    is_review = needs_write_review(skills) or needs_search_papers(skills)
    print(f"--> [Save] skills={skills} is_review={is_review}")
    if session_id and is_review:
        _rr = state.get("review_report", {}) or {}
        save_conversation_turn(session_id, "assistant", final_output[:3000],
                               {"type": "final_review", "topic": topic,
                                "score": score.get("total_score", "N/A"),
                                "rewrites": state.get("rewrite_count", 0),
                                "low_confidence": len(_rr.get("low_confidence") or [])})
        # 综述摘要记忆由 run_stream 的 LLM-as-Judge 统一分类存储（L1/L2），此处不再重复写入

    # ---- 3. 如果用户指定了保存路径，写入文件 ----
    if save_path:
        sp = Path(save_path)
        if not sp.is_absolute():
            sp = Path.cwd() / save_path
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(final_output, encoding="utf-8")
        print(f"--> [Save] ✅ 已按用户指定路径保存: {sp}")

    return {
        "final_review": final_output,
        "save_path": str(sp) if save_path else "",
    }

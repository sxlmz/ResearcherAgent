# -*- coding: utf-8 -*-
"""CC（Claude Code）风格项目记忆：MEMORY.md 结构化精炼记忆。

机制对照 Claude Code Auto Memory（官方 docs: code.claude.com/docs/en/memory）：
- 纯 markdown、**分类分区**（type 字段：pitfall 踩坑 / command 常用命令 / workflow 流程约定 /
  convention 用户偏好）——对应 CC 的 user/feedback/command/workflow 四类笔记；
- **模型主动写入**：remember_memory 工具（LLM 在工作过程中自主判断"这值得记"），
  成功事件回写（fix_and_rerun 修复成功）也经同一入口；
- **同主题合并更新**（同 type+topic 命中 → 替换内容，而非 append-only 堆积）；
- **预算加载**：注入时截断（CC 的 MEMORY.md 索引只载入前 200 行/25KB）；
- **Dreams 式整理**：consolidate_memory 工具（LLM 重组：合并重复、替换过时、保持格式）；
- 审计留痕：EXPERIENCE.md（innovation_tools 维护）保留为 append-only 原始日志，
  MEMORY.md 是精炼后的注入用记忆——两者并存，职责分离。

读写走沙箱后端（get_env），与 code_exec 工具组一致。
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agent.core import tracing
from agent.sandbox.sandbox_exec import get_env

# ---------------------------------------------------------------------------
# 常量与结构
# ---------------------------------------------------------------------------
MEMORY_FILE = "project/MEMORY.md"

# 分区：type → 标题。对应 CC 四类笔记（user→convention、feedback→pitfall、command、workflow）
SECTIONS = {
    "pitfall": "踩坑经验（pitfall）",
    "command": "常用命令与环境（command）",
    "workflow": "项目流程与约定（workflow）",
    "convention": "用户偏好与项目约定（convention）",
}
_SECTION_ORDER = ["pitfall", "command", "workflow", "convention"]
_SECTION_MAX = 12                 # 每区条目上限（超出丢最旧，配合 consolidate 精修）
_INJECT_MAX_CHARS = 1500          # 注入预算（与 CC MEMORY.md 索引预算同理）
_INJECT_PER_SECTION = 3           # 注入时每区取最近条数

_HEADER = "# 项目记忆（CC 风格）\n"
_HEADER += "> 自动维护：remember_memory 记录，consolidate_memory 整理（Claude Code Auto Memory 模式）。\n"
_HEADER += "> 条目格式：`- **主题**：内容`；分区见各 `##` 标题。\n"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_raw() -> str:
    """读 MEMORY.md 原文（空/不存在返回空串）。"""
    try:
        raw = get_env().read_file(MEMORY_FILE)
    except Exception:
        return ""
    if not raw or raw.startswith("Error reading file"):
        return ""
    return str(raw)


def _write_raw(text: str):
    try:
        get_env().write_file(MEMORY_FILE, text)
    except Exception:
        pass


def _parse(raw: str) -> dict:
    """按分区标题把原文解析成 {type: [entry_str, ...]}（仅条目行，去重保序）。"""
    result = {t: [] for t in SECTIONS}
    cur = None
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            title = stripped[3:].strip()
            cur = next((t for t, h in SECTIONS.items() if h == title), None)
            continue
        if cur and stripped.startswith("- "):
            result[cur].append(stripped)
    return result


def _render(entries: dict) -> str:
    """按分区渲染回 markdown（保留固定分区顺序，空区省略标题）。"""
    blocks = [_HEADER]
    for t in _SECTION_ORDER:
        items = entries.get(t) or []
        if not items:
            continue
        blocks.append(f"## {SECTIONS[t]}")
        blocks.extend(items)
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def read_memory() -> dict:
    """读取当前记忆（解析后的条目 dict）。"""
    return _parse(_read_raw())


def _norm_topic(s: str) -> str:
    """主题归一化（合并键）：去首尾空白、末尾冒号等标点、统一小写——`ModuleNotFoundError:` 与
    `ModuleNotFoundError` 视为同一主题，避免同语义条目堆积。"""
    return s.strip().strip(":：;；。,.！!？?").strip().lower()


def remember(type_: str, topic: str, content: str) -> tuple[str, int]:
    """写入一条记忆：同 type+topic（归一化后）命中 → 更新；否则新增。返回 (动作, 该区当前条数)。

    type_: pitfall / command / workflow / convention（未知类型归入 convention）。
    topic: 短主题（合并键，≤40 字）。
    content: 内容（单行，≤300 字）。
    """
    if type_ not in SECTIONS:
        type_ = "convention"
    topic = (topic or "未命名").strip()[:40]
    content = " ".join((content or "").split())[:300]
    entry = f"- **{topic}**：{content}"

    entries = read_memory()
    items = entries.get(type_) or []
    hit = None
    key = _norm_topic(topic)
    for i, line in enumerate(items):
        # 条目内 **主题**：xxx —— 主题归一化后一致即视为同一记忆（更新而非堆积）
        m = re.match(r"^- \*\*(.+?)\*\*：", line)
        if m and _norm_topic(m.group(1)) == key:
            hit = i
            break
    if hit is not None:
        # 更新：沿用已存在条目的主题（更干净），只替换内容
        old = items[hit]
        m0 = re.match(r"^- \*\*(.+?)\*\*：", old)
        keep_topic = m0.group(1) if m0 else topic
        items[hit] = f"- **{keep_topic}**：{content}"
        action = "updated"
    else:
        items.append(entry)
        action = "added"
    if len(items) > _SECTION_MAX:
        items = items[-_SECTION_MAX:]           # 丢最旧，保留最近
    entries[type_] = items
    _write_raw(_render(entries))
    return action, len(items)


def inject_context(max_chars: int = _INJECT_MAX_CHARS,
                   per_section: int = _INJECT_PER_SECTION) -> str:
    """生成注入文本（预算截断）：每区最近 per_section 条，拼接为一个 prompt 段落。"""
    entries = read_memory()
    blocks = []
    for t in _SECTION_ORDER:
        items = entries.get(t) or []
        if not items:
            continue
        recent = items[-per_section:]
        blocks.append(f"【{SECTIONS[t]}】")
        blocks.extend(recent)
    text = "\n".join(blocks).strip()
    if not text:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return f"## 项目记忆（CC 风格，跨会话沉淀）\n{text}"


def read_memory_for_inject() -> str:
    """供 agent 首步全局注入的完整段落（含标题，预算截断）。"""
    ctx = inject_context()
    return ctx


# ---------------------------------------------------------------------------
# 工具 1：remember_memory —— LLM 主动记录 / 事件回写入口
# ---------------------------------------------------------------------------
@tool
def remember_memory(
    type_: str,
    topic: str,
    content: str,
) -> str:
    """把一条值得跨会话记住的信息写入项目记忆（MEMORY.md）。

    何时使用：你在工作中发现了值得下次继续使用的知识——踩过的坑及修复、
    验证过有效的命令/环境细节、项目流程约定、用户的偏好或纠正。
    （Claude Code Auto Memory 模式：模型主动沉淀，不只在失败时记录。）

    Args:
        type_: 类别。pitfall=踩坑经验（报错/根因/修复）；command=常用命令与环境细节；
               workflow=项目流程与约定；convention=用户偏好与项目约定。
        topic: 短主题（≤40 字，作为合并键——同一主题会更新旧条目而非重复添加）。
        content: 内容（≤300 字，单行；建议含 症状/根因/修复 或 背景/结论）。
    Returns:
        写入结果（新增或更新 + 该区当前条数）。
    """
    action, count = remember(type_, topic, content)
    return f"[memory] {action} 条目（{type_}/{topic}），该区共 {count} 条。"


# ---------------------------------------------------------------------------
# 工具 2：consolidate_memory —— Dreams 式整理（LLM 重组）
# ---------------------------------------------------------------------------
def make_consolidate_memory(llm):
    @tool
    def consolidate_memory(
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
    ) -> Command:
        """整理项目记忆 MEMORY.md（Claude "Dreams" 模式）。

        适用：记忆条目变多、出现重复/过时/矛盾时手动或由模型触发。
        工具把当前 MEMORY.md 全文交给 LLM 重组：合并重复条目、用最新信息替换过时条目、
        保持分区与条目格式（- **主题**：内容），只输出整理后的完整 markdown 并写回。
        （原始审计日志 EXPERIENCE.md 不受影响。）
        """
        raw = _read_raw()
        if not raw.strip():
            return Command(update={"messages": [ToolMessage(
                content="MEMORY.md 尚无内容，暂无需整理。",
                tool_call_id=tool_call_id, status="error")]})
        sys = (
            "你是项目记忆管理员。整理下面的项目记忆文件（Claude Code Auto Memory 的整理环节）：\n"
            "1) 合并重复/相似条目（保留信息最全的一条）；\n"
            "2) 用更新、更准确的表述替换过时或有误的条目；\n"
            "3) 保持分区标题（## 踩坑经验（pitfall）等）与条目格式 `- **主题**：内容` 不变；\n"
            "4) 不丢失任何仍有价值的信息，条数只减不增（或持平）；\n"
            "5) 只输出整理后的完整 markdown，不要解释。"
        )
        with tracing.span("llm", "consolidate_memory", state.get("session_id", "")):
            resp = llm.invoke([SystemMessage(content=sys),
                               HumanMessage(content=f"当前 MEMORY.md：\n\n{raw}")])
        text = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r"```(?:markdown|md)?\s*([\s\S]*?)```", text)
        out = m.group(1).strip() if m else text.strip()
        if not out.startswith("#"):
            out = _HEADER + "\n" + out
        _write_raw(out)
        entries = _parse(out)
        total = sum(len(v) for v in entries.values())
        return Command(update={"messages": [ToolMessage(
            content=f"[memory] 整理完成，当前共 {total} 条（分区："
                    + "、".join(f"{k}={len(v)}" for k, v in entries.items() if v) + "）。",
            tool_call_id=tool_call_id)]})

    return consolidate_memory

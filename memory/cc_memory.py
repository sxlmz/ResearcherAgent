# -*- coding: utf-8 -*-
"""CC（Claude Code）风格项目记忆 v2：摘要索引 + topic 文件按需读。

v2 重构（2026-10 用户决策，对齐 Claude Code Auto Memory 的完整纪律）：
- **摘要索引**：MEMORY.md 只存索引行 `- **主题**（日期）：一行摘要（→ memory/<type>/<slug>.md）`，
  **全量注入**；具体说明落 topic 文件，需要时由 recall_memory 按索引定位读取——不再把全文塞进注入。
- **写前过滤 + 写前查重**：remember 写前一次 LLM 调用，对现有索引做三分类
  ADD（新增）/ UPDATE（更新已有条目）/ NOOP（可从代码推导、已存在或通用知识 → 不写），
  从源头防废话堆积与重复（Mem0 ADD/UPDATE/DELETE/NOOP 的轻量版，去掉 DELETE）。
- **自动整理（Auto Dream）**：run_stream 尾部 maybe_auto_consolidate 双门限自动触发
  （距上次整理 > 24h 且 新增条数 ≥ 3），LLM 重组索引 + 合并 topic，更新 metadata。
- **时间戳 + 新鲜度警告**：索引行与 topic 文件带记录日期；注入时 > 90 天条目附
  「⚠️ 可能过时」警告（CC 对 >1 天记忆提示过时，项目踩坑环境稳定，取 90 天）。
- **审计双轨保留**：EXPERIENCE.md（innovation_tools 维护）仍是 append-only 原始日志。

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

from ResearcherAgent.core import tracing
from ResearcherAgent.sandbox.sandbox_exec import get_env

# ---------------------------------------------------------------------------
# 常量与结构
# ---------------------------------------------------------------------------
MEMORY_FILE = "project/MEMORY.md"
_MEM_DIR = "project/memory"                 # topic 文件根目录（按 type 分子目录）

# 分区：type → 标题。对应 CC 四类笔记（user→convention、feedback→pitfall、command、workflow）
SECTIONS = {
    "pitfall": "踩坑经验（pitfall）",
    "command": "常用命令与环境（command）",
    "workflow": "项目流程与约定（workflow）",
    "convention": "用户偏好与项目约定（convention）",
}
_SECTION_ORDER = ["pitfall", "command", "workflow", "convention"]
_SECTION_MAX = 12                 # 每区索引条目上限（超出丢最旧，配合 consolidate 精修）
_INJECT_MAX_CHARS = 4000          # 注入预算（索引全量注入；行短，4000 字符足够容纳 48 条）
_STALE_DAYS = 90                  # 超过该天数注入时附「可能过时」警告
_META_RE = re.compile(r">\s*last_consolidated:\s*(\S+)")
_META_CNT_RE = re.compile(r">\s*entries_at_consolidation:\s*(\d+)")

_HEADER = "# 项目记忆（CC 风格）\n"
_HEADER += "> 自动维护：remember_memory 记录，consolidate_memory / 会话后自动整理精修。\n"
_HEADER += "> 索引行格式：`- **主题**（日期）：一行摘要（→ memory/<type>/<slug>.md）`；细节按需读 topic 文件。\n"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


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


# ---------------------------------------------------------------------------
# topic 文件读写（具体说明落这里，索引只留摘要）
# ---------------------------------------------------------------------------
def _slug(topic: str) -> str:
    """主题 → 文件名安全 slug（稳定，recall 可反推路径）。"""
    s = _norm_topic(topic)
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", s).strip("-")
    return (s or "untitled")[:40]


def _topic_path(type_: str, topic: str) -> str:
    return f"{_MEM_DIR}/{type_}/{_slug(topic)}.md"


def _read_topic(path: str) -> str:
    try:
        raw = get_env().read_file(path)
    except Exception:
        return ""
    if not raw or raw.startswith("Error reading file"):
        return ""
    return str(raw)


def _write_topic(path: str, text: str):
    try:
        get_env().write_file(path, text)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 索引解析/渲染（旧格式兼容：`- **主题**：内容` 也能解析）
# ---------------------------------------------------------------------------
def _parse(raw: str) -> dict:
    """按分区标题把 MEMORY.md 解析成 {type: [entry_line, ...]}（仅索引行，去重保序）。"""
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


def _render(entries: dict, meta: str = "") -> str:
    """按分区渲染回 markdown（保留固定分区顺序，空区省略标题）。"""
    blocks = [_HEADER]
    if meta:
        blocks.insert(1, meta.rstrip("\n"))
    for t in _SECTION_ORDER:
        items = entries.get(t) or []
        if not items:
            continue
        blocks.append(f"## {SECTIONS[t]}")
        blocks.extend(items)
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def _meta(raw: str) -> str:
    """提取文件头 metadata 行（仅 last_consolidated / entries_at_consolidation）。"""
    out = []
    for line in (raw or "").splitlines()[:8]:
        if _META_RE.match(line) or _META_CNT_RE.match(line):
            out.append(line)
    return "\n".join(out) + "\n" if out else ""


# ---------------------------------------------------------------------------
# 核心写入：remember（确定性，topic 文件 + 索引行）
# ---------------------------------------------------------------------------
def _norm_topic(s: str) -> str:
    """主题归一化（合并键）：去首尾空白、末尾冒号等标点、统一小写——`ModuleNotFoundError:` 与
    `ModuleNotFoundError` 视为同一主题，避免同语义条目堆积。"""
    return s.strip().strip(":：;；。,.！!？?").strip().lower()


def _index_topic(line: str) -> str:
    """从索引行提取主题（兼容新旧两种格式）。"""
    m = re.match(r"^- \*\*(.+?)\*\*", line)
    return _norm_topic(m.group(1)) if m else ""


def _read_memory() -> dict:
    return _parse(_read_raw())


def remember(type_: str, topic: str, content: str,
             summary: str | None = None, ts: str | None = None,
             _migrate: bool = True) -> tuple[str, int]:
    """确定性写入一条记忆：写 topic 文件 + upsert 索引行。

    同 type+topic（归一化后）命中 → 更新（保留旧索引主题，时间戳刷新）；否则新增。
    返回 (动作, 该区当前条数)。摘要缺省取内容前 60 字（LLM 版 remember 会传精炼摘要）。

    _migrate=False：迁移内部调用时关闭（防 remember ↔ _migrate_legacy 递归）。
    """
    if _migrate:
        _migrate_legacy()   # 写入前确保存量已迁到新格式（幂等；迁移内部以 _migrate=False 调用自身）
    if type_ not in SECTIONS:
        type_ = "convention"
    topic = (topic or "未命名").strip()[:40]
    content = " ".join((content or "").split())[:800]
    if summary:
        summary = " ".join(summary.split())[:120]
    else:
        # 无显式摘要：取内容前 60 字，超过则加省略号（LLM 版会传精炼摘要，不会走这里）
        summary = " ".join(content.split())[:60]
        if len(" ".join(content.split())) > 60:
            summary = summary[:59] + "…"
    ts = ts or _today()
    path = _topic_path(type_, topic)
    _write_topic(path, f"# {topic}\n\n> 记录于 {ts} ｜ type: {type_}\n\n{content}\n")

    entries = _read_memory()
    items = entries.get(type_) or []
    hit = None
    key = _norm_topic(topic)
    for i, line in enumerate(items):
        if _index_topic(line) == key:
            hit = i
            break
    entry = f"- **{topic}**（{ts}）：{summary}（→ {path}）"
    if hit is not None:
        old = items[hit]
        m0 = re.match(r"^- \*\*(.+?)\*\*", old)
        keep_topic = m0.group(1) if m0 else topic
        items[hit] = f"- **{keep_topic}**（{ts}）：{summary}（→ {path}）"
        action = "updated"
    else:
        items.append(entry)
        action = "added"
    if len(items) > _SECTION_MAX:
        items = items[-_SECTION_MAX:]           # 丢最旧，保留最近（索引级；topic 文件保留供 recall）
    entries[type_] = items
    _write_raw(_render(entries, meta=_meta(_read_raw())))
    return action, len(items)


# ---------------------------------------------------------------------------
# 写前过滤 + 查重：LLM 三分类（ADD / UPDATE / NOOP）
# ---------------------------------------------------------------------------
def make_remember_memory(llm):
    """CC 纪律版 remember：写前一次 LLM 调用过滤/查重，避免记废话与重复。

    规则（写进 prompt）：
    - 能从项目代码/已有文档直接推导的环境细节（如"入口是 main.py"）→ NOOP；
    - 索引里已有同主题 → UPDATE（合并到旧条目，保留历史有用信息）；
    - 通用知识（"安装依赖需要网络"）→ NOOP；
    - 真正有信息增量的踩坑/命令/流程/偏好 → ADD。
    LLM 失败时回退确定性写入（不阻塞，代价是可能多记一条）。
    """

    @tool
    def remember_memory(
        type_: str,
        topic: str,
        content: str,
    ) -> str:
        """把一条值得跨会话记住的信息写入项目记忆（MEMORY.md 摘要索引 + topic 文件）。

        何时使用：你在工作中发现了值得下次继续使用的知识——踩过的坑及修复、
        验证过有效的命令/环境细节、项目流程约定、用户的偏好或纠正。
        写入前会自动做「过滤 + 查重」：可从代码/文档推导的、已存在或通用知识不写
        （返回 NOOP 说明原因）；同主题自动更新旧条目而非重复堆积。

        Args:
            type_: 类别。pitfall=踩坑经验（报错/根因/修复）；command=常用命令与环境细节；
                   workflow=项目流程与约定；convention=用户偏好与项目约定。
            topic: 短主题（≤40 字，作为合并键——同一主题会更新旧条目而非重复添加）。
            content: 内容（≤300 字，单行；建议含 症状/根因/修复 或 背景/结论）。
        Returns:
            写入结果（新增/更新/不写入 + 原因 + 该区当前条数）。
        """
        raw = _read_raw()
        entries = _read_memory()
        index_lines = "\n".join(
            f"[{t}] {line}" for t in _SECTION_ORDER for line in entries.get(t) or [])
        _c = " ".join((content or "").split())[:800]
        sys = (
            "你是项目记忆管理员（Claude Code Auto Memory 的写入裁决环节）。\n"
            "现有记忆索引如下（[type] 行）：\n"
            f"{index_lines or '（空）'}\n\n"
            f"待写入新记忆：type={type_} topic={topic}\n{_c}\n\n"
            "请裁决本次写入，只输出 JSON（不要解释）：\n"
            '{"action": "add" | "update" | "noop", '
            '"reason": "一句话原因", '
            '"topic": "合并用主题（update 时沿用旧主题或新主题，取更准确的）", '
            '"summary": "一行摘要（≤120 字，含关键结论）", '
            '"content": "精炼后的完整说明（≤300 字，症状/根因/修复 或 背景/结论）"}\n'
            "裁决规则：\n"
            "1) 能从项目代码/已有文档直接推导的（如『入口是 main.py』『用 conda 环境』）→ noop；\n"
            "2) 通用知识（『安装依赖需要网络』『GPU 比 CPU 快』）→ noop；\n"
            "3) 索引中已存在同主题或高度相似的记忆 → update（合并，保留历史有用信息）；\n"
            "4) 其余有信息增量的踩坑/命令/流程/偏好 → add。\n"
            "update 时 content 应尽量保留旧条目仍有效的信息并加入新信息；不确定旧内容时可用 '…' 概括。"
        )
        try:
            with tracing.span("llm", "remember_memory_filter", ""):
                resp = llm.invoke([SystemMessage(content=sys),
                                   HumanMessage(content="请裁决。")])
            text = resp.content if hasattr(resp, "content") else str(resp)
            m = re.search(r"\{[\s\S]*\}", text)
            dec = __import__("json").loads(m.group(0)) if m else {}
            action = str(dec.get("action", "noop")).strip().lower()
            if action == "noop":
                return f"[memory] 不写入（NOOP）：{dec.get('reason', '可从代码/文档推导或已存在')}。"
            if action == "update":
                t2 = str(dec.get("topic") or topic).strip()[:40]
                c2 = str(dec.get("content") or content).strip()
                s2 = str(dec.get("summary") or "").strip()
                # update：先读旧 topic 文件，保留其中仍有效的内容（若有且未包含在新内容里）
                old_path = _topic_path(type_, t2)
                old_raw = _read_topic(old_path)
                old_body = ""
                if old_raw:
                    _body_m = re.search(r"\n\n([\s\S]*)$", old_raw)
                    old_body = _body_m.group(1).strip() if _body_m else ""
                if old_body and old_body not in c2:
                    c2 = f"{c2}\n\n（历史记录）{old_body[:200]}"
                return remember(type_, t2, c2, summary=s2) and (
                    f"[memory] 更新条目（{type_}/{t2}），该区共 "
                    f"{len((_read_memory().get(type_) or []))} 条。")
            # add
            t2 = str(dec.get("topic") or topic).strip()[:40]
            c2 = str(dec.get("content") or content).strip()
            s2 = str(dec.get("summary") or "").strip()
            return remember(type_, t2, c2, summary=s2) and (
                f"[memory] 新增条目（{type_}/{t2}），该区共 "
                f"{len((_read_memory().get(type_) or []))} 条。")
        except Exception as e:
            # LLM 裁决失败 → 确定性回退（直接写入，不阻塞）
            action, count = remember(type_, topic, content)
            return f"[memory] 裁决失败回退 {action}（{type_}/{topic}），该区共 {count} 条（裁决异常: {str(e)[:60]}）。"

    return remember_memory


# ---------------------------------------------------------------------------
# 检索：recall_memory —— 按索引定位 topic 文件（CC「索引 + 按需读」落地）
# ---------------------------------------------------------------------------
@tool
def recall_memory(query: str, top_k: int = 3) -> str:
    """按问题/关键词从项目记忆索引定位并读取具体说明（topic 文件）。

    何时使用：你看到了注入的项目记忆索引（或凭经验知道某个坑/命令存在），
    需要完整细节（症状/根因/修复、环境参数）时调用——索引只有一行摘要，
    具体说明在 memory/<type>/<slug>.md，需要时按索引读取。

    Args:
        query: 问题或关键词（如 "CUDA 设备不一致"、"FFT 周期估计"、"ModuleNotFoundError"）。
        top_k: 最多返回的匹配条目数（默认 3）。
    Returns:
        命中的完整说明（标题 + 记录日期 + 正文）；无命中提示改用更具体关键词。
    """
    entries = _read_memory()
    hits = []  # (score, line)
    q = _norm_topic(query)
    q_tokens = [t for t in re.split(r"[\s,，。.;；:：()（）/\\\-_]+", q) if len(t) >= 2]
    for t in _SECTION_ORDER:
        for line in entries.get(t) or []:
            # 索引行含主题+摘要+路径，全部参与匹配
            score = 0
            if q and q in line.lower():
                score += 5
            for tok in q_tokens:
                if tok in line.lower():
                    score += 2
            if score > 0:
                hits.append((score, t, line))
    hits.sort(key=lambda x: -x[0])
    out = []
    for score, t, line in hits[:top_k]:
        path_m = re.search(r"（→ (.+?)）", line)
        path = path_m.group(1) if path_m else ""
        body = _read_topic(path) if path else ""
        topic_m = re.match(r"^- \*\*(.+?)\*\*", line)
        topic = topic_m.group(1) if topic_m else line
        out.append(f"### {topic}（type: {t}，score={score}）\n{body or '（topic 文件缺失，仅索引）'}")
    if not out:
        return (f"[memory] 索引中未匹配到「{query}」相关记忆。可尝试更具体的关键词"
                f"（如具体报错名、环境名、模块名），或确认该坑尚未被记录。")
    return "## 项目记忆 · 检索命中\n" + "\n\n".join(out)


# ---------------------------------------------------------------------------
# 注入：索引全量 + 新鲜度警告
# ---------------------------------------------------------------------------
def inject_context(max_chars: int = _INJECT_MAX_CHARS,
                   per_section: int | None = None) -> str:
    """生成注入文本：**全量索引**（每区全部条目，不带 topic 正文），超预算兜底截断。

    per_section 参数保留兼容旧调用（传 None 或省略 = 全量；旧调用方 innovation_tools
    传默认值即全量）。> 90 天的条目附「⚠️ 可能过时」。注入前先做存量迁移（幂等）。
    """
    _migrate_legacy()
    entries = _read_memory()
    today = _today()
    blocks = []
    for t in _SECTION_ORDER:
        items = entries.get(t) or []
        if not items:
            continue
        blocks.append(f"【{SECTIONS[t]}】")
        for line in items:
            date_m = re.search(r"（(\d{4}-\d{2}-\d{2})）：", line)
            warn = ""
            if date_m:
                try:
                    days = (datetime.strptime(today, "%Y-%m-%d")
                            - datetime.strptime(date_m.group(1), "%Y-%m-%d")).days
                    if days > _STALE_DAYS:
                        warn = " ⚠️可能过时"
                except Exception:
                    pass
            blocks.append(line + warn)
    text = "\n".join(blocks).strip()
    if not text:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return f"## 项目记忆索引（CC 风格，跨会话沉淀；细节用 recall_memory 读取）\n{text}"


def read_memory_for_inject() -> str:
    """供 agent 首步全局注入的完整段落（索引全量，预算截断）。"""
    return inject_context()


# ---------------------------------------------------------------------------
# 整理：consolidate（Auto Dream）—— 索引 + topic 重组，更新 metadata
# ---------------------------------------------------------------------------
def _reconcile_topics(entries: dict) -> int:
    """整理后把索引行指向的 topic 文件补全（摘要/路径与文件不一致时以索引为准重建文件）。"""
    n = 0
    for t in _SECTION_ORDER:
        for line in entries.get(t) or []:
            m = re.match(r"^- \*\*(.+?)\*\*（(\d{4}-\d{2}-\d{2})）：(.+?)（→ (.+?)）", line)
            if not m:
                continue
            topic, ts, summary, path = m.group(1), m.group(2), m.group(3), m.group(4)
            if not _read_topic(path):
                _write_topic(path, f"# {topic}\n\n> 记录于 {ts} ｜ type: {t}\n\n{summary}\n")
                n += 1
    return n


def make_consolidate_memory(llm):
    @tool
    def consolidate_memory(
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
    ) -> Command:
        """整理项目记忆（Claude "Dreams" 模式）：LLM 重组摘要索引并合并 topic 文件。

        适用：记忆索引条目变多、出现重复/过时/矛盾时手动或由模型触发，或会话后自动整理。
        工具把当前 MEMORY.md 索引全文交给 LLM 重组：合并重复条目、用最新信息替换过时条目、
        保持分区与索引行格式（- **主题**（日期）：摘要（→ 路径）），只输出整理后的索引并写回，
        随后按索引补全 topic 文件。原始审计日志 EXPERIENCE.md 不受影响。
        """
        raw = _read_raw()
        if not raw.strip():
            return Command(update={"messages": [ToolMessage(
                content="MEMORY.md 尚无内容，暂无需整理。",
                tool_call_id=tool_call_id, status="error")]})
        entries = _read_memory()
        total_before = sum(len(v) for v in entries.values())
        sys = (
            "你是项目记忆管理员（Claude Code Auto Memory 的整理环节）。\n"
            "整理下面的项目记忆索引文件：\n"
            "1) 合并重复/相似条目（保留信息最全的一条，主题取更准确的）；\n"
            "2) 用更新、更准确的表述替换过时或有误的条目（保持该行日期为最近记录日期）；\n"
            "3) 保持分区标题（## 踩坑经验（pitfall）等）与索引行格式 "
            "`- **主题**（YYYY-MM-DD）：一行摘要（→ memory/<type>/<slug>.md）` 不变；\n"
            "4) 不丢失任何仍有价值的信息，条数只减不增（或持平）；\n"
            "5) 每条索引行必须保留（→ memory/<type>/<slug>.md）路径，slug 用主题归一化后的短横线形式；\n"
            "6) 只输出整理后的完整 markdown，不要解释。"
        )
        with tracing.span("llm", "consolidate_memory", state.get("session_id", "")):
            resp = llm.invoke([SystemMessage(content=sys),
                               HumanMessage(content=f"当前 MEMORY.md：\n\n{raw}")])
        text = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r"```(?:markdown|md)?\s*([\s\S]*?)```", text)
        out = m.group(1).strip() if m else text.strip()
        if not out.startswith("#"):
            out = _HEADER + "\n" + out
        entries2 = _parse(out)
        total_after = sum(len(v) for v in entries2.values())
        out = _render(entries2, meta=f"> last_consolidated: {_today()}\n"
                                     f"> entries_at_consolidation: {total_after}\n")
        _write_raw(out)
        _reconcile_topics(entries2)
        return Command(update={"messages": [ToolMessage(
            content=f"[memory] 整理完成：{total_before} → {total_after} 条"
                    f"（分区：" + "、".join(f"{k}={len(v)}" for k, v in entries2.items() if v) + "）。",
            tool_call_id=tool_call_id)]})

    return consolidate_memory


# ---------------------------------------------------------------------------
# 自动整理：会话后双门限（Auto Dream 落地）
# ---------------------------------------------------------------------------
_AUTO_CONS_MIN_INTERVAL_H = 24     # 距上次整理 ≥ 24h
_AUTO_CONS_MIN_NEW = 3             # 且 新增索引条数 ≥ 3


def _consolidation_state() -> tuple[str | None, int | None, int]:
    """返回 (last_consolidated, entries_at_consolidation, 当前索引条数)。"""
    raw = _read_raw()
    entries = _read_memory()
    cur = sum(len(v) for v in entries.values())
    lm = _META_RE.search(raw or "")
    cm = _META_CNT_RE.search(raw or "")
    return (lm.group(1) if lm else None,
            int(cm.group(1)) if cm else None,
            cur)


def maybe_auto_consolidate(llm) -> str:
    """会话结束后调用：双门限满足则自动整理记忆（Auto Dream 模式）。

    双门限（CC 社区逆向）：距上次整理 > 24h 且 新增索引条数 ≥ 3（无整理历史时
    以总条数 ≥ 8 触发）。LLM 不可用时跳过（不阻塞主流程）。
    """
    try:
        last, at_cons, cur = _consolidation_state()
        if cur == 0:
            return ""
        if last:
            try:
                days = (datetime.strptime(_today(), "%Y-%m-%d")
                        - datetime.strptime(last, "%Y-%m-%d")).days
            except Exception:
                days = 999
            new_entries = max(0, cur - (at_cons if at_cons is not None else cur))
            if not (days >= 1 and new_entries >= _AUTO_CONS_MIN_NEW):
                return ""
        else:
            if cur < 8:
                return ""
        # 触发：调用整理工具逻辑（复用 consolidate 的核心，不走 tool Command）
        consolidate = make_consolidate_memory(llm)
        # 简化：直接调用内部整理（模拟 tool 输入）
        entries = _read_memory()
        total_before = sum(len(v) for v in entries.values())
        raw = _read_raw()
        sys = (
            "你是项目记忆管理员（Claude Code Auto Memory 的自动整理环节）。\n"
            "整理下面的项目记忆索引文件：\n"
            "1) 合并重复/相似条目（保留信息最全的一条）；\n"
            "2) 用更新、更准确的表述替换过时或有误的条目（保持日期为最近记录日期）；\n"
            "3) 保持分区标题与索引行格式 `- **主题**（YYYY-MM-DD）：一行摘要（→ memory/<type>/<slug>.md）` 不变；\n"
            "4) 不丢失任何仍有价值的信息，条数只减不增（或持平）；\n"
            "5) 每条索引行必须保留（→ memory/<type>/<slug>.md）路径；\n"
            "6) 只输出整理后的完整 markdown，不要解释。"
        )
        with tracing.span("llm", "auto_consolidate_memory", ""):
            resp = llm.invoke([SystemMessage(content=sys),
                               HumanMessage(content=f"当前 MEMORY.md：\n\n{raw}")])
        text = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r"```(?:markdown|md)?\s*([\s\S]*?)```", text)
        out = m.group(1).strip() if m else text.strip()
        if not out.startswith("#"):
            out = _HEADER + "\n" + out
        entries2 = _parse(out)
        total_after = sum(len(v) for v in entries2.values())
        _write_raw(_render(entries2, meta=f"> last_consolidated: {_today()}\n"
                                          f"> entries_at_consolidation: {total_after}\n"))
        _reconcile_topics(entries2)
        return f"自动整理触发：{total_before} → {total_after} 条（距上次 {last or '无'}，新增 {cur - (at_cons if at_cons is not None else 0)}）"
    except Exception as e:
        return f"自动整理跳过：{str(e)[:60]}"


# ---------------------------------------------------------------------------
# 存量迁移：旧格式（条目内联全文）→ 索引 + topic 文件
# ---------------------------------------------------------------------------
def _migrate_legacy() -> bool:
    """把旧格式 MEMORY.md（`- **主题**：内容`，无路径）迁移为索引 + topic 文件。幂等。

    检测：任一索引行不带「（→ memory/…」即旧格式 → 拆成 topic 文件并重写索引。
    返回是否发生了迁移。
    """
    raw = _read_raw()
    if not raw:
        return False
    lines = [l for l in raw.splitlines() if l.strip().startswith("- ")]
    if not lines:
        return False
    if all("（→ memory/" in l for l in lines):
        return False  # 已是新格式
    # 旧条目内容保留在 topic 文件（摘要 = 原内容前 60 字）
    entries = _parse(raw)
    migrated = 0
    for t in _SECTION_ORDER:
        new_items = []
        for line in entries.get(t) or []:
            if "（→ memory/" in line:
                new_items.append(line)
                continue
            m = re.match(r"^- \*\*(.+?)\*\*：(.+)$", line)
            if not m:
                new_items.append(line)
                continue
            topic, content = m.group(1).strip(), m.group(2).strip()
            action, _ = remember(t, topic, content, _migrate=False)  # 防递归
            migrated += 1 if action == "added" else 0
    if migrated:
        print(f"--> [cc_memory] 迁移 {migrated} 条旧格式记忆为「索引 + topic 文件」")
    return migrated > 0

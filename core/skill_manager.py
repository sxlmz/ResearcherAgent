"""
skill_manager.py
================
原子化 Skill 体系 —— 每个 Skill 只做一件事，模型判断此次请求需要哪些 Skill，
组合由 LLM 在 parse 时决定（模型擅长的），调度器只验证前提条件（确定性的）。

设计原则：
1. 原子性 —— 一个 Skill = 一种能力，不隐式捆绑其他能力
2. 前提条件 —— 声明式，调度器只检查条件是否满足，不做启发式猜测
3. 按需组合 —— LLM 输出 skills 列表，不是单选标签
"""

import re
from pathlib import Path
from typing import Optional

# ==========================================
# 0. 原子 Skill 注册表
# ==========================================
SKILL_REGISTRY = {
    # ── 输入类：从外部获取信息 ──
    "search_papers": {
        "group": "input",
        "description": "调用 arXiv / PubMed 等学术数据库检索论文，返回论文列表和摘要。"
                       "适用：用户需要查找特定主题的最新文献。",
        "precondition": None,  # 无前提条件
    },
    "index_file": {
        "group": "input",
        "description": "解析用户上传的文件（PDF/DOCX/TXT等），切分为语义块并建立 RAG 索引，"
                       "供后续回答或写作时引用文件内容。"
                       "适用：用户上传了文件且需要在回答或综述中参考该文件。",
        "precondition": "has_file",  # 前提：当前对话已上传文件
    },

    # ── 生成类：产出文本内容 ──
    "answer_direct": {
        "group": "generate",
        "description": "用 LLM 自身知识 + 全局记忆档案（SOUL/USER/MEMORY.md）直接回答用户问题。"
                       "不依赖论文检索结果，也不依赖上传文件。"
                       "适用：普通问答、概念解释、闲聊——不需要查论文也不需要看文件。",
        "precondition": None,
    },
    "answer_from_context": {
        "group": "generate",
        "description": "基于检索到的论文数据或上传文件的内容回答问题。"
                       "适用于总结、分析、提炼已有材料——但不写综述。"
                       "适用：「这篇论文讲了什么」「这些论文用了什么方法」「总结一下这个文件」。",
        "precondition": None,  # context 在运行时产生，parse 时无法确定——validator 不硬拦
    },
    "write_review": {
        "group": "generate",
        "description": "撰写一篇完整的结构化文献综述，包含引言、方法分类对比、展望和参考文献。"
                       "需要论文数据或文件内容作为素材来源。"
                       "适用：「写一篇关于XX的综述」「做一篇XX领域的文献综述」。",
        "precondition": None,  # 同 answer_from_context——素材在运行时产生
    },

    # ── 质量类：评审验证 ──
    "judge_review": {
        "group": "quality",
        "description": "用独立评委 LLM 对综述从 6 个维度（结构完整性、内容深度、逻辑严谨性、"
                       "学术规范性、即时约束遵守度、记忆偏好遵守度）打分。低于 8.5 分触发自动重写（最多 3 轮）。"
                       "前提：只有请求里已有 write_review 时才选这个。——问「这篇综述讲了什么」时不要选。",
        "precondition": "paired_with_write_review",
    },

    # ── 执行类：运行代码（写代码/跑训练/执行命令）──
    "code_exec": {
        "group": "execution",
        "description": "在工作区沙箱中执行代码：写/读文件、跑 shell 命令、用 pytorch 环境运行训练脚本。"
                       "适用：用户要求写代码实现、按论文方法复刻代码、跑训练/推理/实验脚本、"
                       "调试代码、安装依赖、查看或整理工作区文件。",
        "precondition": None,
    },
    "innovate": {
        "group": "execution",
        "description": "文献→创新点验证流水线：按主题检索论文并精读制卡（复用 search_papers 检索与 "
                       "read_papers 精读卡片机制），LLM 提出 3-5 个潜在创新点，为创新点生成实现代码"
                       "并在沙箱中运行实验，与基线模型（默认演示：时序领域为 PatchTST，其他领域按主题"
                       "从精读文献的常用对比模型中选取）对比精度，判断创新点是否可行。"
                       "适用：用户要求「基于这些论文提出创新点并测试」「从文献里找改进方向并跑实验"
                       "验证」「提出 3-5 个创新点看有没有提升」「文献综述→创新点验证」。",
        "precondition": None,
    },
}

# ==========================================
# 1. 低熵索引（注入 system prompt）
# ==========================================

def build_skill_index(has_file: Optional[bool] = None) -> str:
    """构建原子能力索引，注入 parse 节点的 system prompt。

    与旧版的关键区别：不再让模型在 5 个「套餐」里单选——
    模型输出一个 skills 列表，说明「这个请求需要哪些原子能力」。

    Args:
        has_file: 当前对话是否已上传文件。传入后会在索引顶部固定这一事实。
    """
    lines = []

    # ---- 上下文事实 ----
    if has_file is True:
        lines.append("## 🔒 当前上下文：**用户已上传文件**")
        lines.append("→ 如果有 `index_file` 的需求，前提条件已满足。")
        lines.append("")
    elif has_file is False:
        lines.append("## 🔒 当前上下文：**没有上传文件**")
        lines.append("→ `index_file` 不可用（前提条件不满足），不要选它。")
        lines.append("")

    # ---- 原子能力列表（按 group 分组） ----
    lines.append("## 可用原子能力（按需选择多个，不互斥）")
    lines.append("")
    groups = {"input": "📥 输入类（获取信息）", "generate": "✍️ 生成类（产出内容）", "quality": "⚖️ 质量类（评审验证）", "execution": "⚙️ 执行类（运行代码）"}
    for group_key, group_label in groups.items():
        lines.append(f"### {group_label}")
        for name, skill in SKILL_REGISTRY.items():
            if skill["group"] != group_key:
                continue
            precond = skill["precondition"] or "无"
            lines.append(f"- **`{name}`**：{skill['description']}")
            lines.append(f"  前提条件：{precond}")
        lines.append("")

    # ---- 组合示例（教模型如何搭配） ----
    lines.append("## 典型组合示例")
    lines.append("- 简单问答（模型有把握） → `[\"answer_direct\"]`")
    lines.append("- 检索论文 + 基于结果回答 → `[\"search_papers\", \"answer_from_context\"]`")
    lines.append("- 检索论文 + 写综述 + 评委 → `[\"search_papers\", \"write_review\", \"judge_review\"]`")
    lines.append("- 上传文件 + 总结/问答 → `[\"index_file\", \"answer_from_context\"]`")
    lines.append("- 上传文件 + 按格式写综述 + 评委 → `[\"index_file\", \"write_review\", \"judge_review\"]`")
    lines.append("- 写代码 / 复刻论文方法 / 跑训练 → `[\"code_exec\"]`（可加 `search_papers` 先查文献）")
    lines.append("- 文献综述→创新点验证：检索精读 + 提出创新点 + 跑实验对比基线 → `[\"innovate\"]`（可加 `search_papers`/`code_exec`）")
    lines.append("")
    lines.append("⚠️ 重要：只要用户上传了文件（上下文提示「已上传文件」），任何需要文件内容的请求都必须包含 `index_file`。")
    lines.append("文件刚上传时尚未索引，跳过 `index_file` 会导致系统找不到文件内容。")

    # ---- 关键区分：写综述 vs 问内容 ----
    lines.append("## ⚠️ 最重要的一条区分（写 vs 问）")
    lines.append("- 用户说「写一篇XX综述 / 做一篇文献综述」→ 必须有 `write_review`（可能还带 `judge_review`）")
    lines.append("- 用户说「这篇综述讲了什么 / 主要观点是什么 / 写了哪些方法」→ 是**内容询问**，只需 `answer_from_context`")
    lines.append("- 用户说「总结/概括/归纳一下这个文件」→ 是**内容询问**，只需 `answer_from_context`")
    lines.append("- **不要**给内容询问型请求加 `write_review` 或 `judge_review`——用户不是要写新综述，是在问已有内容。")
    lines.append("")

    # ---- 关键区分：直接回答 vs 检索 ----
    lines.append("## ⚠️ 何时检索（search_papers）")
    lines.append("- 学术论文检索用 `search_papers`（不是通用网页搜索——本系统无 search_web 能力）")
    lines.append("- 纯概念解释、常识、推理、闲聊，且你有把握 → 只用 `answer_direct`，不必检索")
    lines.append("")

    lines.append("## 你的任务边界")
    lines.append("你的工作是**理解并拆分用户的问题**，不是替用户补全需求。")
    lines.append("⚠️ **绝大多数用户只会说一句「帮我写一篇关于 X 的综述」，没有附带任何额外要求**——")
    lines.append("这是**完全正常**的输入，此时 `user_constraints` 的全部字段都应为空/null。")
    lines.append("**只提取用户原话中明确写出的要求**。用户没说的，一律留空；")
    lines.append("不要替用户设想「应该要摘要」「应该控制在几千字」「应该包含参考文献」——")
    lines.append("这些是学术写作的通用规范，由后续写作环节按文体惯例处理，**不属于用户的即时约束**。")
    lines.append("把通用规范写进 user_constraints 会让下游误以为「用户要求过」，进而按编造的规格评分。")
    lines.append("")

    lines.append("## 输出 JSON")
    lines.append("{{")
    lines.append('  "skills": ["search_papers", "write_review", "judge_review"],')
    lines.append('  "topic": "综述主题（answer_direct 时可为空）",')
    lines.append('  "search_query": "检索关键词（英文，search_papers 需要；从用户请求中提取核心关键词而非完整句子，用空格分隔；answer_direct 时可为空）",')
    lines.append('  "user_constraints": {{')
    lines.append('      "word_budget": null,            // 用户明说篇幅时填 [下限, 上限]（单位：字），否则 null')
    lines.append('      "required_parts": [],           // 用户点名要包含的部件（如 ["摘要","对比表格"]），没点名则空数组')
    lines.append('      "sections": [],                 // 用户点名要覆盖的章节/方向，没点名则空数组')
    lines.append('      "citation_style": null,         // 用户明说引用体例时填 "numeric" 或 "author-year"，否则 null')
    lines.append('      "language": null,               // 用户明说语言时填，否则 null')
    lines.append('      "other": []                     // 用户明说的其它要求，逐条原文摘录，没有则空数组')
    lines.append('  }},')
    lines.append('  "save_path": "保存路径（无则为空）",')
    lines.append('  "review_structure": "用户明确指定的综述章节结构（无则为空）"')
    lines.append("}}")
    lines.append("只输出 JSON。再强调一次：**空字段是正常且预期的结果**。")
    lines.append("举例：用户只说「帮我写一篇关于扩散模型的综述」→ "
                 '`"user_constraints": {{"word_budget": null, "required_parts": [], "sections": [], '
                 '"citation_style": null, "language": null, "other": []}}`')

    return "\n".join(lines)


# ==========================================
# 2. 调度器：只做前提条件校验
# ==========================================

def validate_skills(
    skills: list[str],
    has_file: bool,
) -> tuple[list[str], str]:
    """校验 LLM 输出的 skills 列表的合法性。

    只做确定性的前提条件检查——不做启发式猜测，不穷举每种用户句式。
    模型负责组合（上限高），调度器负责验证（下限稳）。

    校验规则：
    1. 过滤掉不在注册表中的未知 skill
    2. index_file 的前提条件：has_file 必须为 True
    3. judge_review 的前提条件：skills 中必须有 write_review（否则去掉）
    4. answer_direct 与需要外部信息的 skill 冲突 ——
       后者需要运行时上下文，如果同时选了就保留后者（更需要外部信息的一方）
    5. 有输入类 skill（search_papers/index_file）但没有任何生成类 skill →
       兜底补 answer_from_context（否则抓/查了信息却没人合成答案）
    6. 按执行顺序排列（search_* → index_file → answer*/write → judge）

    Returns:
        (ordered_valid_skills, reason)
    """
    skills = [s.strip() for s in skills if s and s.strip() in SKILL_REGISTRY]

    if not skills:
        return (["answer_direct"], "空列表 → 兜底为 answer_direct")

    # ---- 校验 1: 过滤未知 skill ----
    known = [s for s in skills if s in SKILL_REGISTRY]
    dropped = set(skills) - set(known)

    # ---- 校验 2: index_file 前提 = has_file ----
    if "index_file" in known and not has_file:
        known.remove("index_file")
        dropped.add("index_file")

    # ---- 校验 2b: has_file + 需要文件内容 → 兜底补 index_file ----
    # 模型有时会漏选 index_file（以为文件已索引），调度器确保第一次对话能正确索引。
    if has_file and "index_file" not in known:
        needs_file_content = "answer_from_context" in known or "write_review" in known
        if needs_file_content:
            known.append("index_file")
            print("--> [Scheduler] 自动补 index_file（有文件且需要文件内容）")

    # ---- 校验 3: judge_review 前提 = 需要 write_review ----
    if "judge_review" in known and "write_review" not in known:
        known.remove("judge_review")
        dropped.add("judge_review")

    # ---- 校验 4: answer_direct 冲突消解 ----
    # answer_direct 意味着"不需要外部信息"。如果同时选了需要外部信息的 skill，
    # 说明模型矛盾了——保留后者（需要 paper/file 的那些），去掉 answer_direct。
    needs_context = any(s in known for s in
                        ("answer_from_context", "write_review", "search_papers", "index_file"))
    if "answer_direct" in known and needs_context:
        known.remove("answer_direct")
        dropped.add("answer_direct")

    # ---- 校验 5: 有输入无生成 → 兜底补 answer_from_context ----
    # 如果选了输入类（检索/索引）但没有任何生成类 skill，信息拿到了却没人合成答案。
    has_input = any(s in known for s in ("search_papers", "index_file"))
    has_generate = any(s in known for s in ("answer_direct", "answer_from_context", "write_review"))
    if has_input and not has_generate:
        known.append("answer_from_context")
        print("--> [Scheduler] 自动补 answer_from_context（有输入 skill 但缺生成 skill）")

    # ---- 校验 6: 按执行顺序排列 ----
    ORDER = ["search_papers", "index_file", "answer_direct", "answer_from_context", "write_review", "judge_review", "code_exec", "innovate"]
    ordered = [s for s in ORDER if s in known]
    # 把不在 ORDER 里的（如果有）追加到末尾
    ordered += [s for s in known if s not in ORDER]

    if not ordered:
        ordered = ["answer_direct"]

    reason_parts = []
    if dropped:
        reason_parts.append(f"移除: {', '.join(sorted(dropped))}")
    reason_parts.append(f"最终: {ordered}")
    reason = " | ".join(reason_parts)

    return (ordered, reason)


# ==========================================
# 3. 便捷查询（供路由函数使用）
# ==========================================

def needs_search_papers(skills: list[str]) -> bool:
    return "search_papers" in skills


def needs_tools(skills: list[str]) -> bool:
    """是否需要走 agent 工具循环（论文检索 / 代码执行 / 创新实验）"""
    return ("search_papers" in skills or "code_exec" in skills
            or "innovate" in skills)


def needs_index_file(skills: list[str]) -> bool:
    return "index_file" in skills


def needs_write_review(skills: list[str]) -> bool:
    return "write_review" in skills

def needs_content_safety(skills: list[str]) -> bool:
    return "content_safety" in skills


def has_input_skill(skills: list[str]) -> bool:
    """是否有输入类 skill（检索论文/索引文件）——需要先从外部获取信息"""
    return any(s in skills for s in ("search_papers", "index_file"))


# ==========================================
# 4. 渐进式披露（L2）：SKILL.md 正文加载
# ==========================================
# 对齐 Anthropic Agent Skills 规范：
#   L1 元数据（name+description）常驻 parse prompt（build_skill_index）
#   L2 SKILL.md 正文在「触发」（skills 被选中）后注入执行节点（build_skill_instructions）
#   L3 资源（references/scripts）按需加载：
#       - references/：详细规格，SKILL.md 正文用「read_file 读取」指针按需加载（模型驱动）
#       - scripts/：确定性操作实现，运行时不进 LLM 上下文（框架/独立运行调用）
# 这样流程不再写死在 graph 节点里，改技能流程 = 改 SKILL.md，不动代码。

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"  # ResearcherAgent/skills

_skill_body_cache: dict[str, str] = {}


def _read_skill_md_body(name: str) -> str:
    """读 skills/<name>/SKILL.md 的正文（frontmatter 之后的部分）。

    frontmatter 用「首行 --- 到下一个 ---」解析；正文为剩余 markdown。
    文件缺失或解析失败返回 ""（调用方回退到代码内最小逻辑，不中断流程）。
    """
    if name in _skill_body_cache:
        return _skill_body_cache[name]
    body = ""
    p = SKILLS_DIR / name / "SKILL.md"
    try:
        raw = p.read_text(encoding="utf-8")
        m = re.match(r"(?s)^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)$", raw)
        body = m.group(2).strip() if m else raw.strip()
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        body = ""
    _skill_body_cache[name] = body
    return body


def load_skill_bodies(skills: list[str]) -> dict[str, str]:
    """选中 skills 的 SKILL.md 正文（渐进式披露 L2）。

    Returns:
        {skill_name: 正文}，只含注册表内且正文存在的 skill；缺失项不出现。
    """
    return {s: _read_skill_md_body(s) for s in skills if s in SKILL_REGISTRY}


def list_skill_resources(name: str) -> list[str]:
    """列 skills/<name>/references/ 下的文件（L3 资源清单，供提示词生成指针）。"""
    d = SKILLS_DIR / name / "references"
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_file() and p.suffix.lower() in (".md", ".txt"))


def load_skill_references(name: str) -> str:
    """读 skills/<name>/references/ 全部资源内容（框架代读，按需注入场景用）。

    Returns:
        拼接的 markdown；无 references 目录或读取失败返回 ""。
    """
    files = list_skill_resources(name)
    if not files:
        return ""
    parts = []
    for f in files:
        try:
            text = (SKILLS_DIR / name / "references" / f).read_text(encoding="utf-8")
            parts.append(f"## 参考资料 {name}/references/{f}\n\n{text.strip()}")
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            continue
    return "\n\n".join(parts)


def build_skill_instructions(skills: list[str]) -> str:
    """渲染「已选技能执行规范」区块，注入 agent / write / judge / safety 等节点 system prompt。

    每个被选中的 skill 若存在 SKILL.md，其正文即为该能力的标准操作流程——
    LLM 按此执行，而不是依赖代码里写死的步骤描述。无正文时返回空串。
    """
    bodies = load_skill_bodies(skills)
    parts = []
    for s in skills:
        b = bodies.get(s)
        if b:
            parts.append(f"## 技能执行规范 · {s}\n{b}")
    if not parts:
        return ""
    return ("\n\n### 📋 已选技能执行规范（严格按各技能 SKILL.md 的流程与规则执行，"
            "与本提示其他内容冲突时以技能规范为准）\n\n" + "\n\n".join(parts))

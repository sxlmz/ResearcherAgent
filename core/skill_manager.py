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
    "search_web": {
        "group": "input",
        "description": "联网抓取网页内容获取实时/最新信息。"
                       "适用：需要时效性信息（新闻、最新进展、当前状态）、模型自身知识不确定或可能过时、"
                       "用户明确要求联网查询、或问题涉及具体网站/链接的内容。"
                       "注意：这是通用网页信息，不是学术论文检索（那是 search_papers）。",
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
                       "学术规范性、即时约束遵守度、记忆偏好遵守度）打分。低于 7 分触发自动重写（最多 3 轮）。"
                       "前提：只有请求里已有 write_review 时才选这个。——问「这篇综述讲了什么」时不要选。",
        "precondition": "paired_with_write_review",
    },
    "content_safety": {
        "group": "quality",
        "description": "对生成的内容进行安全合规审核（政治合规、用语文明、信息真实性、客观中立）。"
                       "低于 7 分触发自动重写（最多 3 轮）。"
                       "适用：用户要求就事件发表评论/观点、生成公开可传播的文本（评论文、分析文、新闻稿等）。"
                       "纯粹的知识问答、文献综述（已有 judge_review）、文件总结不需要这个。",
        "precondition": "paired_with_content_generation",
        # 校验时：必须有 answer_from_context 或 write_review 才生效
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
                       "并在沙箱中运行实验，与基线模型（PatchTST）对比精度，判断创新点是否可行。"
                       "适用：用户要求「基于这些论文提出创新点并测试」「从文献里找改进方向并跑实验"
                       "验证」「提出 3-5 个创新点看有没有提升」「文献综述→创新点验证」。",
        "precondition": None,
    },

    # ── 元技能：控制流程（LLM 不选，调度器自动补）──
    "human_review": {
        "group": "meta",
        "description": "在检索完成后暂停，让用户审核找到的资料（标题+摘要），"
                       "用户可接受、要求重新检索（带补充说明）、或跳过审核直接生成。"
                       "适用：需要基于检索结果撰写文稿时（write_review + 输入类 skill 同时存在）。"
                       "调度器自动补充，LLM 无需手动选择。",
        "precondition": "paired_with_write_review_and_input",
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
    lines.append("- 联网查实时信息 + 基于结果回答 → `[\"search_web\", \"answer_from_context\"]`")
    lines.append("- 检索论文 + 基于结果回答 → `[\"search_papers\", \"answer_from_context\"]`")
    lines.append("- 检索论文 + 写综述 + 评委 → `[\"search_papers\", \"write_review\", \"judge_review\"]`")
    lines.append("- 上传文件 + 总结/问答 → `[\"index_file\", \"answer_from_context\"]`")
    lines.append("- 上传文件 + 按格式写综述 + 评委 → `[\"index_file\", \"write_review\", \"judge_review\"]`")
    lines.append("- 联网查事件 + 写评论 + 安全审核 → `[\"search_web\", \"write_review\", \"content_safety\"]`")
    lines.append("- 写代码 / 复刻论文方法 / 跑训练 → `[\"code_exec\"]`（可加 `search_papers` 先查文献）")
    lines.append("- 文献综述→创新点验证：检索精读 + 提出创新点 + 跑实验对比基线 → `[\"innovate\"]`（可加 `search_papers`/`code_exec`）")
    lines.append("- 联网查信息 + 回答 + 安全审核 → `[\"search_web\", \"answer_from_context\", \"content_safety\"]`")
    lines.append("")
    lines.append("⚠️ 重要：只要用户上传了文件（上下文提示「已上传文件」），任何需要文件内容的请求都必须包含 `index_file`。")
    lines.append("文件刚上传时尚未索引，跳过 `index_file` 会导致系统找不到文件内容。")

    # ---- 何时需要安全审核 ----
    lines.append("## ⚠️ 何时需要 content_safety（安全审核）")
    lines.append("- 用户要求**就某个事件/话题发表评论、写评论文章、写观点** → 加 `content_safety`")
    lines.append("- 用户生成的是**公开可传播的文本**（评论文、分析文、新闻稿、公告）→ 加 `content_safety`")
    lines.append("- 纯粹的知识问答、文献综述、文件总结 → **不需要** `content_safety`")
    lines.append("- 注意：`judge_review` 管学术质量，`content_safety` 管内容安全，两者不互斥，可同时使用")

    # ---- 关键区分：写综述 vs 问内容 ----
    lines.append("## ⚠️ 最重要的一条区分（写 vs 问）")
    lines.append("- 用户说「写一篇XX综述 / 做一篇文献综述」→ 必须有 `write_review`（可能还带 `judge_review`）")
    lines.append("- 用户说「这篇综述讲了什么 / 主要观点是什么 / 写了哪些方法」→ 是**内容询问**，只需 `answer_from_context`")
    lines.append("- 用户说「总结/概括/归纳一下这个文件」→ 是**内容询问**，只需 `answer_from_context`")
    lines.append("- **不要**给内容询问型请求加 `write_review` 或 `judge_review`——用户不是要写新综述，是在问已有内容。")
    lines.append("")

    # ---- 关键区分：直接回答 vs 联网 ----
    lines.append("## ⚠️ 何时联网（search_web）")
    lines.append("- 问题涉及**时效性**（今天/最近/最新/当前价格/近期新闻/版本号）→ 加 `search_web`")
    lines.append("- 你对答案**不确定**、或信息可能已过时（人物现状、产品动态、事件结果）→ 加 `search_web`")
    lines.append("- 用户明确要求「联网 / 上网查 / 搜一下 / 查查最新」→ 加 `search_web`")
    lines.append("- 问题涉及某个具体网址/网站内容 → 加 `search_web`")
    lines.append("- 纯概念解释、常识、推理、闲聊，且你有把握 → 只用 `answer_direct`，不必联网")
    lines.append("- 学术论文检索用 `search_papers`，不是 `search_web`；两者可同时选")
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
    lines.append('  "search_query": "检索关键词（英文，search_papers 和 search_web 都需要；从用户请求中提取核心关键词而非完整句子，用空格分隔；answer_direct 时可为空）",')
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
    5. 有输入类 skill（search_web/search_papers/index_file）但没有任何生成类 skill →
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

    # ---- 校验 3b: content_safety 前提 = 需要内容生成 skill ----
    if "content_safety" in known:
        has_content_gen = "answer_from_context" in known or "write_review" in known
        if not has_content_gen:
            known.remove("content_safety")
            dropped.add("content_safety")

    # ---- 校验 4: answer_direct 冲突消解 ----
    # answer_direct 意味着"不需要外部信息"。如果同时选了需要外部信息的 skill，
    # 说明模型矛盾了——保留后者（需要 web/paper/file 的那些），去掉 answer_direct。
    needs_context = any(s in known for s in
                        ("answer_from_context", "write_review", "search_papers", "search_web", "index_file"))
    if "answer_direct" in known and needs_context:
        known.remove("answer_direct")
        dropped.add("answer_direct")

    # ---- 校验 5: 有输入无生成 → 兜底补 answer_from_context ----
    # 如果选了输入类（抓取/检索/索引）但没有任何生成类 skill，信息拿到了却没人合成答案。
    has_input = any(s in known for s in ("search_web", "search_papers", "index_file"))
    has_generate = any(s in known for s in ("answer_direct", "answer_from_context", "write_review"))
    if has_input and not has_generate:
        known.append("answer_from_context")
        print("--> [Scheduler] 自动补 answer_from_context（有输入 skill 但缺生成 skill）")

    # ---- 校验 6: 人机审核门 —— write_review + 输入类 skill 时自动插入 ----
    # 仅当需要"写文稿"且从外部获取信息时才触发。普通问答、纯直接回答不触发。
    if "write_review" in known and has_input_skill(known) and "human_review" not in known:
        w_idx = known.index("write_review")
        known.insert(w_idx, "human_review")
        print("--> [Scheduler] 自动补 human_review（write_review + 输入类 skill → 审核门）")

    # ---- 校验 7: 按执行顺序排列 ----
    ORDER = ["search_web", "search_papers", "index_file", "answer_direct", "answer_from_context", "human_review", "write_review", "judge_review", "content_safety", "code_exec", "innovate"]
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
    """是否需要走 agent 工具循环（联网抓取 / 论文检索 / 代码执行 / 创新实验）"""
    return ("search_web" in skills or "search_papers" in skills or "code_exec" in skills
            or "innovate" in skills)


def needs_index_file(skills: list[str]) -> bool:
    return "index_file" in skills


def needs_write_review(skills: list[str]) -> bool:
    return "write_review" in skills

def needs_content_safety(skills: list[str]) -> bool:
    return "content_safety" in skills


def needs_human_review(skills: list[str]) -> bool:
    """是否需要人机审核门（检索结果给用户过目）"""
    return "human_review" in skills


def has_input_skill(skills: list[str]) -> bool:
    """是否有输入类 skill（联网/检索论文/索引文件）——需要先从外部获取信息"""
    return any(s in skills for s in ("search_web", "search_papers", "index_file"))

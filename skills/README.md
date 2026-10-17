# ResearcherAgent Skills（项目能力规范）

本目录把 `ResearcherAgent` 的 8 个原子能力规范化为 **Anthropic Agent Skills 标准目录包**（`skill-name/{SKILL.md + references/ + scripts/ + examples/}`），并已**接入运行时**（渐进式披露 L1/L2/L3）：

- **L1 元数据**（name+description）：`core/skill_manager.py` `build_skill_index` 渲染进 parse prompt，LLM 判断并选出 skills。
- **L2 正文注入**：`core/skill_manager.py` `build_skill_instructions(skills)` 读取 `skills/<name>/SKILL.md` 正文，注入 agent 循环 / write / judge 节点及 innovate 工具内部 LLM 调用的 system prompt——**流程不再写死在代码里**，改技能流程 = 改 SKILL.md。
- **L3 资源按需加载**：详细规格放 `references/`（SKILL.md 正文给出指针，模型需要细节时用 `read_file` 读取）；确定性操作放 `scripts/`（独立可复现实现，运行时不进 LLM 上下文）；可直接拷贝的示例放 `examples/`。
- 兜底：SKILL.md 缺失时返回空串，节点保留最小执行契约，不中断流程。

## 目录包结构

```
skills/
├── README.md
├── <skill-name>/
│   ├── SKILL.md          # 必需：YAML frontmatter + 精简操作规程（用途/流程概要/硬约束/坑位/L3 指针）
│   ├── references/       # 可选：按需加载的详细规格（模型 read_file 读取）
│   ├── scripts/          # 可选：确定性操作独立实现（运行时不进 LLM 上下文）
│   └── examples/         # 可选：可直接拷贝的示例（输出契约/模板）
```

## 目录

| Skill | group | 目录包内容 | 对应代码（权威实现） | 一句话职责 |
|---|---|---|---|---|
| search_papers | input | SKILL.md + references/sources-and-fallback.md | `tools/agent_tools.py` 检索 MCP 包装、`retrieval/perspective_dialogue.py` `_search` | 学术论文检索（多源搭配 + 降级链） |
| index_file | input | SKILL.md | `core/graph_nodes.py` `process_file`/`file_qa` 节点 | 上传文件 RAG 索引与问答 |
| answer_direct | generate | SKILL.md | `core/graph_nodes.py` `simple_chat` 节点 | 自身知识 + 记忆直接回答 |
| answer_from_context | generate | SKILL.md | `core/graph_nodes.py` agent 循环 | 基于检索/文件内容回答（不写综述） |
| write_review | generate | SKILL.md + references/（flow-details、writing-guidelines） | `core/graph_nodes.py` `_agent_system_prompt`/`_write`/`route_after_agent` | 结构化文献综述（含维度化建池） |
| judge_review | quality | SKILL.md + references/review-rubric.md + examples/judge-output-example.md | `core/graph_nodes.py` `_judge`/`route_after_judge` | 6 维评审，<8.5 分重写 ≤3 轮 |
| code_exec | execution | SKILL.md + references/sandbox-details.md + examples/experiment-script-template.md | `sandbox/code_executor.py` `CODE_EXEC_TOOLS` | 沙箱代码执行（本地/Docker 双后端） |
| innovate | execution | SKILL.md + references/experiment-contract.md + examples/metric-example.md + scripts/（experience_log、novelty_check） | `innovate/innovation_tools.py` `INNOVATE_TOOLS` | 创新点挖掘 + 沙箱实验验证流水线 |

## 与代码的关系（重要）

1. **注册表**：`core/skill_manager.py` 的 `SKILL_REGISTRY` 是本目录的权威来源（description/precondition/group 必须与之一致）。
2. **路由**：`core/skill_manager.py` `validate_skills` / `needs_*` + `core/graph_nodes.py` `route_after_parse` 消费 skills 列表。
3. **工具暴露**：`tools/agent_tools.py` `select_tools` 按 skills 从 60+ 工具中挑子集（write_review 已暴露 `read_file`，供 L3 按需读取 references）。
4. **流程提示**：`core/graph_nodes.py` `_agent_system_prompt`（综述 is_review 分支）与各工具 docstring 是本目录正文的权威来源。

**变更纪律**：改任何 skill 的 description/precondition/流程 → 必须同步修改 `SKILL_REGISTRY` 与相关提示词，或反之。本目录是"项目能力地图"，防止能力散落各处后失忆。

## 渐进式披露实现说明（已落地）

| 层 | 内容 | 实现位置 |
|---|---|---|
| L1 元数据 | name+description 常驻 parse prompt | `build_skill_index`（skill_manager.py） |
| L2 正文 | 选中 skills 的 SKILL.md 正文注入执行节点 | `load_skill_bodies` / `build_skill_instructions`（skill_manager.py）→ `_agent_system_prompt`、`write_review_node`、`judge_review_node`、`content_safety_node`、`implement_innovation` / `judge_innovation` / `fix_and_rerun` 内部 sys |
| L3 资源 | references 按需读取（模型 `read_file`）+ scripts 独立运行 | `list_skill_resources` / `load_skill_references`（skill_manager.py）；write_review 工具集含 `read_file` |
| 执行契约 | 节点保留的参数化硬约束（预算数字/引用编号/JSON 解析协议/EXPERIENCE 沉淀） | 各节点与工具内 |

## 与 Anthropic 规范的对应

- **物理形态**：`skill-name/{SKILL.md（必需，frontmatter + 操作程序）+ references/ + scripts/ + examples/（可选）}` —— 已按此重构。
- **L3 渐进披露**：SKILL.md 正文只留"必读的流程骨架与硬约束"，细节进 `references/` 按需读取；`scripts/` 放确定性实现（当前为独立可复现版本，运行时仍走既有模块，避免双实现漂移；后续可平滑迁移为运行时直接调用）。
- **改流程不改代码**：调整综述/评审/审核/实验规则 = 改对应 SKILL.md / references，不动 graph 与工具实现。

## 变更记录

- 2026-10-15：**删除 human_review**（v1.2.7 链路精简：大纲定节已保证每节 ≥5 篇有全文，删除 check_coverage + 按需补检 + 渐进式披露/用户审核门；全部卡片直接用，read_sections 内嵌全池重排）。8 个 skill。
- 2026-10-12：标准目录包重构——4 个大 skill（write_review/judge_review/innovate）拆分 references/（+innovate 的 scripts/、judge_review 与 innovate 与 code_exec 的 examples/）；code_exec/search_papers 补 references/；SKILL.md 精简并加 L3 指针；write_review 工具集暴露 read_file。
- 2026-10-12：**删除多余 skill**——content_safety（评论/新闻稿场景无使用）、search_web（两条主线均用学术库，无联网场景）；注册表/调度校验/提示/工具集同步清理，图节点保留但不可达（needs_content_safety 恒 False）。11 → 9 个。

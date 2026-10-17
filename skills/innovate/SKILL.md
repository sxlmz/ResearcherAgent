---
name: innovate
description: 文献→创新点验证流水线：按主题检索论文并精读制卡（复用 search_papers 检索与 read_papers 精读卡片机制），LLM 提出 3-5 个潜在创新点，为创新点生成实现代码并在沙箱中运行实验，与基线模型（默认演示：时序领域为 PatchTST，其他领域按主题从精读文献常用对比模型中选取）对比精度，判断创新点是否可行。适用：用户要求「基于这些论文提出创新点并测试」「从文献里找改进方向并跑实验验证」「提出 3-5 个创新点看有没有提升」「文献综述→创新点验证」。
group: execution
precondition: null
tools: [read_papers, read_sections, deep_survey, propose_innovations, novelty_check, prepare_reference_repos, code_survey, run_baseline, implement_innovation, fix_and_rerun, judge_innovation, execute_command, read_file, create_file, write_file, list_files, create_directory, gen_code_tree_structure, run_python, remember_memory, consolidate_memory, recall_memory]
references: [experiment-contract.md]
---

# 创新点挖掘与沙箱实验验证

## 用途（何时选）

- 「基于这些论文提出创新点并测试」「从文献里找改进方向并跑实验验证」「提出 3-5 个创新点看有没有提升」。
- 综述→创新点验证的延展链路（先检索建池再创新）。

## 完整流水线（工具描述"前置/下一步"驱动，agent 循环内自主编排）

```
1. read_papers / read_sections   精读 5-10 篇论文 → paper_cards
2. deep_survey                   全文结构化精读：原子学术定义 + 公式原样照抄 + 代码实现线索
                                 （在 propose_innovations 之前调用；产出 state.survey_notes）
3. propose_innovations           两阶段：生成 5 个结构化方案 → 横向比较新颖性/可行性/完整性 → 择优增强 1-2 个
4. novelty_check                 ★ 新颖性查重：多源检索，判定 novel / partially_explored / duplicate
                                 （duplicate → 自动拦截，后续 implement 会跳过实验证明）
5. prepare_reference_repos       按论文标题搜索 GitHub 开源实现 → 浅克隆到沙箱
                                 （论文无开源实现 → 跳过，由 LLM 自写代码）
6. code_survey                   读参考代码库 → 为每个创新点生成实现报告（公式对应真实代码）
7. run_baseline                  跑基线模型（默认演示：时序 PatchTST；其他领域按主题从精读文献
                                 常用对比模型中选取），指标按 dataset|seq_len|pred_len|epochs 缓存
8. implement_innovation          为创新点生成实验代码 → 沙箱运行 → 返回主指标（时序演示：MSE/MAE）
9. 失败 → fix_and_rerun          错误日志回灌 LLM 自动修复重跑（≤3 轮），成功即返回指标，
                                 并把「错误+根因+修复」写入 project/EXPERIENCE.md
10. judge_innovation             对比基线判定可行/存疑/不可行 → state.experiments[id].verdict
11. 经验沉淀                      踩坑写入 EXPERIENCE.md + CC 记忆 MEMORY.md
```

## 查重与闸门（核心规则）

- **novelty_check 判定为 duplicate 的创新点不再做实验证明**（"已经存在的东西没必要再证明"）——implement_innovation 会直接拒绝运行并提示。
- 基线指标缓存缺失时 judge 会提示先 run_baseline 同参数（dataset|seq_len|pred_len|epochs）。

## 与其他 skill 的关系

- 内部自动暴露代码执行组（code_exec）与检索精读（read_papers/read_sections），无需单独选。
- 可配 `search_papers`（先建文献池）；从综述延展时配 `write_review`。
- **领域泛化（v1.2.5 硬规则）**：主题/基线/数据集/指标全部由用户主题与精读文献决定——
  时序仅为演示默认（主题=长时间序列预测、基线=PatchTST、数据集=ETTh1、指标=MSE/MAE）；
  其他领域（视觉/医药/工程等）按领域惯例选取基线、数据集与指标，严禁照抄演示默认。

## 坑位与经验

- 只读论文摘要就提创新点 = 玩具：必须先 deep_survey 拿到**公式级理解**再 propose（防幻觉）。
- 参考代码优先读开源实现（prepare_reference_repos → code_survey），无开源才让 LLM 自写。
- 实验失败自动修复最多 3 轮，超过即放弃该方向并记录，不无限重试。
- 经验沉淀是闭环的一部分：EXPERIENCE.md 会注入后续所有实验 prompt（CC 式记忆跨会话复用）。

## L3 资源（需要细节时用 read_file 读取）

- `references/experiment-contract.md`：实验代码硬性要求 7 条 / 判定标准 / BASELINE_JSON·METRIC_JSON 契约 / EXEC_ARGS 参数
- `examples/metric-example.md`：BASELINE_JSON / METRIC_JSON / EXPERIENCE 条目示例
- `scripts/`：确定性操作独立实现（novelty_check 查重、EXPERIENCE 读写），运行时不进 LLM 上下文

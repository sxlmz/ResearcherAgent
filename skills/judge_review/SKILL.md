---
name: judge_review
description: 用独立评委 LLM 对综述从 6 个维度（结构完整性、内容深度、逻辑严谨性、学术规范性、即时约束遵守度、记忆偏好遵守度）打分。单次评审不重写，输出评审报告（含低置信内容清单与原因）随综述一并交付，低置信点交用户审核。前提：只有请求里已有 write_review 时才选这个。问「这篇综述讲了什么」时不要选。
group: quality
precondition: paired_with_write_review
tools: []
references: [review-rubric.md]
---

# 综述质量评审

## 用途（何时选）

- 只有请求里已有 `write_review` 时才选本 skill（配成 `["search_papers","write_review","judge_review"]`）。
- 「这篇综述讲了什么」等**内容询问**不要选——那是 `answer_from_context`。

## 前置条件（确定性校验）

- 调度器校验：`judge_review` 必须与 `write_review` 同时出现，否则被移除。

## 执行流程（图骨架）

```
write（写作节点）→ judge（异步节点，秒回）→ save（落盘交付）
```

- 独立评委 LLM 从 **6 个维度**打分（聚合自局部 4 维 + 全局 5 维，见 L3 资源）。
- **评审已异步化（v1.2.13，用户决策：judge 曾占总耗时 58.8%）**：judge 节点只启动
  后台线程执行完整评审后**立即返回**，综述**先行交付**用户；评审报告
  （总分/各维/引用真实率/issues/低置信内容清单）后台生成，写入
  `logs/review_report_latest.md/.json`，完成后补交用户审核。
- **单次评审，不做多轮重写**：评审结果以「评审报告」随综述一并交付（异步）——
  报告列出低置信内容（无锚引用 / 无引用事实句 / 与引用论文相似度低等）并说明原因，交用户审核。

## 评审报告（低置信标注，核心输出）

- 评审报告结构：总分 / 各维度分 / 引用真实率 / 缺陷 issues（claim 级）/ **低置信内容清单**。
- 低置信内容来源：确定性证据检查（逐句 verdict）+ 评委 issues 中证据类缺陷，每条含：
  所属节、原文句子、问题类型（no_anchor / no_citation / nli_contradiction / weak_match）、原因说明。
- 用途：用户据此审核哪些论断需要人工核实或修正，不自动重写整篇。

## 与其他 skill 的关系

- `content_safety` 管内容安全（政治/文明/真实/中立），`judge_review` 管学术质量，两者不互斥可同时使用。
- 纯综述不需要 `content_safety`；公开可传播文本（评论/分析/新闻稿）才需要。

## 坑位与经验

- 评审 prompt 里注入用户即时约束（word_budget/required_parts/sections 等）与记忆偏好，防止"按编造的规格评分"——`user_constraints` 为空是正常状态，不代表用户提了要求。
- 断言引用精确率是评审的关键抓手（引用编号与来源必须对上，防止幻觉引用）。

## L3 资源（需要细节时用 read_file 读取）

- `references/review-rubric.md`：评审执行完整规范（分节并行架构 / 局部 4 维 / 全局 5 维 / 确定性证据两档 / 输出契约 / 缺陷分流）
- `examples/judge-output-example.md`：局部 + 全局评审 JSON 输出示例

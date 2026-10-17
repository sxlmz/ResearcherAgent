---
name: answer_from_context
description: 基于检索到的论文数据或上传文件的内容回答问题。适用于总结、分析、提炼已有材料——但不写综述。适用：「这篇论文讲了什么」「这些论文用了什么方法」「总结一下这个文件」。
group: generate
precondition: null
tools: [read_papers, read_sections, file_qa]
---

# 基于已有材料回答

## 用途（何时选）

- 总结、分析、提炼已有材料（检索到的论文 / 上传的文件内容）。
- 典型问法：「这篇论文讲了什么」「这些论文用了什么方法」「总结一下这个文件」「比较这几篇论文」。

## 关键区分（最重要的一条）

- 用户说「**写**一篇XX综述 / 做一篇文献综述」→ 是 `write_review`，不是本 skill。
- 用户说「这篇综述**讲了什么** / 主要观点是什么 / 写了哪些方法」→ 是**内容询问**，用本 skill。
- 用户说「**总结/概括/归纳**一下这个文件」→ 是内容询问，用本 skill。
- **不要**给内容询问型请求加 `write_review` 或 `judge_review`——用户不是要写新综述，是在问已有内容。

## 执行流程（agent 循环）

1. 走 agent 工具循环（skill 含 search_papers/index_file 时工具子集自动暴露）。
2. 精读用 `read_papers`（指定具体论文制卡）或 `read_sections`（按节分配名额）。
3. 基于精读卡片/检索结果回答；引用强制 `[1][2]` + 末尾「参考链接」。
4. 信息不足可结合自身知识但要说明；不编造来源。

## 与其他 skill 的关系

- 有输入类 skill（search_papers / index_file）但没选任何生成类 skill 时，调度器会**自动补本 skill**（否则信息拿到了没人合成答案）。

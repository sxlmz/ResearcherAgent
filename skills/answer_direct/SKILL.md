---
name: answer_direct
description: 用 LLM 自身知识 + 全局记忆档案（SOUL/USER/MEMORY.md）直接回答用户问题。不依赖论文检索结果，也不依赖上传文件。适用：普通问答、概念解释、闲聊——不需要查论文也不需要看文件。
group: generate
precondition: null
tools: []
---

# 直接回答

## 用途（何时选）

- 普通问答、概念解释、闲聊——不需要查论文也不需要看文件。
- 模型对答案有把握、无时效性需求。

## 关键区分

- 与 `answer_from_context` 的区别：后者基于检索到的论文/文件内容回答。
- **冲突消解**：若同时选了需要外部信息的 skill（search_papers/index_file/answer_from_context/write_review），调度器会**移除 `answer_direct`**（保留更需要外部信息的一方），模型不应同时输出两者。

## 执行流程（simple_chat 节点）

1. **记忆注入**：检索全局记忆档案（L0 工作记忆 / L1 档案 / L2 语义 + CC 风格项目记忆 MEMORY.md），按主题注入上下文。
2. **直接回答**：用 LLM 自身知识 + 记忆回答，无检索、无工具调用。
3. 回答结束后落盘保存。

## 坑位与经验

- 伪命题/恶意测试请求会被意图门拦截（见路由），不要在本 skill 内试图"证明"反科学命题。
- 模型不确定、需要文献证据时宁可选 `search_papers` 也不要只用自身知识硬答。

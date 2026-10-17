---
name: index_file
description: 解析用户上传的文件（PDF/DOCX/TXT等），切分为语义块并建立 RAG 索引，供后续回答或写作时引用文件内容。适用：用户上传了文件且需要在回答或综述中参考该文件。
group: input
precondition: has_file
tools: [process_file 节点, file_qa 节点]
---

# 文件索引与问答

## 用途（何时选）

- 用户上传了文件（PDF/DOCX/TXT 等），且需要在回答/综述中引用文件内容。

## 前置条件（确定性校验）

- **必须已上传文件**（`has_file=True`）；否则调度器会移除本 skill。
- 用户上传了文件但模型漏选时，调度器会**自动补上** `index_file`（有文件 + 需要文件内容时）。

## 执行流程

1. **process_file 节点**：用 Infinity 索引/复用上传文档（doc_id 已存在则跳过重复索引），做**混合检索**（稠密 + 全文 BM25）。
2. **file_qa 节点**：基于 RAG 检索结果回答用户问题。
3. 回答后写入记忆（历史格式偏好等，按需）。

## 关键规则

- 文件刚上传时**尚未索引**，跳过 `index_file` 会导致系统找不到文件内容——只要上下文提示"已上传文件"，任何需要文件内容的请求都必须包含本 skill。
- 检索失败时如实报告，不编造文件内容。

## 与其他 skill 的关系

- 常与 `answer_from_context`（总结/问答）或 `write_review`（基于文件写综述）搭配。
- 纯直接回答（不需要文件内容）时不要选本 skill。

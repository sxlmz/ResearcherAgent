---
name: search_papers
description: 调用 arXiv / PubMed 等学术数据库检索论文，返回论文列表和摘要。适用：用户需要查找特定主题的最新文献。
group: input
precondition: null
tools: [search_arxiv, search_europepmc, search_core]
references: [sources-and-fallback.md]
---

# 学术论文检索

## 用途（何时选）

- 用户需要查找特定主题的**学术论文**（不是网页/新闻）。
- 综述、创新点验证、回答"有哪些论文/最新进展"等需要文献证据的场景。

## 关键区分

- **学术论文检索用 `search_papers`**（系统无通用网页搜索能力；时效性/新闻类内容不属于本项目检索范围）。

## 检索源与选择策略（v1.2.6 起只保留能返回全文线索的源）

三个学术源覆盖面不同，**建议多源搭配**（模型在 agent 循环中自行决定换源）：

| 源 | 特点 |
|---|---|
| search_europepmc | 开放获取全文通路（PMCID → JATS 全文 XML 直取） |
| search_arxiv | 偏预印本/最新工作；100% 全文可达（ar5iv） |
| search_core | CORE v3（OA 全文文本，需 CORE_API_KEY；配 key 后稳定） |

规则：
- 某源返回 0 条时**换另一个源**，不要反复重试同一个。
- 源熔断时工具会直接回"该源已熔断 Ns，请改用：…"，照提示换源。

**已移除（不再使用）**：`search_openalex`（本轮 E2E 多次 ReadTimeout 20s 拖慢检索）、
`search_crossref`（多为题录、无全文线索——综述要全文，纯题录浪费时间）；
Semantic Scholar（整体移除，v1.2.8 用户决策：无 key 时 1 req/s 节流 + 429 频繁，
检索与全文通路均不再使用该源；novelty_check 等场景另有独立检索链路，不受影响）。

## 检索词规范

- **英文短语优先**（学术库英文覆盖远好于中文），短关键词（≤30 字符），不是完整句子。
- 超长中文/自然语言 query 会导致部分源报错/召回失真——先拆词再检索。

## 综述链路检索纪律（v1.2.13 流程修订，防多轮建池浪费时间）

综述生成流程中，论文检索**只发生在大纲确定之后**，按**方法节**定向检索一次：

```
survey_evolution（tavily 探索演进方向，不检索论文库）
→ make_outline（按方向写大纲，方法论默认 3 节、最多 4 节，无需预先建池）
→ 按每个方法节定向检索（每节 1 个 query，arXiv/EuropePMC/CORE 三源并发，
  一轮、禁止补检）→ read_sections（精排 + 全文制卡；有全文 <5 篇的节自动删除或并入相邻节）
```

- ❌ **不应该**：在大纲生成前大规模检索建池（如每个演进方向各拆多个 query 反复搜）；
  同一 query 多轮补检；检索后再次检索只为凑篇数。
- ✅ **应该**：一次把该节 query 发到三个源（search_arxiv + search_europepmc + search_core
  并行）；结果并入 paper_pool 后立即交给 read_sections，由它精排过滤与全文制卡。

## 坑位与经验

- 检索源直连（arXiv/Europe PMC/CORE，与综述白名单一致）；源失败自动换下一个，不阻塞链路。
- 检索结果会按"摘要长度优先"排序，保证下游有可引用的细节。
- 结果并入 `paper_pool`（按标题归一化去重，reducer 再做全局去重）。

## L3 资源（需要细节时用 read_file 读取）

- `references/sources-and-fallback.md`：检索源细节 / 降级链 / 拆词规范示例

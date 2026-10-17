# search_papers · 检索源与降级策略详解（L3 参考）

SKILL.md 正文的精读版本。模型检索策略有歧义、或需要决定换源/降级时用 `read_file` 读取本文件。

## 检索源覆盖对比（v1.2.8 定稿：仅 arXiv + Europe PMC + CORE 三源）

| 源 | 特点 | 覆盖 | 摘要比例 |
|---|---|---|---|
| search_europepmc | 开放获取全文通路（PMCID → JATS 全文 XML 直取） | 全学科 OA | 中高 |
| search_arxiv | 预印本/最新工作；100% 全文可达（ar5iv） | CS/数学/物理 | 高 |
| search_core | CORE v3 OA 全文文本（MDPI/DOAJ 系；需 CORE_API_KEY） | OA 期刊 | 中 |

## 主源与降级链（代码层自动处理）

- **主源**：Europe PMC（全文通路最直接）→ **arXiv**（预印本全文保障）→ **CORE**（OA 期刊全文）。
- 单条 query 在单源上最多重试 2 次，第 2 次仍失败直接换源，不做无上限重试。
- 某源持续失败会触发源健康度熔断（per-session），自动从工具集摘除。
- **已移除**：OpenAlex（本轮 E2E 多次 ReadTimeout 20s，网络不稳定拖慢检索）；
  Semantic Scholar（无 key 429 频繁）；Crossref（纯题录无全文）。

## 检索词规范

- **英文短语优先**（学术库英文覆盖远好于中文），短关键词（≤30 字符），不是完整句子。
- 超长中文/自然语言 query 会导致部分源返回 500——先拆词再检索。
- 拆词示例：`"基于 Transformer 和频域增强的长时间序列预测模型"` →
  `["transformer frequency enhancement", "long-term time series forecasting", "frequency domain attention"]`。

## 结果处理

- 检索结果按"摘要长度优先"排序（下游要可引用的细节）。
- 结果并入 `paper_pool`：按标题归一化去重（reducer 再做全局去重）。
- 某源返回 0 条 / 熔断 → 工具直接提示"该源已熔断 Ns，请改用：…"，照提示换源，不反复重试同一源。

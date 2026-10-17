# write_review · 流程细则（L3 参考）

SKILL.md 正文的精读版本。模型在 agent 循环中需要细节时用 `read_file` 读取本文件。

## 五步流水线细则（2026-10 方案 v1.2 + v1.2.6 演进调研前置 + v1.2.7 链路精简）

### 0. survey_evolution（演进脉络调研，v1.2.6 新增第一步）★核心

- **为什么**：大纲凭空列世代（LLM 拍脑袋）容易硬列无全文支撑的空节；先摸清「该方向是怎么
  演进的」，再按演进方向定向检索，大纲与文献一致。
- **怎么执行**（工具内部自动完成，仅 1 次 LLM 调用 + 1 次 Tavily 搜索）：
  1. `tavily_search(f"{topic} research evolution history timeline survey")` 拿网页片段（免费 HTTP）；
  2. LLM 1 次调用梳理 **3-4 个演进方向**（由远及近）：{name, era, methods, search_queries
     （英文带领域限定词）, description}；
  3. 存入 `state.evolution_directions`（幂等：已存在则复用，不重复扣配额）。
- 产出：3-4 个演进方向 + 每方向英文检索词 → 供第 1 步定向检索、第 2 步 make_outline 定节。

### 1. 按方向定向检索建池（v1.2.6）

- 对**每个演进方向**分别用 search_* 定向检索（白名单三源：arXiv / Europe PMC / CORE），
  每方向找 **10 篇有全文**的论文入 paper_pool。
- **检索源白名单固定三源**：Europe PMC（PMCID）/ arXiv（ar5iv）/ CORE（OA 全文文本，需 key）；
  `search_openalex`（本轮 E2E 多次 ReadTimeout 拖慢检索）、`search_crossref`（纯题录无全文）与
  Semantic Scholar（无 key 429 频繁）已移除（v1.2.8 用户决策；novelty_check 等场景另有独立检索链路不受影响）。
- 检索解析（parse_papers）统一给每条论文标注 `fulltext_hint`（arxiv:<id> / pmc:<PMCID> / oa:<url> / ""）；
  并入 pool 前**剔除无全文线索的论文**（全无全文时原样保留并警告换源）——保证定节依据就是「真能精读的文献」。
- 产出：`paper_pool` 按演进方向覆盖；子查询由**真实检索结果引导**，比凭空拆视角更贴文献分布。

### 2. make_outline（演进式大纲，v1.2.4-v1.2.6：**基于有全文论文分布 + 演进方向定节**）

- 输出 5-6 节（引言 1 + 技术迭代 3-4 + 展望 1），**技术世代由远及近、由浅入深**；
  **方法论最好只列三个，除非实在很多，允许有四节大纲**；
  世代划分由 LLM 依领域自适应（few-shot 引导，具体领域变通，不写死）；
  **若 state 已有 evolution_directions，方法节尽量与其对应**（可合并相近方向，不无依据另起世代）。
- **硬要求**：先看 paper_pool 里**可获取全文**（arXiv/PMC/OA）的论文及其主题分布，再定节——
  每个技术迭代节至少要有 **5 篇以上有全文论文**可支撑（**<5 篇 → 删除该节或并入相邻节**）；
  **某方向无全文素材就不单列一节**（并入相邻节或引言/展望带过）；
  **严禁按领域惯例硬列章节**（如池里没有统计类全文，就不要列"经典统计时代"节）。
- make_outline 返回时会附"池内可精读论文 N 篇"提示，供模型审视并主动修正。
- 每节 focus 写清「承接上一代 + 本代**方法族**与**待回答问题** + 局限引出下一代」——
  **待回答问题**是第 4 步定向抽取的关键词 query 依据，缺失会导致抽取偏题。

### 3. read_sections（选文：方法节 top-10，v1.2.7 内嵌重排）

- **每个方法节对池内论文做全池 Cross-Encoder 重排，取 top-10 精读**（v1.2.7：不再依赖
  check_coverage——该步已从链路删除；read_sections 内部直接调 select_section_candidates 重排，
  确定性无 LLM）；引言/展望不单独选文，其素材从 30-40 篇的背景/结论段概括。
- **精读节强制以 `state.review_outline` 节名为准（v1.2.18 防节名漂移）**：长 agent 循环中
  模型传参可能复用旧大纲节名 → 护栏删不到新大纲节、write 归节失败、补写无素材节。
  read_sections 忽略模型传入的旧节名（打印提示），只精读大纲现存方法节。
- 共 3-4 节 → **30-40 篇**（按大纲实际节数浮动，上限 40）。
- 全文**全篇下载不截断**（抽取阶段不吃 LLM 上下文）；超大文档按段落数预算
  （`FULLTEXT_PARAGRAPH_BUDGET`，超过优先保方法/实验节）。
- 逐篇并行（下载并发 6 / 抽取并发 8），卡片带 `extract_mode`：keyword / llm_fallback / abstract_only。
- 重排结果写回 `state.section_candidates`，供「一轮检索后证据不足拒答门」使用。

### 3.5 大纲-真实全文对齐（v1.2.18，用户拍板）

- **每节要么有 ≥5 篇真实可读全文的论文，要么删除/合并该节**：护栏①（候选不足）、
  护栏③（实际精读不足）删除/并入时**同步更新 `state.review_outline`**（卡片 section
  改写为并入后节名，write 归节自然映射）。大纲定节依据是「精排后实际下载到全文」，
  不是检索时的 fulltext_hint 纸面声明。
- **无素材节拒绝生成（write 层兜底，防各种原因导致的空节）**：write 节点对
  「该节无精读卡片」的节（方法节按节卡片、引言/展望按任意卡片判定）**不补写硬编**，
  输出占位说明「本节因缺乏可精读文献支撑，未生成内容」；逐节写作 prompt 同款硬规则
  （无材料直接拒绝，严禁凭知识/池内摘要编造）。

### 4. 字段化卡片（v1.2.7：全部卡片直接用，无渐进式披露；v1.2.17 加 method 兜底）

- 卡片 = {title, id, section, num, fields:{background,method,experiment,conclusion}, extract_mode}，
  字段 **100% 原文段落**（带节号），实验字段表格转 Markdown 保留行列语义，缺失字段标 `"none"`。
- **每节最多 10 篇，全部卡片直接用**——不再生成轻量索引、不再按需 get_card
  （v1.2.7 删除 build_card_index/get_card/ask_user）。
- **method 兜底（v1.2.17 用户决策）**：
  - background/conclusion/experiment 缺失可接受（标 none），**method 必须命中**才走关键词路径；
  - method 未命中 → MQ1 消费者线程内**就地 MQ2 兜底**：LLM 直接读全文（截 60k）提取句式 JSON 块
    （一步直达，等价把写作第四步提前到制卡第三步，不做原文卡片再二次提取）；
    产出「虚拟卡片」（extract_mode=llm_direct，method 字段=LLM 句式）+ direct 块入 MQ3
    （MQ3 消费者见 direct_blocks 直接回显，不二次调用 LLM）；
  - LLM 判定全文无方法论（纯数据集/综述/立场文）→ 该篇丢弃（不入任何卡片、不进写作）；
  - 编号：read_sections 送 MQ1 时统一带 num（global_index），虚拟卡与关键词卡共用同一编号体系。

### 5. 停止

- 模型完成 read_sections 精读后停止调用工具，交 write 节点（自动完成 JSON 块生成 → 拼接 → 润色 → 预算）。

## 定向抽取（第 3 步）口径（v1.2.16 精确标题策略）

- 节类型定位（三层，全部确定性，不调 LLM）：
  1. **标题精确识别**：先剥 `##` 前缀 → 数字节号（`^\d+(\.\d+)*`）优先；无编号标题过
     `_looks_like_heading`（词表命中 / 全大写 / Title-Case 短行），**前置噪音排除**
     `_noise_heading`——含 `|` 表格行、`In ` 参考文献行、`" : "` 论文标题副标题形态、
     无字母词碎片行、标题行后无正文（作者行/页脚）一律不当标题；
  2. **归桶**：基节用词表（中英变体 + 领域变体 few-shot，非穷尽）→ 节号兜底
     （最小基节→背景，最大基节→结论）；**子节继承父节类型**（`4.3` 挂 `4 Experiments`
     下即归 experiment；父节 other/缺失才回退词表）——消除"实验节子节被 TRAINING 抢进
     method"类误伤；
  3. **词表未覆盖标题**（如 `An Embarrassingly Simple Baseline`）→ other 桶节头段落按
     字段 query 做 bge-m3 打分补位（排除 Abstract/References/Acknowledgements 框架节，
     选中节头 top-2 → 整节内容入字段）。全字段仍空 → abstract_only 兜底卡。
- 每字段取相似度 top-2 段落（bge-m3 按「字段语义 + 论文标题」打分）。
- 表格转 Markdown：fulltext 层 HTML 表格已转（保留行列语义），纯文本启发式兜底 PDF 通路。
- 实测（4 篇跨领域 arXiv：U-Net/ViT/BERT/DLinear）：真实章节标题全部识别、四字段齐全率
  4/4、明显归桶错误 0 例（修复前 3 例）；残余噪音为参考文献区论文标题（SimCLR 标题行）。

## 全文通路（v1.2.4 优先级；v1.2.8 移除 Semantic Scholar）

1. **arXiv → ar5iv**（结构化 HTML，最优先，无需 key）
2. **本机直下 PDF + PyPDF2**（对可直连的 OA PDF）
3. **CORE API 全文文本**（MDPI/DOAJ 系 OA 期刊直接给全文；需免费 CORE_API_KEY）
4. **Tavily extract**（绕本机网络限制，对 arXiv 页/可抓 PDF 有效；对 ACM/MDPI/ScienceDirect 反爬源常无效）
5. 出版社源（ACM/MDPI/ScienceDirect PDF 直链）是**系统性全文黑洞**——Tavily 无法稳定抓取；
   遇到这类论文靠 CORE 或换 arXiv/综述型开放论文。

## 检索源策略（写作预算口径）

- 方法节按**输出预算**挑选句式片段（3 节→900/节、4 节→700/节）：放不下的篇目本轮不引用
  （v1.2.3：超预算=减少引用篇数，不压缩不截断）；
- 引言/展望素材按 `LEAD_BLOCK_INPUT_BUDGET` 挑选后一次概括合成（输出 ≤300 字/节）；
- 回退逐节写作路径仍按 `WRITE_SECTION_BUDGET` 注入材料（输入侧），输出预算按用户字数分配下发。
- 每节精读上限由 read_sections 的 per_section 强制（默认 10），选卡以重排分数 + 全文可达为准。

## 降级与兜底

- **MQ 不可用**（未装 pika / 连接失败 / 超时）→ 自动退化 `ThreadPoolExecutor` 直连，链路不阻塞（验收标准）。
- JSON 块生成失败重试 2 次 → 该篇降级为索引提及（进参考文献但不展开）。
- arxiv connect timeout 属境外网络抖动，换源即可，不要死磕。

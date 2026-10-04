"""
Configuration module for the LangGraph agent application.

Contains MCP server configurations, safety instructions, tool failure
detection heuristics, and content safety review thresholds.

记忆系统 v2 说明：旧的 Chroma 向量库 / MD 文件记忆已迁移至
  Infinity（文档 chunk + 对话原文 + L2 语义记忆）+ PostgreSQL（L1 档案/事实）
  + Redis（L0 工作记忆）。相关代码见 memory_stack / memory_orchestrator /
  memory_classifier / memory_integration。

Environment variables (Docker / production):
  DEEPSEEK_API_KEY        — DeepSeek API key
  EMBEDDING_API_KEY       — OpenAI-compatible embedding API key
  EMBEDDING_API_BASE      — OpenAI-compatible embedding API base URL
  TAVILY_API_KEY          — Tavily web search API key
  PAPER_SEARCH_PYTHON     — Python interpreter for paper_search_mcp (default: sys.executable)
"""

import os

# ── MCP server config (paths use env vars in Docker, hardcoded fallback for local dev) ──
_paper_search_python = os.environ.get(
    "PAPER_SEARCH_PYTHON",
    # fallback: use the interpreter on PATH; override with the env var
    "python",
)

_gs_proxy = os.environ.get("GS_PROXY_URL", "").strip()
_tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()

# ── MCP 子进程的代理策略（★ 实测根因，不是猜测）────────────────────────
# 本机 Windows 系统代理指向 127.0.0.1:7890（Clash 类），其**境外出口已失效**。
# `requests` 在 Windows 上会从注册表读系统代理，所以 MCP 子进程默认就走这个坏代理。
# 实测对照（同一时刻、同一 URL，只差走不走代理）：
#     走 7890 ：crossref/openalex/europepmc → SSLError: UNEXPECTED_EOF_WHILE_READING
#               arxiv → 502 Bad Gateway / api.tavily.com → ReadTimeout   → 全灭
#     不走代理：crossref 200(3060B)、openalex 200(45065B)、europepmc 200(1950B)、
#               semantic 200(405B)、tavily 200(7673B, 3 条结果)          → 全部正常
# 结论：「检索源总是不稳定」的主因**不是源在限流，而是子进程在走一个坏代理**——
# 这也解释了日志里为何全是 SSL EOF 而不是 429。
# 故显式注入 NO_PROXY=* 让 MCP 子进程直连；若将来要改走一个**可用**的代理，
# 设 MCP_PROXY=https://host:port 即可（会同时清掉 NO_PROXY）。
# ⚠️ ① 两个 MCP server 都要注入——tavily 同样是走代理就挂、直连就好（实测见上）。
#    ② NO_PROXY 只影响「从环境/注册表推断代理」，不影响代码里显式传的 `proxies=`，
#       因此 google scholar 专用的 *_PROXY_URL 不受影响。
_proxy_env = {"NO_PROXY": "*", "no_proxy": "*"}
_mcp_proxy = os.environ.get("MCP_PROXY", "").strip()
if _mcp_proxy:
    _proxy_env.update({
        "HTTP_PROXY": _mcp_proxy, "HTTPS_PROXY": _mcp_proxy,
        "http_proxy": _mcp_proxy, "https_proxy": _mcp_proxy,
        "NO_PROXY": "", "no_proxy": "",
    })
del _mcp_proxy

MCP_CONFIG = {
    "paper_search_server": {
        "command": _paper_search_python,
        "args": ["-m", "paper_search_mcp.server"],
        "transport": "stdio",
        "env": {
            **_proxy_env,
            "max_results_per_source": "10",
            "PAPER_SEARCH_MCP_GOOGLE_SCHOLAR_PROXY_URL": _gs_proxy,
        },
    },
    "web-search": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "tavily-mcp@latest"],
        "env": {
            **_proxy_env,
            "TAVILY_API_KEY": _tavily_key,
            "DEFAULT_PARAMETERS": "{\"include_images\": true, \"max_results\": 15, \"search_depth\": \"advanced\"}",
        },
    },
}

MAX_REWRITE_ROUNDS = 3          # 最多打回重写次数
MAX_SEARCH_ROUNDS = 3           # 人机审核门最多检索循环次数

# ── 意图门与拒答阈值（入口意图识别 + 一轮检索后证据不足拒答）──
REJECT_MALICIOUS_THRESHOLD = float(os.environ.get("REJECT_MALICIOUS_THRESHOLD", "0.6"))     # malicious ≥ 该值 → 拒答
REJECT_RESEARCHABLE_THRESHOLD = float(os.environ.get("REJECT_RESEARCHABLE_THRESHOLD", "0.3"))  # researchable < 该值 → 拒答
REJECT_RATIO_THRESHOLD = float(os.environ.get("REJECT_RATIO_THRESHOLD", "0.02"))  # 一轮检索后全池相关比例均值 < 该值 → 拒答
# 「证据类缺陷 → 回研究阶段补检索」的**独立配额**。
# ⚠️ 为什么必须独立计数：`judge → agent` 这条边原本不经过任何计数器，
#    于是 route_after_judge 里那句 `rewrite_count < target` 守卫**永远成立**
#    （rewrite_count 在这条路径上不增加），judge 被反复调用直到 needs_research 偶然变 False。
#    实测 judgev3c：judge 跑了 **6 次**而 rewrite_count 只有 3、force_rounds=2，
#    judge 占总耗时 58.8%（1355s/2306s）——基线（6 维时代）只占 1%。
#    默认 1 次：若补了检索仍不忠实，问题多半出在写作而非素材，继续检索是浪费。
MAX_RE_RESEARCH = int(os.environ.get("MAX_RE_RESEARCH", "1"))
JUDGE_PASS_THRESHOLD = float(os.environ.get("JUDGE_PASS_THRESHOLD", "8.5"))  # 评委打分阈值（满分 10；qwen 异族尺度高，默认 8.5）

# ── Agent 护栏（Anthropic《Building Effective Agents》：max steps / token 预算 / HITL）──
# 研究阶段是模型驱动的 ReAct 循环（agent ⟲ tools），"下一步做什么"由模型决定；
# 这里只设硬上限，超限后模型被要求立刻收尾，而不是被代码强行推进。
MAX_AGENT_STEPS = int(os.environ.get("MAX_AGENT_STEPS", "25"))        # 一次请求内 agent↔tools 往返上限
TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET", "400000"))          # 一次请求累计 token 上限（usage_metadata）
TOOL_CONCURRENCY = int(os.environ.get("TOOL_CONCURRENCY", "3"))       # MCP 工具并发上限（学术源频繁 429）
MAX_TOOL_RESULT_CHARS = 1500    # 回给模型的单条工具结果长度（全量结果存 paper_pool，上下文只放摘要+句柄）
KEEP_RECENT_TOOL_MESSAGES = 6   # 上下文里保留原文的最近 ToolMessage 数，更早的替换为一行句柄

# ── 检索量 ↔ 写作预算的对齐参数 ──
# 瓶颈不在「检索了多少篇」，而在「多少篇能真正进写作 prompt」：
# 每节按字符预算渲染，有摘要的论文优先；超出预算的论文对写作贡献为 0。
# 因此 check_coverage 按同一口径回报「可注入篇数」，让模型的继续/停止决策有正确依据。
WRITE_SECTION_BUDGET = 15000    # 分节写作时每节「可引用材料」的字符预算（输入侧）

# ── 检索源健康度与重试 ──
# 机制参考成熟项目（读源码得到，非猜测）：
#   · paper-chaser-mcp: MAX_429_RETRIES=6、SEMANTIC_SCHOLAR_MIN_INTERVAL=1.05s、
#     payload["degraded"]=True（降级必须显式标记，而不是静默返回空）
#   · research-mcp: 主源被限流 → 自动降级备用源；脆弱响应磁盘缓存 + stale fallback
#   · Auto-Sci-Find: 429 退避重试 3 次，指数间隔 5/15/45 秒
#   · agentic-ai-lab: Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
#
# ⚠️ 我们的约束：检索走 MCP **子进程**，各平台客户端把 requests 异常
#    `except RequestException: return []`（实测 crossref.py 就是这么写的），
#    所以拿不到 HTTP 状态码，也无法共享其速率限制器。
#    实测确认（_probe_mcp_shapes.py）：源内部报错时 CallToolResult 仍是
#    `isError=False, structuredContent={'result': []}`——**与「真的没结果」完全同形**。
#    因此只能在包装层做：单次超时 + 退避重试 + 健康度熔断 + 降级显式标记。
#
# 阈值由实测数据定（_probe_sources.py，修好代理后 4 源 × 3 查询 × 2 轮）：
#     search_openalex   6/6 成功   avg 1.1s  max  1.7s   ← 最稳最快
#     search_europepmc  6/6 成功   avg 1.8s  max  6.1s
#     search_crossref   6/6 成功   avg 4.1s  max 17.4s
#     search_arxiv      5/6 成功   avg 18.2s max 70.0s（客户端超时）← 最慢最不稳
#   · 连续 3 次才熔断（不是 2 次）：arxiv 实测 6 次里就失败 1 次，
#     阈值定 2 会把「偶发抖动」误判成「源挂了」，而 3 连败在这份数据里是真信号。
#   · 单次调用超时 45s：覆盖 crossref 的 max 17.4s，同时砍掉 arxiv 那条 70s 长尾
#     （实测 n=... 那条 70s 是客户端超时，不是源返回）。超时用 asyncio.wait_for 实现，
#     已实测「在途取消 MCP 调用后会话仍完全可用」（连取消 3 次后 3 个源照常返回）。
SRC_FAIL_STREAK = int(os.environ.get("SRC_FAIL_STREAK", "3"))       # 连续失败 N 次 → 熔断该源
SRC_COOLDOWN_SEC = int(os.environ.get("SRC_COOLDOWN_SEC", "180"))   # 熔断冷却时长（秒）
MCP_CALL_TIMEOUT_SEC = float(os.environ.get("MCP_CALL_TIMEOUT_SEC", "45"))  # 单次工具调用上限
# 退避重试间隔（秒）。参考 Auto-Sci-Find 的 5/15/45，但按我们的实测延迟压缩：
# 单次已要 1~17s，若照搬 5/15/45 会让一次工具调用逼近 70s，反而拖垮 agent 循环。
RETRY_BACKOFF_SEC = (2.0, 6.0)

# ── Cross-Encoder 重排（检索精排）──
# 为什么需要：检索是宽召回（池内 500+ 篇，无相关性判断），旧代码拿「有没有摘要」当排序代理
# ——筛出来的不是「最相关的 N 篇」，而是「摘要最全的 N 篇」。重排是成熟 RAG 管线的标准第二级。
SECTION_CANDIDATES_K = int(os.environ.get("SECTION_CANDIDATES_K", "15"))  # 每节重排后保留的候选数
RERANK_MIN_SCORE = float(os.environ.get("RERANK_MIN_SCORE", "0.8"))       # 低于此分视为不相关
CARDS_PER_SECTION = int(os.environ.get("CARDS_PER_SECTION", "3"))         # 每节精读几篇生成卡片（证据约束下保证每节有可引素材）

# ── 综述正文的总字数上限（输出侧，硬约束）──
# 为什么必须有：此前不设限，模型会写出 6 万字，而评审只看得到前 1.2 万字（旧代码 draft[:12000]），
# 于是评审判「第 3 章缺失」——它不是判错，是真的看不到后面的内容；通过与否变成抽奖，
# 还白烧 3 轮重写。截断是症状，篇幅失控才是病因。
# 正确做法是从生成端就约束：在预算内**写完整**（章节一个不少、每节写短），而不是写完再截。
MAX_REVIEW_CHARS = int(os.environ.get("MAX_REVIEW_CHARS", "10000"))
# 超预算容忍度：超过 预算×该系数 的章节才触发压缩（避免对轻微超出反复调 LLM）
SECTION_OVERFLOW_TOLERANCE = 1.25
# 评审/安全审核单次能读入的正文上限。正常情况下正文已 ≤MAX_REVIEW_CHARS，不会触发；
# 若模型仍严重超预算，则**截断并显式告知评审「这是节选、实际共 N 字」**，
# 杜绝「没看到」被误判成「没写」。
REVIEW_READ_MAX = MAX_REVIEW_CHARS * 3
#   ⚠️ 配套教训（两轮实测）：覆盖度反馈里**只报数字、不下指令**。
#   曾试过输出「专属带摘要 <N 篇 → 请补检索 / 建议停止检索」这类判断，结果：
#     · 阈值定成「达标线（≥6 篇才停）」时，该指标由 LLM 每次重新判定（paper_idx 每次不同），
#       会涨也会跌、永远收敛不到阈值 → 每次检查都输出「尚未达标」→ 模型被推着检索到
#       撞满 25 步护栏（stop_reason=max_steps），而不加判据的一轮是 21 步自然终止。
#     · 改成「下限（<3 篇才补）」仍是指令，同样把它推向护栏。
#   结论：代码负责把正确的数字摆在模型面前（本节专属带摘要几篇、预算多大），
#   是否继续检索由模型自己判断——这与 todolist「结构交给模型」的原则一致。

# ── 模型模式开关（single / multi）──
# single：所有节点用**同一个**模型（省钱/兜底；换端点只改 SINGLE_* 三个变量）
# multi ：按 ROUTER_/AGENT_/EXTRACT_/WRITER_/JUDGE_/SAFETY_MODEL 各角色分别路由（更强也更贵）
#
# 为什么要保留多模型设计：不同节点的任务性质、token 量、对延迟的敏感度差别极大——
# agent 循环占总量 ~60%、写作决定成品质量、标题生成是用户可感知的串行等待。
# 一个模型包打天下不是最优解，所以模式开关只切"用不用得上"，不拆掉这套路由结构。
def model_mode() -> str:
    """读当前模型模式。**每次调用时读环境变量**，避免 load_dotenv 的导入时序陷阱。"""
    return (os.environ.get("MODEL_MODE") or "multi").strip().lower()


# 渐进式工具加载：search_papers 只暴露这几个轻量检索源（永不暴露 download_*/read_* PDF 工具）
#
# ⚠️ 曾经的错误结论（留档，避免重犯）：先前实测 search_openalex「持续报错、9 次调用 0 条结果」，
#    于是把它从白名单摘掉了。**那是误判**——真正的原因是检索子进程在走一个失效的系统代理
#    （见上方 NO_PROXY 的实测对照），OpenAlex 只是受害者。
#    修好代理后重测：openalex 6/6 成功、avg 1.1s，是**最快最稳**的一个源。
#    教训：把「环境问题」当成「源的问题」会把好源一个个摘掉，越摘越差。
#    现在改成按实测可靠性排序；源若真的挂了，由健康度熔断在运行期自动摘除，
#    而不是靠人手工维护白名单。
#
# 实测（修好代理后，4 源 × 3 查询 × 2 轮，见 _probe_sources.py）：
#     openalex   6/6   avg 1.1s  max  1.7s   ← 最稳最快，作为首选
#     europepmc  6/6   avg 1.8s  max  6.1s   ← 开放获取全文通路，摘要覆盖率高
#     crossref   6/6   avg 4.1s  max 17.4s   ← 覆盖最广（含非英文/会议），但延迟抖动大
#     arxiv      5/6   avg 18.2s max 70.0s   ← 最慢最不稳；预印本为主，窄领域仍有价值
PAPER_SEARCH_TOOL_WHITELIST = (
    "search_openalex",      # 实测最稳最快：首选
    "search_europepmc",     # 开放获取全文通路
    "search_crossref",      # 覆盖最广，延迟抖动大
    "search_arxiv",         # 预印本/最新工作；慢且偶发超时，靠超时+熔断兜住
)
WEB_SEARCH_TOOL_PREFIXES = ("tavily",)   # web-search MCP server 的工具名前缀

# Checkpointer：postgres（跨重启恢复 HITL 断点）| memory（仅进程内）
CHECKPOINTER = os.environ.get("CHECKPOINTER", "postgres")

# 基础安全指令 —— 注入所有生成类节点的 system prompt，作为兜底。
# 不替代 content_safety 节点的深度审核，但能在源头减少违规输出。
_BASE_SAFETY_INSTRUCTION = (
    "\n\n## 内容安全底线（必须遵守）\n"
    "- 遵守中国法律法规，坚持社会主义核心价值观\n"
    "- 不使用侮辱性、歧视性、攻击性语言；不煽动对立情绪\n"
    "- 涉及争议话题时客观呈现多方观点，不偏袒、不炒作\n"
    "- 不传播未经证实的谣言或虚假信息；引用信息时注明来源\n"
    "- 不提供违法、暴力、色情等违规内容的描述或指导\n"
)

# 明确的工具失败短语（区别于论文正文里可能出现的 "error rate"/"reconstruction error" 等正常词）
# 只在「短内容」上做子串启发式，长正文一律以 ToolMessage.status 为准。
_TOOL_FAILURE_PHRASES = (
    "tool execution error",
    "rate limit",
    "rate-limit",
    "timed out",
    "timeout",
    "connection refused",
    "traceback (most recent call last)",
    "未找到相关论文",
    "检索异常",
    "无法检索",
    "请稍后重试",
    "请等待",
)

# 内容启发式只在短于该长度时才生效，避免扫描整篇论文正文导致误判
_TOOL_FAILURE_HEURISTIC_MAX_LEN = 600

_CONTENT_SAFETY_DIMS = ["政治合规性", "用语文明", "信息真实性", "客观中立"]
_CONTENT_SAFETY_THRESHOLD = 7.0

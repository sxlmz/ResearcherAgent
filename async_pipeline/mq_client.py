# -*- coding: utf-8 -*-
"""async_pipeline/mq_client.py —— 卡片→句式JSON 的 RabbitMQ / ThreadPool 双轨实现。

设计（方案 v1.2 第 6 节）：
  · 消息确认：消费者处理成功才 ack；失败 nack 重试（最多 MQ_MAX_RETRIES），超限记失败降级；
  · 结果写回：JSON 块落盘（logs/review_build/）+ 内存返回，主流程不阻塞；
  · 降级：MQ 不可用 / 未装 pika / 连接失败 / 超时 → 自动退化为 ThreadPoolExecutor 直连，
    链路不阻塞（这是验收标准第 5 条：MQ 停掉后 ThreadPool 直连仍可完成全流程）。
  · 纯函数 generate_json_blocks_for_card 与传输解耦：MQ 消费者与 ThreadPool 直连共用同一逻辑。
"""
import json
import math
import os
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ResearcherAgent.core.config import (
    MQ_URL, MQ_QUEUE_CARD_TO_JSON, MQ_QUEUE_LOCATE_FALLBACK, MQ_QUEUE_CARD_BUILD,
    MQ_CONSUMER_CONCURRENCY, MQ_CARD_BUILD_CONCURRENCY, MQ_TASK_TIMEOUT_SEC, MQ_MAX_RETRIES, MQ_RESULT_DIR,
    JSON_BLOCK_CHARS_PER_PAPER,
)

_CARD_FIELD_ORDER = ("background", "method", "experiment", "conclusion")


class MqUnavailable(Exception):
    """MQ 不可用（未装 pika / 连接失败 / 超时）——调用方应降级 ThreadPool。"""


def _result_dir() -> Path:
    d = (MQ_RESULT_DIR or "").strip()
    p = Path(d) if d else Path("logs") / "review_build"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ────────────────────────────────────────────────────────────
# 纯函数：卡片 → 句式 JSON 块（第 4 步核心；MQ 消费者与 ThreadPool 共用）
# ────────────────────────────────────────────────────────────
_JSON_PROMPT = """你是学术综述的内容组装助手。下面是一篇论文的**字段化卡片**
（背景/方法/实验/结论，均为原文段落，带节号锚点；缺失字段标 (none)）。

请把该论文拆成 **2-4 个**「内容块」。**块所属章节不在这里选择**——由系统按
block_type 与论文所属大纲节确定性标注，你只负责产出内容与类型。
**产出规则（硬约束，保证每个章节都有素材可填，禁止少出）**：
- **方法类内容**（method 或 experiment 字段非 (none)）→ **必须**产出至少 1 个
  **method 块**；
- **背景/结论类内容**（background 或 conclusion 字段非 (none)）→ **必须**分别产出
  **intro 块**（服务引言节）与 **conclusion 块**（服务展望节），各至少 1 个；
- 若所有字段均 (none)（纯摘要卡）→ 用摘要概括产出 1 个 intro 块（领域背景/问题）+
  1 个 method 块（摘要中描述的方法，指标标"（未提供实验）"），不产出 conclusion 块；
- 同一内容可拆出多个块，但**不得漏掉**上述任一类型的产出要求。

【块类型（block_type 硬约束，只能取下列三种之一）】：
1. **method（方法块）**：源自 method/experiment 字段；
   - 句式使用**四要素硬契约**（任何措辞必须围绕这四要素，漏一项视为不合格）：
     ① 提出了 XX（方法/模型名）
     ② 通过/用 XX 解决了 XX（机制 + 针对的难题，必须来自 method 字段）
     ③ 在 XX 数据集上达到 XX 指标（数据集 + 数字，**必须来自 experiment 字段/表格，
        禁止凭印象填**）
     ④ （可选）相较 XX 的突破（无卡片证据支撑时不写）
   - 纯理论/无实验论文：降级为「提出了 XX，一种 XXX，通过 XXX 解决了 XXX」并标注
     "未提供实验"（**不得编造指标**）。
2. **intro（引言块）**：源自 background 字段或方法论文的背景铺垫；
   - **不使用四要素句式**：对卡片 background 内容做**忠实转述**（背景问题/领域趋势），
     只许重组卡片原文的措辞，**不得新增卡片之外的论断/事实**，不套用方法句式；
   - 句尾标注本卡片编号 [N]（该编号即「引用」键，供引言节受控组织时溯源）。
3. **conclusion（展望块）**：源自 conclusion 字段或方法论文的总结展望；
   - **不使用四要素句式**：对卡片 conclusion 内容做**忠实转述**（结论/展望），
     只许重组卡片原文的措辞，**不得新增卡片之外的论断/事实**，不套用方法句式；
   - 句尾标注本卡片编号 [N]。

通用约束：
- content 用中文，客观转述，不添加卡片之外的信息；
- **句级引用锚点（硬约束）**：每个以句号结尾的完整论断句，**句尾**标注引用编号 [N]——
  N 必须等于本卡片编号（card 的 num 字段，方括号数字格式）；一句号一句 = 对编号 N
  论文的整句引用，句内以逗号/分号连接的子句**不需要各自标注**；
  未标注编号的完整句会被评审判为「论断无引用锚点」而扣分。
- **禁止**标注本卡片之外的编号（不得写别的论文的 [N]）；系统会在组装期校验纠正，
  写错编号的句子的编号会被强制纠正为本卡片编号；
- paper_ref 固定为**该论文的完整标题**（逐字复制 card 的 title 字段）；
- paper_num 固定为**该论文的池内全局编号**（card 的 num 字段，即「引用」键）；
- evidence 填本块依据的节号锚点（卡片里的 【§x.x】 列表，用于评审证据锚校验）；
- 不输出卡片中没有的实验数字；字段 (none) 时对应要素标"（未提供）"。

只输出 JSON 数组（outline_section/大纲归属由系统按 block_type + 卡片所属大纲节确定性标注，
你不输出）：
[{{"block_type": "method", "content": "……提出……[12]。……解决……[12]。……指标……[12]。",
   "paper_ref": "论文标题", "paper_num": 12, "evidence": ["§1", "§4.2"]}}]

卡片：
{card_json}
"""


def generate_json_blocks_for_card(card: dict, llm, outline: list | None = None) -> list[dict]:
    """单卡片 → 内容块列表；失败返回 []（调用方降级为索引提及）。

    card: {"title","id","fields":{background,method,experiment,conclusion},...}
    outline: 已不再注入 prompt（大纲归属由组装阶段按 block_type + 卡片 section
             代码确定性标注）；保留参数仅为兼容旧调用。

    块 schema：{"block_type": "method|intro|conclusion", "content": "...",
                "paper_ref": "论文标题", "evidence": ["§1", "§4.2"]}
    """
    import re
    from langchain_core.prompts import ChatPromptTemplate

    # v1.2.19 提取块缓存：同一张卡片（论文+字段）只调一次 LLM。
    # 命中直接返回（键 = title+id+fields，不含 num/section——同一篇论文在
    # 不同综述里节/编号不同仍可复用提取结果）。
    from ResearcherAgent.core import cache
    _cached = cache.blocks_get(card)
    if _cached is not None:
        print(f"    [card_to_json] 缓存命中提取块 {card.get('id','')[:24]}（{len(_cached)} 块，跳过 LLM）")
        return _cached

    fields = (card or {}).get("fields") or {}
    card_json = json.dumps({
        "title": (card or {}).get("title", ""),
        "num": (card or {}).get("num"),
        "fields": {f: (fields.get(f) or "(none)") for f in _CARD_FIELD_ORDER},
    }, ensure_ascii=False)
    chain = ChatPromptTemplate.from_messages([
        ("system", "你是严谨的学术综述组装助手，严格按用户要求输出 JSON。"),
        ("user", _JSON_PROMPT),
    ]) | llm
    try:
        resp = chain.invoke({"card_json": card_json})
        content = resp.content if hasattr(resp, "content") else str(resp)
        blocks = _extract_json_array(content)
    except Exception as e:
        print(f"    [card_to_json] 生成失败 {card.get('id','')}: {type(e).__name__}: {str(e)[:100]}")
        return []
    if not blocks:
        return []
    # 兜底规范化（确定性，不依赖 LLM 自觉）：
    #   ① paper_ref 强制 = 论文完整标题；paper_num 强制 = 卡片池内编号（「引用」键）；
    #   ② block_type 非法/缺失 → 按卡片字段来源推断；
    #   ③ v1.2.11/v1.2.13 句级编号：LLM 已把 [N] 直接写进 content（句级锚点）。规范化只做
    #      **校验纠正**：块内所有 [N] 一律纠正为本卡片编号（防 LLM 写错/串号）；
    #      无编号的块 → content 末尾补 [num]（兜底，至少句号级锚点）。
    _title = (card or {}).get("title", "") or ""
    num = (card or {}).get("num")
    for b in blocks:
        b["paper_ref"] = _title
        b["paper_num"] = num
        bt = str(b.get("block_type") or "").strip().lower()
        if bt not in ("method", "intro", "conclusion"):
            bt = _infer_block_type(card)
        b["block_type"] = bt
        c = str(b.get("content") or "").strip()
        if num is not None:
            c = re.sub(r"\[\d{1,3}\]", f"[{num}]", c)          # 纠正为卡片编号
            if not re.search(r"\[\d{1,3}\]", c):
                c = c.rstrip("。") + f"。[{num}]"               # 无锚 → 块尾补
        else:
            c = re.sub(r"\[\d{1,3}\]", "", c).strip()           # 无编号可纠 → 剥离
        b["content"] = c
    # 预算裁剪：单篇 JSON 块总字符上限，防挤爆写作预算
    total, kept = 0, []
    for b in blocks:
        c = str(b.get("content") or "")
        if total + len(c) > JSON_BLOCK_CHARS_PER_PAPER:
            break
        total += len(c)
        kept.append(b)
    if kept:
        cache.blocks_set(card, kept)   # v1.2.19 写提取块缓存（30 天）
    return kept


def _infer_block_type(card: dict) -> str:
    """LLM 未给 block_type 时的兜底推断：method/experiment 非 (none) → method；
    只有 background → intro；只有 conclusion → conclusion；否则 method。"""
    fields = (card or {}).get("fields") or {}
    has_method = any((fields.get(f) or "").strip() not in ("", "(none)")
                     for f in ("method", "experiment"))
    has_bg = (fields.get("background") or "").strip() not in ("", "(none)")
    has_concl = (fields.get("conclusion") or "").strip() not in ("", "(none)")
    if has_method:
        return "method"
    if has_bg and not has_concl:
        return "intro"
    if has_concl and not has_bg:
        return "conclusion"
    return "method"


def _format_outline_json(outline: list | None) -> str:
    """大纲节列表 → prompt 注入文本（精确节名 + type + focus）。"""
    if not outline:
        return "（未提供大纲，节名请按综述惯例给出：引言/方法节/展望）"
    lines = []
    for i, o in enumerate(outline, 1):
        name = (o.get("section") or "").strip() or f"第{i}节"
        typ = o.get("type") or _infer_section_type(name, i, len(outline))
        focus = (o.get("focus") or "").strip()
        line = f'  {i}. "{name}"（type: {typ}）'
        if focus:
            line += f"—— 本节关注：{focus[:120]}"
        lines.append(line)
    return "\n".join(lines)


def _infer_section_type(name: str, index: int, total: int) -> str:
    """按名称/位置推断节类型：引言 / 方法 / 展望。"""
    if any(k in name for k in ("引言", "背景", "问题定义", "introduction", "intro")):
        return "引言"
    if any(k in name for k in ("展望", "结论", "总结", "未来", "挑战", "outlook", "conclusion")):
        return "展望"
    return "方法"


def _normalize_block_sections(blocks: list[dict], outline: list) -> list[dict]:
    """双保险：LLM 输出的节名若不在大纲内（如"方法-XXX"/近似名），剥前缀/别名后对齐。"""
    import re

    def _nk(s: str) -> str:
        return re.sub(r"[\s:：,，、\-_/\\]+", "", (s or "")).lower()

    names = [(o.get("section") or "").strip() for o in outline if o.get("section")]
    if not names:
        return blocks
    for b in blocks:
        bs = str(b.get("outline_section") or "").strip()
        if bs in names or _nk(bs) in {_nk(n) for n in names}:
            continue
        # 剥常见前缀与别名后缀
        cand = bs
        for prefix in ("方法-", "方法：", "方法:", "Method-", "method-"):
            if cand.startswith(prefix):
                cand = cand[len(prefix):]
                break
        cand = cand.split("：")[0].split(":")[0].strip()
        best, best_len = None, 0
        for n in names:
            n_core = n.split("：")[0].split(":")[0].strip()
            if not cand:
                continue
            if cand in n or n_core in cand or cand in n_core or _nk(cand) == _nk(n_core):
                if len(cand) > best_len:
                    best, best_len = n, len(cand)
        if best:
            b["outline_section"] = best
        elif bs:
            # 无法对齐的：引言/展望别名走类型匹配，否则保留原值（下游 _match_outline_section 兜底）
            if any(k in bs for k in ("引言", "背景")):
                for o in outline:
                    if "引言" in (o.get("section") or ""):
                        b["outline_section"] = o["section"]
                        break
            elif any(k in bs for k in ("展望", "结论", "总结", "未来")):
                for o in outline:
                    if any(k in (o.get("section") or "") for k in ("展望", "结论", "总结", "未来")):
                        b["outline_section"] = o["section"]
                        break
    return blocks


def _extract_json_array(content: str) -> list[dict]:
    if not content:
        return []
    import re
    m = re.search(r"\[.*\]", content, re.S)
    if not m:
        return []
    try:
        d = json.loads(m.group(0))
        return [b for b in d if isinstance(b, dict) and b.get("content")]
    except Exception:
        return []


# ────────────────────────────────────────────────────────────
# MQ 客户端（懒连接；任何失败抛 MqUnavailable → 降级）
# ────────────────────────────────────────────────────────────
class _JsonBuildSession:
    """流式批量会话：start → extend（可多次增量入队）→ collect（收结果 + 清理）。

    生命周期跨节点：read_sections 制卡期间每出一张卡就 extend 入队（MQ 消费者
    线程持续运行），write 节点 collect 时统一收齐。任何失败抛 MqUnavailable →
    调用方降级 ThreadPool 直连。
    """

    def __init__(self, client, handler, conn, ch, reply_q, task_ids, pending,
                 results, stop, threads):
        self.client = client
        self.handler = handler
        self.conn = conn
        self.ch = ch
        self.reply_q = reply_q
        self.task_ids = task_ids          # 全部任务 id（含增量）
        self.pending = pending            # set，线程安全（lock）
        self.results = results            # {tid: result}
        self.stop = stop
        self.threads = threads
        self.lock = threading.Lock()
        self.cid_by_tid: dict = {}   # tid → card_id（collect 还原用）
        self.done = False

    # ---- 增量入队（卡片流式：每出一张卡就 publish 一条任务）----
    def extend(self, payloads: list[dict]) -> list[str]:
        """追加任务入队，返回新 task_ids。消费者线程已在跑，无需新启。

        v1.2.15：发布用**独立短连接**——pika channel 非线程安全，本方法可能被
        MQ1 制卡消费者线程跨线程调用（制卡成功自动入 MQ3：extend_json_session）。
        每次 extend 新建连接→发布→关闭；发布只经 broker 路由，reply 仍由会话
        自己的消费者线程接收，语义不变。"""
        new_ids = [f"t{int(time.time() * 1000)}-{i}-{uuid.uuid4().hex[:4]}"
                   for i in range(len(payloads))]
        with self.lock:
            self.task_ids.extend(new_ids)
            self.pending.update(new_ids)
            for tid, p in zip(new_ids, payloads):
                self.cid_by_tid[tid] = (p or {}).get("card_id") or ""
        try:
            cconn = self.client._connection()
            cch = cconn.channel()
            for tid, payload in zip(new_ids, payloads):
                cch.basic_publish(
                    exchange="", routing_key=self.client.queue,
                    body=json.dumps({"task_id": tid, "data": payload, "attempt": 0},
                                    ensure_ascii=False))
            cconn.close()
        except Exception:
            with self.lock:
                for tid in new_ids:
                    self.pending.discard(tid)
        return new_ids

    # ---- 收集（write 节点调用；等全部完成，超时抛 MqUnavailable）----
    def collect(self, timeout: float | None = None) -> dict:
        """等待全部任务完成并收集 reply。返回 {tid: result}。"""
        if self.done:
            return dict(self.results)
        deadline = time.time() + (timeout if timeout is not None else self.client.timeout)
        while True:
            with self.lock:
                n_pending = len(self.pending)
            if n_pending == 0 or time.time() >= deadline:
                break
            try:
                self.conn.process_data_events(time_limit=0.2)
            except Exception:
                break
            try:
                method, props, body = self.ch.basic_get(self.reply_q, auto_ack=False)
                while method:
                    self._on_reply(method, props, body)
                    method, props, body = self.ch.basic_get(self.reply_q, auto_ack=False)
            except Exception:
                break
            time.sleep(0.05)
        self.stop.set()
        for t in self.threads:
            t.join(timeout=5)
        try:
            self.conn.close()
        except Exception:
            pass
        # v1.2.18 会话结束清理隔离队列（card_to_json__jbs_* / card_build__cbs_*），
        # 防止长期运行堆积空队列；base 队列（批量 RPC 用）不删。
        if "__" in self.client.queue:
            try:
                _d = self.client._connection()
                _dch = _d.channel()
                _dch.queue_delete(queue=self.queue)
                _d.close()
            except Exception:
                pass
        self.done = True
        with self.lock:
            n_pending = len(self.pending)
        if n_pending:
            raise MqUnavailable(f"MQ 收集超时（完成 {len(self.results)}/{len(self.task_ids)}）")
        return dict(self.results)

    def _on_reply(self, method, props, body):
        try:
            d = json.loads(body.decode())
            with self.lock:
                self.results[d.get("task_id")] = d.get("result")
                self.pending.discard(d.get("task_id"))
            self.ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception:
            try:
                self.ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception:
                pass

    def _publish_result(self, cch, tid: str, result: dict):
        """落盘 + 回发 reply（用消费者自己的 channel——pika channel 非线程安全）。"""
        try:
            (self.client.result_dir / f"{tid}.json").write_text(
                json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        try:
            cch.basic_publish(
                exchange="", routing_key=self.reply_q,
                body=json.dumps({"task_id": tid, "result": result}, ensure_ascii=False))
        except Exception:
            pass

    def _process_one(self, cch, method, props, body):
        """处理一条任务：成功回发 reply；失败按 attempt 重试（ack 原消息 + 单条重发）。"""
        try:
            payload = json.loads(body.decode())
            tid = payload.get("task_id", "")
            attempt = int(payload.get("attempt", 0))
            try:
                out = self.handler(payload.get("data"))
                result = {"ok": True, "data": out}
            except Exception as e:
                result = {"ok": False, "error": str(e)[:200]}
                print(f"    [{self.client.queue}] 消费者处理失败 {tid}: {type(e).__name__}: {str(e)[:120]}")
            if not result.get("ok") and attempt < self.client.retries:
                cch.basic_ack(delivery_tag=method.delivery_tag)
                cch.basic_publish(
                    exchange="", routing_key=self.client.queue,
                    body=json.dumps({**payload, "attempt": attempt + 1},
                                    ensure_ascii=False))
                return
            cch.basic_ack(delivery_tag=method.delivery_tag)
            self._publish_result(cch, tid, result)
        except Exception:
            try:
                cch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception:
                pass


class _MqClient:
    def __init__(self, url: str | None = None, queue: str | None = None,
                 result_dir: str | None = None, retries: int | None = None,
                 timeout: float | None = None):
        self.url = (url or "").strip() or MQ_URL
        self.queue = (queue or "").strip() or MQ_QUEUE_CARD_TO_JSON
        self.result_dir = Path(result_dir) if result_dir else _result_dir()
        self.retries = MQ_MAX_RETRIES if retries is None else retries
        self.timeout = MQ_TASK_TIMEOUT_SEC if timeout is None else timeout
        self._params = None
        self._lock = threading.Lock()

    def _connection(self):
        """懒建立 pika 连接（每次新建，简单可靠；失败抛 MqUnavailable）。"""
        if self._params is None:
            try:
                import pika
                import urllib.parse
                u = urllib.parse.urlparse(self.url)
                self._params = pika.ConnectionParameters(
                    host=u.hostname or "localhost",
                    port=u.port or 5672,
                    credentials=pika.PlainCredentials(u.username or "guest",
                                                      u.password or "guest"),
                    heartbeat=30,
                )
            except ImportError:
                raise MqUnavailable("pika 未安装，降级 ThreadPool")
            except Exception as e:
                raise MqUnavailable(f"MQ 参数解析失败: {str(e)[:80]}")
        try:
            import pika
            return pika.BlockingConnection(self._params)
        except Exception as e:
            raise MqUnavailable(f"MQ 连接失败: {str(e)[:80]}")

    # ---- RPC 模式：session 化（start → extend 增量 → collect）----
    def start_session(self, payloads: list[dict], handler,
                      concurrency: int | None = None) -> _JsonBuildSession:
        """建会话：连接 + reply_q + 消费者线程 + publish 首批。失败抛 MqUnavailable。
        concurrency: 消费者并发数；缺省用 MQ_CONSUMER_CONCURRENCY（card_to_json），
        card_build（MQ1 制卡）由调用方传 MQ_CARD_BUILD_CONCURRENCY。"""
        try:
            import pika
            conn = self._connection()
        except Exception as e:
            raise MqUnavailable(f"MQ 不可用（pika 缺失/连接失败）: {str(e)[:80]}")
        try:
            ch = conn.channel()
            ch.queue_declare(queue=self.queue, durable=True)
            # ⚠️ 不用服务端生成的 amq.gen-* 临时队列：① RabbitMQ 禁止客户端显式声明 amq. 保留前缀
            #    （实测 403 ACCESS_REFUSED）；② exclusive 队列只对声明它的连接可见，消费者线程是
            #    另一条连接会收不到。改为固定命名、auto_delete、非 exclusive 的回复队列。
            reply_q = f"rpc_reply_{uuid.uuid4().hex}"
            ch.queue_declare(queue=reply_q, durable=False, auto_delete=True)

            task_ids = [f"t{int(time.time() * 1000)}-{i}-{uuid.uuid4().hex[:4]}"
                        for i in range(len(payloads))]
            pending = set(task_ids)
            results: dict = {}
            stop = threading.Event()
            session = _JsonBuildSession(self, handler, conn, ch, reply_q, list(task_ids),
                                        pending, results, stop, [])

            # 消费者线程：各自独立连接 + channel，basic_get 轮询（RabbitMQ 天然分发，
            # 每条消息同一时刻只被一个消费者取到）；不设 idle 提前退出——extend 会
            # 持续追加新任务（重试消息也可能稍后发布），只以 collect 的 stop 退出。
            # v1.2.10 固定消费者并发：流式首批可能只有 1 张卡，
            # 若按首批数量起线程会导致后续增量全部串行——固定并发数保证 28 篇
            # 卡片增量入队时始终被 N 个消费者并行处理。
            n_consumers = concurrency if concurrency else MQ_CONSUMER_CONCURRENCY

            def _consumer(_idx: int):
                """消费者线程：独立连接 + channel，basic_get 轮询（队列按会话隔离后
                只消费本会话消息）。连接/循环异常 → 打印 + 重连继续（v1.2.18：
                此前 except:pass 静默死亡 → 长存活会话（read_sections→write 间隔
                170s+）消费者死光 → collect 0/N → 降级重提交，双倍等待）。"""
                while not stop.is_set():
                    try:
                        cconn = self._connection()
                        cch = cconn.channel()
                        cch.queue_declare(queue=self.queue, durable=True)
                        cch.queue_declare(queue=reply_q, durable=False, auto_delete=True)
                    except Exception as e:
                        print(f"    [{self.queue}] 消费者#{_idx} 连接失败 "
                              f"{type(e).__name__}: {str(e)[:60]}，2s 后重连")
                        time.sleep(2)
                        continue
                    try:
                        while not stop.is_set():
                            method, props, body = cch.basic_get(
                                queue=self.queue, auto_ack=False)
                            if not method:
                                time.sleep(0.1)
                                continue
                            session._process_one(cch, method, props, body)
                    except Exception as e:
                        print(f"    [{self.queue}] 消费者#{_idx} 循环异常 "
                              f"{type(e).__name__}: {str(e)[:60]}，重连")
                    finally:
                        try:
                            cconn.close()
                        except Exception:
                            pass

            threads = [threading.Thread(target=_consumer, args=(i,), daemon=True)
                       for i in range(n_consumers)]
            for t in threads:
                t.start()
            session.threads = threads

            # publish 首批
            for tid, payload in zip(task_ids, payloads):
                session.cid_by_tid[tid] = (payload or {}).get("card_id") or ""
                ch.basic_publish(exchange="", routing_key=self.queue,
                                 body=json.dumps({"task_id": tid, "data": payload,
                                                  "attempt": 0}, ensure_ascii=False))
            return session
        except Exception as e:
            try:
                conn.close()
            except Exception:
                pass
            if isinstance(e, MqUnavailable):
                raise
            raise MqUnavailable(f"MQ 会话启动失败: {str(e)[:80]}")

    def rpc_call(self, payloads: list[dict], handler) -> list:
        """一次性批量 RPC（兼容入口）= start_session + collect。
        返回与 payloads 等长的结果列表（失败条目为 None）。
        """
        session = self.start_session(payloads, handler)
        replies = session.collect()
        return [replies.get(tid) for tid in session.task_ids[:len(payloads)]]


# ────────────────────────────────────────────────────────────
# 对外接口
# ────────────────────────────────────────────────────────────
def _locate_llm(paras, title, llm):
    """locate_fallback 消费者：复用 graph_nodes 的纯定位函数。"""
    from ResearcherAgent.core.graph_nodes import _locate_sections_with_llm
    return _locate_sections_with_llm(llm, paras, title)


def submit_cards_for_json(cards: list, llm, mq_url: str | None = None,
                          queue: str | None = None,
                          result_dir: str | None = None,
                          outline: list | None = None) -> dict:
    """批量卡片 → 内容块。返回 {card_id: blocks}（失败的卡片为 []，
    调用方降级为索引提及——该篇进参考文献但不展开）。
    MQ 可用 → RPC 异步；不可用/失败/超时 → ThreadPool 直连（自动降级，链路不阻塞）。
    outline: 已不注入 prompt（大纲归属组装期代码标注），保留参数仅为兼容。"""
    if not cards:
        return {}
    payloads = [{"card_id": c.get("id", ""), "card": c} for c in cards]

    def _handler(data):
        if data.get("direct_blocks"):   # v1.2.17 MQ2 直接提取的块：不做卡片二次提取
            return data["direct_blocks"]
        return generate_json_blocks_for_card(data.get("card"), llm)

    client = _MqClient(url=mq_url, queue=queue, result_dir=result_dir)
    try:
        replies = client.rpc_call(payloads, _handler)
        out = {}
        for r, p in zip(replies, payloads):
            if r and r.get("ok"):
                out[p["card_id"]] = r.get("data") or []
            else:
                out[p["card_id"]] = []
        print(f"    [card_to_json] MQ 完成 {len(out)} 篇（降级 ThreadPool 前先走 MQ）")
        return out
    except MqUnavailable as e:
        print(f"    [card_to_json] {e} → 降级 ThreadPool 直连（{len(cards)} 篇）")
        return _build_via_threads(cards, llm)


def _build_via_threads(cards: list, llm, outline: list | None = None) -> dict:
    with ThreadPoolExecutor(max_workers=MQ_CONSUMER_CONCURRENCY) as ex:
        results = list(ex.map(
            lambda c: (c.get("id", ""),
                       generate_json_blocks_for_card(c, llm)), cards))
    return {cid: blk for cid, blk in results if blk}


# ────────────────────────────────────────────────────────────
# 流式会话对外接口（read_sections 边制卡边送 MQ → write 节点收集）
# ────────────────────────────────────────────────────────────
_SESSIONS: dict = {}  # session_id → _JsonBuildSession（模块级注册表，跨节点共享）


def start_json_session(cards: list, llm, mq_url: str | None = None,
                       queue: str | None = None,
                       result_dir: str | None = None) -> str:
    """启动卡片→内容块的流式会话（首批入队 + 消费者线程开跑）。
    返回 session_id（存 state，write 节点用其 collect）；MQ 不可用返回 ""（走降级）。
    v1.2.15：cards 允许为空——read_sections 先建会话，MQ1 制卡消费者边制卡边
    extend_json_session 增量入队，write 节点 collect 时统一收齐。"""
    payloads = [{"card_id": c.get("id", ""), "card": c} for c in (cards or [])]

    def _handler(data):
        if data.get("direct_blocks"):   # v1.2.17 MQ2 直接提取的块：不做卡片二次提取
            return data["direct_blocks"]
        return generate_json_blocks_for_card(data.get("card"), llm)

    # v1.2.18 队列按会话隔离：多会话（read_sections 流式 + write 重提交）并存时
    # 若共用 MQ_QUEUE_CARD_TO_JSON，消费者互相抢任务、reply 发错会话 → collect 0/N。
    # 会话专属队列 = 基队列 + sid 后缀；extend/collect/消费者全走 session.client.queue。
    sid = f"jbs_{uuid.uuid4().hex[:12]}"
    client = _MqClient(url=mq_url, queue=f"{queue or MQ_QUEUE_CARD_TO_JSON}__{sid}",
                       result_dir=result_dir)
    try:
        session = client.start_session(payloads, _handler)
    except MqUnavailable as e:
        print(f"    [card_to_json] 流式会话启动失败 {str(e)[:60]} → write 节点将降级 ThreadPool")
        return ""
    _SESSIONS[sid] = session
    return sid


def extend_json_session(session_id: str, cards: list, mq_url: str | None = None,
                        queue: str | None = None,
                        result_dir: str | None = None,
                        direct_blocks: list | None = None) -> bool:
    """向既有会话增量入队（每制出一张卡调用）。会话不存在/失败 → False（调用方忽略，
    该卡片将在 write 节点随 submit_cards_for_json 兜底补提交）。
    v1.2.17：direct_blocks 非空 → 该任务带 direct 标记（MQ3 消费者直接回显）。"""
    session = _SESSIONS.get(session_id or "")
    if not session or not cards:
        return False
    payloads = [{"card_id": c.get("id", ""), "card": c,
                 **({"direct_blocks": direct_blocks} if direct_blocks else {})}
                for c in cards]
    try:
        session.extend(payloads)
        return True
    except Exception as e:
        print(f"    [card_to_json] 流式 extend 失败 {str(e)[:60]}（write 节点兜底补提交）")
        return False


def collect_json_session(session_id: str, timeout: float | None = None,
                         mq_url: str | None = None,
                         queue: str | None = None,
                         result_dir: str | None = None) -> dict:
    """收集会话全部结果 {card_id: blocks}；成功后清理注册表。
    会话不存在或收集超时 → 抛 MqUnavailable（调用方降级 ThreadPool 补做）。"""
    session = _SESSIONS.get(session_id or "")
    if not session:
        raise MqUnavailable(f"流式会话 {session_id} 不存在")
    try:
        replies = session.collect(timeout=timeout)
    finally:
        _SESSIONS.pop(session_id, None)
    out: dict = {}
    for tid, r in replies.items():
        cid = session.cid_by_tid.get(tid) or ""
        if r and r.get("ok") and cid:
            out[cid] = r.get("data") or []
    n_ok = sum(1 for r in replies.values() if r and r.get("ok"))
    print(f"    [card_to_json] 流式会话收集完成 {n_ok}/{len(replies)} 篇（MQ 增量）")
    return out


def submit_locate_fallback(paras: list, title: str, llm,
                           mq_url: str | None = None,
                           queue: str | None = None,
                           result_dir: str | None = None) -> dict:
    """关键词/节号失效时 LLM 定位节（第 2 步兜底）。MQ 优先，自动降级直连。
    返回 {field: sec}（如 {"method": "3", "experiment": "4"}）。"""
    if not paras:
        return {}
    client = _MqClient(url=mq_url, queue=queue or MQ_QUEUE_LOCATE_FALLBACK,
                       result_dir=result_dir)
    try:
        replies = client.rpc_call(
            [{"title": title, "paras": paras}],
            lambda d: _locate_llm(d.get("paras") or [], d.get("title") or "", llm))
        r = replies[0] if replies else None
        if r and r.get("ok"):
            return r.get("data") or {}
        raise MqUnavailable("locate_fallback 无有效回复")
    except MqUnavailable as e:
        print(f"    [locate_fallback] {e} → 降级直连")
        return _locate_llm(paras, title, llm)


# ────────────────────────────────────────────────────────────
# v1.2.15 制卡流水线（MQ1 论文全文→字段化卡片 / MQ3 卡片→JSON 块）
# 链路：read_sections 下载完成 → start_card_build_session（MQ1）→ 每篇 extend →
#       MQ1 消费者关键词制卡（skip_llm_fallback，并发；纯正则/embedding，不调 LLM）
#       → 制卡成功 → 卡片自动入 MQ3（card_to_json 流式会话，read_sections 预先建好）。
# read_sections 用 collect_card_build_session 等制卡全部完成（护栏③复核删节/合并）。
# 关键词表覆盖不到的字段标 "none"，由写作阶段按「未精读/缺字段」处理。
# （v1.2.15 用户决策：去掉 MQ2 LLM 兜底制卡——submit_locate_fallback / locate_fallback
#   队列保留为独立能力，但不再接入制卡流水线。）
# MQ 不可用 → start 返回 ""，read_sections 降级同步制卡（fetch_and_card 旧路径）。
# ────────────────────────────────────────────────────────────
def _submit_card_to_json(data: dict, card: dict, direct_blocks: list | None = None) -> None:
    """MQ1 制卡成功（或 MQ2 直接提取）后自动入 MQ3（card_to_json 流式会话，
    read_sections 预建）。direct_blocks 非空 → MQ3 消费者直接回显（不再调 LLM）。

    json_session_id 由 read_sections 塞进 block（消息 data={"block": {...}}），
    MQ1 消费者从 block 内取值；顶层兼容兜底。"""
    blk = data.get("block") or {}
    jsid = data.get("json_session_id") or blk.get("json_session_id") or ""
    if not jsid:
        return
    try:
        extend_json_session(jsid, [card], direct_blocks=direct_blocks)
    except Exception as e:
        print(f"    [card_build] 送 MQ3 失败（jsid={jsid[:12]}）"
              f"{type(e).__name__}: {str(e)[:80]}（该卡 write 节点兜底补提交）")


def _build_card_from_paper(data: dict, llm) -> dict:
    """MQ1 制卡消费者 handler：论文全文 → 字段化卡片（纯关键词/embedding 确定性路径）。

    data: {"block": {"id","title","section","text","num","json_session_id"}, ...}
    返回 {"card": card, "via": "mq1_keyword"|"mq2_direct", "dropped": bool, "reason"?: ...}

    v1.2.15：MQ2（LLM 兜底制卡）整体去掉——制卡只走关键词路径（_map_to_cards
    skip_llm_fallback=True：切段 → 字段关键词分桶 → embedding 打分）；背景/结论/实验
    缺失标 "none" 由写作阶段处理，**method 必须要有**。
    v1.2.17（用户决策兜底策略）：
    - method 命中 → 直接入 MQ3（不管其他字段缺不缺）；
    - method 未命中 → 就地 MQ2 兜底：LLM 直接提取句式 JSON 块（一步直达，不做原文
      卡片再提取）→ 产出「虚拟卡片」（method 字段=LLM 内容）+ direct 块入 MQ3；
      LLM 判定全文无方法论 → dropped（丢弃该篇，不入任何卡片）。
    """
    from ResearcherAgent.core.graph_nodes import _map_to_cards
    blk = data.get("block") or {}
    cards = _map_to_cards([blk], llm, skip_llm_fallback=True)
    card = cards[0] if cards else None
    method_ok = bool(card and (card.get("fields") or {}).get("method") not in ("", "none"))
    if method_ok:
        if card.get("num") is None and blk.get("num") is not None:
            card["num"] = blk.get("num")
        _submit_card_to_json(data, card)
        return {"card": card, "via": "mq1_keyword"}
    # method 未命中 → MQ2 直接 LLM 提取（v1.2.17）
    return _extract_method_direct(data, llm)


def _extract_method_direct(data: dict, llm) -> dict:
    """MQ2 兜底（v1.2.17）：method 关键词未命中 → LLM 直接提取句式 JSON 块。

    产出「虚拟卡片」（method 字段=LLM 提取内容，id/section/num 与原文一致——
    read_sections/组装统一按卡片概念处理）+ direct 块自动入 MQ3；
    LLM 判定全文无方法论 → {"dropped": True} 丢弃（不入任何卡片）。
    """
    from ResearcherAgent.core.graph_nodes import extract_method_json_direct
    blk = data.get("block") or {}
    out = extract_method_json_direct(blk.get("text") or "", blk.get("title") or "",
                                     blk.get("section") or "", blk.get("num"), llm)
    if out.get("drop"):
        return {"dropped": True, "via": "mq2_direct",
                "reason": out.get("reason") or "LLM 判定无方法论"}
    blocks = out.get("blocks") or []
    if not blocks:
        return {"dropped": True, "via": "mq2_direct", "reason": "LLM 未产出方法块"}
    content = "\n".join(str(b.get("content") or "") for b in blocks)
    card = {"title": blk.get("title") or "", "id": blk.get("id") or "",
            "section": blk.get("section") or "", "num": blk.get("num"),
            "card": content, "passages": [],
            "fields": {"background": "none", "method": content,
                       "experiment": "none", "conclusion": "none"},
            "extract_mode": "llm_direct", "code_url": ""}
    _submit_card_to_json(data, card, direct_blocks=blocks)
    return {"card": card, "via": "mq2_direct", "blocks": blocks}


def start_card_build_session(papers: list, llm, mq_url: str | None = None,
                             queue: str | None = None,
                             result_dir: str | None = None) -> str:
    """MQ1 制卡流水线会话：论文全文 → 字段化卡片（并发制卡，失败转 MQ2）。
    返回 session_id；MQ 不可用返回 ""（read_sections 降级同步制卡）。
    papers: [{"id","title","section","text"}]（text=全文）。
    """
    payloads = [{"block": p} for p in papers]

    def _handler(data):
        return _build_card_from_paper(data, llm)

    # v1.2.18 队列按会话隔离（同 start_json_session 理由）：MQ1 制卡会话独享队列，
    # 不与 MQ3 会话/批量 RPC 抢消息。
    sid = f"cbs_{uuid.uuid4().hex[:12]}"
    client = _MqClient(url=mq_url, queue=f"{queue or MQ_QUEUE_CARD_BUILD}__{sid}",
                       result_dir=result_dir, retries=1)
    try:
        session = client.start_session(payloads, _handler,
                                       concurrency=MQ_CARD_BUILD_CONCURRENCY)
    except MqUnavailable as e:
        print(f"    [card_build] MQ1 会话启动失败 {str(e)[:60]} → 降级同步制卡")
        return ""
    _SESSIONS[sid] = session
    return sid


def extend_card_build_session(session_id: str, papers: list,
                              mq_url: str | None = None,
                              queue: str | None = None,
                              result_dir: str | None = None) -> bool:
    """向 MQ1 会话增量入队（每下载完成一篇即调用）。会话不存在/失败 → False。"""
    session = _SESSIONS.get(session_id or "")
    if not session or not papers:
        return False
    try:
        session.extend([{"block": p} for p in papers])
        return True
    except Exception as e:
        print(f"    [card_build] 流式 extend 失败 {str(e)[:60]}（read_sections 兜底同步制卡）")
        return False


def collect_card_build_session(session_id: str, timeout: float | None = None,
                               mq_url: str | None = None,
                               queue: str | None = None,
                               result_dir: str | None = None) -> dict:
    """收集 MQ1 制卡结果 {tid: {"ok":True,"data":{"card":{...},"via":...}} | {"ok":False,...}}；
    等待全部完成（含 MQ2 兜底），成功后清理注册表。会话不存在 → 抛 MqUnavailable。"""
    session = _SESSIONS.get(session_id or "")
    if not session:
        raise MqUnavailable(f"MQ1 会话 {session_id} 不存在")
    try:
        replies = session.collect(timeout=timeout)
    finally:
        _SESSIONS.pop(session_id, None)
    n_ok = sum(1 for r in replies.values() if r and r.get("ok") and
               (r.get("data") or {}).get("card"))
    print(f"    [card_build] MQ1 制卡完成 {n_ok}/{len(replies)} 篇"
          f"（纯关键词路径，并发 {MQ_CARD_BUILD_CONCURRENCY}）")
    return replies

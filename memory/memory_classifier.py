# -*- coding: utf-8 -*-
"""
memory_classifier.py —— LLM-as-Judge 记忆分类与存储

每轮对话结束后调用：LLM 评估对话中是否有值得长期保留的信息，
按「关于模型 / 关于用户 / 底线 / 会话事实」分类，并路由到对应存储：

  personality_preference → Postgres user_profiles(SOUL)        跨会话
  user_preference / important_fact → Postgres(USER) + Infinity mem  跨会话
  zero_tolerance          → Postgres user_profiles(MEMORY)     跨会话
  personal_experience / project_context / technical → Infinity mem  会话级

运行环境：Python 3.11。LLM 用 OpenAI 兼容接口直连（不依赖 langchain）。
"""
import os
import json

from openai import OpenAI

from ResearcherAgent.memory.memory_stack import MemoryStack


# 对话模型（记忆分类用）——归 router 角色。
# ⚠️ **每次调用时**读环境变量，不在导入时固化：本模块通常早于 load_dotenv() 被导入，
#    模块级常量会读到空值。这与 rerank.py 是同一类时序陷阱。
#
# ⚠️ 优先走代理（OPENAI_API_BASE），只有没配代理时才退回 DeepSeek 官方端点：
#    本机官方余额已耗尽（402），而记忆分类的异常是被吞掉的——
#    实测表现是「分类静默不工作」，日志里只有一行「提取跳过」，很难发现。
def _chat_cfg() -> tuple[str, str, str]:
    """(base, key, model) —— 跟随 MODEL_MODE 走。

    single 模式用 SINGLE_*；multi 模式归 router 角色（轻量结构化任务）。
    """
    if (os.environ.get("MODEL_MODE") or "multi").strip().lower() == "single":
        return (os.environ.get("SINGLE_BASE") or os.environ.get("DEEPSEEK_API_BASE")
                or "https://api.deepseek.com",
                os.environ.get("SINGLE_KEY") or os.environ.get("DEEPSEEK_API_KEY") or "",
                os.environ.get("SINGLE_MODEL") or "deepseek-v4-flash")
    return (os.environ.get("OPENAI_API_BASE")
            or os.environ.get("DEEPSEEK_API_BASE") or "https://api.deepseek.com",
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY") or "",
            os.environ.get("ROUTER_MODEL")
            or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-flash")

EMBED_BASE = os.environ.get("EMBEDDING_API_BASE", "https://az.gptplus5.com/v1")
EMBED_KEY = os.environ.get("EMBEDDING_API_KEY", "")
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")

_CLASSIFY_PROMPT = """你是一个记忆管理助手。请判断下面这段对话中，是否有值得长期保留的信息。

判断依据（一条信息属于哪类）：
- personality_preference：用户希望「模型」成为什么样（语气/性格/自称/行为风格）——关于模型
- user_preference：用户的偏好/习惯（格式、语言、称谓、写作风格）——关于用户
- important_fact：用户的重要身份/事实（职业、研究方向、联系方式）
- zero_tolerance：零容错底线（硬约束、不可违反的规则）
- personal_experience：用户的个人经历/场景
- project_context：正在进行的项目背景
- technical：技术性上下文（会话级短期记忆）
- 闲聊/寒暄/百科问答 → 不提取

要求：
1. 只输出 JSON 数组，如 [{"type":"user_preference","summary":"用户偏好APA引用格式"}]
2. summary 用 1 句话精炼概括（≤40 字），不包含对话原文
3. 一条信息只能归一类；无值得保留的信息输出 []
4. 只输出 JSON，不要任何解释文字

对话：
【用户】{user}
【助手】{assistant}"""


def make_deterministic_embed(dim=None):
    """确定性离线 embedding（哈希+归一化）。用于测试/降级兜底，不依赖网络。

    相同文本 → 相同向量（可复现）；不同文本余弦接近 0（无语义，仅保功能）。
    """
    import hashlib as _hashlib
    dim = dim or int(os.environ.get("EMBED_DIM", "3072"))

    def embed(text):
        h = _hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(((h[i % 32] + (i * 7) % 256) % 256) / 255.0) * 2 - 1 for i in range(dim)]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    embed.degraded = True
    return embed


def make_embed_fn():
    """返回 embedding 函数（text-embedding-3-large，3072 维）。

    - timeout 15s：避免代理端点慢/挂时阻塞关键路径
    - 重试 3 次（退避 1s/2s）：代理连接偶发 SSL/EOF 错误，重试可恢复
    - 远程持久不可用 → 回退确定性伪向量（`embed.degraded = True` 标记），
      写入不崩、检索走全文 BM25 兜底
    """
    import time as _time
    dim = int(os.environ.get("EMBED_DIM", "3072"))
    client = OpenAI(base_url=EMBED_BASE, api_key=EMBED_KEY, timeout=15.0)
    fallback = make_deterministic_embed(dim)

    def embed(text):
        # ---- L2 缓存：同一段文本反复 embed 是常态（对话轮次、文档 chunk）----
        _cache = None
        try:
            from ResearcherAgent.core import cache as _cache
            hit = _cache.embed_get(EMBED_MODEL, text)
            if hit is not None:
                return hit
        except Exception:
            _cache = None
        last = None
        for i in range(3):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=text)
                vec = resp.data[0].embedding
                try:
                    if _cache is not None:
                        _cache.embed_set(EMBED_MODEL, text, vec)
                except Exception:
                    pass
                return vec
            except Exception as e:
                last = e
                if i < 2:
                    _time.sleep(1 * (i + 1))
        print(f"⚠️ 远程 embedding 不可用（{type(last).__name__}），降级为确定性向量")
        return fallback(text)

    embed.degraded = False
    return embed


def make_chat_fn():
    """返回 LLM 对话函数（OpenAI 兼容接口）。

    注意：`OpenAI()` 客户端在**调用时**才建（惰性），否则 load_dotenv 之前的
    base_url/api_key 会是空值。
    """
    def chat(prompt, temperature=0.2):
        base, key, model = _chat_cfg()
        client = OpenAI(base_url=base, api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    return chat


def classify_and_store(user_request, assistant_reply, user_id, session_id,
                       stack=None, chat_fn=None, embed_fn=None):
    """LLM-as-Judge：分类一段对话并写入记忆存储。返回存储报告 dict。

    Args:
        user_request: 用户消息
        assistant_reply: 助手回复
        user_id / session_id: 归属
        stack: MemoryStack 实例（缺省自动创建）
        chat_fn / embed_fn: 可注入（便于测试）
    """
    if chat_fn is None:
        chat_fn = make_chat_fn()
    if embed_fn is None:
        embed_fn = make_embed_fn()
    if stack is None:
        stack = MemoryStack(embed_fn=embed_fn)

    # 用 replace 而非 format：prompt 里 JSON 示例的 {} 不能被当作占位符
    prompt = (_CLASSIFY_PROMPT
              .replace("{user}", (user_request or "")[:2000])
              .replace("{assistant}", (assistant_reply or "")[:3000]))
    raw = chat_fn(prompt)

    items = _parse_items(raw)
    report = {"raw": raw[:500], "items": items, "stored": []}

    for it in items:
        typ = it.get("type", "")
        summary = (it.get("summary", "") or "").strip()
        if not summary:
            continue
        anchor = {"thread_id": session_id, "session_id": session_id}
        try:
            if typ == "personality_preference":
                stack.pg.upsert_profile(user_id, "SOUL", summary, session_id)
                report["stored"].append(f"{typ}→SOUL.md")
            elif typ in ("user_preference", "important_fact"):
                stack.pg.upsert_profile(user_id, "USER", summary, session_id)
                stack.infinity.store_memory(user_id, typ, summary, anchor, session_id, embed_fn)
                report["stored"].append(f"{typ}→USER.md + Infinity")
            elif typ == "zero_tolerance":
                stack.pg.upsert_profile(user_id, "MEMORY", summary, session_id)
                report["stored"].append(f"{typ}→MEMORY.md")
            else:  # personal_experience / project_context / technical（会话级）
                stack.infinity.store_memory(user_id, typ, summary, anchor, session_id, embed_fn)
                report["stored"].append(f"{typ}→Infinity(会话级)")
        except Exception as e:
            report.setdefault("errors", []).append(f"{typ}: {e}")

    return report


def _parse_items(raw: str) -> list:
    """解析 LLM 输出的 JSON 数组（容错：截取 [] 之间的内容）。"""
    if not raw:
        return []
    try:
        text = raw.strip()
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1:
            data = json.loads(text[start:end + 1])
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
    except Exception:
        pass
    return []

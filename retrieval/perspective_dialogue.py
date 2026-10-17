# -*- coding: utf-8 -*-
"""
perspective_dialogue.py —— 多视角对话式检索（移植自 STORM 的知识整理阶段）

STORM 原文机制（knowledge_storm/storm_wiki/modules/knowledge_curation.py）：
  persona 生成 → 每个视角并行跑 ConvSimulator（最多 max_conv_turn 轮）：
     WikiWriter（带 persona 的作者，一次只问一个问题）
     → TopicExpert（把问题拆成搜索 query → 检索 → 句句带引用回答）
  全部对话轮次 + 检索结果统一进信息表，供大纲生成与分节写作使用。

按中文综述场景适配：
  · persona 生成 + 提问 = qwen（QWEN_EVAL_*，qwen3.7-flash@maas，用户指定）
  · 专家拆词 + 回答   = deepseek（SINGLE_*/MAIN_MODEL，主 LLM）
  · 检索              = 直连学术源（OpenAlex 主源 / Europe PMC / arXiv，
                        与综述检索白名单一致；NO_PROXY 绕开失效系统代理；
                        Semantic Scholar 已移出，不再使用）
  · 输出              = 对话日志 + 每轮检索证据（标题/年份/来源），可落盘 JSON

用法（CLI 试跑）：
  python ResearcherAgent/retrieval/perspective_dialogue.py "时序预测" [--max-persona 3] [--max-turn 3]
"""
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents
             if (p / "ResearcherAgent" / "__init__.py").exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 直连学术源：本机系统代理（Clash 7890）境外出口已失效，显式绕开（与 config.py 的实测结论一致）
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import dotenv
import requests
from openai import OpenAI

from ResearcherAgent.core.paths import REPO_ROOT

# ── 模型配置（与 agent_runtime / memory_classifier 同一套环境变量）───────────
def _ds_cfg():
    dotenv.load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("SINGLE_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
    base = (os.environ.get("SINGLE_BASE") or "https://api.deepseek.com").strip()
    model = (os.environ.get("SINGLE_MODEL") or os.environ.get("MAIN_MODEL")
             or "deepseek-v4-flash").strip()
    return key, base, model

def _qwen_cfg():
    dotenv.load_dotenv(REPO_ROOT / ".env")
    key = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    base = (os.environ.get("QWEN_EVAL_BASE")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
    model = (os.environ.get("QWEN_EVAL_MODEL") or "qwen3.7-flash").strip()
    return key, base, model


def _call(client, model, messages, temperature=0.7, max_tokens=2048, retry=4):
    """单次 LLM 调用 + 指数退避重试。

    ⚠️ deepseek-v4-flash 是推理模型：max_tokens 过小时全部消耗在
    reasoning_content 上，正式 content 为空且 finish_reason=length。
    因此：content 为空且 finish_reason=length → 翻倍 max_tokens 重试。
    """
    last = None
    for i in range(retry):
        try:
            r = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=messages)
            msg = r.choices[0].message
            content = (msg.content or "").strip()
            if content:
                return content
            # 推理模型被截断：翻倍配额再试一次（上限 32768）
            if r.choices[0].finish_reason == "length" and max_tokens < 32768:
                print(f"    ⚠️ 推理 token 占满（{max_tokens}），翻倍到 {max_tokens * 2} 重试…")
                max_tokens *= 2
                continue
            last = f"content 为空（finish_reason={r.choices[0].finish_reason}）"
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:160]}"
            print(f"    ⚠️ LLM 第 {i+1} 次失败：{last}；退避重试…")
        time.sleep(2 ** (i + 1))
    raise RuntimeError(f"LLM 调用失败（{retry} 次）：{last}")


# ── 检索：requests 直连学术 API（Europe PMC / arXiv / CORE，与综述白名单三源一致）──
# 不依赖 paper_search_mcp 第三方包；NO_PROXY 绕开失效系统代理。
# Semantic Scholar / OpenAlex 已移除（429 频繁 / 本轮 E2E 多次 ReadTimeout 拖慢检索）。
import xml.etree.ElementTree as ET

def _fetch_json(url: str, params: dict, timeout: float = 20.0) -> dict | None:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    ⚠️ 检索失败 {url.split('/')[2]}：{type(e).__name__}: {str(e)[:100]}")
        return None

def _search_core(query: str, max_results: int) -> list[dict]:
    """CORE v3 API（OA 全文文本源；需要 CORE_API_KEY，Bearer 认证）。"""
    key = (os.environ.get("CORE_API_KEY") or os.environ.get("PAPER_SEARCH_MCP_CORE_API_KEY") or "").strip()
    if not key:
        return []
    import re as _re
    try:
        r = requests.get("https://api.core.ac.uk/v3/search/works",
                         params={"q": query, "limit": max_results},
                         headers={"Authorization": "Bearer " + key,
                                  "User-Agent": "ReviewAgent/1.0"}, timeout=25)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"    ⚠️ 检索失败 core.ac.uk：{type(e).__name__}: {str(e)[:100]}")
        return []
    out = []
    for w in data.get("results", []) or []:
        ab = str(w.get("abstract") or "")
        ab = _re.sub(r"<[^>]+>", " ", ab).strip()[:600]
        out.append({
            "title": str(w.get("title") or "").strip(),
            "year": w.get("yearPublished") or "",
            "venue": str(w.get("publisher") or "").strip(),
            "url": f"https://doi.org/{w['doi']}" if w.get("doi") else "",
            "abstract": ab,
        })
    return out

def _search_europepmc(query: str, max_results: int) -> list[dict]:
    """Europe PMC（开放获取全文通路：命中即含 PMCID，可直取 JATS 全文）。"""
    data = _fetch_json("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                       {"query": query, "format": "json", "pageSize": max_results,
                        "resultType": "core"})
    if not data:
        return []
    out = []
    for item in (data.get("resultList") or {}).get("result", []) or []:
        out.append({
            "title": str(item.get("title") or "").strip(),
            "year": str(item.get("pubYear") or ""),
            "venue": str(item.get("journalTitle") or "").strip(),
            "url": f"https://doi.org/{item['doi']}" if item.get("doi") else
                   f"https://europepmc.org/article/PMC/{item['pmcid']}" if item.get("pmcid") else "",
            "abstract": str(item.get("abstractText") or "").strip()[:600],
            "pmcid": item.get("pmcid") or "",
        })
    return out

def _search_arxiv(query: str, max_results: int) -> list[dict]:
    try:
        r = requests.get("http://export.arxiv.org/api/query",
                         params={"search_query": f"all:{query}",
                                 "max_results": max_results}, timeout=25)
        r.raise_for_status()
    except Exception as e:
        print(f"    ⚠️ 检索失败 arxiv：{type(e).__name__}: {str(e)[:100]}")
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.content)
    out = []
    for entry in root.findall("a:entry", ns):
        title = (entry.findtext("a:title", "", ns) or "").strip().replace("\n", " ")
        out.append({
            "title": title,
            "year": (entry.findtext("a:published", "", ns) or "")[:4],
            "venue": "arXiv",
            "url": entry.findtext("a:id", "", ns) or "",
            "abstract": (entry.findtext("a:summary", "", ns) or "").strip()[:600],
        })
    return out

def _search(query: str, max_results: int = 5) -> list[dict]:
    """检索论文：Europe PMC（全文通路）→ arXiv → CORE（OA 全文，与综述白名单三源一致，
    Semantic Scholar / OpenAlex 已移除）；去重、摘要优先。"""
    out: dict[str, dict] = {}
    for fn in (_search_europepmc, _search_arxiv, _search_core):
        try:
            for row in fn(query, max_results=max_results) or []:
                title = str(row.get("title") or "").strip()
                if len(title) < 8:
                    continue
                key = title.lower().replace(" ", "")
                if key in out:
                    continue
                out[key] = row
        except Exception as e:
            print(f"    ⚠️ 检索源异常：{type(e).__name__}: {str(e)[:100]}")
    rows = list(out.values())
    # 摘要优先：带摘要（≥80 字）的排前面，保证专家回答有可引用细节
    rows.sort(key=lambda r: (len(r.get("abstract") or "") >= 80), reverse=True)
    return rows[:max_results * 2]


# ── persona 生成（qwen）──────────────────────────────────────────────
PERSONA_PROMPT = (
    "你是一群准备共同撰写一篇关于《{topic}》学术综述的研究者。"
    "请从不同研究视角分工，列出 {max_persona} 个视角，每个视角代表一个身份/侧重点，"
    "并给出一句话说明该视角关注什么。只输出编号列表，不要多余文字。格式：\n"
    "1. 视角名：关注点\n2. 视角名：关注点\n..."
)

def generate_personas(topic: str, max_persona: int = 3) -> list[str]:
    key, base, model = _qwen_cfg()
    if not key:
        raise RuntimeError("未找到 DASHSCOPE_API_KEY（.env）——qwen 提问模型无法启动")
    client = OpenAI(api_key=key, base_url=base, timeout=60)
    text = _call(client, model,
                 [{"role": "system", "content": "你是学术综述选题策划专家。"},
                  {"role": "user", "content": PERSONA_PROMPT.format(
                      topic=topic, max_persona=max_persona)}],
                 temperature=0.7, max_tokens=1024)
    personas = []
    for line in text.splitlines():
        m = re.match(r"^\s*\d+[.、]\s*(.*)", line)
        if m:
            s = m.group(1).strip()
            if 4 <= len(s) <= 80:
                personas.append(s)
    return personas[:max_persona]


# ── WikiWriter：带视角的提问（qwen）──────────────────────────────────
WRITER_PROMPT = (
    "你是一名经验丰富的综述撰写者，正在与一位领域专家对话，为撰写《{topic}》综述收集信息。"
    "除综述撰写者身份外，你还专注于特定视角：{persona}\n"
    "请基于对话历史提出一个好问题，获取更有用的信息。规则：\n"
    "1. 每次只问一个问题，不要问之前问过的问题；\n"
    "2. 问题应与综述主题相关，且尽量深入（基于专家上一轮回答追问）；\n"
    "3. 当你确认没有更多值得问的问题时，只输出「谢谢你的帮助，我没有其他问题了！」结束。\n\n"
    "对话历史：\n{conv}\n\n请只输出你的问题："
)

def ask_question(topic: str, persona: str, turns: list[dict]) -> str:
    key, base, model = _qwen_cfg()
    client = OpenAI(api_key=key, base_url=base, timeout=60)
    conv_lines = []
    for t in turns:
        conv_lines.append(f"你（综述撰写者）：{t['question']}")
        conv_lines.append(f"专家：{t['answer']}")
    conv = "\n".join(conv_lines) or "（对话刚开始，还没有历史）"
    conv = conv[-2500:]
    text = _call(client, model,
                 [{"role": "system", "content": "你是严谨的学术综述资料收集者。"},
                  {"role": "user", "content": WRITER_PROMPT.format(
                      topic=topic, persona=persona, conv=conv)}],
                 temperature=0.6, max_tokens=1024)
    return text


# ── TopicExpert：拆词（deepseek）→ 检索 → 带引用回答（deepseek）────────
QUERY_PROMPT = (
    "You want to search academic paper databases to answer the question below. "
    "What search phrases would you type?\n"
    "Requirements:\n"
    "- **English phrases only** (academic databases have much better English coverage);\n"
    "- Each phrase must be a **short key phrase** (≤ 30 characters), not a full sentence;\n"
    "- Do not copy the whole question; do not add punctuation or quotes.\n"
    "Output at most 3 phrases in this format:\n- phrase 1\n- phrase 2\n- phrase 3\n\n"
    "Topic under discussion: {topic}\nQuestion to answer: {question}"
)

def _norm_query(q: str, max_len: int = 60) -> str:
    """检索词规范化：去标点/引号/句尾语气，超长截断——避免 OpenAlex 对长自然语言 query 返回 500。"""
    q = re.sub(r"[，。？！、；：“”\"'（）：…\n\t]+", " ", q).strip()
    return q[:max_len]

EXPERT_PROMPT = (
    "你是一位能够有效利用信息的领域专家，正在与一位综述撰写者对话，"
    "他将撰写一篇关于《{topic}》的综述。你已收集到如下检索到的论文信息。\n"
    "请基于这些信息回答问题：\n"
    "1. 尽可能信息丰富，**每一句都要有检索信息支持**，句末用 [n] 标注来源编号；\n"
    "2. 若检索信息与问题不直接相关，给出基于现有信息最相关的回答；\n"
    "3. 若无法形成合适回答，请回复「我无法基于现有信息回答这个问题」并说明局限；\n"
    "4. 不要编造编号之外的论文。\n\n"
    "检索到的论文信息：\n{info}\n\n问题：{question}\n\n请给出回答："
)

def _split_queries(topic: str, question: str) -> list[str]:
    """问题 → 搜索词（deepseek QUERY_PROMPT）：英文短语 ≤30 字符，最多 3 条。
    供专家回答与维度化建池复用；deepseek 推理模型直接给足 token 预算。"""
    key, base, model = _ds_cfg()
    if not key:
        raise RuntimeError("未找到 DeepSeek API key（.env 的 SINGLE_KEY / DEEPSEEK_API_KEY）")
    client = OpenAI(api_key=key, base_url=base, timeout=90)
    q_text = _call(client, model,
                   [{"role": "user", "content": QUERY_PROMPT.format(
                       topic=topic, question=question)}],
                   temperature=0.3, max_tokens=8192)
    queries = [q.lstrip("-•* ").strip() for q in q_text.splitlines() if q.strip()]
    queries = [q for q in queries if 2 <= len(q) <= 80][:3]
    queries = [_norm_query(q) for q in queries]
    if not queries:            # 兜底：拆词失败直接用问题原文作检索词
        queries = [_norm_query(question)]
        print("    ⚠️ 拆词结果为空，直接用问题原文作检索词")
    return queries


def expert_answer(topic: str, question: str) -> dict:
    key, base, model = _ds_cfg()
    if not key:
        raise RuntimeError("未找到 DeepSeek API key（.env 的 SINGLE_KEY / DEEPSEEK_API_KEY）")
    client = OpenAI(api_key=key, base_url=base, timeout=90)

    # ① 拆搜索词（deepseek 推理模型：直接给足 token 预算，避免翻倍浪费轮次）
    queries = _split_queries(topic, question)

    # ② 检索
    results = []
    for q in queries:
        results.extend(_search(q))
    seen = set()
    uniq = []
    for r in results:
        if r["title"] not in seen:
            seen.add(r["title"])
            uniq.append(r)
    results = uniq[:10]

    # ③ 带引用回答
    info_lines = []
    for i, r in enumerate(results, 1):
        info_lines.append(
            f"[{i}]《{r['title']}》（{r.get('year') or '年份未知'}，{r.get('venue') or '来源未知'}）\n"
            f"    摘要：{r.get('abstract') or '（无摘要）'}")
    info = "\n".join(info_lines) or "（检索未返回任何结果）"
    answer = _call(client, model,
                   [{"role": "system", "content": "你是严谨的领域专家，绝不编造来源。"},
                    {"role": "user", "content": EXPERT_PROMPT.format(
                        topic=topic, question=question, info=info)}],
                   temperature=0.4, max_tokens=4096)

    return {
        "queries": queries,
        "results": results,
        "answer": answer,
    }


# ── 主入口：多视角对话式检索 ──────────────────────────────────────────
def run_perspective_dialogue(topic: str, max_persona: int = 3,
                             max_turn: int = 3) -> dict:
    """返回 {topic, personas, conversations:[{persona, turns:[{question, queries, results, answer}]}]}"""
    print(f"\n══ 多视角对话式检索 · 主题「{topic}」══")
    print("① 生成研究视角（qwen3.7-flash）…")
    personas = generate_personas(topic, max_persona=max_persona)
    if not personas:
        print("  ⚠️ 视角生成失败，回退为单个默认视角")
        personas = ["整体综述视角"]
    for i, p in enumerate(personas, 1):
        print(f"  视角{i}: {p}")

    conversations = []
    for pi, persona in enumerate(personas, 1):
        print(f"\n② 视角{pi}/{len(personas)}「{persona}」对话模拟（提问=qwen / 专家=deepseek）…")
        turns = []
        for turn in range(1, max_turn + 1):
            print(f"  轮次{turn}/{max_turn}：")
            question = ask_question(topic, persona, turns)
            if "谢谢你的帮助" in question or "没有其他问题" in question:
                print(f"    ✋ 作者表示没有问题可问了：{question}")
                break
            print(f"    作者提问：{question}")
            out = expert_answer(topic, question)
            print(f"    专家检索词：{out['queries']}")
            print(f"    检索到 {len(out['results'])} 篇论文")
            print(f"    专家回答：{out['answer'][:160]}{'…' if len(out['answer']) > 160 else ''}")
            turns.append({"question": question, **out})
        conversations.append({"persona": persona, "turns": turns})

    return {"topic": topic, "personas": personas,
            "conversations": conversations}


# ── 接入点②：维度化检索建池（多视角对话 → 按维度分桶 → 并入综述检索池）──
def _pool_item(row: dict, persona: str) -> dict:
    """检索结果 → 综述 paper_pool 条目（字段与 agent_tools.parse_papers 对齐）。"""
    return {
        "title": str(row.get("title") or "").strip(),
        "year": row.get("year") or "",
        "venue": str(row.get("venue") or "").strip(),
        "url": row.get("url") or "",
        "abstract": str(row.get("abstract") or "").strip(),
        "source": "dimension_probe",
        "dimension": persona,
    }

def _norm_title(t: str) -> str:
    return re.sub(r"\s+", "", str(t or "").lower())

def build_dimension_pool(topic: str, max_persona: int = 3,
                         max_turn: int = 1) -> dict:
    """维度化检索建池：多视角对话产出"问题/搜索词"→ 按视角分桶检索 → 维度化论文池。

    对应 STORM 知识收集的"结构发现"价值，但**不经过大纲**：
      每个视角的问题 = 主题的一个隐藏维度；用该视角拆出的 query 驱动检索，
      保证检索池覆盖主题各面（算法/应用/数据/评估/挑战…），而不是只搜主题词。

    返回 {topic, personas, dimensions:[{persona, questions, queries, papers}],
           merged_pool:[...]}；merged_pool 已去重，字段对齐综述 paper_pool。
    max_turn>1 时，下一轮提问以上一轮专家回答为上下文（更深入，成本更高）。
    """
    print(f"\n══ 维度化检索建池 · 主题「{topic}」══")
    print("① 生成研究视角（qwen3.7-flash）…")
    personas = generate_personas(topic, max_persona=max_persona)
    if not personas:
        print("  ⚠️ 视角生成失败，回退为单个默认视角")
        personas = ["整体综述视角"]
    for i, p in enumerate(personas, 1):
        print(f"  视角{i}: {p}")

    dimensions = []
    seen: dict[str, dict] = {}          # 全池去重（跨维度、跨轮次）
    for pi, persona in enumerate(personas, 1):
        print(f"\n② 视角{pi}/{len(personas)}「{persona}」：提问 → 拆词 → 按维度检索…")
        turns: list[dict] = []          # 对话上下文（仅用于下一轮提问）
        dim_papers: dict[str, dict] = {}  # 本维度去重
        dim_queries: list[str] = []
        n_asked = 0
        for turn in range(1, max_turn + 1):
            question = ask_question(topic, persona, turns)
            if "谢谢你的帮助" in question or "没有其他问题" in question:
                print(f"    ✋ 视角{pi} 作者表示没有问题可问了")
                break
            n_asked += 1
            queries = _split_queries(topic, question)
            dim_queries.extend(queries)
            print(f"    轮{turn} 提问：{question[:80]}")
            print(f"      搜索词：{queries}")
            rows = []
            for q in queries:
                rows.extend(_search(q, max_results=5))
            for r in rows:
                item = _pool_item(r, persona)
                k = _norm_title(item["title"])
                if len(k) < 8:
                    continue
                dim_papers[k] = item
                seen[k] = item
            # 专家回答作为下一轮提问上下文（建池本身不消费回答，只消费 query 检索结果）
            if max_turn > 1 and turn < max_turn:
                try:
                    ans = expert_answer(topic, question)
                    turns.append({"question": question, "answer": ans["answer"]})
                    for r in ans.get("results") or []:
                        item = _pool_item(r, persona)
                        k = _norm_title(item["title"])
                        if len(k) >= 8:
                            dim_papers[k] = item
                            seen[k] = item
                    dim_queries.extend(ans.get("queries") or [])
                except Exception as e:
                    print(f"    ⚠️ 专家回答失败（不影响建池）：{str(e)[:100]}")
                    break
        dim_list = list(dim_papers.values())
        print(f"  视角{pi} 维度池：{len(dim_list)} 篇（提问 {n_asked} 轮）")
        dimensions.append({"persona": persona, "queries": dim_queries,
                           "papers": dim_list})

    merged = list(seen.values())
    print(f"\n══ 建池完成：{len(personas)} 个维度 / 共 {len(merged)} 篇（去重）══")
    return {"topic": topic, "personas": personas,
            "dimensions": dimensions, "merged_pool": merged}


def dump_and_print(result: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{result['topic']}.json"
    fp.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"\n══ 结果已落盘：{fp} ══")
    print(f"视角数：{len(result['personas'])}｜总问答轮数："
          f"{sum(len(c['turns']) for c in result['conversations'])}")
    for c in result["conversations"]:
        print(f"\n【视角】{c['persona']}")
        for t in c["turns"]:
            print(f"  Q: {t['question']}")
            print(f"    queries: {t['queries']}")
            print(f"    papers : {len(t['results'])} 篇（"
                  + "；".join(r["title"][:40] for r in t["results"][:3]) + "…）")
            print(f"    A: {t['answer']}")
    return fp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("topic", nargs="?", default="",
                    help="研究主题（示例演示默认：时序预测；实际使用传入任意领域主题）")
    ap.add_argument("--max-persona", type=int, default=3)
    ap.add_argument("--max-turn", type=int, default=3)
    ap.add_argument("--build-pool", action="store_true",
                    help="维度化建池模式：每视角提问→拆词→按维度检索（默认对话模式）")
    args = ap.parse_args()

    t0 = time.time()
    if args.build_pool:
        result = build_dimension_pool(args.topic, max_persona=args.max_persona,
                                      max_turn=max(1, min(args.max_turn, 3)))
        out = REPO_ROOT / "logs" / "eval" / "dimension_pool"
        out.mkdir(parents=True, exist_ok=True)
        fp = out / f"{result['topic']}.json"
        fp.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        print(f"\n══ 维度池已落盘：{fp} ══")
        print(f"维度数：{len(result['dimensions'])}｜合并池：{len(result['merged_pool'])} 篇")
        for d in result["dimensions"]:
            print(f"  【{d['persona']}】{len(d['papers'])} 篇｜queries: {d['queries']}")
    else:
        result = run_perspective_dialogue(args.topic, max_persona=args.max_persona,
                                          max_turn=args.max_turn)
        out = REPO_ROOT / "logs" / "eval" / "perspective_dialogue"
        fp = dump_and_print(result, out)
    print(f"\n总耗时 {time.time() - t0:.1f}s")

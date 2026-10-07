# -*- coding: utf-8 -*-
"""
discover_directions.py —— 方向级热点发现（从文献中发现"新的研究方向"）。

照搬成熟 agent 体系（Deep Ideation / SciAgents / Scideator / SciMON / VIRSCI）的分层逻辑：
  ① 候选提取（LLM，只读新文献）—— LLM 的强项：读摘要识别主题概念
  ② 统计验证（确定性代码，全池）—— LLM 的弱项：数数/算增长由代码裁决
  ③ 决策（LLM，必须引用具体论文）—— Scideator 式 novelty 评估，防编造
  ④ 落地（代码）—— 批准的方向 merge 进 subtopics

分工原则：LLM 负责"提出"与"判断"，代码负责"数数"与"落地"。
"""
import json
import os
import re
import sys
import time
from pathlib import Path

# --- repo-root bootstrap（本模块既被 import，也有 `python ResearcherAgent/research/discover_directions.py` 的 CLI 用法）---
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ResearcherAgent" / "__init__.py").exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dotenv
from openai import OpenAI

from ResearcherAgent.core.paths import REPO_ROOT

CACHE_DIR = Path("logs") / "eval" / "discover_directions"

# ═══ ① 候选提取 ═══
CANDIDATE_SYSTEM = """你是学术文献分析助手。下面给出近期（近 2-3 年）论文的标题与摘要列表。

任务：从中提取「近期研究热点方向/概念」候选——即近期频繁出现在这些论文中、
且可能值得成为综述里一个独立小节的研究方向。

要求：
- 提取的是**研究方向/技术概念**（如 "Speculative Decoding"、"Retrieval-Augmented Generation"、
  "Mixture of Experts"、"Test-Time Compute"），不是具体论文、不是作者名、不是期刊名
- 每个候选附 1 句理由（为什么它近期受关注）
- 每个候选给出 2-5 个 example_titles，**必须逐字来自给定论文标题列表**，用于后续统计验证
- 不要提空泛通用词（如 "LLM"、"transformer"、"deep learning"、"performance"、
  "model"、"paper" 这类在所有论文里都出现的词）
- 如果近期文献里没有明显的热点方向，输出空数组，不要硬编

只输出一个 JSON 数组，不要输出任何其他文字。格式：
[{"concept": "候选方向名", "reason": "一句话理由", "example_titles": ["论文标题1", "论文标题2"]}]"""


def _load_env() -> tuple[str, str, str]:
    dotenv.load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("SINGLE_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
    base = (os.environ.get("SINGLE_BASE") or "https://api.deepseek.com").strip()
    model = (os.environ.get("SINGLE_MODEL") or "deepseek-v4-flash").strip()
    return key, base, model


def _extract_json(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I).strip()
    s = t.find("[")
    e = t.rfind("]")
    if s == -1 or e == -1 or e <= s:
        return None
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return None


def _extract_json_obj(text: str):
    """解析 JSON 对象（decide 用）：容忍围栏与前后杂音。"""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I).strip()
    s = t.find("{")
    e = t.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return None


def _chat(system: str, user: str, temperature: float = 0.3,
          max_retry: int = 2, json_mode: bool = True) -> str:
    key, base, model = _load_env()
    if not key:
        raise RuntimeError("未找到 API key（.env 的 SINGLE_KEY / DEEPSEEK_API_KEY）")
    client = OpenAI(api_key=key, base_url=base)
    last = None
    for i in range(max_retry + 1):
        try:
            r = client.chat.completions.create(
                model=model, temperature=temperature,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return r.choices[0].message.content or ""
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"  ⚠️ LLM 第 {i+1} 次尝试失败：{last}；重试…")
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"LLM 调用失败（{max_retry+1} 次）：{last}")


def _year_of(p: dict) -> int:
    """兼容多种字段的年份提取：year / publication_year / published_date(含 ISO 日期或 arXiv yymmdd)。"""
    for k in ("year", "publication_year"):
        v = p.get(k)
        if v:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    d = str(p.get("published_date") or "")
    m = re.search(r"(19|20)\d{2}", d)
    if m:
        return int(m.group(0))
    pid = str(p.get("paper_id") or "")
    m = re.match(r"(\d{4})", pid)  # arXiv id 如 1609.04846v1 → 2016
    if m and len(pid) > 5:
        y = int(m.group(1))
        if 1900 < y < 2100:
            return y
    return 0


def extract_candidates(pool: list[dict], min_year: int = 2024,
                       max_papers: int = 60) -> list[dict]:
    """① 候选提取：只把较新的论文喂给 LLM（按被引数取 top），避免提出全历史通用词。"""
    recent = [p for p in pool if _year_of(p) >= min_year]
    recent.sort(key=lambda p: -(p.get("citations") or 0))
    recent = recent[:max_papers]
    if not recent:
        return []
    lines = []
    for i, p in enumerate(recent):
        t = (p.get("title") or "").strip()
        a = (p.get("abstract") or "").strip()
        lines.append(f"[{i}] ({_year_of(p)}) {t}\n  摘要: {a[:600]}")
    user = "近期论文列表（标题+摘要）：\n\n" + "\n\n".join(lines)
    raw = _chat(CANDIDATE_SYSTEM, user)
    val = _extract_json(raw) or []
    out = []
    for it in val:
        if not isinstance(it, dict):
            continue
        concept = str(it.get("concept") or "").strip()
        if len(concept) < 3:
            continue
        titles = [str(t).strip() for t in (it.get("example_titles") or []) if t]
        if not titles:
            continue
        out.append({"concept": concept,
                    "reason": str(it.get("reason") or "")[:200],
                    "example_titles": titles})
    if len(out) > 5:
        print(f"  ⚠️ 候选 {len(out)} 个，截断到 5 个（研究方向最多列举 5 个）")
        out = out[:5]
    print(f"  ① 候选提取: 输入新文献 {len(recent)} 篇 → 候选 {len(out)} 个")
    return out


_STOP = {"and", "or", "for", "in", "on", "with", "of", "the", "a", "an", "to",
         "based", "using", "via", "towards", "toward", "from", "into", "across",
         "over", "under", "at", "by", "as", "is", "are", "was", "were"}


def _concept_tokens(concept: str) -> set[str]:
    """概念名分词（≥3 字母、去停用词）→ 用于词级弱匹配。"""
    toks = re.findall(r"[A-Za-z]{3,}", concept)
    return {t.lower() for t in toks if t.lower() not in _STOP}


def _hit(p: dict, concept: str, min_frac: float = 0.6) -> bool:
    """论文是否命中概念：完整短语子串命中 或 词级部分命中（≥min_frac 核心词）。"""
    text = f"{p.get('title') or ''} {p.get('abstract') or ''}".lower()
    if concept.lower() in text:
        return True
    toks = _concept_tokens(concept)
    if len(toks) < 3:
        return False
    cnt = sum(1 for t in toks if t in text)
    return cnt >= max(2, int(len(toks) * min_frac))


def _norm_title(t: str) -> str:
    t = re.sub(r"\s*\(\d{4}\)", "", t)      # 剥离 (2024) 年份括号（LLM 抄写时可能带上）
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def build_evidence(candidates: list[dict], pool: list[dict],
                   min_year: int = 2024) -> list[dict]:
    """② 统计验证：在**全池**数每个候选的出现篇数 + 新旧年份分布（代码裁决，LLM 不数数）。

    双通道命中：概念名字符串/词级匹配 ∪ LLM 提供的 example_titles 在池内核验。
    """
    pool_by_title = {_norm_title(p.get("title") or ""): p for p in pool}
    cards = []
    for c in candidates:
        concept = c["concept"]
        hits = [p for p in pool if _hit(p, concept)]
        # 通道 B：LLM 声称包含该概念的论文标题 → 池内核验（标题归一化匹配）
        for t in c.get("example_titles") or []:
            p = pool_by_title.get(_norm_title(t))
            if p and not any(h is p for h in hits):
                hits.append(p)
        recent = [p for p in hits if _year_of(p) >= min_year]
        old = [p for p in hits if _year_of(p) < min_year]
        # 增长信号：近两年出现为主，且(旧文献几乎不出现 或 近期出现数相对可观)
        growth = (len(recent) >= 3 and len(old) <= 2) or \
                 (len(recent) >= 5 and len(recent) >= 2 * max(len(old), 1))
        cards.append({
            "concept": concept,
            "reason": c.get("reason", ""),
            "total": len(hits), "recent": len(recent), "old": len(old),
            "growth_signal": growth,
            "support_papers": [{"title": p.get("title") or "", "year": _year_of(p)}
                               for p in sorted(hits, key=lambda q: -(q.get("citations") or 0))[:6]],
        })
    return cards


# ═══ ③ 决策（照 Scideator 的 novelty 评估：LLM 判断必须引用具体论文）═══
DECISION_SYSTEM = """你是综述项目的 Planner 评审员。研究员提交了一批"候选新研究方向"及其统计证据
（每个方向在文献池中的出现篇数、新旧年份分布、代表性论文）。

你的职责：判断每个候选是否值得**新增为综述的子主题**，或应被拒绝/并入已有子主题。

判断标准：
1. 与已有子主题明显重叠 → 拒绝或并入（注明并入哪个已有子主题）
2. 文献支撑不足（evidence.total < 3，或近两年 recent == 0）→ 拒绝
3. growth_signal 为 true 且不重叠 → 接受（这是"新兴方向"的核心信号）
4. 空泛通用词（LLM/transformer/performance 等）→ 拒绝

只输出一个 JSON 对象，不要输出其他文字。格式：
{"decisions": [
  {"concept": "候选名", "verdict": "accept|reject|merge",
   "target_subtopic": "并入的已有子主题名（仅 merge 时填，否则空字符串）",
   "reason": "判断理由，必须引用证据卡中的论文标题作为依据",
   "support_papers": ["证据卡中出现的论文标题（accept 时填 2-5 个，必须逐字来自证据卡）"]}]}"""


def decide(cards: list[dict], topic: str, subtopics: list[dict]) -> list[dict]:
    """③ 决策：LLM 拿统计证据卡判断（必须引用具体论文，防编造频率）。"""
    if not cards:
        return []
    sub_lines = [f"- {it.get('subtopic')}" for it in subtopics if it.get("subtopic")]
    ev_lines = []
    for c in cards:
        sp = "；".join(f"{p['title']}({p['year']})" for p in c["support_papers"])
        ev_lines.append(
            f"• {c['concept']} | 全池出现 {c['total']} 篇 | 近两年 {c['recent']} 篇 | "
            f"更早 {c['old']} 篇 | 增长信号 {c['growth_signal']}\n"
            f"  理由: {c['reason']}\n  代表论文: {sp}")
    user = (f"综述主题：{topic}\n\n已有子主题：\n{chr(10).join(sub_lines) or '(无)'}\n\n"
            f"候选方向统计证据：\n\n" + "\n".join(ev_lines))
    raw = _chat(DECISION_SYSTEM, user)
    val = _extract_json_obj(raw)
    if not isinstance(val, dict) or not isinstance(val.get("decisions"), list):
        return []
    out = []
    for d in val["decisions"]:
        if not isinstance(d, dict):
            continue
        concept = str(d.get("concept") or "").strip()
        verdict = str(d.get("verdict") or "").strip()
        if verdict not in ("accept", "reject", "merge"):
            continue
        out.append({
            "concept": concept,
            "verdict": verdict,
            "target_subtopic": str(d.get("target_subtopic") or "").strip(),
            "reason": str(d.get("reason") or "")[:300],
            "support_papers": [str(t).strip() for t in (d.get("support_papers") or []) if t],
        })
    return out


# ═══ ④ 落地：accept 的方向 merge 进 subtopics ═══
def apply_decisions(decisions: list[dict], subtopics: list[dict],
                    pool: list[dict],
                    evidence: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    """返回 (更新后的 subtopics, 新增的子主题列表)。

    落地论文来源（按可靠性排序）：
      ① 证据卡论文（代码在池内核实、概念命中的论文）——首选，100% 在池内
      ② LLM 的 support_papers（归一化匹配池内标题）——兜底
    两者都取不到则不落地（防幻觉污染）。
    """
    ev_by_concept = {}
    for e in evidence or []:
        ev_by_concept.setdefault(_norm_title(e.get("concept") or ""), e)
    new_subs = []
    for d in decisions:
        if d.get("verdict") != "accept":
            continue
        papers = []
        ev = ev_by_concept.get(_norm_title(d.get("concept") or ""))
        if ev:
            for p in (ev.get("support_papers") or [])[:4]:
                if p.get("title"):
                    papers.append({"title": p["title"], "year": p.get("year"),
                                   "subtopic": d["concept"]})
        if not papers:
            for t in d.get("support_papers") or []:
                t = str(t or "").strip()
                if not t:
                    continue
                m = next((p for p in pool
                          if _norm_title(p.get("title") or "") == _norm_title(t)), None)
                if m:
                    papers.append({"title": m["title"], "year": _year_of(m),
                                   "subtopic": d["concept"]})
        if not papers:
            print(f"  ⚠️ {d['concept']} 无池内核实的论文可落地，跳过")
            continue
        new_subs.append({"subtopic": d["concept"], "papers": papers,
                         "origin": "discovered"})
    merged = list(subtopics or []) + new_subs
    return merged, new_subs


def run_discovery(pool: list[dict], topic: str, subtopics: list[dict],
                  min_year: int = 2024, force: bool = False) -> dict:
    """完整流程：提取候选 → 统计验证 → 决策 → 落地。返回全流程证据（可落盘/渲染）。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ckey = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", topic)[:60]
    cf = CACHE_DIR / f"{ckey}.json"
    if cf.exists() and not force:
        try:
            return json.loads(cf.read_text(encoding="utf-8"))
        except Exception:
            pass
    print(f"--> [discover_directions] 主题: {topic[:40]} | 池 {len(pool)} 篇 | 已有子主题 {len(subtopics)}")
    candidates = extract_candidates(pool, min_year=min_year)
    cards = build_evidence(candidates, pool, min_year=min_year)
    decisions = decide(cards, topic, subtopics)
    merged, new_subs = apply_decisions(decisions, subtopics, pool, evidence=cards)
    result = {
        "topic": topic, "pool_size": len(pool), "min_year": min_year,
        "candidates": candidates, "evidence": cards, "decisions": decisions,
        "new_subtopics": new_subs, "updated_subtopics": merged,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cf.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="大模型在代码生成中的应用")
    ap.add_argument("--min-year", type=int, default=2024)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    # 独立入口：从现有 run 的 checkpoint 或日志读 pool —— 简单模式：直接读 eval JSON
    import json as _j
    pool = []
    for f in Path("logs/eval").glob("*_paper_pool.json"):
        try:
            pool = _j.loads(f.read_text(encoding="utf-8"))
            if pool:
                print(f"从 {f.name} 载入论文池 {len(pool)} 篇")
                break
        except Exception:
            continue
    if not pool:
        raise SystemExit("未找到论文池文件 logs/eval/*_paper_pool.json，请先跑主流程。")
    res = run_discovery(pool, args.topic, [], min_year=args.min_year, force=args.force)
    print("\n═══ 决策结果 ═══")
    for d in res["decisions"]:
        print(f"  [{d['verdict']}] {d['concept']} | {d['reason'][:80]}")
    print(f"\n新增子主题 {len(res['new_subtopics'])} 个")
    for s in res["new_subtopics"]:
        print(f"  + {s['subtopic']} → {len(s['papers'])} 篇论文")

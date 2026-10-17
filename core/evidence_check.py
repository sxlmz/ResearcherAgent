# -*- coding: utf-8 -*-
"""
evidence_check.py —— 确定性证据检查（引用支持度下沉，DeepSurvey 思路）

在 LLM judge 之前，用规则 + embedding（bge-m3）对正文做确定性事实校验：
  1) 锚点存在性：论断引用的 [N] 是否在引用精确映射中（缺失 → no_anchor）
  2) 语义支持度：论断句子 vs 其引用论文摘要的 embedding 余弦（低于阈值 → low_support）

输出按节的检查报告，注入 judge 局部评审作为硬信号（judge 据此从严扣分），
不再把「引用是否支持论断」完全交给 LLM 自由裁量。
"""
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from ResearcherAgent.core.paths import REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

# ════════════════════════════════════════════════════════════════
# ⚠️ 关于余弦阈值：**它作为硬信号是失效的，已改为对比式边距**（2026-09 实测）
#
# 实测（judgev3c 的稿子，74 个有锚句）：
#   真实配对（句子 vs 它引用的论文摘要）：min 0.426  中位 0.620  max 0.729
#   随机配对（句子 vs 池里随便另一篇）  ：min 0.464  中位 0.550  max 0.662
#   → **两个分布几乎完全重叠**。bge-m3 对「中文论断 vs 英文摘要」即使毫不相关
#     也给 ~0.55 的基线相似度，所以旧的绝对阈值 0.40 **低于噪声地板、永远不触发**
#     （实测效果：`确定性证据检查：0/8 节发现问题`，而同一次 judge 的 LLM
#      用读摘要的方式找出了 5 处证据问题 —— 确定性那条腿等于没有）。
#
# 改法：不再问「像不像」，改问「**引用的那篇是不是比别的更像**」——
#   边距 = sim(句, 引用论文) − max(sim(句, 若干非引用论文))
#   实测边距中位仅 +0.047，16/74 为负 → 它**也只能算辅助信号**，不是硬判据。
#   因此 verdict 改名为 `weak_match`（弱匹配），措辞上不再声称"该引用是错的"，
#   而是"这篇引用未必最贴切，请对照摘要核对"。
# ════════════════════════════════════════════════════════════════
WEAK_MATCH_MARGIN = float(os.environ.get("WEAK_MATCH_MARGIN", "0.0"))  # 边距 ≤ 该值 → weak_match
REF_SAMPLE_K = int(os.environ.get("REF_SAMPLE_K", "6"))   # 每句对比的非引用参考论文数
MAX_SENT_CHARS = 220          # 过长句子截断（只保留主句部分）
MAX_EVIDENCE_CHARS = 800      # 单句证据（引用论文摘要）上限

# 「这句话是不是在陈述一个**需要出处的事实**」的确定性判据。
# 为什么需要：实测 23%~30% 的论断**根本没有引用标记**，而旧实现对这些句子
# `if not anchors: continue` 直接跳过 —— 幻觉最高发的地方确定性检查偏偏不看。
# 但也不能把所有无锚句都算问题：定义句/过渡句（"推荐系统是缓解信息过载的服务"）
# 本来就不需要引用。所以只挑**带可验证事实标记**的。
_CLAIM_VERBS = ("表明", "显示", "证明", "发现", "提出", "达到", "优于", "超过",
                "提升", "降低", "下降", "增长", "首次", "证明了", "实验")
_CLAIM_NUM = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|倍|万|亿)")


def _looks_like_claim(sent: str) -> bool:
    """无引用标记的句子，是否在陈述需要出处的事实（数字/量化结论/方法声称）。"""
    if len(sent) < 25:
        return False
    if _CLAIM_NUM.search(sent):                      # 含百分比/倍数量化
        return True
    return sum(1 for v in _CLAIM_VERBS if v in sent) >= 1 and len(sent) >= 40


def _split_sentences(text: str):
    """按**句号级**切句（v1.2.13：去掉分号切分点）。

    锚点口径 = 一句号一句 = 对一篇论文的引用：一个以句号结尾的完整论断句
    （内部可用逗号/分号连接多个小句），句尾有 [N] 即为有效锚定。
    分号（；）不再是句子边界——"XX提出了XX；解决了XX；指标XX[12]。"是同一句，
    中间小句不要求各自标注，避免把完整论断误切碎报 no_citation。
    """
    out = []
    for chunk in re.split(r"(?<=[。！？.!?])\s*|\n+", text or ""):
        c = chunk.strip()
        if c:
            out.append(c)
    return out


def build_cite_map_from_pool(draft, pool):
    """参考表 [N] → pool 论文（利用 global_index 稳定编号反查，无需标题匹配）。

    pool 论文 dict 需含 title（+ abstract/paper/summary 之一）。这是确定性证据检查
    的**正确口径**：正文引用 71 个编号 → 71 篇池论文全可定位，不再误报 no_anchor。
    """
    from ResearcherAgent.tools.agent_tools import global_index, paper_key
    gidx = global_index(pool or [])
    rev = {}
    for p in (pool or []):
        if not isinstance(p, dict):
            continue
        i = gidx.get(paper_key(p))
        if i:
            rev[i] = p
    ref_marks = list(re.finditer(r"(?m)^\s*#{0,3}\s*(?:参考文献|References)", draft))
    ref_sec = draft[ref_marks[-1].end():] if ref_marks else ""
    out = {}
    for line in ref_sec.splitlines():
        m = re.match(r"^\[?\s*(\d{1,3})\]?", line.strip())
        if not m:
            continue
        num = int(m.group(1))
        if num in rev:
            out[num] = rev[num]
    return out


def build_cite_map_from_refs(draft, cards):
    """正文参考表 [N] → 论文 dict（title + card 摘要）。与 judge 的引用精确映射同口径。"""
    ref_marks = list(re.finditer(r"(?m)^\s*#{0,3}\s*(?:参考文献|References)", draft))
    ref_sec = draft[ref_marks[-1].end():] if ref_marks else ""
    norm_map = {}
    for c in (cards or [])[:200]:
        t = str(c.get("title") or "").strip()
        if t:
            norm_map[re.sub(r"[^a-z0-9]+", "", t.lower())] = c
    out = {}
    for line in ref_sec.splitlines():
        line = line.strip()
        m = re.match(r"^\[?\s*(\d{1,3})\]?", line)
        if not m:
            continue
        num = int(m.group(1))
        if num > 200:
            continue
        cands = re.findall(r"[A-Za-z][A-Za-z0-9 ,\-:]{15,}", line)
        t = max(cands, key=len).strip() if cands else ""
        tn = re.sub(r"[^a-z0-9]+", "", t.lower()) if t else ""
        hit = None
        if tn:
            if tn in norm_map:
                hit = norm_map[tn]
            else:
                for k, c2 in norm_map.items():
                    if tn[:60] == k[:60] or tn in k or k in tn:
                        hit = c2
                        break
        if hit:
            out[num] = hit
    return out


def _cos(v1, v2):
    n1 = sum(a * a for a in v1) ** 0.5
    n2 = sum(b * b for b in v2) ** 0.5
    if not n1 or not n2:
        return 0.0
    return sum(a * b for a, b in zip(v1, v2)) / (n1 * n2)


def _ev_text(p: dict, limit: int = MAX_EVIDENCE_CHARS) -> str:
    # 判定证据优先级：卡片（含原文证据段落摘录）→ 摘要 → 其他。
    # 摘要级证据太粗，论断细节（数值/参数）无法被 NLI 核对 → 忠实度判不出来（实测 3.x 根因）。
    return (str(p.get("title") or "") + " "
            + str(p.get("card") or p.get("abstract") or p.get("paper")
                  or p.get("summary") or "")).strip()[:limit]


def _embed_cached(text: str, embed_fn, cos_cache):
    """向量缓存（cos_cache 里以 ("#vec#", text) 为键存单文本向量）。"""
    k = ("#vec#", text)
    if k in cos_cache:
        return cos_cache[k]
    try:
        v = embed_fn(text)
    except Exception:
        v = None
    cos_cache[k] = v
    return v


def check_section_deterministic(section_text, cite_map, embed_fn, cos_cache,
                                ref_sample=None, use_nli: bool = True,
                                card_nums: set | None = None):
    """一节内的确定性证据检查。返回 [{"sentence","anchors","sim","verdict"}]。

    verdict ∈ no_anchor / no_citation / weak_evidence / nli_contradiction /
             nli_neutral / weak_match / no_evidence_text / ok

    card_nums（2026-10 新增）：池内有原文证据卡片（精读）的论文编号集合。
    引用编号全部落在 card_nums 之外 → `weak_evidence`（摘要级弱证据）：
    综述引用池内未精读论文是正常实践，但论断细节无法被原文核对，标记进低置信
    清单交用户审核；不再对摘要跑 NLI（摘要太粗、忠实度判不出来），也不再砍编号。

    NLI 判定（use_nli=True，默认开）：用多语言蕴含模型（mDeBERTa-XNLI）判断
    「论断句 hypothesis vs 被引论文证据 premise」：
      · entailment    → 支持 → ok
      · contradiction → `nli_contradiction`（**强信号**：论断与所引论文语义矛盾）
      · neutral       → `nli_neutral`（弱信号：既非支持也非矛盾，通用句常见）
    模型固定权重、输出可复现 → 确定性指标（不靠 LLM 主观评分，弃 RAGAS 口径）。
    NLI 不可用时回退到 embedding 对比式边距（weak_match，仅辅助信号）。
    """
    findings = []
    ref_sample = ref_sample or []
    for sent in _split_sentences(section_text):
        anchors = sorted(set(int(x) for x in re.findall(r"\[(\d{1,3})\]", sent)))
        # ── ① 没有任何引用标记：只在它像"需要出处的事实陈述"时才报 ──
        if not anchors:
            if _looks_like_claim(sent):
                findings.append({"sentence": sent[:MAX_SENT_CHARS], "anchors": [],
                                 "sim": None, "verdict": "no_citation"})
            continue
        # ── ② 引用了池里不存在的编号：确定性，最该修的一类 ──
        missing = [n for n in anchors if n not in cite_map]
        if missing:
            findings.append({"sentence": sent[:MAX_SENT_CHARS], "anchors": anchors,
                             "sim": None, "verdict": f"no_anchor[{missing}]"})
            continue
        ev = " ".join(_ev_text(cite_map[n]) for n in anchors).strip()[:MAX_EVIDENCE_CHARS]
        if not ev:
            findings.append({"sentence": sent[:MAX_SENT_CHARS], "anchors": anchors,
                             "sim": None, "verdict": "no_evidence_text"})
            continue
        # ── ②b 全部引用均无精读卡片 → 摘要级弱证据（2026-10）──
        if card_nums is not None and not any(n in card_nums for n in anchors):
            findings.append({"sentence": sent[:MAX_SENT_CHARS], "anchors": anchors,
                             "sim": None, "verdict": "weak_evidence",
                             "nli_label": None, "nli_score": None})
            continue
        # ── ③ NLI 确定性支持度判定（优先于 embedding；失败才回退边距）──
        nli = None
        if use_nli:
            try:
                from ResearcherAgent.core.nli_evidence import nli_verdict
                nli = nli_verdict(ev, sent)
            except Exception:
                nli = None
        if nli and not nli.get("error"):
            _lbl = nli.get("label")
            if _lbl == "contradiction":
                findings.append({"sentence": sent[:MAX_SENT_CHARS], "anchors": anchors,
                                 "sim": None, "verdict": "nli_contradiction",
                                 "nli_label": "contradiction", "nli_score": nli.get("score")})
                continue
            if _lbl == "neutral":
                findings.append({"sentence": sent[:MAX_SENT_CHARS], "anchors": anchors,
                                 "sim": None, "verdict": "nli_neutral",
                                 "nli_label": "neutral", "nli_score": nli.get("score")})
                continue
            # entailment（支持）→ ok，不再跑 embedding 边距（弱信号无需复核强支持）
            findings.append({"sentence": sent[:MAX_SENT_CHARS], "anchors": anchors,
                             "sim": None, "verdict": "ok",
                             "nli_label": "entailment", "nli_score": nli.get("score")})
            continue
        if embed_fn is None or not ref_sample:
            findings.append({"sentence": sent[:MAX_SENT_CHARS], "anchors": anchors,
                             "sim": None, "verdict": "ok"})
            continue
        # ── ④ NLI 不可用时的回退：对比式边距（辅助信号，非硬判据）──
        vs = _embed_cached(sent, embed_fn, cos_cache)
        ve = _embed_cached(ev, embed_fn, cos_cache)
        if vs is None or ve is None:
            continue
        sim = _cos(vs, ve)
        best_other = 0.0
        for rt, re_txt in ref_sample:
            vr = _embed_cached(re_txt, embed_fn, cos_cache)
            if vr is not None:
                best_other = max(best_other, _cos(vs, vr))
        margin = sim - best_other
        verdict = "weak_match" if margin <= WEAK_MATCH_MARGIN else "ok"
        findings.append({"sentence": sent[:MAX_SENT_CHARS], "anchors": anchors,
                         "sim": round(sim, 3), "margin": round(margin, 3),
                         "verdict": verdict})
    return findings


def build_ref_sample(cite_map, k: int = REF_SAMPLE_K):
    """挑 k 篇**非**被引论文做对比参照（确定性：按编号排序取，保证可复现）。"""
    items = sorted((n, p) for n, p in (cite_map or {}).items() if p)
    if not items:
        return []
    step = max(1, len(items) // max(k, 1))
    out = []
    for i in range(0, len(items), step):
        p = items[i][1]
        t = str(p.get("title") or "").strip()
        if t:
            out.append((t, _ev_text(p)))
        if len(out) >= k:
            break
    return out


def summarize_findings(findings):
    """把检查报告压缩成给 judge 的信号文本。

    ⚠️ 措辞已修正：不再声称这些是"硬信号/引用错误"。其中 `weak_match`（余弦边距）
    实测区分度很弱（中位边距 +0.047），必须让 judge 知道**哪些是确定的、哪些只是提示**，
    否则 judge 会把弱信号当铁证去扣分。
    """
    flagged = [f for f in findings if f["verdict"] != "ok"]
    if not flagged:
        return ""
    hard = [f for f in flagged if str(f["verdict"]).startswith("no_anchor")
            or f["verdict"] == "no_citation"
            or f["verdict"] == "nli_contradiction"]
    # NLI 校准：neutral 不当辅助提示注入 judge（跨语言下 neutral 偏高、非缺陷），
    # 只保留 contradiction 作为硬信号；neutral 仍留在 det_raw 供指标统计。
    soft = [f for f in flagged if f not in hard and f.get("verdict") != "nli_neutral"]
    lines = []
    if hard:
        lines.append(f"【确定性问题 {len(hard)} 处：必须据此扣分并列入 issues】")
        for f in hard[:8]:
            v = str(f["verdict"])
            if v.startswith("no_anchor"):
                t = "引用了池中不存在的编号"
            elif v == "no_citation":
                t = "陈述事实但无任何引用"
            else:
                t = f"NLI 判定论断与引用论文**矛盾**（score={f.get('nli_score')}）"
            lines.append(f"- [{v}] {t}：{f['sentence'][:110]}")
    if soft:
        lines.append(f"【辅助提示 {len(soft)} 处：**弱信号**——相似度边距偏低或仅摘要级引用，"
                     f"需你判断后再决定是否扣分】")
        for f in soft[:6]:
            v = str(f.get("verdict"))
            if v == "weak_evidence":
                lines.append(f"- [weak_evidence] 所引论文仅摘要级（未精读原文）：{f['sentence'][:110]}")
            else:
                mg = f.get("margin")
                lines.append(f"- [weak_match margin={mg}] {f['sentence'][:110]}")
    return "确定性证据检查：\n" + "\n".join(lines)


def render_writer_evidence(evidence_report, cite_map, abstract_chars: int = 260,
                           max_items: int = 10, max_chars: int = 4000):
    """把逐句确定性检查明细渲染成**给 writer 的修改指令**。

    与 `summarize_findings` 的分工（这是本函数存在的全部理由）：
      · `summarize_findings` → 给 **judge** 看：只需知道「这里有 3 处低支持」，据此扣分；
      · 本函数 → 给 **writer** 看：必须带上**对应论文的标题 + 摘要**。
        否则 writer 只知道「这句不被支持」，不知道「应该支持成什么」——
        于是只能把句子改写得更笼统，越改越滑，实测忠实度随重写轮次**净跌**
        （judgev3c 六轮 5.2→4.7）。给出论文原文摘要后，writer 才有「照抄事实」的依据。

    `cite_map`：{编号: 论文 dict}，由 `build_cite_map_from_pool` 产出。
    """
    if not evidence_report or not cite_map:
        return ""
    by_sev = []

    def _short(s, n):
        s = " ".join(str(s or "").split())
        return s[:n] + ("…" if len(s) > n else "")

    for sec, findings in (evidence_report or {}).items():
        for f in (findings or []):
            if f.get("verdict") == "ok":
                continue
            by_sev.append((sec, f))
    if not by_sev:
        return ""
    # 排序：确定性最强的排前面（改起来最明确）
    #   no_anchor（编号不存在）→ no_citation（有事实无出处）→ 其余（弱信号）
    def _rank(f):
        v = str(f.get("verdict") or "")
        return (0 if v.startswith("no_anchor") else
                1 if v == "no_citation" else
                2 if v == "nli_contradiction" else
                3 if v in ("no_evidence_text", "nli_neutral") else 4)
    by_sev.sort(key=lambda x: _rank(x[1]))

    hard = sum(1 for _, f in by_sev if _rank(f) <= 2)
    out = [f"## 🔍 逐句证据核查（共 {len(by_sev)} 处，其中 {hard} 处是**确定性**问题）"]
    used = 0
    for sec, f in by_sev[:max_items]:
        verdict = str(f.get("verdict") or "")
        sim, margin = f.get("sim"), f.get("margin")
        if verdict.startswith("no_anchor"):
            head = f"### 【{sec}】引用了不存在的编号：{verdict}（**必须修**）"
        elif verdict == "no_citation":
            head = (f"### 【{sec}】陈述了事实但**没有任何引用**（**必须补引用或删掉量化表述**）")
        elif verdict == "nli_contradiction":
            head = (f"### 【{sec}】NLI 判定论断与引用论文**矛盾**"
                    f"（score={f.get('nli_score')}，**必须改写论断或改引**）")
        elif verdict == "nli_neutral":
            head = (f"### 【{sec}】NLI 判定为中性（未证实也未矛盾，score={f.get('nli_score')}）"
                    f"——请核对论断是否被引用论文支持")
        elif verdict == "no_evidence_text":
            head = f"### 【{sec}】引用的论文没有可核对的摘要（{verdict}）"
        elif verdict == "weak_evidence":
            head = (f"### 【{sec}】所引论文仅摘要级信息（未精读原文）"
                    f"——论断细节未被原文核对，**请确认未展开实验数据**")
        else:      # weak_match —— 辅助信号，措辞必须留有余地
            head = (f"### 【{sec}】引用的论文未必最贴切"
                    f"（相似度 {sim}，比参照论文高 {margin}）——**请核对后再决定**")
        block = [head, f"  · 原句：{_short(f.get('sentence'), 160)}"]
        if verdict == "no_citation":
            block.append("  · 这句给出了可验证的事实/量化结论，却没有 `[N]` 标注。"
                         "请在「可引用材料」中找出真正支持它的论文并补上编号；"
                         "找不到就**删掉该量化表述**，不要留无出处的断言。")
        for n in (f.get("anchors") or [])[:3]:
            p = cite_map.get(n)
            if not p:
                block.append(f"  · 引用 [{n}]：**该编号在文献池中不存在**"
                             f"（正文引了池里没有的编号 → 删掉这个标记，或改引真实存在的编号）")
                continue
            ab = str(p.get("abstract") or p.get("paper") or p.get("card")
                     or p.get("summary") or "").strip()
            block.append(f"  · 引用 [{n}]：《{_short(p.get('title'), 90)}》"
                         f"（{p.get('year') or '?'}）")
            if ab:
                block.append(f"    该论文实际内容：{_short(ab, abstract_chars)}")
        if not verdict.startswith("no_citation"):
            block.append("  · 要求：使论断与该论文实际结论一致；若论文并不支持该论断，"
                         "**改写论断使其成立，或改引到真正支持它的论文**，不要保留无法溯源的表述。")
        cand = "\n".join(block)
        if used + len(cand) > max_chars:
            out.append(f"（另有 {len(by_sev) - len(out) + 1} 处未列出，同类处理）")
            break
        used += len(cand)
        out.append(cand)
    return "\n\n".join(out)


def enforce_citation_domain(draft: str, evidence_nums: set) -> tuple:
    """【写后确定性约束】正文 [N] 必须属于「有原文段落证据」的论文集合。

    证据约束只控写作输入（材料只给证据论文）是不够的——模型会凭记忆写出
    材料之外的编号（实测 27 条引用 vs 8 张卡片）。写完后必须确定性强制：
      · 正文里不在 evidence_nums 的 [N] → 删除标记（该句退回「无出处事实」，下轮修）；
      · 参考文献表同步删除对应编号条目。
    返回 (new_draft, removed_nums)。"""
    if not draft or not evidence_nums:
        return draft, []
    ref_marks = list(re.finditer(r"(?m)^\s*#{0,3}\s*(?:参考文献|References)", draft))
    body = draft[:ref_marks[-1].start()] if ref_marks else draft
    ref_sec = draft[ref_marks[-1].end():] if ref_marks else ""
    removed = []

    def _drop_or_keep(m):
        n = int(m.group(1))
        if n in evidence_nums:
            return m.group(0)
        if n not in removed:
            removed.append(n)
        return ""
    new_body = re.sub(r"\[(\d{1,3})\]", _drop_or_keep, body)
    if ref_marks and ref_sec:
        kept = []
        for line in ref_sec.splitlines():
            lm = re.match(r"^\[?\s*(\d{1,3})\]?", line.strip())
            n = int(lm.group(1)) if lm else None
            if n is not None and n in removed:
                continue
            kept.append(line)
        new_ref = "\n".join(kept)
        new_draft = new_body + f"\n## 参考文献\n{new_ref}"
    else:
        new_draft = new_body
    return new_draft, sorted(removed)

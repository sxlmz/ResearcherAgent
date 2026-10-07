# -*- coding: utf-8 -*-
"""
must_read.py —— 用「种子论文 + 引文网络扩展」自动生成某方向的**代理必读清单**。

解决的评测难题：CQF1 的 recall、Completeness 的 IR 都需要一份「该领域必引文献清单」，
而让专家全量标注不现实。折中方案（也是本模块的全部意义）：
  用户只给 3-5 篇自己已知的核心文献 → 系统自动扩展出候选池 → 排序取 Top-K
  → 得到一份**可复核**的代理清单，够用来算引用覆盖率。

设计要点（每条都有实测依据，见 `_probe` 结论）：
  · 数据源用 **OpenAlex 直连**（不走 MCP）。理由：MCP 的 search_* 只做文本检索，
    拿不到 `referenced_works`（引文边）和 `cited_by_count`；OpenAlex 有
    `filter=cites:W...`（谁引用它）与工作对象的 `referenced_works`（它引用谁），
    还有 `sources.summary_stats.2yr_mean_citedness`（近似影响因子）。
    ⚠️ 实测：MCP 子进程与本机脚本都要绕开失效的系统代理，故默认 `proxies=None`。
  · **必须去重**：实测 LightGCN 的 OpenAlex 记录指向的是 **arXiv 预印本**
    （`type=repository`、2yr_mean_citedness=0.17），同一个工作在 OpenAlex 里
    常同时存在「预印本记录」和「正会记录」。不去重会让清单被预印本记录污染，
    且 venue 分会把 CS 论文整体压低。→ 按 DOI/规范化标题合并，保留被引更高的那条。
  · 排序**不能只看被引数**：老论文天然被引多。用「年均被引」+「种子共现」+「venue 影响力」。
  · **种子共现数是核心信号**：被 3 篇种子都引用的论文，比只被 1 篇引用的更"核心"。
  · 被引数只是**相对**信号：实测 OpenAlex 给 LightGCN 414 次，远低于 Google Scholar
    的数千次。所以只能用于**同一个方向内的排序**，不能当绝对重要性。

用法：
  python ResearcherAgent/research/must_read.py \
      --seeds "LightGCN: Simplifying and Powering Graph Convolution Network" \
              "Neural Graph Collaborative Filtering" \
              "Graph Neural Networks for Recommender Systems: Challenges, Methods, and Directions" \
      --k 20 --out logs/eval/mustread_gnn_rec.json
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import requests

# ── HTTP 配置 ─────────────────────────────────────────────────────
OPENALEX = "https://api.openalex.org"
MAILTO = os.environ.get("OPENALEX_MAILTO", "litreview-eval@example.org")
_REQ_TIMEOUT = 30
_RETRY = (1.0, 3.0)          # 退避重试两次（与 agent_tools 的 MCP 包装同风格）
DEFAULT_K = 20
CACHE_DIR = Path("logs") / "eval" / "mustread_cache"


def _proxy_kw() -> dict:
    """默认直连；要改走可用代理时设 EVAL_HTTP_PROXY。

    为什么要显式写：本机系统代理（注册表里）的境外出口已失效，
    而 `requests` 在 Windows 上会读注册表 → 不显式禁用就会走坏代理。
    """
    p = os.environ.get("EVAL_HTTP_PROXY", "").strip()
    if p:
        return {"proxies": {"http": p, "https": p}}
    return {"proxies": {"http": None, "https": None}}


_session = requests.Session()
_session.headers.update({"User-Agent": f"litreview-eval/1.0 (mailto:{MAILTO})"})


_THROTTLE = float(os.environ.get("OPENALEX_THROTTLE", "0.35"))   # 请求间隔（秒）
_MAX_429_RETRIES = 6          # 参考 paper-chaser-mcp 的 MAX_429_RETRIES=6
_MAX_429_WAIT = 45.0         # 单次 429 退避上限（含 Retry-After 响应头的夹取）
_MAX_429_TOTAL_WAIT = 120.0  # 同一请求累计 429 等待上限，超了就放弃而不是无限退避


def _get(path: str, **params) -> dict:
    """带磁盘缓存的 GET。缓存键 = 完整请求 URL 的哈希（同参数不重复打 OpenAlex）。

    ⚠️ 429 必须特殊处理：构建 8 个方向的清单会打出上百次请求（每篇种子一次精确
    标题过滤 + 正向引用分页 + venue 批量），实测**无退避重试会直接撞 429 并让整个
    方向构建失败**。做法照成熟项目：优先听 `Retry-After` 响应头，否则指数退避。
    """
    params.setdefault("mailto", MAILTO)
    url = f"{OPENALEX}{path}?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cf = CACHE_DIR / (hashlib.md5(url.encode()).hexdigest() + ".json")
    if cf.exists():
        try:
            return json.loads(cf.read_text(encoding="utf-8"))
        except Exception:
            pass
    last = ""
    waited = 0.0
    attempts = max(len(_RETRY) + 1, _MAX_429_RETRIES)
    for i in range(attempts):
        try:
            r = _session.get(url, timeout=_REQ_TIMEOUT, **_proxy_kw())
            if r.status_code == 200:
                data = r.json()
                try:
                    cf.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
                time.sleep(_THROTTLE)     # 礼貌节流，别把开放 API 打疼
                return data
            last = f"HTTP {r.status_code}"
            if r.status_code == 429:
                # ⚠️ 外部给的 Retry-After **必须设上限**。实测 OpenAlex 会回
                #    `Retry-After: 61821`（≈17 小时）——初版写成
                #    `max(header, min(backoff, 30))`，那个 30s 上限只约束了退避部分、
                #    没约束响应头，于是进程在 429 退避里**静默睡了一整轮**，
                #    表现为"构建跑一半就没动静了"且不报错（排查成本极高）。
                #    教训与写断点/重试的通用规则一致：**外部输入一律先夹到合理区间**。
                ra = r.headers.get("Retry-After")
                try:
                    hdr = float(ra) if ra else 0.0
                except (TypeError, ValueError):
                    hdr = 0.0
                hdr = min(hdr, _MAX_429_WAIT)
                wait = min(max(hdr, min(2.0 * (2 ** i), _MAX_429_WAIT)), _MAX_429_WAIT)
                waited += wait
                if waited > _MAX_429_TOTAL_WAIT:
                    raise RuntimeError(
                        f"OpenAlex 持续 429（累计已等 {waited:.0f}s，单次上限 "
                        f"{_MAX_429_WAIT:.0f}s）→ 放弃该请求，避免无限退避")
                print(f"    ⏳ OpenAlex 429，等待 {wait:.0f}s 后重试"
                      f"（{i+1}/{attempts}，累计 {waited:.0f}s；"
                      f"服务端 Retry-After={ra!r} 已夹到 {_MAX_429_WAIT:.0f}s 上限）")
                time.sleep(wait)
                continue
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:80]}"
        if i < len(_RETRY):
            time.sleep(_RETRY[i])
    raise RuntimeError(f"OpenAlex 请求失败 [{last}] {url[:120]}")


def _wid(openalex_url: str) -> str:
    return (openalex_url or "").rsplit("/", 1)[-1]


# ══════════════════════════════════════════════════════════════════
# 规范化与去重
# ══════════════════════════════════════════════════════════════════
_STOP = {"a", "an", "the", "of", "for", "and", "on", "in", "to", "with", "via", "using"}


def norm_title(t: str) -> str:
    """标题规范化：小写、去标点、去停用词、压空白。用于跨记录判定「同一篇论文」。"""
    t = re.sub(r"[^a-z0-9\s]", " ", (t or "").lower())
    return " ".join(w for w in t.split() if w not in _STOP)


def _norm_doi(d) -> str:
    return (d or "").lower().replace("https://doi.org/", "").strip()


# ══════════════════════════════════════════════════════════════════
# ① 种子解析
# ══════════════════════════════════════════════════════════════════
def clean_title(t: str) -> str:
    """OpenAlex 有不少记录标题被平台加过前缀（实测 nnU-Net 那条是 `Abstract: nnU-Net: ...`）。"""
    return re.sub(r"^\s*(abstract|preprint|title)\s*[:\-]\s*", "", (t or ""), flags=re.I).strip()


def strip_annotation(t: str) -> str:
    """去掉标题里的括号注释，如 `...(DDPM)` / `...(AlexNet)` / `...(SAT)` / `...(RSA)`。

    ⚠️ 必须先去：单词重合度打分会被括号里的**缩写**带偏——实测输入
    `Denoising Diffusion Probabilistic Models (DDPM)` 命中了
    `UNIT-DDPM: UNpaired Image Translation with Denoising Diffusion ...`，
    因为候选标题正好包含 `ddpm` 这个词，重合度反而比真正的 DDPM 更高。
    用户写缩写是为了自己看，不是标题的一部分。
    """
    return re.sub(r"[（(][^（()）]*[)）]", " ", t or "").strip()


def _match_score(query: str, cand_title: str, year_hint: int | None,
                 cand_year: int | None) -> float:
    """种子匹配打分：全等 > 包含关系 > Jaccard 相似度；命中年份提示再小幅加权。

    ⚠️ 为什么用 **Jaccard 而不是「查询词被覆盖的比例」**：后者对"超集"没有惩罚——
    实测 `Attention Is All You Need` 与 `Is Space-Time Attention All You Need for
    Video Understanding` 的查询词覆盖率是 **1.0**（查询的 5 个词全在里面），
    于是一篇 2021 年的视频理解论文和 2017 年的正版 Transformer 论文同分。
    Jaccard 会因为候选多出 4 个词而把分数降到 0.56，正版自然胜出。
    """
    q, c = norm_title(strip_annotation(query)), norm_title(clean_title(cand_title))
    if not q or not c:
        return 0.0
    tq, tc = set(q.split()), set(c.split())
    if q == c:
        sc = 1.0
    elif q in c or c in q:                     # 一方是另一方的子串 → 强信号
        sc = 0.95
    else:
        sc = len(tq & tc) / len(tq | tc)       # Jaccard：对多出来的词有惩罚
    if year_hint and cand_year and abs(int(cand_year) - int(year_hint)) <= 1:
        sc = min(sc + 0.15, 1.0)
    return sc


def resolve_seed(seed, year_hint: int | None = None) -> dict | None:
    """把一条种子（标题 / DOI / OpenAlex id / dict）解析成工作记录。

    标题搜索**取前 10 条候选而不是第一条**，并按「标题词重合度 + 年份提示」排序。
    ⚠️ 为什么必须这样：实测输入 `Segment Anything in 3D Medical Images with Text
    Prompts (SAT), 2023` 时，只取首条会命中另一篇 2025 年的论文
    （`Large-vocabulary segmentation for medical images with text prompts`）——
    **一篇错的种子会污染整个方向的清单，而清单是往后的评分标准**。
    因此清单构建方要把 md 里的年份一起传进来做消歧。
    """
    if isinstance(seed, dict):
        year_hint = seed.get("year", year_hint)
        seed = seed.get("title") or seed.get("doi") or ""
    s = (seed or "").strip()
    if not s:
        return None

    m = re.search(r"10\.\d{4,9}/[^\s]+", s)
    if m:
        d = _get("/works", filter=f"doi:{m.group(0)}", per_page=1)
        w = (d.get("results") or [None])[0]
        return {"work": w, "match_score": 1.0, "year_hint": year_hint} if w else None
    if re.fullmatch(r"(https?://openalex\.org/)?[Ww]\d+", s):
        w = _get("/works/" + _wid(s))
        return ({"work": w, "match_score": 1.0, "year_hint": year_hint}
                if isinstance(w, dict) and w.get("id") else None)

    # 查询串去掉括号注释（用户写缩写是给自己看的，不是标题的一部分）
    q = strip_annotation(s)
    # ⚠️ filter 值有语法限制：实测含 `?` / `,` 的标题（"Quo Vadis, Action Recognition? ..."）
    #   直接拼进 URL 会 400（`?` 是查询串分隔符、`,` 是 OpenAlex filter 分隔符）。
    #   处理：把非字母数字的标点替换成空格（display_name.search 是词搜索，标点无信息量），
    #   再对 URL 保留字符做编码；空格与冒号保留（实测可用，且保持与既有缓存键一致）。
    q = re.sub(r"[^A-Za-z0-9: ]+", " ", q).strip()
    q_enc = quote(q, safe=": ")
    # ⚠️ 优先用 `filter=display_name.search:`（**精确标题过滤**），失败才退回 `search=`。
    #    实测差异很大：`search=An Analysis of Linear Time Series Forecasting Models Toner`
    #    返回一堆风马牛不相及的论文，而 `display_name.search:analysis of linear time series
    #    forecasting models` 第一条就正中目标。
    rs = (_get("/works", filter=f"display_name.search:{q_enc}", per_page=10).get("results") or [])
    if not rs:
        rs = (_get("/works", search=q_enc, per_page=10).get("results") or [])
    if not rs:
        return None
    best, best_key = None, (-1.0, 1e9)
    for r in rs:
        y = r.get("publication_year")
        sc = _match_score(s, r.get("display_name", ""), year_hint, y)
        # 同分时优先年份提示更接近的（实测有同名不同年的记录）
        dy = abs(int(y) - int(year_hint)) if (year_hint and y) else 1e9
        key = (sc, -dy)
        if key > best_key:
            best, best_key = r, key
    wy = best.get("publication_year")
    # ★ 元数据腐坏标记：实测 OpenAlex 的 Attention 记录（W2626778328）作者/标题都对，
    #   但 **publication_year 写成了 2025、doi 是垃圾值 10.65215/...**。
    #   这类记录仍可用于引文扩展（引文边是真实工作的），但年份/DOI 不可信 → 显式标记。
    corrupt = bool(year_hint and wy and abs(int(wy) - int(year_hint)) > 2)
    return {"work": best, "match_score": round(best_key[0], 3), "year_hint": year_hint,
            "year_mismatch": corrupt,
            "matched_title": clean_title(best.get("display_name", ""))}


OVERRIDE_PATH = Path("docs/seed_overrides.json")


def load_overrides() -> dict:
    """种子覆盖表：`{"原输入标题": {"openalex": "Wxxx" 或 "title": "订正后的标题"}}`。

    ⚠️ 为什么必须有：实测 24 篇种子里有 3 篇**靠标题无法从 OpenAlex 解析**——
    一篇标题在源清单里就是错的（Shou 2016）、一篇 OpenAlex 未收录（SAT）、
    一篇搜出来的全是重名论文。没有覆盖机制就只能眼睁睁看着整个方向的清单被污染。
    """
    try:
        return json.loads(OVERRIDE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_seeds(seeds: list) -> list[dict]:
    out = []
    ov = load_overrides()
    for s in seeds:
        key = s.get("title") if isinstance(s, dict) else s
        fix = ov.get(key) or {}
        if fix.get("openalex"):
            s = {"title": fix["openalex"], "year": (s.get("year") if isinstance(s, dict) else None)}
            print(f"  ↻ 使用覆盖（OpenAlex id）: {str(key)[:52]}")
        elif fix.get("title"):
            orig_year = s.get("year") if isinstance(s, dict) else None
            s = {"title": fix["title"], "year": fix.get("year", orig_year)}
            print(f"  ↻ 使用覆盖（订正标题）: {str(key)[:40]} → {fix['title'][:44]}")
        r = resolve_seed(s)
        if not r:
            print(f"  ⚠️ 种子未解析到: {str(s)[:70]}")
            continue
        w = r["work"]
        w["display_name"] = clean_title(w.get("display_name", ""))
        label = s.get("title") if isinstance(s, dict) else s
        if r.get("year_mismatch"):
            print(f"  ⚠️ 年份不符：输入提示 {r.get('year_hint')}，OpenAlex 记录是 "
                  f"{w.get('publication_year')}（该记录元数据可能腐坏，引文边仍可用）")
        if r["match_score"] < 0.6:
            print(f"  ⚠️ 种子匹配度低({r['match_score']}): 输入「{str(label)[:44]}」"
                  f" → 命中「{w.get('display_name','')[:52]}」({w.get('publication_year')}) "
                  f"← 需人工确认是否同一篇")
        print(f"  {'✓' if r['match_score'] >= 0.6 else '?'} 种子 [{_wid(w['id'])}] "
              f"{w.get('publication_year')} 被引{w.get('cited_by_count')} | "
              f"{w.get('display_name','')[:56]}")
        r["input"] = key if isinstance(key, str) else str(key)
        r["match_score"] = 1.0 if (fix.get("openalex") or fix.get("title")) else r["match_score"]
        out.append(r)
    return out


# ══════════════════════════════════════════════════════════════════
# ② 引文网络扩展
# ══════════════════════════════════════════════════════════════════
def expand(seed_works: list[dict], per_seed_cap: int = 200,
           max_candidates: int = 1500, direction: str = "both") -> dict:
    """从种子出发扩展候选池。

    `direction`：backward = 只取种子引用的（奠基性工作）；
                 forward  = 只取引用种子的（后续发展）；
                 both     = 两者都取（默认，覆盖「经典 + 前沿」）。
    返回 {wid: {work, seed_links, link_kind}}，`seed_links` = 被几个种子牵到（核心信号）。
    """
    cand: dict[str, dict] = {}
    seed_ids = {_wid(w["id"]) for w in seed_works}

    def touch(w: dict, kind: str, seed_id: str):
        wid = _wid(w["id"])
        if not wid or wid in seed_ids:
            return
        e = cand.setdefault(wid, {"work": w, "seed_links": set(), "link_kind": set()})
        e["seed_links"].add(seed_id)
        e["link_kind"].add(kind)
        # 保留字段更全的那条记录（预印本记录往往缺 venue）
        if (w.get("cited_by_count") or 0) > (e["work"].get("cited_by_count") or 0):
            e["work"] = w

    for sw in seed_works:
        sid = _wid(sw["id"])
        if direction in ("backward", "both"):
            refs = [_wid(x) for x in (sw.get("referenced_works") or []) if x]
            for i in range(0, len(refs), 100):        # openalex_id 批量取，每批 ≤100
                batch = refs[i:i + 100]
                try:
                    d = _get("/works", filter="openalex_id:" + "|".join(batch),
                             per_page=100)
                except Exception as e:
                    print(f"    ⚠️ 反向取参考失败: {e}")
                    continue
                for w in d.get("results") or []:
                    touch(w, "ref", sid)
        if direction in ("forward", "both"):
            page = 1
            got = 0
            while got < per_seed_cap:
                try:
                    d = _get("/works", filter=f"cites:{sid}", per_page=200, page=page,
                             sort="cited_by_count:desc")
                except Exception as e:
                    print(f"    ⚠️ 正向取引用失败: {e}")
                    break
                rs = d.get("results") or []
                if not rs:
                    break
                for w in rs:
                    touch(w, "cited_by", sid)
                got += len(rs)
                if len(rs) < 200:
                    break
                page += 1
        if len(cand) > max_candidates:
            print(f"    ℹ️ 候选池超过 {max_candidates}，提前停止扩展")
            break
    return cand


# ══════════════════════════════════════════════════════════════════
# ③ 去重 + 打分 + Top-K
# ══════════════════════════════════════════════════════════════════
def _is_seed_dup(nt: str, seed_titles: list[str]) -> bool:
    """候选是否是某个种子的**重复记录**（而非不同论文）。

    ⚠️ 只比"标题全等"不够：实测 LightGCN 在 OpenAlex 里有一条残记录标题就是
    `"LightGCN"`（被引 4418），与种子的完整长标题规范化后不相等，于是它作为
    "候选"混进了清单——清单里出现用户自己给的种子是明显的自相矛盾。
    所以还要判断「短标题的词集完全被长标题包含」。
    """
    if not nt:
        return False
    tw = set(nt.split())
    for st in seed_titles:
        if nt == st:
            return True
        sw = set(st.split())
        if not sw:
            continue
        if tw <= sw and len(tw) <= 4:            # 例：{'lightgcn'} ⊆ 完整标题
            return True
        if len(tw) >= 3 and len(tw & sw) / len(tw) >= 0.8:
            return True
    return False


def dedupe(cand: dict, seed_titles: list[str] | None = None) -> list[dict]:
    """合并「预印本记录」与「正式发表记录」等同物，并剔除种子的重复记录。

    判据：DOI 相同，或规范化标题相同。合并时取被引更高的记录，seed_links 取并集。
    """
    seed_titles = seed_titles or []
    by_key: dict[str, dict] = {}
    for wid, e in cand.items():
        w = e["work"]
        nt = norm_title(w.get("display_name", ""))
        if _is_seed_dup(nt, seed_titles):
            continue                     # 是种子本身（另一条记录）→ 不要
        doi = _norm_doi(w.get("doi"))
        key = f"doi:{doi}" if doi else f"t:{nt}"
        if not nt and not doi:
            continue                     # 没标题也没 DOI，丢弃
        prev = by_key.get(key)
        cur = {"wid": wid, "work": w, "seed_links": set(e["seed_links"]),
               "link_kind": set(e["link_kind"])}
        if prev is None:
            by_key[key] = cur
        else:
            cur["seed_links"] |= prev["seed_links"]
            cur["link_kind"] |= prev["link_kind"]
            if (cur["work"].get("cited_by_count") or 0) < (prev["work"].get("cited_by_count") or 0):
                cur["work"] = prev["work"]          # 保留被引更高的那条记录
                cur["wid"] = prev["wid"]
            by_key[key] = cur
    return list(by_key.values())


def fetch_source_stats(items: list[dict]) -> dict[str, dict]:
    """批量取 venue 元数据（type / 影响力）。

    ⚠️ 为什么单独取：`/works` 返回的 `primary_location.source` 里**没有**
    `summary_stats`——实测它对所有记录都缺失，导致 venue 分恒等于中位数、
    整个 venue 维度形同虚设。必须走 `/sources?filter=openalex_id:S1|S2`。
    """
    sids = []
    for it in items:
        sid = _wid(((it["work"].get("primary_location") or {}).get("source") or {}).get("id") or "")
        if sid.startswith("S"):
            sids.append(sid)
    sids = sorted(set(sids))
    out: dict[str, dict] = {}
    for i in range(0, len(sids), 100):
        batch = sids[i:i + 100]
        try:
            d = _get("/sources", filter="openalex_id:" + "|".join(batch), per_page=100)
        except Exception as e:
            print(f"    ⚠️ 取 venue 元数据失败（venue 维度降级为中性）: {e}")
            continue
        for s in d.get("results") or []:
            out[_wid(s["id"])] = {
                "name": s.get("display_name"), "type": (s.get("type") or "").lower(),
                "v2y": (s.get("summary_stats") or {}).get("2yr_mean_citedness"),
                "h": (s.get("summary_stats") or {}).get("h_index"),
            }
    return out


# 排序权重——每一项都对应一个"核心性"假设，写在注释里便于以后调
W_SEED = 3.0        # 种子共现：被 N 篇种子牵到 → 最强信号
W_CITE = 1.0        # 年均被引（对数）：经典性
W_VENUE = 0.6       # venue 影响力（对数）：发表层级


def score(items: list[dict], now_year: int, src_stats: dict[str, dict]) -> list[dict]:
    import math
    # 预印本记录（type=repository）常是正式版本的**同物异记录**，venue 分不该为 0，
    # 取候选池里正式发表 venue 的中位数作为中性值——否则整个 CS 方向的 venue 分被系统性压低。
    vals = sorted(s["v2y"] for s in src_stats.values()
                  if s.get("v2y") and s.get("type") in ("journal", "conference"))
    venue_median = vals[len(vals) // 2] if vals else 1.0
    print(f"  venue 元数据覆盖 {len(src_stats)} 个来源，"
          f"正式发表 venue 影响力中位数={round(venue_median, 3)}")

    out = []
    for it in items:
        w = it["work"]
        year = w.get("publication_year") or now_year
        age = max(now_year - year + 1, 1)
        cites = w.get("cited_by_count") or 0
        cite_s = math.log1p(cites / age)                      # 年均被引，抗"老论文天然多引"
        seed_s = float(len(it["seed_links"]))
        sid = _wid(((w.get("primary_location") or {}).get("source") or {}).get("id") or "")
        st = src_stats.get(sid) or {}
        stype, v2y = st.get("type"), st.get("v2y")
        if stype == "repository" or not v2y:
            venue_s, neutral = math.log1p(venue_median), True
        else:
            venue_s, neutral = math.log1p(float(v2y)), False
        # ★ 可疑记录防护：实测 OpenAlex 存在明显错并/污染的记录——2025 年的论文挂着
        #   16,307 次被引。这类记录用年龄一算年均被引就冲到榜首，而它与主题毫无关系。
        #   宁可漏掉一个真·爆款，也不让污染记录定义"必读"。
        suspect = age <= 3 and cites > 3000
        it.update({"year": year, "age": age, "cited_by_count": cites,
                   "cite_score": round(cite_s, 3), "seed_score": seed_s,
                   "venue_score": round(venue_s, 3), "venue_neutral": neutral,
                   "venue": st.get("name"), "venue_type": stype or None,
                   "suspect": suspect,
                   "total": round(W_SEED * seed_s + W_CITE * cite_s + W_VENUE * venue_s, 4)})
        out.append(it)
    out.sort(key=lambda x: -x["total"])
    return out


def top_k(items: list[dict], k: int = DEFAULT_K, min_year: int | None = None,
          min_seed_links: int = 2, max_per_venue: float = 0.3) -> tuple[list[dict], dict]:
    """取 Top-K。

    `min_seed_links=2` 是**关键过滤器**：只要 ≥1 篇种子引用过就收，会把
    「所有论文都会引的通用基座」一并收入——实测曾把 *Deep Residual Learning*、
    *Adam* 放进 GNN 推荐方向的必读清单。要求被 ≥2 篇种子牵到，噪声几乎消失。
    同一 venue 不超过 k×max_per_venue，避免清单一家人包揽。
    """
    dropped = {}
    pool = []
    for it in items:
        if min_year is not None and (it.get("year") or 0) < min_year:
            dropped.setdefault("year", []).append(it)
            continue
        if len(it["seed_links"]) < min_seed_links:
            dropped.setdefault("seed_links", []).append(it)
            continue
        if it.get("suspect"):
            dropped.setdefault("suspect", []).append(it)
            continue
        pool.append(it)
    cap = max(1, int(k * max_per_venue))
    per_venue: dict[str, int] = {}
    picked = []
    for it in pool:
        v = it.get("venue")
        # ⚠️ 只有**已知 venue** 才计入集中度。曾把 venue 为空的记录都归到 "?" 一个桶里，
        #    结果 6 条缺 venue 元数据的记录占满了 30% 的名额，把 PinSage、
        #    Self-supervised Graph Learning（SGL）、Neural FM 这些真·必读挤出了清单——
        #    去重/限流的桶键用「缺失值」兜底，等于凭空制造了一个虚假的"同一家"。
        if v and per_venue.get(v, 0) >= cap:
            dropped.setdefault("venue_cap", []).append(it)
            continue
        if v:
            per_venue[v] = per_venue.get(v, 0) + 1
        picked.append(it)
        if len(picked) >= k:
            break
    stats = {k2: len(v) for k2, v in dropped.items()}
    return picked, {"dropped": stats,
                    "dropped_examples": {k2: [x["work"].get("display_name", "")[:60]
                                              for x in v[:4]]
                                         for k2, v in dropped.items()}}


# ══════════════════════════════════════════════════════════════════
# ④ 用清单评 Agent 的引用覆盖率（CQF1 的 recall 侧）
# ══════════════════════════════════════════════════════════════════
def _flat(it: dict) -> dict:
    """把条目统一成拍平格式。

    同时兼容两种输入：`checklist`/`must_read` 的输出格式（title 在顶层），
    以及内部格式（标题在 `it["work"]["display_name"]`）——否则调用方要自己转换，
    而 `coverage()` 最常被用在**已落盘的清单**上。
    """
    if "title" in it or "display_name" in it:
        return {"wid": it.get("wid"), "title": it.get("title") or it.get("display_name"),
                "year": it.get("year"), "cited_by_count": it.get("cited_by_count"),
                "doi": it.get("doi"), "venue": it.get("venue"),
                "is_seed": bool(it.get("is_seed")),
                "seed_links": it.get("seed_links") if isinstance(it.get("seed_links"), int)
                else len(it.get("seed_links") or [])}
    w = it.get("work") or {}
    return {"wid": it.get("wid"), "title": w.get("display_name"),
            "year": w.get("publication_year"), "cited_by_count": w.get("cited_by_count"),
            "doi": w.get("doi"),
            "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
            "is_seed": False, "seed_links": len(it.get("seed_links") or [])}


def coverage(must_read: list[dict], review_refs: list[str]) -> dict:
    """计算综述对必读清单的覆盖率（CQF1 的 recall 侧）。

    `review_refs`：综述参考文献的条目文本（或标题）列表。
    匹配顺序：DOI → 规范化标题**全等** → 规范化标题词重合 ≥0.8。
    不做低于该阈值的模糊匹配——**宁可报"漏"也不要报"命中"**：漏了只是低估覆盖率，
    虚高则会让这个指标彻底失去意义（这是本项目栽过跟头的地方）。
    """
    have_d = set()
    have_t = []
    for r in review_refs or []:
        d = re.search(r"10\.\d{4,9}/[^\s\])。]+", str(r))
        if d:
            have_d.add(_norm_doi(d.group(0)))
        nt = norm_title(str(r))
        if len(nt.split()) >= 3:
            have_t.append(nt)

    hit, missed = [], []
    for raw in must_read:
        it = _flat(raw)
        doi = _norm_doi(it["doi"])
        nt = norm_title(it["title"] or "")
        tw = set(nt.split())
        ok = bool(doi and doi in have_d)
        if not ok:
            for h in have_t:
                hw = set(h.split())
                # 综述参考表条目含作者/年份，标题是其子串 → 用「清单标题被引用条目包含」判定
                if nt and (nt == h or nt in h or len(tw & hw) / max(len(tw), 1) >= 0.8):
                    ok = True
                    break
        (hit if ok else missed).append(it)

    return {"must_read_k": len(must_read), "hit": len(hit), "missed": len(missed),
            "recall": round(len(hit) / len(must_read), 3) if must_read else None,
            "hit_items": hit, "missed_items": missed}


# ══════════════════════════════════════════════════════════════════
def build(seeds: list[str], k: int = DEFAULT_K, min_year: int | None = None,
          direction: str = "both", max_candidates: int = 1500,
          min_seed_links: int = 2) -> dict:
    print(f"\n① 解析 {len(seeds)} 篇种子…")
    sr = resolve_seeds(seeds)
    if not sr:
        raise SystemExit("没有任何种子被解析成功，请检查标题/DOI 是否准确")
    seed_works = [x["work"] for x in sr]
    seed_titles = {norm_title(w.get("display_name", "")) for w in seed_works}

    print(f"\n② 引文网络扩展（方向={direction}）…")
    cand = expand(seed_works, direction=direction, max_candidates=max_candidates)
    print(f"  原始候选 {len(cand)} 篇")
    items = dedupe(cand, seed_titles)
    print(f"  去重后 {len(items)} 篇（合并预印本/正式版本等同物，已排除种子自身）")

    print(f"\n③ 取 venue 元数据并打分…")
    stats = fetch_source_stats(items)
    now_year = time.gmtime().tm_year
    items = score(items, now_year, stats)
    # ★ 自适应阈值：种子共现 ≥2 是抑制「通用基座论文」的关键过滤器，但它假设
    #   三篇种子共享可观的引文邻居。**实测失效场景**：某方向的种子含极新论文
    #   （MedSAM3 2025 被引 1 次）时，扩展出的候选几乎没有共同邻居 →
    #   要求 ≥2 会滤掉全部候选、清单只剩种子本身。所以先按设定值取，
    #   结果过少就放宽到 1 并**显式记录本次用的是哪个阈值**（不静默降级）。
    picks, drops = top_k(items, k=k, min_year=min_year, min_seed_links=min_seed_links)
    msl_used = min_seed_links
    if len(picks) < max(8, k // 2) and min_seed_links > 1:
        print(f"  ⚠️ min_seed_links={min_seed_links} 仅得 {len(picks)} 篇 → "
              f"放宽到 1 重取（种子间共同邻居过少，常见于种子含极新论文）")
        picks, drops = top_k(items, k=k, min_year=min_year, min_seed_links=1)
        msl_used = 1

    # 被过滤掉的东西必须**显式报告**，否则"清单短了"无从解释
    if drops["dropped"]:
        print(f"  过滤统计: {drops['dropped']}")
        for why, ex in drops["dropped_examples"].items():
            if ex:
                print(f"    - {why} 示例: {' / '.join(x for x in ex[:3])}")

    def _entry(it: dict) -> dict:
        return {"wid": it["wid"], "title": it["work"].get("display_name"),
                "year": it["year"], "cited_by_count": it["cited_by_count"],
                "doi": it["work"].get("doi"), "venue": it.get("venue"),
                "venue_type": it.get("venue_type"),
                "seed_links": len(it["seed_links"]),
                "link_kind": sorted(it["link_kind"]),
                "score": it["total"], "is_seed": False,
                "score_parts": {"cite": it["cite_score"], "seed": it["seed_score"],
                                "venue": it["venue_score"]},
                "openalex": f"https://openalex.org/{it['wid']}"}

    must_read = [_entry(it) for it in picks]
    # checklist = 用户给的种子（标注 is_seed）+ 扩展出来的清单。
    # 覆盖率必须在**这个全集**上算：种子是用户自己认定的核心文献，综述理应引用它们，
    # 只对扩展部分算 recall 会系统性低估。
    checklist = [{"wid": _wid(w["id"]), "title": w.get("display_name"),
                  "year": w.get("publication_year"),
                  "cited_by_count": w.get("cited_by_count"), "doi": w.get("doi"),
                  "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                  "venue_type": ((w.get("primary_location") or {}).get("source") or {}).get("type"),
                  "seed_links": len(seed_works), "link_kind": ["seed"], "score": None,
                  "is_seed": True,
                  "openalex": w.get("id")} for w in seed_works] + must_read

    return {
        "seeds": [{"wid": _wid(x["work"]["id"]), "title": x["work"].get("display_name"),
                   "year": x["work"].get("publication_year"),
                   "cited_by_count": x["work"].get("cited_by_count"),
                   "doi": x["work"].get("doi"),
                   "match_score": x["match_score"], "year_hint": x.get("year_hint"),
                   "input": (x.get("input") or x["work"].get("display_name"))}
                  for x in sr],
        "params": {"k": k, "min_year": min_year, "direction": direction,
                   "min_seed_links": msl_used, "min_seed_links_configured": min_seed_links,
                   "weights": {"seed": W_SEED, "cite": W_CITE, "venue": W_VENUE},
                   "now_year": now_year, "max_candidates": max_candidates},
        "candidates_raw": len(cand), "candidates_dedup": len(items),
        "filtered_out": drops,
        "must_read": must_read,
        "checklist": checklist,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True, help="3-5 篇种子论文（标题/DOI/OpenAlex id）")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--min-year", type=int, default=None, help="滤掉太老的背景文献")
    ap.add_argument("--min-seed-links", type=int, default=2,
                    help="至少被几篇种子牵到才算候选（默认 2：滤掉通用基座论文）")
    ap.add_argument("--direction", default="both", choices=["both", "backward", "forward"])
    ap.add_argument("--out", default="")
    ap.add_argument("--review-refs", default="", help="综述参考文献文件（每行一条）→ 顺带算覆盖率")
    args = ap.parse_args()

    res = build(args.seeds, k=args.k, min_year=args.min_year, direction=args.direction,
                min_seed_links=args.min_seed_links)

    print("\n" + "=" * 92)
    print(f"代理必读清单 Top-{len(res['must_read'])}"
          f"（候选 {res['candidates_raw']} → 去重 {res['candidates_dedup']}）")
    print("=" * 92)
    print(f"{'#':>3} {'年份':>5} {'被引':>7} {'种子':>4} {'venue类型':>10} 标题")
    for i, it in enumerate(res["must_read"], 1):
        print(f"{i:3d} {it['year'] or '?':>5} {it['cited_by_count'] or 0:7d} "
              f"{it['seed_links']:>4} {str(it['venue_type'] or '-'):>10} {it['title'][:50]}")
    print(f"\n权重：{res['params']['weights']}")

    if args.review_refs:
        refs = [l for l in Path(args.review_refs).read_text(encoding="utf-8").splitlines() if l.strip()]
        cov = coverage(res["checklist"], refs)
        res["coverage"] = cov
        print("\n" + "=" * 92)
        print(f"引用覆盖率（种子+扩展共 {cov['must_read_k']} 篇）："
              f"命中 {cov['hit']} → recall = {cov['recall']}")
        print("=" * 92)
        for it in cov["missed_items"]:
            print(f"  ✗ 漏  {it['title'][:64]}  ({it['year']}, 被引{it['cited_by_count']})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已存 {args.out}")


if __name__ == "__main__":
    main()

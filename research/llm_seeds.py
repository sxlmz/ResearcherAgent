# -*- coding: utf-8 -*-
"""
llm_seeds.py —— 用 LLM 把研究方向拆成子主题，并为每个子主题给出代表性论文（种子）。

泛用性设计：不针对任何具体方向写死种子；输入任意研究方向名，
LLM 按「经典基础 → 主流深度 → 近两年热点 → 交叉方向」拆 **3-5 个子主题**，
每个子主题 1-2 篇高被引代表作。结果落盘缓存（logs/eval/llm_seeds/<topic>.json），
重复调用不重复花钱。

⚠️ 这里的「3-5」必须与 `SYSTEM_PROMPT` 里的「最多 5 个」保持一致——
docstring 曾写「8-12 个」，与 prompt 矛盾（prompt 一直是对的，所以不影响实际效果，
但会误导读代码的人）。`generate()` 另有两处防御性截断兜底（读旧缓存 / 新返回超限）。
"""
import json
import os
import re
import sys
import time
from pathlib import Path

# --- repo-root bootstrap（本模块既被 import，也有 `python ResearcherAgent/research/llm_seeds.py <主题>` 的 CLI 用法）---
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ResearcherAgent" / "__init__.py").exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dotenv
from openai import OpenAI

from ResearcherAgent.core.paths import REPO_ROOT

CACHE_DIR = Path("logs") / "eval" / "llm_seeds"

SYSTEM_PROMPT = """你是学术文献综述专家，负责把一个研究方向拆解成子主题并给出每个子主题的代表性论文。

子主题拆分原则：
- **最多 5 个**（3~5 个），覆盖四类：经典基础方法、主流深度学习方法、近 2-3 年热点方向、与相邻领域的交叉方法
- 子主题之间界限清晰、互不重叠

代表性论文原则：
- 必须真实存在、是该子主题学术界公认的高被引论文
- 标题必须与论文原文标题一致（会拿去数据库精确检索，自创/改写的标题会导致检索失败）
- 每个子主题给 1-2 篇，优先选最经典、被引最高的一篇
- 不同子主题之间论文不要重复

只输出一个 JSON 数组，不要输出任何其他文字。格式：
[{"subtopic": "子主题名", "papers": [{"title": "论文完整标题", "year": 发表年份}]}]"""


def _load_env() -> tuple[str, str, str]:
    dotenv.load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("SINGLE_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
    base = (os.environ.get("SINGLE_BASE") or "https://api.deepseek.com").strip()
    model = (os.environ.get("SINGLE_MODEL") or "deepseek-v4-flash").strip()
    return key, base, model


def _extract_json(text: str):
    """健壮解析：容忍 ```json 围栏、前后杂音；取第一个完整 JSON 数组。"""
    t = text.strip()
    # 去 markdown 围栏
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I).strip()
    s = t.find("[")
    e = t.rfind("]")
    if s == -1 or e == -1 or e <= s:
        return None
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return None


def _validate(raw) -> list[dict]:
    """校验并清洗 LLM 输出 → [{subtopic, papers:[{title, year}]}]"""
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        sub = str(item.get("subtopic") or "").strip()
        papers = []
        for p in item.get("papers") or []:
            if not isinstance(p, dict):
                continue
            title = str(p.get("title") or "").strip()
            if len(title) < 8:      # 太短大概率是幻觉
                continue
            year = p.get("year")
            try:
                year = int(year) if year else None
            except (TypeError, ValueError):
                year = None
            papers.append({"title": title, "year": year})
        if sub and papers:
            out.append({"subtopic": sub, "papers": papers})
    return out


def generate(topic: str, force: bool = False, temperature: float = 0.3,
             max_retry: int = 2) -> list[dict]:
    """LLM 生成子主题 + 种子。缓存命中直接返回；force=True 强制重跑。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cf = CACHE_DIR / f"{topic}.json"
    if cf.exists() and not force:
        try:
            val = json.loads(cf.read_text(encoding="utf-8"))
            if len(val) > 5:                       # 旧缓存可能超 5 个，统一截断
                val = val[:5]
                cf.write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")
            return val
        except Exception:
            pass

    key, base, model = _load_env()
    if not key:
        raise RuntimeError("未找到 DeepSeek API key（.env 的 DEEPSEEK_API_KEY）")
    client = OpenAI(api_key=key, base_url=base)

    prompt = f"研究方向：{topic}\n\n请按系统要求输出子主题与代表性论文的 JSON 数组。"
    last = None
    for i in range(max_retry + 1):
        try:
            r = client.chat.completions.create(
                model=model, temperature=temperature,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}])
            raw = _extract_json(r.choices[0].message.content or "")
            val = _validate(raw)
            if val:
                if len(val) > 5:
                    print(f"  ⚠️ LLM 返回 {len(val)} 个子主题，截断到 5 个")
                    val = val[:5]
                cf.write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")
                return val
            last = f"解析失败: {(r.choices[0].message.content or '')[:120]}"
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
        print(f"  ⚠️ LLM 第 {i+1} 次尝试失败：{last}；重试…")
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"LLM 生成子主题种子失败（3 次）：{last}")


def flatten(topic_seeds: list[dict]) -> list[dict]:
    """[{subtopic, papers:[...]}] → [{"title","year","subtopic"}] 平铺种子列表"""
    out = []
    for item in topic_seeds:
        for p in item.get("papers") or []:
            out.append({"title": p["title"], "year": p.get("year"),
                        "subtopic": item.get("subtopic")})
    return out


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "时序预测"
    res = generate(t)
    flat = flatten(res)
    print(f"\n{t} 子主题数: {len(res)}｜种子总数: {len(flat)}")
    for item in res:
        for p in item["papers"]:
            print(f"  [{item['subtopic']}] {p.get('year')} | {p['title'][:70]}")

# -*- coding: utf-8 -*-
"""
innovation_tools.py — 「文献综述 → 创新点验证」流水线工具集（skill: innovate）

流水线（模型在 agent 循环中自由编排，工具只提供能力）：
  1. search_papers（复用 MCP 检索）          → paper_pool
  2. read_papers / read_sections（复用精读制卡）→ paper_cards
  3. deep_survey（新增，照搬 AI-Researcher Paper Survey）→ 把主题拆成原子学术定义，
     从卡片提取数学公式/理论 → state.survey_notes
  4. propose_innovations（改造，照搬 idea_agent）→ 基于 survey_notes 生成 5 个结构化
     proposal（Challenges/Motivation/Proposed Method/数学公式/Expected Outcomes）
     → 横向比较择优增强 → state.innovations
  5. novelty_check                          → 多源检索查重，duplicate 自动拦截
  6. code_survey（新增，照搬 Code Survey Agent）→ 创新点 + 参考代码库 → 实现报告
     （每个关键机制 → 可复用代码 + 需新增部分）→ state.code_survey_reports
  7. run_baseline                            → 确定性工具：跑 PatchTST 基线，得基线指标
  8. implement_innovation                    → LLM 为指定创新点生成实验代码（注入实现报告）
     → 沙箱运行 → 实验指标
  9. judge_innovation                        → LLM 对比基线 vs 实验，判断可行性

复用约定：
- 检索/精读/卡片机制完全复用现有组件（MCP 检索工具 + read_papers + paper_cards）。
- 代码执行复用 code_exec 工具组（create_file / run_python，后端由 LANGGRAPH_EXEC_BACKEND 决定）。

指标协议：训练脚本结束打印一行  METRIC_JSON={"dataset":...,"pred_len":...,"epochs":...,
"mse":...,"mae":...,"device":...,"best_val":...,"seconds":...}，工具用正则提取解析。
"""

import asyncio
import json
import os
import re
from datetime import datetime
from typing import Annotated, Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from ResearcherAgent.core import tracing
from ResearcherAgent.sandbox import code_executor as ce
from ResearcherAgent.sandbox.sandbox_exec import PROJECT_ROOT, get_env
from ResearcherAgent.memory import cc_memory
from ResearcherAgent.retrieval.fulltext import fetch_arxiv_fulltext

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
INNOVATE_TOOLS = ("deep_survey", "propose_innovations", "prepare_reference_repos", "run_baseline",
                  "implement_innovation", "fix_and_rerun", "judge_innovation",
                  "novelty_check", "code_survey")

_METRIC_RE = re.compile(r"METRIC_JSON\s*=\s*(\{.*?\})\s*$", re.M | re.S)
_CARD_MAX_CHARS = 1800          # 每张卡片注入 LLM 的截断长度
_LOG_TAIL_CHARS = 4000          # 实验日志尾部（回给模型）
_BASELINE_CACHE = "project/.baseline_cache.json"

# —— 踩坑经验库（方案 1：轻量 EXPERIENCE.md，fix_and_rerun 自动沉淀，implement/fix 注入参考）——
_EXP_FILE = "project/EXPERIENCE.md"
_EXP_TAIL_N = 5                 # 每次注入 prompt 的最近条数
_EXP_KEEP = 60                  # 文件保留的最大条数

# ---------------------------------------------------------------------------
# 小工具：JSON 解析 / 指标提取 / 缓存
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> Any:
    """从 LLM 输出里提取第一个 JSON 数组/对象（兼容 ```json 围栏与裸输出）。"""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    raw = m.group(1) if m else text
    starts = [i for i in (raw.find("["), raw.find("{")) if i >= 0]
    if not starts:
        return None
    st = min(starts)
    open_ch, close_ch = ("[", "]") if raw[st] == "[" else ("{", "}")
    depth, end = 0, -1
    for i in range(st, len(raw)):
        if raw[i] == open_ch:
            depth += 1
        elif raw[i] == close_ch:
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None
    try:
        return json.loads(raw[st:end + 1])
    except Exception:
        return None


def _extract_metric(text: str) -> dict | None:
    m = _METRIC_RE.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def _cards_context(cards: list) -> str:
    """把 paper_cards 压成 LLM 可读的上下文（title + 方法段落）。"""
    if not cards:
        return ""
    lines = []
    for i, c in enumerate(cards, 1):
        title = (c.get("title") or "").strip() or f"论文{i}"
        body = (c.get("card") or "").strip()
        if len(body) > _CARD_MAX_CHARS:
            body = body[:_CARD_MAX_CHARS] + "…"
        lines.append(f"[{i}] 《{title}》\n{body}")
    return "\n\n".join(lines)


def _load_cache() -> dict:
    try:
        raw = ce.read_file.invoke({"file_path": _BASELINE_CACHE})
        # read_file 返回 "Error reading file..." 或内容
        if raw.startswith("Error reading file"):
            return {}
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


def _write_cache(cache: dict):
    try:
        ce.create_file.invoke({"path": _BASELINE_CACHE,
                               "content": json.dumps(cache, ensure_ascii=False, indent=1)})
    except Exception:
        pass


def _ensure_dataset(dataset: str) -> str:
    """确保数据集已拷贝进沙箱工作区，返回工作区内的数据目录（相对 workspace 根）。

    领域无关：拷贝 datasets/{dataset}/ 下**全部数据文件**（不再写死 ETTh1.csv）；
    具体用哪个文件由基线/实验脚本的 EXEC_ARGS 决定。
    """
    import glob as _glob
    src_dir = os.path.join(PROJECT_ROOT, "datasets", dataset)
    # 本机源目录存在且非空 → 整目录同步进沙箱（copy /Y 幂等）
    if os.path.isdir(src_dir) and _glob.glob(os.path.join(src_dir, "*")):
        ce.execute_command.invoke({"command": (
            f'if not exist "datasets\\{dataset}" mkdir "datasets\\{dataset}" && '
            f'copy /Y "{src_dir}\\*" "datasets\\{dataset}\\"')})
    else:
        # 源目录不在本机（可能只存在于沙箱内）→ 仅确保目录存在
        ce.execute_command.invoke({"command": (
            f'if not exist "datasets\\{dataset}" mkdir "datasets\\{dataset}"')})
    return f"datasets/{dataset}"


# ---------------------------------------------------------------------------
# 踩坑经验库：EXPERIENCE.md 读写（方案 1）
# 格式：每行一条 JSON，如
#   - {"time":"2026-10-02T13:20:00","innovation_id":"inn_01",
#      "error":"ModuleNotFoundError: numpy","cause":"沙箱未装 numpy","fix":"pip install numpy 后重跑"}
# ---------------------------------------------------------------------------
def _load_experiences(limit: int = _EXP_KEEP) -> list:
    """读取经验库全部条目（底层接口，避免工具层截断）。"""
    try:
        raw = get_env().read_file(_EXP_FILE)
    except Exception:
        return []
    if not raw or raw.startswith("Error reading file"):
        return []
    entries = []
    for line in str(raw).splitlines():
        line = line.strip()
        if line.startswith("- "):
            try:
                d = json.loads(line[2:])
                if isinstance(d, dict):
                    entries.append(d)
            except Exception:
                pass
    return entries[-limit:]


def _append_experience(entry: dict):
    """追加一条经验（时间戳 + 创新点 id + 错误/根因/修复），文件按 _EXP_KEEP 截断。"""
    try:
        entries = _load_experiences(_EXP_KEEP)
    except Exception:
        entries = []
    entries.append(entry)
    entries = entries[-_EXP_KEEP:]
    lines = [
        "# 踩坑经验库（fix_and_rerun 自动记录；implement_innovation / fix_and_rerun 会注入最近条数供参考）",
        "# 每行一条 JSON：{\"time\", \"innovation_id\", \"error\", \"cause\", \"fix\"}",
        "",
    ]
    lines += [f"- {json.dumps(e, ensure_ascii=False)}" for e in entries]
    try:
        get_env().write_file(_EXP_FILE, "\n".join(lines))
    except Exception:
        pass


def _experiences_context(n: int = _EXP_TAIL_N) -> str:
    """把最近 n 条经验压成 prompt 注入文本（空则返回空串）。"""
    entries = _load_experiences(_EXP_KEEP)[-n:]
    if not entries:
        return ""
    lines = []
    for i, e in enumerate(entries, 1):
        lines.append(f"{i}. 错误：{e.get('error', '')}｜根因：{e.get('cause', '')}｜修复：{e.get('fix', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PatchTST 基线脚本（自包含模板，写入沙箱后运行）
# ⚠️ 领域说明：PatchTST/ETT/滑窗是**时序领域的演示默认基线**；
#    其他领域由 implement_innovation 的 LLM 按主题上下文生成对应基线，本模板仅作参考骨架。
# ---------------------------------------------------------------------------
BASELINE_PY = r'''# -*- coding: utf-8 -*-
"""PatchTST 基线（ICLR 2023, "A Time Series is Worth 64 Words"）—— 自包含。
时序领域演示默认基线；其他领域按实验主题替换数据加载/模型/指标部分。
参数经环境变量 EXEC_ARGS 传入（格式: |data_dir|seq_len|pred_len|epochs），缺省用默认值。
输出: 每 epoch 一行日志 + 末尾一行 METRIC_JSON={...}
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

_args = (os.environ.get("EXEC_ARGS") or "").split("|")
SEQ_LEN = int(_args[2]) if len(_args) > 2 else 336
PRED_LEN = int(_args[3]) if len(_args) > 3 else 96
EPOCHS = int(_args[4]) if len(_args) > 4 else 50
_PROJ = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = _args[1] if len(_args) > 1 else os.path.join(_PROJ, "..", "datasets", "ETTH1")
# 数据文件自动匹配：优先 ETTh1.csv（时序演示默认），否则取目录内第一个 CSV（领域无关）
import glob as _glob
_CSV_CAND = os.path.join(DATA_DIR, "ETTh1.csv")
if not os.path.exists(_CSV_CAND):
    _found = _glob.glob(os.path.join(DATA_DIR, "*.csv"))
    _CSV_CAND = _found[0] if _found else _CSV_CAND
CSV = _CSV_CAND
PATCH_LEN, STRIDE = 16, 8
D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT = 16, 4, 3, 128, 0.2
BATCH, LR = 128, 0.001
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---- 数据（60/20/20 划分 + train 统计量标准化）----
df = pd.read_csv(CSV)
data = df.drop(columns=["date"]).values.astype(np.float32)   # (N, M)
n_vars = data.shape[1]
N = len(data)
train_end, val_end = int(N * 0.6), int(N * 0.8)
mean = data[:train_end].mean(axis=0)
std = data[:train_end].std(axis=0) + 1e-8
data = (data - mean) / std


class TSDataset(Dataset):
    def __init__(self, x, s, h):
        self.x, self.s, self.h = x, s, h

    def __len__(self):
        return len(self.x) - self.s - self.h + 1

    def __getitem__(self, i):
        return (torch.tensor(self.x[i:i + self.s].T, dtype=torch.float32),
                torch.tensor(self.x[i + self.s:i + self.s + self.h].T, dtype=torch.float32))


def loader(a, b, shuffle):
    return DataLoader(TSDataset(data[a:b], SEQ_LEN, PRED_LEN), batch_size=BATCH, shuffle=shuffle)


# ---- 模型：Channel Independence + Patch + Transformer（ETTh1 缩小版）----
class PatchTST(nn.Module):
    def __init__(self, n_vars):
        super().__init__()
        self.patch_num = (SEQ_LEN - PATCH_LEN) // STRIDE + 2   # N = floor((L-P)/S)+2
        self.embed = nn.Linear(PATCH_LEN, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, self.patch_num, D_MODEL) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEADS,
                                         dim_feedforward=D_FF, dropout=DROPOUT,
                                         activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=N_LAYERS)
        self.head = nn.Linear(self.patch_num * D_MODEL, PRED_LEN)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):                                     # (B, M, L)
        B, M, L = x.shape
        mu, sd = x.mean(-1, keepdim=True), x.std(-1, keepdim=True) + 1e-5
        x = (x - mu) / sd
        x = torch.nn.functional.pad(x, (0, STRIDE))
        pat = x.unfold(-1, PATCH_LEN, STRIDE)                 # (B, M, N, P)
        pm = pat.reshape(B * M, self.patch_num, PATCH_LEN)    # Channel Independence
        z = self.embed(pm) + self.pos
        z = self.encoder(z)
        out = self.head(z.reshape(B * M, -1)).reshape(B, M, PRED_LEN)
        return out * sd + mu


# ---- 训练 / 评估 ----
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PatchTST(n_vars).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
crit = nn.MSELoss()


def evaluate(loader):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            tot += crit(model(x), y).item() * len(x)
            n += len(x)
    return tot / n


t0 = time.time()
last_val = None
for ep in range(1, EPOCHS + 1):
    model.train()
    tr, n = 0.0, 0
    for x, y in loader(0, train_end, True):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        tr += loss.item() * len(x)
        n += len(x)
    last_val = evaluate(loader(train_end, val_end, False))
    print(f"epoch {ep}: train_mse={tr / n:.6f} val_mse={last_val:.6f}", flush=True)

test_mse = evaluate(loader(val_end, N, False))
model.eval()
mae, n = 0.0, 0
with torch.no_grad():
    for x, y in loader(val_end, N, False):
        x, y = x.to(device), y.to(device)
        mae += (model(x) - y).abs().mean().item() * len(x)
        n += len(x)
metric = {"dataset": os.path.basename(DATA_DIR), "seq_len": SEQ_LEN, "pred_len": PRED_LEN,
          "epochs": EPOCHS, "mse": round(test_mse, 6), "mae": round(mae / n, 6),
          "device": device, "best_val": round(last_val, 6) if last_val else None,
          "seconds": round(time.time() - t0, 1)}
print("METRIC_JSON=" + json.dumps(metric, ensure_ascii=False))
'''


# ---------------------------------------------------------------------------
# 1. deep_survey —— 原子学术定义拆解 + 论文公式提取（照搬 AI-Researcher Paper Survey Agent）
#    输入论文卡片，把研究主题拆成可实现的「原子学术定义」，逐一定义提取数学公式/理论基础，
#    输出 state.survey_notes，作为 propose_innovations 的公式级输入。
#
# 升级（PDF 导航精读等价）：拆解完成后，对每个定义去来源论文**原文**（ar5iv 结构化全文，
# 保留 LaTeX 公式）中定位精确公式并回填——不再只依赖精读卡片（卡片压缩可能丢公式细节）。
# ---------------------------------------------------------------------------
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")

def _extract_arxiv_id(s: str) -> str:
    m = _ARXIV_ID_RE.search(s or "")
    return m.group(1) if m else ""

_PAPER_NOTES_SYS = (
    "你是科研精读专家。下面是论文《{title}》的完整全文（含 LaTeX 公式，形如 $...$ 或 \\[ ... \\]）。\n"
    "请**通读全文、边读边理解**，提取该论文中最值得作为「创新点构建基石」的原子学术定义。\n"
    "原子学术定义的标准：\n"
    "1) 单个自包含概念——示例（仅示意「单个概念」的含义，领域变通，不要照抄）："
    "『Patch 嵌入与通道独立注意力』『可分解注意力』『频域损失项』；"
    "不要合并多个概念；\n"
    "2) 有明确的数学形式——公式必须**原样照抄全文**（LaTeX 或文字公式），并标注出处章节；\n"
    "3) 可被代码实现——给出实现要点（伪代码/张量形状/关键变换/损失项）；\n"
    "4) 只提取该论文**确实提出或详细论述**的概念，不要泛泛的背景概念。\n"
    "直接输出 1-4 个定义的 JSON 数组，**不要复述格式说明**。字段：\n"
    '   [{{"definition": "原子学术定义", "math_formula": "原样公式（LaTeX）", '
    '"theory": "理论基础（2-3 句）", "code_hints": "实现要点（伪代码/张量形状）", '
    '"section": "公式出处章节"}}]'
)

def _read_paper_notes(llm, aid: str, title: str, session_id: str) -> list[dict]:
    """单篇论文全文结构化精读：通读全文 → 一次提取该论文的全部原子定义（公式原样照抄）。

    读全文不再截断式取前 20000 字符（那会漏掉靠后章节的公式，如 FEDformer §3.3）；
    ar5iv 全文上限放宽到 80000 字符、单节 8000，覆盖绝大多数论文正文。
    失败/无全文返回空列表（由调用方降级）。
    """
    ft = fetch_arxiv_fulltext(aid, max_chars=80000, section_max=8000)
    if len(ft) < 500:
        print(f"    [deep_survey] 《{str(title)[:28]}》arXiv {aid} 无全文，降级卡片提取")
        return []
    user = (f"论文《{title}》（arXiv:{aid}）完整全文如下：\n\n{ft[:80000]}")
    try:
        with tracing.span("llm", "deep_survey_paper_read", session_id, id=aid):
            resp = llm.invoke([SystemMessage(content=_PAPER_NOTES_SYS.format(title=title)),
                               HumanMessage(content=user)])
        raw_txt = resp.content if hasattr(resp, "content") else str(resp)
        data = _extract_json_block(raw_txt)
        if not isinstance(data, list):
            print(f"    [deep_survey] 《{str(title)[:28]}》未返回定义 JSON，跳过")
            return []
        out = []
        for d in data[:4]:
            if not isinstance(d, dict) or not d.get("definition"):
                continue
            out.append({
                "definition": str(d.get("definition") or "").strip()[:120],
                "math_formula": str(d.get("math_formula") or "").strip()[:800],
                "theory": str(d.get("theory") or "").strip()[:400],
                "code_hints": str(d.get("code_hints") or "").strip()[:500],
                "reference_papers": [str(title).strip()],
                "arxiv_id": aid,
                "section": str(d.get("section") or "").strip()[:80],
                "formula_source": f"arXiv:{aid} 原文精读",
            })
        print(f"    [deep_survey] 《{str(title)[:28]}》全文精读 → {len(out)} 个定义（arXiv:{aid}）")
        return out
    except Exception as e:
        print(f"    [deep_survey] 《{str(title)[:28]}》精读失败：{type(e).__name__}: {str(e)[:120]}")
        return []

def _norm_def(s: str) -> str:
    """定义名归一化（去空白/标点/大小写），用于跨论文聚合去重。"""
    return re.sub(r"[\s\u3000:：,，.。()（）\-_/“”\"']+", "", s or "").lower()[:60]

def _merge_notes(all_notes: list[dict], max_definitions: int) -> list[dict]:
    """跨论文聚合：同义定义合并（保留多个来源、互补公式），输出上限 max_definitions。"""
    merged: list[dict] = []
    for n in all_notes:
        k = _norm_def(n.get("definition", ""))
        if not k:
            continue
        hit = None
        for m in merged:
            mk = _norm_def(m.get("definition", ""))
            if k == mk or (len(k) >= 6 and (k in mk or mk in k)):
                hit = m
                break
        if hit:
            src = n.get("reference_papers") or []
            if src and src[0] not in (hit.get("reference_papers") or []):
                hit.setdefault("reference_papers", []).append(src[0])
            if not hit.get("math_formula") and n.get("math_formula"):
                hit["math_formula"] = n["math_formula"]
        else:
            merged.append(dict(n))
    return merged[:max_definitions]

def _cards_fallback_notes(llm, cards: list, theme: str, session_id: str) -> list[dict]:
    """降级路径：无论文可读全文时，从精读卡片提取定义（老逻辑）。"""
    ctx = _cards_context(cards)
    sys = (
        "你是理论功底扎实的科研调研专家。任务：把研究主题拆解为若干「原子学术定义」。\n"
        "原子学术定义的标准：\n"
        "1) 单个自包含概念——示例（仅示意「单个概念」的含义，领域变通，不要照抄）："
        "『Patch 嵌入与通道独立注意力』『可分解注意力』『频域损失项』；"
        "不要合并多个概念；\n"
        "2) 有明确的数学形式（公式/伪代码/变换定义）；\n"
        "3) 可被代码实现（PyTorch 可写）；\n"
        "4) 可追溯到具体论文。\n"
        "对每个定义，从给定的论文精读卡片（含原文段落证据）中提取：数学公式、理论基础、"
        "来源论文、实现要点提示。卡片中没有明确公式的定义，基于卡片内容做忠实归纳并标注"
        "『（卡片未给公式，基于描述归纳）』。\n"
        "只输出一个 JSON 数组，字段：\n"
        '   [{"definition": "原子学术定义", "math_formula": "数学公式（LaTeX 或文字公式）", '
        '"theory": "理论基础（2-3 句）", "reference_papers": ["来源论文"], '
        '"code_hints": "实现要点（结构/张量形状/损失项）"}]'
    )
    try:
        with tracing.span("llm", "deep_survey_card_fallback", session_id, id="cards"):
            resp = llm.invoke([SystemMessage(content=sys),
                               HumanMessage(content=f"研究主题：{theme}\n\n论文精读卡片：\n{ctx}")])
        data = _extract_json_block(resp.content if hasattr(resp, "content") else str(resp))
        if not isinstance(data, list):
            return []
        return [d for d in data if isinstance(d, dict) and d.get("definition")]
    except Exception as e:
        print(f"    [deep_survey] 卡片降级提取失败：{type(e).__name__}: {str(e)[:120]}")
        return []

def make_deep_survey(llm):
    @tool
    def deep_survey(
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        topic: str | None = None,
        max_definitions: int = 6,
        max_papers: int = 5,
    ) -> Command:
        """深度调研：对已精读论文做「全文结构化精读」——通读每篇论文原文，
        一次提取该论文的原子学术定义（公式原样照抄 + 代码实现线索），跨论文聚合去重。

        适用：read_papers/read_sections 精读完成、paper_cards 已有卡片后调用（在 propose_innovations 之前）。
        输出写入 state.survey_notes（definition / math_formula / theory / reference_papers /
        code_hints / formula_source / section），让后续创新点提出建立在「原文公式级理解」之上。
        """
        cards = state.get("paper_cards") or []
        if not cards:
            return Command(update={"messages": [ToolMessage(
                content="paper_cards 为空：请先调用 read_papers（精读 5-10 篇）再做深度调研。",
                tool_call_id=tool_call_id, status="error")]})
        theme = topic or state.get("topic") or state.get("user_request") or "该领域"
        session_id = state.get("session_id", "")

        # 1) 按 arXiv id 去重选论文（无 id 的卡片也保留，走降级）
        papers, seen_ids = [], set()
        for c in cards:
            aid = _extract_arxiv_id(str(c.get("id") or ""))
            if aid and aid in seen_ids:
                continue
            if aid:
                seen_ids.add(aid)
            papers.append({"aid": aid, "title": (c.get("title") or "").strip() or "论文",
                           "card": c})
        papers = papers[:max_papers]

        # 2) 全文结构化精读（主路径）
        all_notes = []
        for p in papers:
            if p["aid"]:
                all_notes.extend(_read_paper_notes(llm, p["aid"], p["title"], session_id))
        # 3) 主路径产出不足 → 降级：从卡片提取
        if not all_notes:
            print("    [deep_survey] 无可精读全文，降级从精读卡片提取定义")
            notes = _cards_fallback_notes(llm, cards, theme, session_id)
        else:
            notes = _merge_notes(all_notes, max_definitions)

        if not notes:
            return Command(update={"messages": [ToolMessage(
                content="深度调研未产出有效原子定义（全文/卡片均失败），请重试或调整精读范围。",
                tool_call_id=tool_call_id, status="error")]})

        lines = [f"深度调研完成：全文精读 {len(papers)} 篇论文，聚合出 {len(notes)} 个原子学术定义："]
        for i, d in enumerate(notes, 1):
            formula = (d.get("math_formula") or "")[:80]
            src = d.get("formula_source") or "〔卡片归纳〕"
            lines.append(f"{i}. {d.get('definition', '未命名')}（{src}）")
            lines.append(f"   公式：{formula}")
            if d.get("section"):
                lines.append(f"   出处：{d.get('section')}")
        lines.append("")
        lines.append("👉 下一步：调用 propose_innovations —— 它将基于这些公式级笔记"
                     "生成 5 个结构化创新点方案并择优增强。")
        return Command(update={
            "survey_notes": notes,
            "messages": [ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)],
        })

    return deep_survey


# ---------------------------------------------------------------------------
# 2. propose_innovations —— 基于精读卡片提出 3-5 个创新点
# ---------------------------------------------------------------------------
def make_propose_innovations(llm):
    @tool
    def propose_innovations(
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        topic: str | None = None,
        count: int = 5,
    ) -> Command:
        """基于精读卡片/深度调研笔记提出创新点，存入 state.innovations（两阶段：生成→择优增强）。

        照搬 AI-Researcher idea_agent 逻辑：先基于 survey_notes（或论文卡片）生成
        多个结构化 proposal（挑战/现有方法/动机/方法/数学公式/预期结果），再横向比较
        新颖性/可行性/完整性，择优增强 1-2 个为最终创新点。
        适用：deep_survey 之后（或 read_papers 完成后）；topic：研究主题；count：最终数量。
        """
        cards = state.get("paper_cards") or []
        notes = state.get("survey_notes") or []
        if not cards and not notes:
            return Command(update={"messages": [ToolMessage(
                content="paper_cards 与 survey_notes 均为空：请先调用 read_papers 精读，"
                        "建议再调用 deep_survey 做原子定义拆解后再提出创新点。",
                tool_call_id=tool_call_id, status="error")]})

        theme = topic or state.get("topic") or state.get("user_request") or "该领域"
        evidence = ""
        if notes:
            evidence = "【深度调研笔记（原子学术定义 + 数学公式）】\n" + json.dumps(notes, ensure_ascii=False, indent=1)[:_CARD_MAX_CHARS * 3]
        evidence += "\n\n【论文精读卡片】\n" + _cards_context(cards)
        # ---- 阶段 1：生成 count*2 个结构化 proposal ----
        sys1 = (
            "你是科研方法论创新专家。基于给定的深度调研笔记（含原子定义与数学公式）与论文卡片，"
            "为研究主题生成多个**结构化创新点方案**。\n"
            "要求：\n"
            "1) 创新点必须具体可实现：基于 PyTorch、可直接在**该领域的标准实验管线**上实现"
            "（数据划分/基线模型由主题上下文给定；若无显式基线，从精读论文中最常被对比的模型中选一个作基线，"
            "并在方案里写明用哪个基线、在什么数据集上对比、用什么指标）。\n"
            "2) 每个方案必须给出：挑战（现有方法瓶颈）、现有方法、动机、具体方法（结构/训练改动）、"
            "数学公式（尽可能完整）、预期结果、基于哪些论文、难度。\n"
            "3) 至少 1 个低难度（数据/训练层面小改动）、至少 1 个中高难度（结构创新）。\n"
            '4) 只输出一个 JSON 数组，元素字段：\n'
            '   [{"id": "inn_01", "name": "方案名称", "challenges": "现有瓶颈", '
            '"existing_methods": "现有方法", "motivation": "动机", '
            '"proposed_method": "具体方法（300字内）", "math_formula": "数学公式（尽量完整）", '
            '"expected_outcomes": "预期结果", "based_on": "基于哪些论文", "difficulty": "低/中/高"}]'
        )
        user1 = f"研究主题：{theme}\n\n{evidence}"
        with tracing.span("llm", "propose_stage1", state.get("session_id", "")):
            resp1 = llm.invoke([SystemMessage(content=sys1), HumanMessage(content=user1)])
        proposals = _extract_json_block(resp1.content if hasattr(resp1, "content") else str(resp1))
        if not isinstance(proposals, list) or not proposals:
            return Command(update={"messages": [ToolMessage(
                content="模型未返回有效创新点方案 JSON，请重试或调整调研范围。",
                tool_call_id=tool_call_id, status="error")]})
        proposals = proposals[: max(6, count * 2)]
        # ---- 阶段 2：择优增强 ----
        brief = []
        for i, p in enumerate(proposals, 1):
            brief.append(f"[{i}] id={p.get('id')} 名称={p.get('name','')} 难度={p.get('difficulty','')}\n"
                         f"    方法={p.get('proposed_method','')[:180]}\n"
                         f"    公式={p.get('math_formula','')[:120]}")
        sys2 = (
            "你是科研想法评审专家。请横向比较下面的创新点方案，按 新颖性 > 可行性 > 完整性 > 影响力 "
            "排序，选出最有潜力的 1-2 个，并对每个补全/增强为最终创新点。\n"
            "输出一个 JSON 数组（最多 2 个元素），元素字段（兼容下游 novelty_check / implement）：\n"
            '   [{"id": "inn_01", "name": "创新点名称", '
            '"hypothesis": "假设：为什么能提升", '
            '"method": "最终实现思路（结构/模块/训练改动，含关键公式描述，300字内）", '
            '"math_formula": "数学公式（尽量完整）", "based_on": "基于哪些论文", '
            '"expected_gain": "预期在哪个指标/哪类序列上提升", "difficulty": "低/中/高"}]'
        )
        user2 = "研究主题：{}\n\n候选方案：\n{}".format(theme, "\n".join(brief))
        with tracing.span("llm", "propose_select", state.get("session_id", "")):
            resp2 = llm.invoke([SystemMessage(content=sys2), HumanMessage(content=user2)])
        data = _extract_json_block(resp2.content if hasattr(resp2, "content") else str(resp2))
        if not isinstance(data, list) or not data:
            # 择优失败则退回阶段 1 前 count 个（转成兼容字段）
            data = []
            for p in proposals[:count]:
                data.append({"id": p.get("id"), "name": p.get("name", ""),
                             "hypothesis": p.get("motivation", ""),
                             "method": p.get("proposed_method", ""),
                             "math_formula": p.get("math_formula", ""),
                             "based_on": p.get("based_on", ""),
                             "expected_gain": p.get("expected_outcomes", ""),
                             "difficulty": p.get("difficulty", "中")})
        data = data[: count] if count >= 1 else data[:1]
        lines = [f"已提出并择优 {len(data)} 个创新点："]
        for i, d in enumerate(data, 1):
            lines.append(f"{i}. [{d.get('difficulty', '中')}] {d.get('name', '未命名')}"
                         f"（{d.get('based_on', '')[:40]}）")
            lines.append(f"   假设：{d.get('hypothesis', '')[:100]}")
            if d.get("math_formula"):
                lines.append(f"   公式：{d['math_formula'][:100]}")
        lines.append("")
        lines.append("👉 下一步：调用 novelty_check 对创新点做新颖性查重（duplicate 会被自动拦截），"
                     "再为 novel / partially_explored 的创新点调用 code_survey 获取参考代码库实现报告，"
                     "最后跑实验。")
        return Command(update={
            "innovations": data,
            "messages": [ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)],
        })

    return propose_innovations


# ---------------------------------------------------------------------------
# 3. code_survey —— 参考代码库调查 → 创新点实现报告（照搬 AI-Researcher Code Survey Agent）
#    对选定创新点，读取参考代码库（工作区内目录，如 project/ 或用户提供的实现），
#    输出每个创新点的「实现报告」：关键机制 → 参考代码中可复用实现 + 需新增部分。
# ---------------------------------------------------------------------------
_CODE_SURVEY_MAX_FILES = 6
_CODE_SURVEY_CHARS = 3000


def _pick_code_files(tree: str, base: str, max_files: int) -> list[str]:
    """从代码树文本中按启发式挑选要精读的文件路径（相对工作区根）。

    用 tree 缩进（每级 4 字符）重建完整相对路径，正确处理子目录：
      project/
      ├── dir/
      │   └── main.py   →  "dir/main.py"
      └── model.py      →  "model.py"
    """
    cands = []
    dir_stack: list[str] = []
    for line in (tree or "").splitlines():
        if not line.strip():
            continue
        m = re.match(r"^([ │]*)[├└]──\s?(.*)$", line)
        if not m:
            continue
        depth = len(m.group(1)) // 4
        name = m.group(2).strip()
        if name.endswith("/"):
            dir_stack = dir_stack[:depth] + [name[:-1]]
            continue
        if name.endswith(".py"):
            cands.append("/".join(dir_stack[:depth] + [name]))
    keys = ("main", "run", "model", "train", "test", "data", "README", "readme")
    scored = []
    for c in cands:
        cl = c.lower()
        score = 0
        for k in keys:
            if k in cl:
                score += 1
        if cl.endswith("__init__.py"):
            score -= 1
        scored.append((score, len(c), c))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [c for _, _, c in scored[:max_files]]


def _parse_github_url(url: str) -> tuple | None:
    """把论文标注的 GitHub 链接解析为 (owner, repo)；不支持/非 github 返回 None。"""
    u = (url or "").strip()
    m = re.match(
        r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)"
        r"(?:\.git)?/?[\s.,;:)\]）】]*$", u, re.I)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"git@github\.com:([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?[\s.]*$", u, re.I)
    if m:
        return m.group(1), m.group(2)
    return None


def _github_search_repos(query: str, limit: int = 2, token: str | None = None) -> list[dict]:
    """GitHub 仓库搜索（对齐 HKUDS search_github_repos：标题关键词 + stars 排序）。

    无 token 走匿名限额（10 次/分），够创新点场景；失败返回空列表（视为无开源）。
    """
    import urllib.parse
    import urllib.request

    q = urllib.parse.quote(f"{query} in:name,description")
    url = f"https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page=10"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "ResearcherAgent",
    })
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return []
    out = []
    for item in (data.get("items") or [])[:limit]:
        out.append({
            "name": f"{item['owner']['login']}/{item['name']}",
            "stars": item.get("stargazers_count", 0),
            "url": item.get("html_url", ""),
            "clone_url": item.get("clone_url", ""),
            "description": (item.get("description") or "")[:120],
            "language": item.get("language"),
        })
    return out


def _github_fetch_py_files(owner: str, repo: str, max_files: int = 12,
                          token: str | None = None) -> dict[str, str]:
    """GitHub Contents API 拉取关键 .py 文件（git clone 不可达时的降级通道）。

    用 git/trees/HEAD?recursive=1 一次拿全树 → 启发式挑关键 .py → contents API 逐个拉内容。
    返回 {相对路径: 源码}；失败或无可拉文件返回空 dict。
    """
    import base64
    import urllib.parse
    import urllib.request

    def _gh(url: str) -> dict | None:
        req = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github+json", "User-Agent": "ResearcherAgent"})
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            return None

    tree = _gh(f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1")
    if not tree or not tree.get("tree"):
        return {}
    paths = [t.get("path", "") for t in tree["tree"]
             if t.get("type") == "blob" and t.get("path", "").endswith(".py")]
    if not paths:
        return {}
    # 启发式挑关键实现文件（同 _pick_code_files 打分）
    keys = ("main", "run", "model", "train", "test", "data", "utils", "module", "network")
    def _score(p: str):
        pl = p.lower()
        s = sum(1 for k in keys if k in pl)
        if pl.endswith("__init__.py") or "/tests/" in pl or pl.startswith("tests/"):
            s -= 2
        if "/experiments/" in pl or "/configs/" in pl:
            s -= 1
        return s
    chosen = sorted(paths, key=lambda p: (-_score(p), len(p)))[:max_files]
    out: dict[str, str] = {}
    for p in chosen:
        data = _gh(f"https://api.github.com/repos/{owner}/{repo}/contents/{urllib.parse.quote(p)}")
        if not data or not data.get("content"):
            continue
        try:
            out[p] = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        except Exception:
            continue
    return out


def _fetch_reference_repo(clone_url: str, owner: str, repo: str, dest: str,
                          token: str | None = None) -> str:
    """拉取参考仓库到沙箱；返回 'git'（git clone）/ 'api'（API 拉关键文件）/ 'CLONE_FAIL: ...'。"""
    # 通道 A：git clone（网络可达时最完整）
    out = ce.execute_command.invoke(
        {"command": f"git clone --depth 1 --single-branch {clone_url} {dest}"})
    if "Exit code: 0" in out:
        return "git"
    # 通道 B：GitHub Contents API 拉关键 .py（api.github.com 通常可达）
    files = _github_fetch_py_files(owner, repo, token=token)
    if files:
        n_ok = 0
        for rel, content in files.items():
            try:
                ce.write_file.invoke({"path": f"{dest}/{rel}", "content": content})
                n_ok += 1
            except Exception:
                pass
        if n_ok:
            return f"api({n_ok} files)"
    return f"CLONE_FAIL: git={out[:120]}"


def make_prepare_reference_repos(llm):
    @tool
    def prepare_reference_repos(
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        max_repos_per_paper: int = 2,
    ) -> Command:
        """按论文标题搜索 GitHub 开源实现并浅克隆到沙箱（对齐 AI-Researcher Prepare Agent）。

        适用：deep_survey 之后调用。对每篇论文卡标题调 GitHub Search API，
        按 stars 选 top-N 仓库，浅克隆到工作区 reference_repos/{owner}__{name}。
        结果写入 state.reference_repos（论文标题 → 仓库列表，含本地路径与 cloned 状态）。
        论文无开源实现（搜索无结果/克隆失败）时该论文标注 no_open_source，
        后续 code_survey 将回退本地参考库（LLM 自行生成实现路线）。
        """
        cards = state.get("paper_cards") or []
        titles = [c.get("title", "").strip() for c in cards if c.get("title")]
        if not titles:
            return Command(update={"messages": [ToolMessage(
                content="paper_cards 为空：请先调用 read_papers 精读论文制卡，再准备参考仓库。",
                tool_call_id=tool_call_id, status="error")]})
        token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_AI_TOKEN") or None)
        refs = dict(state.get("reference_repos") or {})
        lines = [f"参考代码库准备完成（{len(titles)} 篇论文，GitHub 检索 + 浅克隆）："]
        for title in titles:
            entry = []
            card = next((c for c in cards if (c.get("title") or "").strip() == title), None)
            code_url = ((card or {}).get("code_url") or "").strip()
            parsed = _parse_github_url(code_url) if code_url else None
            # ① 论文标注的代码链接最权威，优先拉取
            if parsed:
                owner, repo_name = parsed
                dest = f"reference_repos/{owner}__{repo_name}"
                mode = _fetch_reference_repo(f"https://github.com/{owner}/{repo_name}.git",
                                             owner, repo_name, dest, token)
                ok = not mode.startswith("CLONE_FAIL")
                entry.append({"name": f"{owner}/{repo_name}", "stars": 0,
                              "url": f"https://github.com/{owner}/{repo_name}",
                              "clone_url": f"https://github.com/{owner}/{repo_name}.git",
                              "description": "论文标注代码链接", "language": "",
                              "source": "paper_code_url",
                              "local_path": dest if ok else "", "cloned": ok,
                              "fetch_mode": mode if ok else ""})
                if ok:
                    lines.append(f"• 《{title[:36]}》→ 论文标注 {owner}/{repo_name} 已就绪（{mode}）→ {dest}")
                else:
                    lines.append(f"• 《{title[:36]}》→ 论文标注 {owner}/{repo_name} ⚠️ 拉取失败："
                                 f"{mode[11:120]}，改试 GitHub 搜索兜底…")
            # ② 无标注链接 / 链接拉取失败 → 标题搜索（stars 排序）兜底
            if not entry or not entry[0].get("cloned"):
                repos = _github_search_repos(title, max_repos_per_paper, token)
                for r in repos:
                    if not r.get("clone_url"):
                        continue
                    dest = f"reference_repos/{r['name'].replace('/', '__')}"
                    owner, repo_name = r["name"].split("/", 1)
                    mode = _fetch_reference_repo(r["clone_url"], owner, repo_name, dest, token)
                    ok = not mode.startswith("CLONE_FAIL")
                    entry.append({**r, "source": "github_search",
                                  "local_path": dest if ok else "", "cloned": ok,
                                  "fetch_mode": mode if ok else ""})
                    if ok:
                        lines.append(f"• 《{title[:36]}》→ 搜索 {r['name']} ⭐{r['stars']} "
                                     f"已就绪（{mode}）→ {dest}")
                    else:
                        lines.append(f"• 《{title[:36]}》→ 搜索 {r['name']} ⚠️ 拉取失败：{mode[11:120]}")
            if not entry or not any(e.get("cloned") for e in entry):
                lines.append(f"• 《{title[:36]}》→ ⚠️ 未找到开源实现（no_open_source，"
                             "将回退本地参考库并由 LLM 生成）")
            refs[title] = entry
        lines.append("")
        lines.append("👉 下一步：调用 code_survey 时优先读这些开源仓库；"
                     "无开源实现的论文回退本地参考库 project/（LLM 生成路线）。")
        return Command(update={
            "reference_repos": refs,
            "messages": [ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)],
        })

    return prepare_reference_repos


def make_code_survey(llm):
    @tool
    def code_survey(
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        innovation_ids: str | None = None,
        reference_dir: str = "project",
        max_files: int = _CODE_SURVEY_MAX_FILES,
    ) -> Command:
        """参考代码库调查：读取工作区内的参考实现，生成每个创新点的实现报告。

        适用：propose_innovations + novelty_check 之后、implement_innovation 之前。
        innovation_ids：逗号分隔的创新点 id（缺省=全部）；reference_dir：参考代码库目录
        （相对工作区根，默认 project/，含基线实现）。输出写入 state.code_survey_reports，
        implement_innovation 会注入对应报告。
        """
        innovs = state.get("innovations") or []
        if not innovs:
            return Command(update={"messages": [ToolMessage(
                content="innovations 为空：请先调用 propose_innovations 提出创新点。",
                tool_call_id=tool_call_id, status="error")]})
        targets = innovs
        if innovation_ids:
            ids = {s.strip() for s in innovation_ids.split(",") if s.strip()}
            targets = [d for d in innovs if d.get("id") in ids]
        if not targets:
            return Command(update={"messages": [ToolMessage(
                content=f"未找到 innovation_ids={innovation_ids} 对应的创新点。",
                tool_call_id=tool_call_id, status="error")]})

        # 参考目录解析：优先 prepare_reference_repos 克隆的开源仓库，回退本地参考库（LLM 生成路线）
        ref_repos = state.get("reference_repos") or {}
        ref_dirs, src_notes = [], []
        for title, entry in ref_repos.items():
            for r in entry:
                if r.get("cloned") and r.get("local_path"):
                    ref_dirs.append(r["local_path"])
                    src_notes.append(f"《{title[:30]}》开源仓库 {r['name']}")
        if not ref_dirs:
            ref_dirs = [reference_dir]
            src_notes = [f"本地参考库 {reference_dir}/（论文未开源或无克隆，回退 LLM 自行生成路线）"]

        code_bundle, seen_dirs = [], set()
        for rd in ref_dirs:
            if rd in seen_dirs:
                continue
            seen_dirs.add(rd)
            tree = ce.gen_code_tree_structure.invoke({"directory": rd})
            files = _pick_code_files(tree, rd, max_files)
            for f in files:
                content = ce.read_file.invoke({"file_path": f"{rd}/{f}"})
                if content.startswith("Error reading file"):
                    continue
                code_bundle.append(f"===== {rd}/{f} =====\n{content[:_CODE_SURVEY_CHARS]}")
        if not code_bundle:
            return Command(update={"messages": [ToolMessage(
                content=(f"参考代码库（{'、'.join(ref_dirs) or reference_dir}）未读到可用 .py 文件"
                         f"（tree 输出：\n{tree[:500]}）。请确认目录存在，"
                         "或改用 code_exec 的 list_files 查看工作区。"),
                tool_call_id=tool_call_id, status="error")]})

        brief = []
        for i, d in enumerate(targets, 1):
            brief.append(f"[{i}] id={d.get('id')} 名称={d.get('name','')}\n"
                         f"    方法={d.get('method','')[:250]}\n"
                         f"    公式={d.get('math_formula','')[:150]}")
        notes = state.get("survey_notes") or []
        notes_txt = ("\n【深度调研笔记】\n" + json.dumps(notes, ensure_ascii=False)[:3000]) if notes else ""
        sys = (
            "你是资深代码评审工程师。给定创新点与参考代码库内容，输出每个创新点的**实现报告**：\n"
            "1) 关键机制清单：把创新点的核心机制拆成 2-4 个可实现要点；\n"
            "2) 每个要点：参考代码中**可复用**的具体实现（类/函数名 + 如何复用 + 所在文件），"
            "以及**需新增/修改**的部分（含关键张量形状/公式对应代码）。\n"
            "3) 明确标注哪些机制参考代码**没有现成实现**（需从零写）。\n"
            "4) 只输出一个 JSON 数组，元素字段：\n"
            '   [{"innovation_id": "inn_01", "report": "完整实现报告（含可复用清单与需新增清单）"}]'
        )
        user = ("创新点：\n" + "\n".join(brief) + notes_txt +
                "\n\n参考代码来源：" + "；".join(src_notes) +
                "\n参考代码库内容：\n" + "\n".join(code_bundle))
        with tracing.span("llm", "code_survey", state.get("session_id", "")):
            resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=user)])
        data = _extract_json_block(resp.content if hasattr(resp, "content") else str(resp))
        if not isinstance(data, list) or not data:
            return Command(update={"messages": [ToolMessage(
                content="模型未返回有效实现报告 JSON，请重试。",
                tool_call_id=tool_call_id, status="error")]})
        reports = dict(state.get("code_survey_reports") or {})
        for r in data:
            if isinstance(r, dict) and r.get("innovation_id"):
                reports[r["innovation_id"]] = r.get("report", "")
        lines = [f"实现报告完成（{len(data)} 个创新点）："]
        for r in data:
            if isinstance(r, dict):
                rid = r.get("innovation_id", "?")
                rep_txt = (r.get("report") or "")[:120].replace("\n", " ")
                lines.append(f"• {rid}: {rep_txt}...")
        lines.append("")
        lines.append("👉 下一步：调用 implement_innovation 实现实验（报告会自动注入 prompt，"
                     "LLM 将优先复用参考代码）。")
        return Command(update={
            "code_survey_reports": reports,
            "messages": [ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)],
        })

    return code_survey


# ---------------------------------------------------------------------------
# 4. novelty_check —— 创新点新颖性查重（AI-Scientist Novelty Check 轻量版）
#     verdict=duplicate 的创新点标记 status='duplicate'，implement_innovation 拒绝为其跑实验。
# ---------------------------------------------------------------------------
_NOVELTY_SOURCES = ("search_openalex", "search_europepmc", "search_crossref", "search_arxiv")
_NOVELTY_HIT_MAX = 4            # 每个检索式每源取前 N 条
_NOVELTY_ABSTRACT_CHARS = 280   # 命中论文摘要注入 LLM 的截断长度


def _rule_queries(entry: dict, topic: str = "") -> list[str]:
    """规则兜底检索式（LLM 生成失败时用）：名称 + 领域词（从主题/上下文中取，不再写死时序）。"""
    name = (entry.get("name") or "").strip()
    method = (entry.get("method") or "").strip()
    base = name or method[:40]
    if not base:
        base = (topic or "research").strip()[:40] or "research"
    return [base, base]  # 领域限定由 LLM 检索式承担；规则兜底只保证非空


async def _build_search_queries(llm, targets: list[dict], session_id: str) -> dict:
    """LLM 为每个创新点生成 2 个英文检索式（含核心方法术语，用于查重）。"""
    brief = []
    for i, t in enumerate(targets, 1):
        brief.append(f"[{i}] id={t.get('id')} 名称={t.get('name','')}\n"
                     f"    方法={t.get('method','')[:200]}")
    sys = (
        "你是科研检索专家。为每个创新点生成 **2 个英文检索式**，用于在学术库（OpenAlex/arXiv/Crossref/"
        "EuropePMC）中查找**可能已提出过该创新点**的文献。检索式应包含核心方法术语（结构/训练/损失改动），"
        "不要过于宽泛（如只有领域最泛的上位词，示例：时序领域的 'time series'），也不要带引号/操作符。\n"
        '只输出 JSON 对象：{"<id>": ["query1", "query2"], ...}'
    )
    try:
        with tracing.span("llm", "novelty_queries", session_id):
            resp = llm.invoke([SystemMessage(content=sys), HumanMessage(
                content="创新点列表：\n" + "\n".join(brief))])
        data = _extract_json_block(resp.content if hasattr(resp, "content") else str(resp))
        if isinstance(data, dict):
            out = {}
            for k, v in data.items():
                if isinstance(v, list):
                    out[k] = [str(q)[:120] for q in v[:2] if str(q).strip()]
            if out:
                return out
    except Exception as e:
        print(f"--> [novelty] 检索式生成失败: {e}")
    return {}


async def _mcp_search_papers(queries: dict, per_query: int = _NOVELTY_HIT_MAX) -> dict:
    """按 {id: [query, ...]} 检索白名单 4 源，返回 {id: [命中论文 dict, ...]}（去重、截断）。"""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
    from ResearcherAgent.core.config import MCP_CONFIG

    tools: dict = {}
    try:
        client = MultiServerMCPClient({"paper_search_server": MCP_CONFIG["paper_search_server"]})
        async with client.session("paper_search_server") as session:
            raw = await load_mcp_tools(session)
            tools = {t.name: t for t in raw if t.name in _NOVELTY_SOURCES}
    except Exception as e:
        print(f"--> [novelty] MCP 连接失败: {e}")
        return {}
    if not tools:
        print("--> [novelty] 白名单检索工具为空（MCP 加载失败）")
        return {}

    from ResearcherAgent.tools.agent_tools import parse_papers
    sem = asyncio.Semaphore(3)

    async def _one(tool, query: str) -> list[dict]:
        async with sem:
            for kw in ({"query": query, "max_results": per_query},
                       {"query": query, "limit": per_query},
                       {"query": query}):
                try:
                    parsed = parse_papers(await tool.ainvoke(kw),
                                          source_hint=tool.name.replace("search_", ""))
                    return parsed or []
                except Exception:
                    continue
            return []

    tasks = [(iid, _one(tool, q)) for iid, qs in queries.items()
             for q in qs for tool in tools.values()]
    results = await asyncio.gather(*(t for _, t in tasks))
    hits: dict[str, list[dict]] = {}
    seen: dict[str, set] = {}
    for (iid, _), papers in zip(tasks, results):
        bucket = hits.setdefault(iid, [])
        keys = seen.setdefault(iid, set())
        for p in papers:
            title = (p.get("title") or "").strip()
            if not title:
                continue
            key = title.lower()[:80]
            if key in keys:
                continue
            keys.add(key)
            bucket.append({
                "title": title[:200],
                "year": p.get("year", ""),
                "abstract": str(p.get("abstract") or p.get("summary") or "")[:_NOVELTY_ABSTRACT_CHARS],
            })
    return hits


async def _judge_novelty(llm, targets: list[dict], hits: dict, session_id: str) -> list[dict]:
    """LLM 判定每个创新点：novel / partially_explored / duplicate。"""
    blocks = []
    for t in targets:
        iid = t.get("id", "")
        papers = hits.get(iid) or []
        ptext = "\n".join(
            f"- 《{p.get('title', '')}》 ({p.get('year', '')}) {p.get('abstract', '')}"
            for p in papers[:12]
        ) or "（检索未命中文献）"
        blocks.append(f"[{iid}] 创新点：{t.get('name', '')}\n"
                      f"方法：{t.get('method', '')[:250]}\n检索到的相关文献：\n{ptext[:2200]}")
    sys = (
        "你是科研新颖性审查专家（Novelty Checker，对应 AI-Scientist 的 harsh critic）。"
        "基于**检索到的真实文献**判断每个创新点是否已被提出。判定标准：\n"
        "- **novel**：命中文献没有一篇覆盖该创新点的核心方法或核心假设。\n"
        "- **partially_explored**：有文献做了相近工作（相邻方法/类似假设），但未完全覆盖——必须给出差异化建议。\n"
        "- **duplicate**：已有文献明确提出过该创新点（核心方法一致），基本没有新意。\n"
        "只输出 JSON 数组，不要输出其他文字，元素：\n"
        '[{"id": "...", "verdict": "novel|partially_explored|duplicate", "'
        '"evidence": "依据（引用命中论文标题+理由，100字内）", "confidence": "high|medium|low", "'
        '"differentiation": "仅 partially_explored 必填，如何与现有工作差异化"}]\n'
        "宁可错杀（过度标记 duplicate）也不可放过——创新点必须经得起审稿人查重。"
    )
    try:
        with tracing.span("llm", "novelty_judge", session_id):
            resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content="\n\n".join(blocks))])
        data = _extract_json_block(resp.content if hasattr(resp, "content") else str(resp))
        if isinstance(data, list) and data:
            return [d for d in data if isinstance(d, dict) and d.get("id")]
    except Exception as e:
        print(f"--> [novelty] 判定失败: {e}")
    return []


def make_novelty_check(llm):
    @tool
    async def novelty_check(
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        innovation_ids: list[str] | None = None,
        per_query: int = _NOVELTY_HIT_MAX,
    ) -> Command:
        """对已提出的创新点做新颖性查重（AI-Scientist Novelty Check）。

        用方法关键词检索学术库（OpenAlex/arXiv/Crossref/EuropePMC），把命中论文交给 LLM 判定
        每个创新点是否已被提出。verdict=duplicate 的创新点标记 status='duplicate'，
        implement_innovation 会拒绝为它生成实验代码（已被提出的创新点无需再跑实验证明）。
        返回：每个创新点的判定（novel / partially_explored / duplicate + 证据论文 + 置信度）。
        应在 propose_innovations 之后、implement_innovation 之前调用。
        innovation_ids：要查重的创新点 id（如 "inn_01"），或序号（如 "1" 表示第一个）；
        不传则检查全部。
        """
        innovations = state.get("innovations") or []
        if not innovations:
            return Command(update={"messages": [ToolMessage(
                content="innovations 为空：请先调用 propose_innovations 提出创新点，再做新颖性查重。",
                tool_call_id=tool_call_id, status="error")]})
        targets = innovations
        if innovation_ids:
            targets = []
            for _v in innovation_ids:
                _t = next((x for x in innovations if str(x.get("id")) == str(_v)), None)
                if _t is None and str(_v).strip().isdigit():   # 序号容错："1" → 第 1 个
                    _i = int(_v) - 1
                    if 0 <= _i < len(innovations):
                        _t = innovations[_i]
                if _t is not None and _t not in targets:
                    targets.append(_t)
        if not targets:
            return Command(update={"messages": [ToolMessage(
                content=f"指定的创新点 {innovation_ids} 不在 state.innovations 中（id 或序号均不匹配）。",
                tool_call_id=tool_call_id, status="error")]})

        session_id = state.get("session_id", "")
        queries = await _build_search_queries(llm, targets, session_id)
        if not queries:
            queries = {t.get("id"): _rule_queries(t) for t in targets}

        hits = await _mcp_search_papers(queries, per_query=per_query)
        if not hits:
            return Command(update={"messages": [ToolMessage(
                content="检索未返回任何文献（源可能临时不可用）。请稍后重试 novelty_check，"
                        "或调整检索词。", tool_call_id=tool_call_id, status="error")]})

        verdicts = await _judge_novelty(llm, targets, hits, session_id)
        if not verdicts:
            return Command(update={"messages": [ToolMessage(
                content="新颖性判定失败（LLM 未返回有效 JSON）。请重试。",
                tool_call_id=tool_call_id, status="error")]})

        by_idx = {t.get("id"): i for i, t in enumerate(innovations)}
        lines, dup_n = [], 0
        for v in verdicts:
            iid = v.get("id")
            idx = by_idx.get(iid)
            if idx is None:
                continue
            innovations[idx] = dict(innovations[idx])
            innovations[idx]["novelty"] = v
            verdict = str(v.get("verdict") or "").lower()
            name = innovations[idx].get("name", "")
            conf = v.get("confidence", "-")
            if verdict == "duplicate":
                innovations[idx]["status"] = "duplicate"
                dup_n += 1
                lines.append(f"✗ [{iid}] {name} → **duplicate（已被提出，跳过实验）** 置信度{conf}"
                             f"｜{str(v.get('evidence', ''))[:100]}")
            elif verdict == "partially_explored":
                diff = str(v.get("differentiation") or "")[:80]
                lines.append(f"△ [{iid}] {name} → partially_explored（需差异化：{diff}）")
            else:
                lines.append(f"✓ [{iid}] {name} → novel（置信度{conf}）"
                             f"｜{str(v.get('evidence', ''))[:100]}")
        msg = "创新点新颖性查重结果（基于真实学术检索）：\n" + "\n".join(lines) or "（无判定结果）"
        if dup_n:
            msg += f"\n\n已标记 {dup_n} 个已被提出的创新点（status=duplicate），"
            msg += "implement_innovation 将拒绝为其生成实验代码。"
        return Command(update={
            "innovations": innovations,
            "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
        })

    return novelty_check


# ---------------------------------------------------------------------------
# 2. run_baseline —— 跑 PatchTST 基线（确定性工具，复用 code_exec）
# ---------------------------------------------------------------------------
@tool
def run_baseline(
    dataset: str = "ETTH1",
    seq_len: int = 336,
    pred_len: int = 96,
    epochs: int = 50,
) -> str:
    """运行基线模型，在指定数据集上训练并返回评估指标。

    基线脚本（project/baseline_patchtst.py）首次调用时自动写入沙箱工作区；
    数据从项目 datasets/ 拷入沙箱。同一 (dataset, seq_len, pred_len, epochs)
    的结果会缓存（.baseline_cache.json），再次调用直接返回缓存。

    dataset：数据集目录名（默认演示：ETTH1/ETTH2/ETTm1...，对应 datasets/{dataset}/ 下数据文件；
    其他领域可指定任意数据集目录）；seq_len/pred_len：回看/预测窗口（时序演示默认 336/96，
    其他领域按领域惯例解释或忽略）；epochs：训练轮数（默认 50）。
    """
    data_dir = _ensure_dataset(dataset)
    cache_key = f"{dataset}|{seq_len}|{pred_len}|{epochs}"
    cache = _load_cache()
    if cache_key in cache:
        c = cache[cache_key]
        return (f"[基线缓存] PatchTST {dataset} seq={seq_len} pred={pred_len} epochs={epochs}：\n"
                f"mse={c['mse']} mae={c['mae']} device={c.get('device')} seconds={c.get('seconds')}")

    # 确保基线脚本存在
    probe = ce.read_file.invoke({"file_path": "project/baseline_patchtst.py"})
    if probe.startswith("Error reading file"):
        ce.create_file.invoke({"path": "project/baseline_patchtst.py", "content": BASELINE_PY})

    log = ce.run_python.invoke({
        "code_path": "project/baseline_patchtst.py",
        "cwd": ".",
        "env_vars": {"EXEC_ARGS": f"|{data_dir}|{seq_len}|{pred_len}|{epochs}"},
    })
    metric = _extract_metric(log)
    if not metric:
        tail = (log or "")[-_LOG_TAIL_CHARS:]
        return (f"基线运行完成但未解析到 METRIC_JSON（可能运行出错）：\n{tail}")
    cache[cache_key] = metric
    _write_cache(cache)
    tail = (log or "").strip().splitlines()
    tail = "\n".join(tail[-8:])
    return (f"[基线] PatchTST {dataset} seq={seq_len} pred={pred_len} epochs={epochs}：\n"
            f"test_mse={metric['mse']} mae={metric['mae']} device={metric['device']} "
            f"用时={metric.get('seconds')}s\n最近日志：\n{tail}")


# ---------------------------------------------------------------------------
# 3. implement_innovation —— LLM 生成创新实验代码并运行
# ---------------------------------------------------------------------------
def make_implement_innovation(llm):
    @tool
    def implement_innovation(
        innovation_id: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        epochs: int = 50,
        seq_len: int = 336,
        pred_len: int = 96,
        dataset: str = "ETTH1",
    ) -> Command:
        """为指定创新点生成实现代码（复用基线数据管线），在沙箱运行并返回实验指标。

        innovation_id：创新点 id（来自 propose_innovations 的列表）。
        会写 project/innov_{id}/main.py 并运行；模型可重复调用本工具迭代修改代码
        （运行失败时日志会返回，模型可先用 code_exec 的 read_file/修改 再重跑）。
        """
        innovs = state.get("innovations") or []
        innov = next((d for d in innovs if d.get("id") == innovation_id), None)
        if innov is None:
            known = "、".join(d.get("id", "?") for d in innovs) or "（无）"
            return Command(update={"messages": [ToolMessage(
                content=f"未找到创新点 {innovation_id}。当前创新点：{known}。"
                        "请先调用 propose_innovations。",
                tool_call_id=tool_call_id, status="error")]})

        # —— 新颖性闸门：已被 novelty_check 判定为 duplicate 的创新点不跑实验 ——
        if str(innov.get("status") or "").lower() == "duplicate":
            return Command(update={"messages": [ToolMessage(
                content=(f"[{innovation_id}] {innov.get('name', '')} 已被 novelty_check 判定为 "
                         "duplicate（核心方法已在文献中提出），跳过实验证明。\n"
                         "如需继续验证，请先用 novelty_check 重新检索判定，或人工确认差异点。"),
                tool_call_id=tool_call_id, status="error")]})

        data_dir = _ensure_dataset(dataset)
        base_path = "project/baseline_patchtst.py"
        probe = ce.read_file.invoke({"file_path": base_path})
        baseline_hint = ("基线脚本位于 project/baseline_patchtst.py，数据加载/训练骨架可整体复制。"
                         if not probe.startswith("Error reading file") else
                         "（基线脚本尚未生成，请按**该领域标准实验管线**自写数据加载："
                         "时序 = 60/20/20 划分 + train 统计量标准化 + 滑窗样本；其他领域按领域惯例变通。）")
        exp_ctx = cc_memory.inject_context()

        # 渐进式披露 L2：实验代码硬性要求来自 innovate/SKILL.md，不再写死在工具里
        from ResearcherAgent.core.skill_manager import load_skill_bodies
        _innov_skill = load_skill_bodies(["innovate"]).get("innovate", "")
        sys = (
            "你是深度学习科研工程师。根据给定的创新点，编写一个**完整、可直接运行**的 PyTorch "
            "实验脚本，验证该创新点相对**基线模型**（由 run_baseline 或参考代码库确定；"
            "默认演示示例：时序领域为 PatchTST，其他领域按上下文选常用基线）的精度提升。\n"
            + (_innov_skill if _innov_skill else
               "（技能规范缺失——遵循该领域标准实验约定：单一文件 / 数据划分与标准化按领域惯例 / "
               "同一脚本同时训练基线与创新版 / 标准化尺度上算指标 / 末尾输出 BASELINE_JSON 与 "
               "METRIC_JSON / EXEC_ARGS 读参数。）")
            + "\n按「技能执行规范 · innovate」的「实验代码硬性要求」执行（数据管线/评估口径/输出契约）。"
            "本次具体参数由 EXEC_ARGS 与下方配置给定（时序演示默认：读 CSV，滑窗 "
            "(seq_len={seq_len}, pred_len={pred_len})，batch=128，Adam lr=0.001，MSE loss，epochs={epochs}；"
            "其他领域按领域惯例调整数据加载与指标）；同一脚本同时训练基线版与创新版"
            "（同数据划分/超参/epochs 保证公平）；参数从环境变量 EXEC_ARGS 读取"
            "（|data_dir|seq_len|pred_len|epochs，容忍缺省 336/96/50）；末尾打印两行 "
            "BASELINE_JSON / METRIC_JSON。\n"
            "已知踩坑经验（避免重犯；与本创新点无关可忽略）：\n"
            f"{exp_ctx or '（暂无）'}"
        )
        cs_reports = state.get("code_survey_reports") or {}
        cs_hint = ""
        if cs_reports.get(innovation_id):
            cs_hint = (
                "\n\n【参考代码库实现报告（code_survey 产出，务必优先复用其中指出的可复用实现）】\n"
                f"{cs_reports[innovation_id][:_CARD_MAX_CHARS * 2]}"
            )
        user = (
            f"创新点（id={innovation_id}）：\n{json.dumps(innov, ensure_ascii=False, indent=1)}\n\n"
            f"数据集：{dataset}（数据目录={data_dir}，具体文件由脚本按 EXEC_ARGS/领域约定读取；"
            f"时序演示默认 datasets/{dataset} 下的 CSV）\n"
            f"配置：seq_len={seq_len}, pred_len={pred_len}, epochs={epochs}\n"
            f"{baseline_hint}{cs_hint}\n"
            f"请直接输出完整 Python 代码。"
        )
        with tracing.span("llm", "implement_innovation", state.get("session_id", ""), id=innovation_id):
            resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=user)])
        code = _extract_code_block(resp.content if hasattr(resp, "content") else str(resp))
        if not code:
            return Command(update={"messages": [ToolMessage(
                content="模型未输出有效代码块，请重试。",
                tool_call_id=tool_call_id, status="error")]})

        dir_name = f"project/innov_{innovation_id}"
        ce.create_file.invoke({"path": f"{dir_name}/main.py", "content": code})

        log = ce.run_python.invoke({
            "code_path": f"{dir_name}/main.py",
            "cwd": ".",
            "env_vars": {"EXEC_ARGS": f"|{data_dir}|{seq_len}|{pred_len}|{epochs}"},
        })
        metric = _extract_metric(log)
        baseline_m = re.search(r"BASELINE_JSON\s*=\s*(\{.*?\})\s*$", log or "", re.M)
        baseline_metric = None
        if baseline_m:
            try:
                baseline_metric = json.loads(baseline_m.group(1))
            except Exception:
                baseline_metric = None

        tail = (log or "")[-_LOG_TAIL_CHARS:]
        rec = {"innovation": innov, "metric": metric, "baseline_in_script": baseline_metric,
               "log_tail": tail, "dataset": dataset, "seq_len": seq_len,
               "pred_len": pred_len, "epochs": epochs}
        experiments = dict(state.get("experiments") or {})
        experiments[innovation_id] = rec
        if metric:
            msg = (f"创新点 [{innovation_id}] {innov.get('name', '')} 实验完成：\n"
                   f"创新版: mse={metric.get('mse')} mae={metric.get('mae')} "
                   f"device={metric.get('device')} epochs={metric.get('epochs')}\n")
            if baseline_metric:
                msg += (f"脚本内基线: mse={baseline_metric.get('mse')} "
                        f"mae={baseline_metric.get('mae')}\n")
            msg += f"日志尾部：\n{tail}"
        else:
            msg = (f"创新点 [{innovation_id}] 代码已写入并尝试运行，但未解析到 METRIC_JSON"
                   f"（大概率运行报错）。日志尾部：\n{tail}\n"
                   f"→ 直接调用 fix_and_rerun(innovation_id=\"{innovation_id}\") 自动修复重跑"
                   f"（工具会把报错回灌给 LLM 修改代码，最多 3 轮；成功后自动沉淀经验）；"
                   f"或先用 code_exec 的 read_file 查看 {dir_name}/main.py 定位错误，"
                   f"修改后重新调用本工具。")
        return Command(update={"experiments": experiments,
                               "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]})

    return implement_innovation


def _extract_code_block(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# 4. judge_innovation —— 对比基线/实验，判断可行性
# ---------------------------------------------------------------------------
def make_judge_innovation(llm):
    @tool
    def judge_innovation(
        innovation_id: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
    ) -> Command:
        """对比创新点实验指标与基线（run_baseline 缓存，默认演示：PatchTST）指标，
        判断该创新点是否可行。

        innovation_id：创新点 id（须先 implement_innovation 跑出实验指标）。
        基线指标从 run_baseline 的缓存读取；没有缓存则提示先跑基线。
        结论存入 state.experiments[innovation_id].verdict。
        """
        experiments = state.get("experiments") or {}
        rec = experiments.get(innovation_id)
        if not rec or not rec.get("metric"):
            return Command(update={"messages": [ToolMessage(
                content=f"还没有创新点 [{innovation_id}] 的实验结果，请先调用 implement_innovation。",
                tool_call_id=tool_call_id, status="error")]})

        innov = rec.get("innovation") or {}
        exp_m = rec["metric"]
        dataset, seq_len, pred_len, epochs = (rec.get("dataset", "ETTH1"), rec.get("seq_len", 336),
                                              rec.get("pred_len", 96), rec.get("epochs", 50))
        cache_key = f"{dataset}|{seq_len}|{pred_len}|{epochs}"
        base_m = _load_cache().get(cache_key)
        if not base_m:
            return Command(update={"messages": [ToolMessage(
                content=f"基线指标缓存缺失（{cache_key}）。请先调用 run_baseline "
                        f"dataset={dataset} seq_len={seq_len} pred_len={pred_len} epochs={epochs}。",
                tool_call_id=tool_call_id, status="error")]})

        # 渐进式披露 L2：判定标准来自 innovate/SKILL.md，不再写死在工具里
        from ResearcherAgent.core.skill_manager import load_skill_bodies
        _innov_skill = load_skill_bodies(["innovate"]).get("innovate", "")
        sys = (
            "你是严格的科研实验评审。对比创新点实验与基线（run_baseline 产出的基线指标，"
            "默认演示为 PatchTST，其他领域为所选常用基线）的指标，判断该创新点"
            "是否构成**可行的改进**。\n\n"
            + (_innov_skill if _innov_skill else
               "（技能规范缺失——判断标准：创新版主指标（如 MSE/MAE 或该领域标准指标）相对基线"
               "下降 ≥5% 且辅助指标同步改善 → 可行；0~5% → 存疑；不降反升 → 不可行。）")
            + "\n按「技能执行规范 · innovate」的「判定标准」执行（相对提升 (base-exp)/base ≥5% 且主指标 "
            "同步改善 → 可行；0~5% 存疑；不降反升 → 不可行），并核查：训练是否稳定收敛（日志有无 NaN/"
            "发散）；实现是否忠实于创新点描述；对比是否公平（同数据/同 epochs/同超参）。\n"
            "只输出一个 JSON 对象，字段：\n"
            '   {"id": "...", "feasible": true|false, "gain": "相对基线MSE提升百分比(带正负号)", '
            '"basis": "判断依据（100字内）", "risks": "风险/局限", "suggestion": "下一步建议"}\n'
            "不要输出 JSON 之外的文字。"
        )
        user = (
            f"创新点：{json.dumps(innov, ensure_ascii=False, indent=1)}\n"
            f"基线指标：{json.dumps(base_m, ensure_ascii=False)}\n"
            f"实验指标：{json.dumps(exp_m, ensure_ascii=False)}\n"
            f"实验日志尾部：\n{(rec.get('log_tail') or '')[-1500:]}"
        )
        with tracing.span("llm", "judge_innovation", state.get("session_id", ""), id=innovation_id):
            resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=user)])
        verdict = _extract_json_block(resp.content if hasattr(resp, "content") else str(resp))
        if not isinstance(verdict, dict) or "feasible" not in verdict:
            return Command(update={"messages": [ToolMessage(
                content="评委未返回有效判定 JSON，请重试。",
                tool_call_id=tool_call_id, status="error")]})

        rec["verdict"] = verdict
        experiments = dict(state.get("experiments") or {})
        experiments[innovation_id] = rec
        label = {True: "✅ 可行", False: "❌ 不可行"}.get(verdict.get("feasible"), "🤔 存疑")
        msg = (f"创新点 [{innovation_id}] {innov.get('name', '')} 判定：{label}\n"
               f"相对基线 MSE 变化：{verdict.get('gain', '?')}\n"
               f"依据：{verdict.get('basis', '')}\n"
               f"风险：{verdict.get('risks', '')}\n"
               f"建议：{verdict.get('suggestion', '')}")
        return Command(update={"experiments": experiments,
                               "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]})

    return judge_innovation


# ---------------------------------------------------------------------------
# 5. fix_and_rerun —— 失败日志回灌 LLM，自动修复重跑（方案 2）
#    工具内最多迭代 max_attempts 轮；成功后把「错误+根因+修复」写入 EXPERIENCE.md。
# ---------------------------------------------------------------------------
def make_fix_and_rerun(llm):
    @tool
    def fix_and_rerun(
        innovation_id: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        max_attempts: int = 3,
    ) -> Command:
        """修复并重跑失败的创新点实验代码。

        适用：implement_innovation 运行失败（未解析到 METRIC_JSON）后调用。
        工具内部把最近一次错误日志回灌给 LLM，自动生成修复版代码并重跑，
        最多迭代 max_attempts 轮（默认 3）；任一轮成功即返回指标，
        并把「错误摘要+根因+修复要点」写入经验库 project/EXPERIENCE.md，
        供后续 implement_innovation / fix_and_rerun 注入参考，避免重犯同类坑。

        innovation_id：创新点 id（须已 implement_innovation 生成过脚本）。
        max_attempts：修复-重跑最大轮数（1-5）。
        """
        experiments = dict(state.get("experiments") or {})
        rec = experiments.get(innovation_id)
        if not rec:
            known = "、".join(experiments.keys()) or "（无）"
            return Command(update={"messages": [ToolMessage(
                content=f"未找到创新点 {innovation_id} 的实验记录（当前有：{known}）。"
                        "请先调用 implement_innovation 生成实验脚本。",
                tool_call_id=tool_call_id, status="error")]})

        innov = rec.get("innovation") or {}
        dir_name = f"project/innov_{innovation_id}"
        code_path = f"{dir_name}/main.py"
        probe = ce.read_file.invoke({"file_path": code_path})
        if probe.startswith("Error reading file"):
            return Command(update={"messages": [ToolMessage(
                content=f"未找到实验脚本 {code_path}。请先调用 implement_innovation "
                        f"生成 {innovation_id} 的代码。",
                tool_call_id=tool_call_id, status="error")]})

        dataset = rec.get("dataset", "dataset")
        seq_len = int(rec.get("seq_len", 336))
        pred_len = int(rec.get("pred_len", 96))
        epochs = int(rec.get("epochs", 50))
        data_dir = _ensure_dataset(dataset)
        exp_ctx = cc_memory.inject_context()
        last_log = rec.get("log_tail") or "（无日志）"
        attempts_done = 0

        for attempt in range(1, max(1, min(int(max_attempts), 5)) + 1):
            attempts_done = attempt
            # 渐进式披露 L2：实验硬性要求来自 innovate/SKILL.md，工具只保留解析契约
            from ResearcherAgent.core.skill_manager import load_skill_bodies
            _innov_skill = load_skill_bodies(["innovate"]).get("innovate", "")
            sys = (
                "你是科研代码调试工程师。给定创新点实验脚本及其最近一次运行的错误日志，"
                "修复代码并输出**完整、可直接运行的单一 Python 文件**（不是补丁，是全文）。\n"
                + (_innov_skill if _innov_skill else
                   "（技能规范缺失——遵循该领域标准实验约定：单一文件 / 数据划分与标准化按领域惯例 / "
                   "标准化尺度上算主指标。）")
                + "\n按「技能执行规范 · innovate」的「实验代码硬性要求」执行，并保持工具输出协议：\n"
                "1) 末尾打印两行\n"
                '   BASELINE_JSON={{"mse":..., "mae":...}}\n'
                '   METRIC_JSON={{"mse":..., "mae":..., "device":..., "epochs":..., "name": "创新点名称"}}\n'
                "2) 数据管线与基线一致：读数据集文件（时序演示默认：CSV，60/20/20 划分，train 统计量"
                "标准化，滑窗样本，batch=128，Adam lr=0.001，MSE loss；其他领域按领域惯例调整）；"
                "参数从环境变量 EXEC_ARGS 读取"
                "（|data_dir|seq_len|pred_len|epochs）；评估在标准化尺度上算主指标（禁止先反归一化）。\n"
                "3) 只使用 torch / numpy / pandas；不要引入未安装的库。\n"
                "4) 已知踩坑经验（避免重犯）：\n"
                f"{exp_ctx or '（暂无）'}\n"
                "5) 如果本次修复成功（你确信代码能跑通），在回复末尾单独附一行：\n"
                '   EXPERIENCE={"error": "错误摘要(40字内)", "cause": "根因(60字内)", "fix": "修复要点(60字内)"}\n'
                "   工具会在运行成功后把它写入经验库。"
            )
            user = (
                f"创新点（id={innovation_id}）：\n{json.dumps(innov, ensure_ascii=False, indent=1)}\n"
                f"脚本路径：{code_path}\n"
                f"数据集：{dataset}（数据目录={data_dir}，具体文件由脚本按 EXEC_ARGS/领域约定读取）\n"
                f"配置：seq_len={seq_len}, pred_len={pred_len}, epochs={epochs}\n"
                f"当前是第 {attempt} 轮修复（上限 {max(1, min(int(max_attempts), 5))} 轮）。\n\n"
                f"最近一次运行错误日志（尾部）：\n{str(last_log)[-3000:]}"
            )
            with tracing.span("llm", "fix_and_rerun", state.get("session_id", ""),
                              id=innovation_id, attempt=attempt):
                resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=user)])
            text = resp.content if hasattr(resp, "content") else str(resp)
            code = _extract_code_block(text)
            if not code:
                last_log = str(last_log) + (f"\n[fix_and_rerun] 第{attempt}轮：模型未输出代码块，"
                                            "跳过本轮。")
                continue

            ce.write_file.invoke({"path": code_path, "content": code})
            log = ce.run_python.invoke({
                "code_path": code_path,
                "cwd": ".",
                "env_vars": {"EXEC_ARGS": f"|{data_dir}|{seq_len}|{pred_len}|{epochs}"},
            })
            metric = _extract_metric(log)
            baseline_m = re.search(r"BASELINE_JSON\s*=\s*(\{.*?\})\s*$", log or "", re.M)
            baseline_metric = None
            if baseline_m:
                try:
                    baseline_metric = json.loads(baseline_m.group(1))
                except Exception:
                    baseline_metric = None

            if metric:
                # 修复成功 → 沉淀经验
                m_exp = re.search(r"EXPERIENCE\s*=\s*(\{.*?\})\s*$", text, re.S)
                if m_exp:
                    try:
                        entry = json.loads(m_exp.group(1))
                        if isinstance(entry, dict) and entry.get("error"):
                            _append_experience({
                                **entry,
                                "innovation_id": innovation_id,
                                "time": datetime.now().isoformat(timespec="seconds"),
                            })
                            # CC 式记忆：双写 MEMORY.md（pitfall 区，同主题合并更新）
                            cc_memory.remember(
                                "pitfall",
                                str(entry.get("error", ""))[:20] or f"inn_{innovation_id}",
                                f"根因：{entry.get('cause', '')}；修复：{entry.get('fix', '')}",
                            )
                    except Exception:
                        pass
                rec.update({"metric": metric, "baseline_in_script": baseline_metric,
                            "log_tail": (log or "")[-_LOG_TAIL_CHARS:],
                            "fix_attempts": attempts_done})
                experiments[innovation_id] = rec
                msg = (f"修复成功（第 {attempts_done} 轮重跑通过）：{code_path}\n"
                       f"创新版: mse={metric.get('mse')} mae={metric.get('mae')} "
                       f"device={metric.get('device')} epochs={metric.get('epochs')}\n")
                if baseline_metric:
                    msg += (f"脚本内基线: mse={baseline_metric.get('mse')} "
                            f"mae={baseline_metric.get('mae')}\n")
                msg += f"日志尾部：\n{(log or '')[-_LOG_TAIL_CHARS:]}"
                return Command(update={"experiments": experiments,
                                       "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]})

            last_log = (log or "")[-_LOG_TAIL_CHARS:]

        rec["log_tail"] = last_log
        experiments[innovation_id] = rec
        tail = str(last_log)[-1500:]
        return Command(update={"experiments": experiments, "messages": [ToolMessage(
            content=(f"修复未成功（已尝试 {attempts_done} 轮）。最后日志尾部：\n{tail}\n"
                     f"→ 可人工查看 {code_path} 定位问题；或调整创新点描述后重新 "
                     "implement_innovation。"),
            tool_call_id=tool_call_id, status="error")]})

    return fix_and_rerun

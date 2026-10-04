# -*- coding: utf-8 -*-
"""
innovation_tools.py — 「文献综述 → 创新点验证」流水线工具集（skill: innovate）

流水线（模型在 agent 循环中自由编排，工具只提供能力）：
  1. search_papers（复用 MCP 检索）          → paper_pool
  2. read_papers / read_sections（复用精读制卡）→ paper_cards
  3. propose_innovations                     → LLM 基于卡片提出 3-5 个创新点（state.innovations）
  4. run_baseline                            → 确定性工具：跑 PatchTST 基线，得基线指标
  5. implement_innovation                    → LLM 为指定创新点生成实验代码 → 沙箱运行 → 实验指标
  6. judge_innovation                        → LLM 对比基线 vs 实验，判断可行性

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

from agent.core import tracing
from agent.sandbox import code_executor as ce
from agent.sandbox.sandbox_exec import PROJECT_ROOT, get_env
from agent.memory import cc_memory

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
INNOVATE_TOOLS = ("propose_innovations", "run_baseline", "implement_innovation",
                  "fix_and_rerun", "judge_innovation", "novelty_check")

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
    """确保数据集已拷贝进沙箱工作区，返回工作区内的数据目录（相对 workspace 根）。"""
    probe = ce.execute_command.invoke({"command": f'dir /b "datasets\\{dataset}" 2>nul'})
    if "ETTh1.csv" not in probe:
        src_abs = os.path.join(PROJECT_ROOT, "datasets", dataset, "ETTh1.csv")
        if os.path.exists(src_abs):
            ce.execute_command.invoke({"command": (
                f'if not exist "datasets\\{dataset}" mkdir "datasets\\{dataset}" && '
                f'copy /Y "{src_abs}" "datasets\\{dataset}\\"')})
        else:
            # 尝试工作区内已有副本
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
# ---------------------------------------------------------------------------
BASELINE_PY = r'''# -*- coding: utf-8 -*-
"""PatchTST 基线（ICLR 2023, "A Time Series is Worth 64 Words"）—— 自包含。
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
CSV = os.path.join(DATA_DIR, "ETTh1.csv")
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
# 1. propose_innovations —— 基于精读卡片提出 3-5 个创新点
# ---------------------------------------------------------------------------
def make_propose_innovations(llm):
    @tool
    def propose_innovations(
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        topic: str | None = None,
        count: int = 5,
    ) -> Command:
        """基于已精读的论文卡片提出 3-5 个潜在创新点，存入 state.innovations。

        适用：read_papers/read_sections 精读完成、paper_cards 已有卡片后调用。
        topic：研究主题（默认用用户请求的主题）；count：创新点数量（3-5）。
        每个创新点含：名称、假设、具体方法思路、基于哪些论文、预期提升、实现难度。
        """
        cards = state.get("paper_cards") or []
        if not cards:
            return Command(update={"messages": [ToolMessage(
                content="paper_cards 为空：还没有可依据的论文卡片。请先调用 read_papers（精读 5-10 篇）"
                        "再提出创新点。",
                tool_call_id=tool_call_id, status="error")]})

        theme = topic or state.get("topic") or state.get("user_request") or "该领域"
        ctx = _cards_context(cards)
        sys = (
            "你是科研方法论创新专家。基于给定的论文精读卡片（原文段落证据），为研究主题提出潜在创新点。"
            "要求：\n"
            "1) 创新点必须**具体可实现**：基于 PyTorch、可直接在现有数据管线（长序列时间序列预测，"
            "多变量，60/20/20 划分）上实现并对比基线（PatchTST）。\n"
            "2) 每个创新点要有明确假设（为什么能提升）、具体方法思路（模型结构/训练方式的改动），"
            "并注明基于哪些论文卡片。\n"
            "3) 只输出一个 JSON 数组，不要输出其他文字。数组元素字段：\n"
            '   [{"id": "inn_01", "name": "创新点名称", "hypothesis": "假设：为什么能提升", '
            '"method": "具体实现思路（结构/模块/训练改动，300字内）", "based_on": "基于哪些论文", '
            '"expected_gain": "预期在哪个指标/哪类序列上提升", "difficulty": "低/中/高"}]\n'
            "4) 数量 3-5 个；至少 1 个低难度（数据/训练层面的小改动），至少 1 个中高难度（结构创新）。"
        )
        user = f"研究主题：{theme}\n\n论文精读卡片：\n{ctx}"
        with tracing.span("llm", "propose_innovations", state.get("session_id", "")):
            resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=user)])
        data = _extract_json_block(resp.content if hasattr(resp, "content") else str(resp))
        if not isinstance(data, list) or not data:
            return Command(update={"messages": [ToolMessage(
                content="模型未返回有效创新点 JSON，请重试或调整精读范围。",
                tool_call_id=tool_call_id, status="error")]})
        data = data[: count] if count >= 3 else data
        lines = [f"已提出 {len(data)} 个潜在创新点："]
        for i, d in enumerate(data, 1):
            lines.append(f"{i}. [{d.get('difficulty', '中')}] {d.get('name', '未命名')}"
                         f"（{d.get('based_on', '')[:40]}）")
            lines.append(f"   假设：{d.get('hypothesis', '')[:100]}")
        lines.append("")
        lines.append("👉 下一步：调用 novelty_check 对这些创新点做新颖性查重（检索已有文献，"
                     "判定是否已被提出），再为 novel / partially_explored 的创新点跑实验。")
        return Command(update={
            "innovations": data,
            "messages": [ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)],
        })

    return propose_innovations


# ---------------------------------------------------------------------------
# 2. novelty_check —— 创新点新颖性查重（AI-Scientist Novelty Check 轻量版）
#     verdict=duplicate 的创新点标记 status='duplicate'，implement_innovation 拒绝为其跑实验。
# ---------------------------------------------------------------------------
_NOVELTY_SOURCES = ("search_openalex", "search_europepmc", "search_crossref", "search_arxiv")
_NOVELTY_HIT_MAX = 4            # 每个检索式每源取前 N 条
_NOVELTY_ABSTRACT_CHARS = 280   # 命中论文摘要注入 LLM 的截断长度


def _rule_queries(entry: dict) -> list[str]:
    """规则兜底检索式（LLM 生成失败时用）：名称 + 领域词。"""
    name = (entry.get("name") or "").strip()
    method = (entry.get("method") or "").strip()
    base = name or method[:40] or "time series forecasting"
    return [base + " time series forecasting", base]


async def _build_search_queries(llm, targets: list[dict], session_id: str) -> dict:
    """LLM 为每个创新点生成 2 个英文检索式（含核心方法术语，用于查重）。"""
    brief = []
    for i, t in enumerate(targets, 1):
        brief.append(f"[{i}] id={t.get('id')} 名称={t.get('name','')}\n"
                     f"    方法={t.get('method','')[:200]}")
    sys = (
        "你是科研检索专家。为每个创新点生成 **2 个英文检索式**，用于在学术库（OpenAlex/arXiv/Crossref/"
        "EuropePMC）中查找**可能已提出过该创新点**的文献。检索式应包含核心方法术语（结构/训练/损失改动），"
        "不要过于宽泛（如只有 'time series'），也不要带引号/操作符。\n"
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
    from agent.core.config import MCP_CONFIG

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

    from agent.tools.agent_tools import parse_papers
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
    """运行基线模型（PatchTST）在指定数据集上训练并返回评估指标。

    基线脚本（project/baseline_patchtst.py）首次调用时自动写入沙箱工作区；
    数据从项目 datasets/ 拷入沙箱。同一 (dataset, seq_len, pred_len, epochs)
    的结果会缓存（.baseline_cache.json），再次调用直接返回缓存。

    dataset：数据集目录名（ETTH1/ETTH2/ETTm1...，对应 datasets/{dataset}/ETTh*.csv）；
    seq_len：回看窗口（默认 336）；pred_len：预测长度（默认 96）；epochs：训练轮数（默认 50）。
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
                         "（基线脚本尚未生成，请按标准 ETT 数据管线自写数据加载："
                         "60/20/20 划分 + train 统计量标准化 + 滑窗样本。）")
        exp_ctx = cc_memory.inject_context()

        sys = (
            "你是深度学习科研工程师。根据给定的创新点，编写一个**完整、可直接运行**的 PyTorch "
            "实验脚本，验证该创新点相对基线模型（PatchTST）的精度提升。\n"
            "硬性要求：\n"
            "1) 输出**单一 Python 文件**，不要输出多个文件或解释文字（可附在注释里）。\n"
            "2) 数据管线必须与基线一致：读 {CSV}，60/20/20 划分（train/val/test），"
            "用 train 统计量标准化，滑窗样本 (seq_len={seq_len}, pred_len={pred_len})，"
            "batch=128，Adam lr=0.001，MSE loss。\n"
            "3) 必须实现基线 PatchTST 作为对照骨架（从基线脚本复制模型定义，可整体保留），"
            "然后在其上实现创新点改动。**同一份脚本里同时训练基线版与创新版**，"
            "确保对比公平（相同数据划分/超参/epochs）。\n"
            "4) 每轮打印日志；最后打印两行：\n"
            '   BASELINE_JSON={{"mse":..., "mae":...}}\n'
            '   METRIC_JSON={{"mse":..., "mae":..., "device":..., "epochs":..., "name": "创新点名称"}}\n'
            "5) 参数从环境变量 EXEC_ARGS 读取（格式 |data_dir|seq_len|pred_len|epochs，"
            "以 | 分隔），脚本需容忍缺省（缺省 seq_len=336, pred_len=96, epochs=50）。\n"
            "6) 只使用 torch / numpy / pandas，不依赖未安装的库。\n"
            "7) 已知踩坑经验（避免重犯；与本创新点无关可忽略）：\n"
            f"{exp_ctx or '（暂无）'}"
        )
        user = (
            f"创新点（id={innovation_id}）：\n{json.dumps(innov, ensure_ascii=False, indent=1)}\n\n"
            f"数据集：{dataset}（CSV={data_dir}/ETTh1.csv）\n"
            f"配置：seq_len={seq_len}, pred_len={pred_len}, epochs={epochs}\n"
            f"{baseline_hint}\n"
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
        """对比创新点实验指标与基线（PatchTST）指标，判断该创新点是否可行。

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

        sys = (
            "你是严格的科研实验评审。对比创新点实验与基线（PatchTST）的指标，判断该创新点"
            "是否构成**可行的改进**。判断标准：\n"
            "1) 相对提升：创新版 MSE 相比基线是否下降、下降多少（(base-exp)/base）；"
            "下降 5% 以上且 MAE 同步改善 → 可行；0~5% → 存疑；不降反升 → 不可行。\n"
            "2) 训练是否稳定收敛（日志有无 NaN/发散）。\n"
            "3) 实现是否忠实于创新点描述、对比是否公平（同数据/同 epochs/同超参）。\n"
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

        dataset = rec.get("dataset", "ETTH1")
        seq_len = int(rec.get("seq_len", 336))
        pred_len = int(rec.get("pred_len", 96))
        epochs = int(rec.get("epochs", 50))
        data_dir = _ensure_dataset(dataset)
        exp_ctx = cc_memory.inject_context()
        last_log = rec.get("log_tail") or "（无日志）"
        attempts_done = 0

        for attempt in range(1, max(1, min(int(max_attempts), 5)) + 1):
            attempts_done = attempt
            sys = (
                "你是科研代码调试工程师。给定创新点实验脚本及其最近一次运行的错误日志，"
                "修复代码并输出**完整、可直接运行的单一 Python 文件**（不是补丁，是全文）。\n"
                "硬性要求：\n"
                "1) 保持输出协议：脚本末尾打印两行\n"
                '   BASELINE_JSON={{"mse":..., "mae":...}}\n'
                '   METRIC_JSON={{"mse":..., "mae":..., "device":..., "epochs":..., "name": "创新点名称"}}\n'
                "2) 数据管线必须与基线一致：读 CSV，60/20/20 划分，train 统计量标准化，"
                "滑窗样本，batch=128，Adam lr=0.001，MSE loss；参数从环境变量 EXEC_ARGS 读取"
                "（格式 |data_dir|seq_len|pred_len|epochs）。\n"
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
                f"数据集：{dataset}（CSV={data_dir}/ETTh1.csv）\n"
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

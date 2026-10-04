# -*- coding: utf-8 -*-
"""
nli_evidence.py —— 确定性 NLI 引用支持度判定（不靠 LLM 主观评分，弃 RAGAS 口径）

用多语言蕴含模型（mDeBERTa-v3-base-mnli-xnli，支持中文）判断：
    hypothesis（综述论断句）是否被 premise（被引论文证据文本）蕴含/矛盾/中立。

模型是固定权重、输入输出确定、可复现 → 属于确定性指标（确定性 ≠ 零错误，但
同一输入永远同一输出，不会因"评审模型心情"波动，也不受自评偏置影响）。

调用约定：
    nli_verdict(premise, hypothesis) -> {"label": "entailment|contradiction|neutral", "score": float}
    nli_support_rate(pairs) -> {"support": n, "contradict": n, "neutral": n,
                                 "support_rate": support/total, ...}
"""
import os
import threading

_model = None
_tokenizer = None
_lock = threading.Lock()
_MODEL_NAME = os.environ.get(
    "NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")


def _ensure_model():
    """惰性加载（首次调用下载/载入，之后复用）。线程安全。"""
    global _model, _tokenizer
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        # 只走本地缓存：模型已预先下载到 HF 缓存（hf-mirror），
        # 加载时禁止联网（Windows 代理失效会 ConnectTimeout 卡死整个判定）。
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, local_files_only=True)
        _model = AutoModelForSequenceClassification.from_pretrained(
            _MODEL_NAME, local_files_only=True)
        _model.eval()


def nli_verdict(premise: str, hypothesis: str) -> dict:
    """单条判定：hypothesis 是否被 premise 支持。

    返回 {"label": "entailment"|"contradiction"|"neutral", "score": <softmax 概率>}。
    空输入/异常 → {"label": "neutral", "score": 0.0, "error": "..."}（保守：不算支持也不算矛盾）。
    """
    if not premise or not hypothesis:
        return {"label": "neutral", "score": 0.0, "error": "empty-input"}
    try:
        _ensure_model()
        import torch
        with torch.inference_mode():
            inputs = _tokenizer(premise[:1500], hypothesis[:500],
                                return_tensors="pt", truncation=True)
            logits = _model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        # XNLI 标签顺序：0=contradiction, 1=neutral, 2=entailment
        p = probs.tolist()
        label = ["contradiction", "neutral", "entailment"][int(p.index(max(p)))]
        return {"label": label, "score": round(max(p), 4)}
    except Exception as e:
        return {"label": "neutral", "score": 0.0, "error": f"{type(e).__name__}: {str(e)[:120]}"}


def nli_support_rate(pairs: list[tuple]) -> dict:
    """批量判定：pairs = [(premise, hypothesis), ...]

    返回确定性指标：
      support_rate   = entailment 数 / 总数（有判定的）
      contradict_rate= contradiction 数 / 总数
      neutral_rate   = neutral 数 / 总数
    与 RAGAS/LLM 主观判定的差别：同一对句子在任何时间、任何模型参数下输出一致。
    """
    if not pairs:
        return {"total": 0, "support": 0, "contradict": 0, "neutral": 0,
                "support_rate": 0.0, "contradict_rate": 0.0, "neutral_rate": 1.0,
                "error": "no-pairs"}
    stats = {"support": 0, "contradict": 0, "neutral": 0, "error": 0}
    for prem, hypo in pairs:
        v = nli_verdict(prem, hypo)
        if v.get("error"):
            stats["error"] += 1
            continue
        if v["label"] == "entailment":
            stats["support"] += 1
        elif v["label"] == "contradiction":
            stats["contradict"] += 1
        else:
            stats["neutral"] += 1
    total = sum(stats.values())
    judged = stats["support"] + stats["contradict"] + stats["neutral"]
    return {
        "total": total,
        **stats,
        "support_rate": round(stats["support"] / judged, 4) if judged else 0.0,
        "contradict_rate": round(stats["contradict"] / judged, 4) if judged else 0.0,
        "neutral_rate": round(stats["neutral"] / judged, 4) if judged else 0.0,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # 自测：中文蕴含/矛盾/中立样例
    cases = [
        ("Transformer 使用自注意力机制处理序列", "Transformer 使用自注意力机制"),
        ("本文报告 BLEU 提升 2.1 个点", "本文报告 BLEU 提升 2.1 个点"),
        ("实验表明方法 A 优于方法 B", "实验表明方法 B 优于方法 A"),
        ("本文只评估了英文数据集", "本文评估了中英文数据集"),
    ]
    for p, h in cases:
        print(nli_verdict(p, h))
    print(nli_support_rate(cases))

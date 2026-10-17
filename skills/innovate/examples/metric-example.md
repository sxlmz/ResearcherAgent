# innovate · 输出与经验条目示例（L3 examples）

## BASELINE_JSON / METRIC_JSON（脚本末尾固定打印两行）

```bash
BASELINE_JSON={"model": "PatchTST", "test_mse": 0.4206, "test_mae": 0.4123, "epochs": 2, "seq_len": 336, "pred_len": 96}
METRIC_JSON={"model": "innovation_idea_1", "test_mse": 0.4654, "test_mae": 0.4301, "epochs": 2, "seq_len": 336, "pred_len": 96}
```

- judge 以 `BASELINE_JSON=` / `METRIC_JSON=` 前缀行解析，取 `{...}` 段。
- 相对提升 `gain = (0.4206 - 0.4654) / 0.4206 = -0.1065` → 创新版比基线差 → 不可行（本例）。

## EXPERIENCE.md 条目（fix_and_rerun 自动追加，每行一条 JSON）

```json
- {"time": "2026-10-05T14:30:00", "innovation_id": "inn_01", "error": "mat1 and mat2 shapes cannot be multiplied (32x336x7 and 336x128)", "cause": "patch 层期望 (B, L, C)，传入 (B, C, L)", "fix": "数据统一为 (B, L, C)，embed 后再转 (B, num_patches, d_model)"}
```

- 文件头两行是注释（`# 踩坑经验库...`），随后每行 `- {json}` 一条，最多保留 `_EXP_KEEP`（默认 20）条。

## 实验脚本骨架（单文件、自包含）

```python
import os, numpy as np, pandas as pd, torch

def load_data(data_dir="datasets"):
    df = pd.read_csv(os.path.join(data_dir, "ETTh1.csv"), usecols=["date", "HUFL", "HULL", "MXFL"])
    # 60/20/20 划分 → train/val/test → 标准化 → 滑窗 (336, 96)

def evaluate(model, loader):
    # 标准化尺度上算 MSE/MAE；基线与创新版共用本函数
    ...

def patchtst(...): ...
def innovation_idea_1(...): ...

if __name__ == "__main__":
    args = os.environ.get("EXEC_ARGS", "").split("|")  # |data_dir|seq_len|pred_len|epochs
    # 同数据/同超参训练两个模型 → 末尾打印 BASELINE_JSON / METRIC_JSON
```

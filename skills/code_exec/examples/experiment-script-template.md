# code_exec · 单文件实验脚本模板（L3 examples）

实验脚本必须**单一 Python 文件、自包含、可运行**；只依赖 torch / numpy / pandas。
参数从环境变量 `EXEC_ARGS` 读取（`|data_dir|seq_len|pred_len|epochs`），容忍缺省。

```python
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def load_data(data_dir: str = "datasets"):
    """读 ETT 系列 CSV，60/20/20 划分，train 统计量标准化，滑窗 (seq_len, pred_len)。"""
    df = pd.read_csv(os.path.join(data_dir, "ETTh1.csv"))
    # 取多变量特征列（排除 date），60/20/20 切分
    n = len(df)
    train, val, test = df.iloc[:int(n * 0.6)], df.iloc[int(n * 0.6):int(n * 0.8)], df.iloc[int(n * 0.8):]
    mean, std = train.mean(), train.std() + 1e-8
    for part in (train, val, test):
        part = (part - mean) / std
    # 返回 (X, y) 滑窗张量：X (N, seq_len, C), y (N, pred_len, C)
    ...


def evaluate(model, loader):
    """在标准化尺度上算 MSE/MAE。基线与创新版必须调用同一个 evaluate。"""
    ...


class PatchTST(nn.Module):
    """基线：通道独立 patch embedding + 共享 Transformer encoder + 平坦化 head。"""


class Innovation(nn.Module):
    """创新版：在 PatchTST 基础上加入本次创新点（分解/频域增强/门控等）。"""


if __name__ == "__main__":
    parts = os.environ.get("EXEC_ARGS", "").split("|")
    data_dir = parts[0] if len(parts) > 0 and parts[0] else "datasets"
    seq_len = int(parts[1]) if len(parts) > 1 and parts[1] else 336
    pred_len = int(parts[2]) if len(parts) > 2 and parts[2] else 96
    epochs = int(parts[3]) if len(parts) > 3 and parts[3] else 50

    # 同一数据管线/超参同时训练基线与创新版，末尾打印固定格式两行：
    # BASELINE_JSON={"model": "PatchTST", "test_mse": ..., "test_mae": ..., "epochs": ..., "seq_len": ..., "pred_len": ...}
    # METRIC_JSON={"model": "innovation_<id>", "test_mse": ..., "test_mae": ..., ...}
```

## 关键约束（违反会导致判定失败）

- **禁止先反归一化再算 MSE**——preds 与 y 都是标准化值，直接算 MSE；反归一化会放大几十倍、与基线不可比。
- 基线与创新版必须**同数据划分 / 同超参 / 同 epochs / 同一个 evaluate 函数**，保证对比公平。
- 末尾两行 `BASELINE_JSON=` / `METRIC_JSON=` 必须存在（judge 靠它解析）。

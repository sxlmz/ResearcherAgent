# innovate · 实验契约完整规范（L3 参考，implement_innovation / judge_innovation 执行依据）

SKILL.md 正文的精读版本。实现与判定环节执行时若有歧义，以本文件为准。

## 实验代码硬性要求（implement_innovation 内置规则）

> **领域泛化（v1.2.5 硬规则）**：以下 2/3/4 条中的"ETT/60·20·20/滑窗/PatchTST/MSE/MAE"全部是
> **时序领域演示默认**。其他领域（视觉/医药/工程等）由 implement_innovation 按主题上下文
> 采用该领域标准数据管线、常用基线与标准指标，**严禁照抄时序演示默认**。

1. 输出**单一 Python 文件**，直接可运行；只依赖 torch / numpy / pandas。
2. 数据管线与基线一致（时序演示默认：读 ETT 系列 CSV，**60/20/20 划分**（train/val/test），train 统计量标准化，滑窗样本 (seq_len=336, pred_len=96)，batch=128，Adam lr=0.001，MSE loss；其他领域按领域惯例调整数据加载/划分/标准化）。
3. **同一份脚本同时训练基线版 + 创新版**（基线=默认演示 PatchTST，其他领域=所选常用基线），确保对比公平（同数据划分/超参/epochs）。
4. **评估口径强制**：在标准化后的尺度上算主指标（时序演示：test MSE/MAE，preds 与 y 都是标准化值直接算）；**禁止先反归一化再算**（会放大几十倍、与基线不可比）；基线与创新版必须调用同一个 evaluate 函数。
5. 末尾打印两行固定格式：`BASELINE_JSON={...}`、`METRIC_JSON={...}`（judge 靠它解析）。
6. 参数从环境变量 EXEC_ARGS 读取（`|data_dir|seq_len|pred_len|epochs`），容忍缺省（seq_len=336, pred_len=96, epochs=50）。
7. 已知踩坑经验（EXPERIENCE.md）注入 prompt，避免重犯。

## 判定标准（judge_innovation 内置规则）

- 相对提升 `(base-exp)/base`：下降 **5% 以上且主指标同步改善 → 可行**；0~5% → 存疑；不降反升 → 不可行。
- 训练是否稳定收敛（无 NaN/发散）；实现是否忠实于创新点描述；对比是否公平。
- 输出 JSON：`{feasible, gain, basis, risks, suggestion}`。

## 输出契约

### BASELINE_JSON（run_baseline 打印）

```json
{"model": "PatchTST", "test_mse": 0.4206, "test_mae": 0.4123, "epochs": 2, "seq_len": 336, "pred_len": 96}
```

### METRIC_JSON（implement_innovation 打印）

```json
{"model": "innovation_<id>", "test_mse": 0.4654, "test_mae": 0.4301, "epochs": 2, "seq_len": 336, "pred_len": 96}
```

- judge 解析这两行：`gain = (baseline_mse - exp_mse) / baseline_mse`。
- 基线缓存键 = `dataset|seq_len|pred_len|epochs`；命中直接复用，不重跑。

## EXPERIENCE 沉淀规则（CC 式项目记忆）

- 每次 fix_and_rerun 成功修复，追加一条到 `project/EXPERIENCE.md`：

```markdown
## [YYYY-MM-DD] <问题摘要>
- 错误：<报错原文（截断）>
- 根因：<定位到的问题>
- 修复：<具体改动>
```

- 后续所有实验 prompt 会注入 EXPERIENCE.md 全文（"已知踩坑经验，避免重犯"）。
- 与 MEMORY.md（CC 记忆，跨会话全局）互补：EXPERIENCE.md 是**实验领域**的坑位库。

## 数据与默认值（⚠️ 全部为时序演示默认，其他领域按主题变通）

- 数据集缺省：ETTh1（`datasets/ETTh1.csv`，基线脚本会自动匹配目录内任一 CSV）；`data_dir` 由 EXEC_ARGS 指定，缺省回退到工作区 datasets/。
- 主题缺省：长时间序列预测（多变量）；基线 PatchTST（ICLR 2023，通道独立 patch embedding + 共享 Transformer）。

# code_exec · 沙箱双后端机制详解（L3 参考）

SKILL.md 正文的精读版本。模型在 agent 循环中执行代码/训练任务需要细节时用 `read_file` 读取本文件。

## 双后端机制

### 本地后端（默认，LANGGRAPH_EXEC_BACKEND=local）

- `run_python` 用 `CONDA_PYTHON` 指定解释器执行（如 `D:\Software\anaconda\envs\pytorch-2.7.1\python.exe`）——训练环境即用户本地 conda 环境。
- `execute_command` 在本地 shell 执行（安装依赖、git 操作、跑脚本）。
- **特点**：直接复用本机已有环境（torch/numpy 等），零容器开销；隔离性弱于 Docker。

### Docker 后端（LANGGRAPH_EXEC_BACKEND=docker）

- 容器生命周期管理（创建/启动/停止/销毁）+ 工作目录挂载。
- 通过 Socket 连接 + 流式协议下发命令、实时回传训练日志——执行环境隔离、过程可观测。
- 依赖本机已安装 Docker；镜像需预装 torch/numpy/pandas（或镜像内 pip 安装）。

### 环境变量

| 变量 | 作用 | 默认 |
|---|---|---|
| LANGGRAPH_EXEC_BACKEND | local / docker | local |
| CONDA_PYTHON | 本地后端训练解释器路径 | 见 .env.example |
| EXEC_ARGS | 实验参数 `\|data_dir\|seq_len\|pred_len\|epochs` | 见 innovate skill |

## 使用规则（agent 循环）

- 先看工作区现状（list_files / gen_code_tree_structure）再动手。
- 改代码用 read_file → write_file 的闭环；跑脚本用 run_python / execute_command。
- 实验脚本要求**单一文件、自包含、可运行**；日志重定向避免淹没上下文。
- 运行失败：读错误日志 → 修复 → 重跑；同类错误重复两次即换实现路径。

## 配套记忆

- `remember_memory` / `consolidate_memory`（CC 风格项目记忆）随 code_exec 一起暴露：踩过的坑写入 `MEMORY.md` 并全局注入，跨会话复用。
- 创新点实验的坑位还沉淀到 `project/EXPERIENCE.md`（见 innovate skill）。

## 与其他 skill 的关系

- 可配 `search_papers`（先查文献再复刻/实现）。
- `innovate` skill 内部自带代码执行组；单独 `code_exec` 适用于一般写代码/复刻请求。
- 意图识别：写代码/复刻论文方法/跑训练 → `["code_exec"]`（可加 search_papers）。

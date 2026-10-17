---
name: code_exec
description: 在工作区沙箱中执行代码：写/读文件、跑 shell 命令、用 pytorch 环境运行训练脚本。适用：用户要求写代码实现、按论文方法复刻代码、跑训练/推理/实验脚本、调试代码、安装依赖、查看或整理工作区文件。
group: execution
precondition: null
tools: [execute_command, read_file, create_file, write_file, list_files, create_directory, gen_code_tree_structure, run_python, remember_memory, consolidate_memory, recall_memory]
references: [sandbox-details.md]
---

# 代码执行沙箱

## 用途（何时选）

- 用户要求**写代码实现**、**按论文方法复刻代码**、跑训练/推理/实验脚本、调试代码、安装依赖、查看或整理工作区文件。
- 创新点验证流水线（`innovate`）内部也依赖本工具组执行实验代码。

## 工具组（CODE_EXEC_TOOLS）

| 工具 | 作用 |
|---|---|
| execute_command | 执行一条 shell 命令（运行程序、安装依赖、git 操作） |
| read_file | 读取文件完整内容（检查现有文件） |
| create_file | 新建文件并写入初始内容 |
| write_file | 覆盖写入已存在文件（修改代码/配置） |
| list_files | 列出目录下的文件/子目录 |
| create_directory | 创建目录（不存在时） |
| gen_code_tree_structure | 生成代码树结构（快速了解代码库整体结构） |
| run_python | 运行 Python 脚本（本地后端用 `CONDA_PYTHON` 指定环境；Docker 后端用容器内环境） |

## 双后端机制

- **本地后端**（默认）：`LANGGRAPH_EXEC_BACKEND=local`，用 `CONDA_PYTHON`（如 `D:\Software\anaconda\envs\pytorch-2.7.1\python.exe`）跑训练脚本——训练环境即用户本地 conda 环境。
- **Docker 后端**：`LANGGRAPH_EXEC_BACKEND=docker`，容器生命周期管理 + 工作目录挂载，隔离更强。
- 训练/实验耗时由脚本内 epochs 等参数控制（创新点实验默认 epochs=50）。

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

## L3 资源（需要细节时用 read_file 读取）

- `references/sandbox-details.md`：双后端机制详解（本地 CONDA_PYTHON / Docker 挂载 + 环境变量表 / 使用规则 / 配套记忆）
- `examples/experiment-script-template.md`：单文件实验脚本模板（EXEC_ARGS 参数解析 / 评估口径 / 输出契约）

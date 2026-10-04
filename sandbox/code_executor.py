# -*- coding: utf-8 -*-
"""代码执行 LangChain 工具层。

照搬 HKUDS/AI-Researcher `research_agent/inno/tools/terminal_tools.py` 的工具集
（execute_command / read_file / create_file / write_file / list_files /
create_directory / gen_code_tree_structure / run_python），做最小适配：
- @register_tool / @process_terminal_response 装饰器替换为 LangChain @tool；
- 文件操作不再走 shell 命令，改为调用后端统一接口（本地=Python 实现，Docker=原版命令）；
- 输出直接返回文本（含 exit code），不再写临时文件分页。

执行后端由 sandbox_exec.get_env() 提供，环境变量 LANGGRAPH_EXEC_BACKEND=local|docker 切换。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from langchain_core.tools import tool

from .sandbox_exec import get_env, MAX_OUTPUT_CHARS

# 工作区根（容器风格 /workplace 在本地后端会被解析到 workspace_exec）
_WORKPLACE = "/workplace"


def _format(response: Dict) -> str:
    """统一格式化后端返回的 {"status","result"}"""
    status, result = response.get("status"), response.get("result", "")
    result = (result or "").strip()
    if not result:
        result = " "
    return f"Exit code: {status}\nOutput: \n{result}"


@tool
def execute_command(command: str) -> str:
    """在代码执行环境（沙箱）中执行一条 shell 命令，用于运行程序、安装依赖、git 操作等。

    Args:
        command: 要执行的 shell 命令。本地后端为 Windows 语法；Docker 后端为 Linux 语法。
    Returns:
        命令的退出码与全部输出。
    """
    env = get_env()
    response = env.run_command(command)
    return _format(response)


@tool
def read_file(file_path: str) -> str:
    """读取一个文件的完整内容。用于检查现有文件。

    Args:
        file_path: 文件路径（相对工作区根，如 "project/model.py"；也接受 "/workplace/..." 容器风格）。
    Returns:
        文件内容字符串。
    """
    try:
        content = get_env().read_file(file_path)
        return content[:MAX_OUTPUT_CHARS] + (f"\n...[文件过长，已截断，共 {len(content)} 字符]" if len(content) > MAX_OUTPUT_CHARS else "")
    except Exception as e:
        return f"Error reading file: {file_path}: {e}"


@tool
def create_file(path: str, content: str) -> str:
    """在工作区内创建一个新文件并写入初始内容。用于新建脚本/配置。

    Args:
        path: 新文件路径（相对工作区根）。
        content: 文件初始内容。
    Returns:
        创建结果。
    """
    try:
        return get_env().write_file(path, content)
    except Exception as e:
        return f"Error creating file: {str(e)}"


@tool
def write_file(path: str, content: str) -> str:
    """覆盖写入一个已存在文件的内容。用于修改代码/配置。

    Args:
        path: 文件路径（相对工作区根）。
        content: 要写入的完整新内容。
    Returns:
        写入结果。
    """
    try:
        return get_env().write_file(path, content)
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool
def list_files(path: str) -> str:
    """列出指定目录下的文件和子目录名。用于查看目录结构。

    Args:
        path: 目录路径（相对工作区根）。
    Returns:
        目录内容列表。
    """
    try:
        return get_env().list_files(path)
    except Exception as e:
        return f"Error listing files: {str(e)}"


@tool
def create_directory(path: str) -> str:
    """创建目录（不存在时）。用于新建项目目录结构。

    Args:
        path: 目录路径（相对工作区根）。
    Returns:
        创建结果。
    """
    try:
        return get_env().create_directory(path)
    except Exception as e:
        return f"Error creating directory: {str(e)}"


@tool
def gen_code_tree_structure(directory: str) -> str:
    """生成指定目录的代码树结构（树状图）。用于快速了解代码库整体结构。

    Args:
        directory: 目录路径（相对工作区根）。
    Returns:
        树状结构文本。
    """
    try:
        return get_env().gen_tree(directory)
    except Exception as e:
        return f"Error running tree {directory}: {str(e)}"


@tool
def run_python(code_path: str, cwd: Optional[str] = None, env_vars: Optional[Dict[str, str]] = None) -> str:
    """运行一个 Python 脚本（本地后端用 CONDA_PYTHON 指定的 Python 环境；Docker 后端用容器内环境）。

    Args:
        code_path: 脚本路径（绝对或相对工作区根，如 "project/run_training_testing.py"）。
        cwd: 脚本工作目录（可选；缺省为脚本所在目录。若运行指令里有 cd 命令，请显式传该目录）。
        env_vars: 运行前要设置的环境变量字典（可选）。
    Returns:
        退出码与脚本全部输出。
    """
    try:
        response = get_env().run_python_script(code_path, cwd=cwd, env_vars=env_vars)
        return _format(response)
    except Exception as e:
        return f"Error running python script: {e}"


# 工具清单：select_tools 按这个集合从 local_tools 里取
CODE_EXEC_TOOLS = [
    "execute_command",
    "read_file",
    "create_file",
    "write_file",
    "list_files",
    "create_directory",
    "gen_code_tree_structure",
    "run_python",
]

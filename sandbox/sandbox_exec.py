# -*- coding: utf-8 -*-
"""代码执行后端（照搬 HKUDS/AI-Researcher `research_agent/inno/environment/docker_env.py` 的沙箱体系）。

两种后端，接口一致（都提供 run_command / 文件操作 / run_python_script）：
- LocalEnv : 本地 conda 环境直跑（Windows 适配）。无容器隔离，
             靠「工作区白名单 + 路径穿越防护 + 危险命令拦截 + 超时 + 输出截断」兜底。
- DockerEnv: 原版 Docker 沙箱。docker CLI 创建/启动容器 + socket 流式协议
             （{"type":"chunk"} -> {"type":"final",status}），容器内常驻命令服务执行。
             与原版唯一差异：去掉对 `constant.py` 的导入，常量改由环境变量/构造参数提供。

切换：环境变量 LANGGRAPH_EXEC_BACKEND = local（默认）| docker
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Union

# ---------------------------------------------------------------------------
# 配置（环境变量可覆盖）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 本地模式：conda python 解释器（用户本地环境）
CONDA_PYTHON = os.environ.get("CONDA_PYTHON", "python").strip()
# 本地模式：用 CONDA_PYTHON 指定的解释器（默认 PATH 上的 python；
# 训练需要 pytorch 等环境时，通过环境变量/启动脚本注入对应解释器）
# 本地模式：工作区根目录（容器里对应 /workplace）
LOCAL_WORKSPACE = os.environ.get(
    "EXEC_WORKSPACE",
    str(PROJECT_ROOT / "workspace_exec"),
)
# Docker 模式（对应原版 constant.py）
BASE_IMAGES = os.environ.get("BASE_IMAGES", "tjbtech1/airesearcher:v1")
GPUS = os.environ.get("GPUS", "")          # 例如 "device=0"；留空则不带 --gpus
PLATFORM = os.environ.get("PLATFORM", "linux/amd64")
GITHUB_AI_TOKEN = os.environ.get("GITHUB_AI_TOKEN", None)
AI_USER = os.environ.get("AI_USER", "user")

DEFAULT_TIMEOUT = int(os.environ.get("EXEC_TIMEOUT", "300"))       # 秒
MAX_OUTPUT_CHARS = int(os.environ.get("EXEC_MAX_OUTPUT", "20000")) # 输出截断
BLOCK_DANGEROUS = os.environ.get("EXEC_BLOCK_DANGEROUS", "1") != "0"


# ---------------------------------------------------------------------------
# 本地模式：危险命令拦截（仅本地后端生效；Docker 后端容器内放行）
# ---------------------------------------------------------------------------
_BLOCKED_PATTERNS = [
    re.compile(r"rm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+[/\\][/\\]?"),   # rm -rf /
    re.compile(r"rm\s+-rf\s+[a-zA-Z]:[\\/]"),                        # rm -rf C:\
    re.compile(r"format\s+[a-zA-Z]:"),                               # format C:
    re.compile(r"shutdown\s+[/-]\s*s"),                              # shutdown /s
    re.compile(r"reg\s+delete\s+HKLM", re.IGNORECASE),               # 改注册表
    re.compile(r"del\s+[/-]\s*[sq]?\s+[a-zA-Z]:[\\/]", re.IGNORECASE),  # del /s C:\
    re.compile(r"taskkill\s+[/-]\s*f", re.IGNORECASE),               # taskkill /f
    re.compile(r"rd\s+[/-]\s*[sq]?\s+[a-zA-Z]:[\\/]", re.IGNORECASE),   # rd /s C:\
]


def _check_dangerous(command: str) -> Optional[str]:
    if not BLOCK_DANGEROUS:
        return None
    for pat in _BLOCKED_PATTERNS:
        if pat.search(command):
            return f"危险命令已被本地后端拦截（匹配 {pat.pattern!r}）。本地模式无容器隔离，如需执行请用 Docker 后端。"
    return None


def _truncate(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[输出过长，已截断，共 {len(text)} 字符]"


# ---------------------------------------------------------------------------
# LocalEnv：本地 conda 直跑（Windows 适配）
# ---------------------------------------------------------------------------
class LocalEnv:
    """本地执行后端。run_command 用系统 shell（Windows cmd）执行；
    文件操作与 python 脚本直接用 Python / conda python 实现，不依赖 Linux 命令。
    """

    def __init__(
        self,
        python_path: str = CONDA_PYTHON,
        workspace: str = LOCAL_WORKSPACE,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.python_path = python_path
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.docker_workplace = "/workplace"   # 与 Docker 后端路径风格一致
        self.timeout = timeout

    # ---- 路径解析：兼容容器风格 /workplace/xxx 与相对路径 ----
    def _resolve(self, path: str) -> Path:
        p = str(path).strip().replace("\\", "/")
        if p.startswith("/workplace"):
            rel = p[len("/workplace"):].lstrip("/")
        elif p.startswith(str(self.workspace).replace("\\", "/")):
            rel = os.path.relpath(p, str(self.workspace).replace("\\", "/"))
        else:
            rel = p.lstrip("/")
        real = (self.workspace / rel).resolve()
        if not real.is_relative_to(self.workspace):
            raise ValueError(f"路径超出工作区（拒绝访问）: {path}")
        return real

    # ---- 核心：执行 shell 命令（透传，流式收集）----
    def run_command(self, command: str, stream_callback: Optional[Callable[[str], None]] = None) -> Dict:
        blocked = _check_dangerous(command)
        if blocked:
            return {"status": -1, "result": blocked}
        proc = subprocess.Popen(
            command, shell=True, cwd=str(self.workspace),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        lines, output = [], []
        try:
            for line in proc.stdout:
                if stream_callback:
                    stream_callback(line)
                lines.append(line)
                if len(output) < MAX_OUTPUT_CHARS:
                    output.append(line)
            proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            return {
                "status": -1,
                "result": f"[超时终止] 命令超过 {self.timeout}s，已强制结束。\n前 50 行输出：\n" + "".join(output[:50]),
            }
        result = "".join(output)
        return {"status": proc.returncode, "result": _truncate(result)}

    # ---- 文件操作（Python 实现，等价于原版 cat/ls/mkdir/tree/base64 命令）----
    def read_file(self, path: str) -> str:
        fp = self._resolve(path)
        return fp.read_text(encoding="utf-8", errors="replace")

    def write_file(self, path: str, content: str) -> str:
        fp = self._resolve(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"文件已写入: {fp.relative_to(self.workspace)}"

    def list_files(self, path: str) -> str:
        fp = self._resolve(path)
        if fp.is_file():
            return f"{fp.relative_to(self.workspace)} 是文件，请提供目录路径。"
        if not fp.exists():
            return f"目录不存在: {path}"
        names = [p.name + ("/" if p.is_dir() else "") for p in sorted(fp.iterdir())]
        return "\n".join(names) if names else "(空目录)"

    def create_directory(self, path: str) -> str:
        fp = self._resolve(path)
        fp.mkdir(parents=True, exist_ok=True)
        return f"目录已创建: {fp.relative_to(self.workspace)}"

    def gen_tree(self, directory: str, max_depth: int = 4) -> str:
        fp = self._resolve(directory)
        if not fp.exists():
            return f"目录不存在: {directory}"
        lines: list[str] = []

        def walk(d: Path, prefix: str, depth: int):
            if depth > max_depth:
                lines.append(prefix + "...")
                return
            entries = sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            for i, e in enumerate(entries):
                last = i == len(entries) - 1
                branch = "└── " if last else "├── "
                lines.append(prefix + branch + e.name + ("/" if e.is_dir() else ""))
                if e.is_dir() and depth < max_depth:
                    walk(e, prefix + ("    " if last else "│   "), depth + 1)

        lines.append(fp.name + "/")
        walk(fp, "", 0)
        return "\n".join(lines)

    # ---- 运行 python 脚本（conda 环境）----
    def run_python_script(
        self,
        code_path: str,
        cwd: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict:
        fp = self._resolve(code_path)
        if not fp.exists():
            return {"status": -1, "result": f"脚本不存在: {code_path}"}
        workdir = self._resolve(cwd) if cwd else fp.parent
        module_path = str(fp.relative_to(workdir).with_suffix("")).replace(os.sep, ".")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(workdir)
        if env_vars:
            env.update({k: str(v) for k, v in env_vars.items()})
        proc = subprocess.Popen(
            [self.python_path, "-u", "-m", module_path], cwd=str(workdir),
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        output = []
        try:
            for line in proc.stdout:
                if stream_callback:
                    stream_callback(line)
                if len(output) < MAX_OUTPUT_CHARS:
                    output.append(line)
            proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            return {"status": -1, "result": f"[超时终止] 脚本超过 {self.timeout}s，已强制结束。\n" + "".join(output[:50])}
        return {"status": proc.returncode, "result": _truncate("".join(output))}


# ---------------------------------------------------------------------------
# DockerEnv：照搬原版 docker_env.py（去掉 constant 依赖，常量参数化）
# ---------------------------------------------------------------------------
@dataclass
class DockerConfig:
    container_name: str
    workplace_name: str
    communication_port: int                       # 宿主机端口 -> 容器 8000
    local_root: str = field(default=str(PROJECT_ROOT / "workplace_exec"))
    base_image: str = field(default=BASE_IMAGES)
    gpus: str = field(default=GPUS)
    platform: str = field(default=PLATFORM)


class DockerEnv:
    """原版 Docker 沙箱。init_container 创建/启动容器，run_command 走 socket 流式协议。"""

    def __init__(self, config: Union[DockerConfig, Dict]):
        if isinstance(config, Dict):
            config = DockerConfig(**config)
        self.workplace_name = config.workplace_name
        self.local_workplace = os.path.join(config.local_root, config.workplace_name)
        self.docker_workplace = f"/{config.workplace_name}"
        self.container_name = config.container_name
        self.communication_port = config.communication_port
        self.base_image = config.base_image
        self.gpus = config.gpus
        self.platform = config.platform

    def init_container(self, timeout: int = 60):
        container_check_command = [
            "docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Names}}",
        ]
        existing_container = subprocess.run(container_check_command, capture_output=True, text=True)
        os.makedirs(self.local_workplace, exist_ok=True)

        if existing_container.stdout.strip() == self.container_name:
            running_check_command = [
                "docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}",
            ]
            running_container = subprocess.run(running_check_command, capture_output=True, text=True)
            if running_container.stdout.strip() == self.container_name:
                print(f"Container '{self.container_name}' is already running. Skipping creation.")
                return
            start_command = ["docker", "start", self.container_name]
            subprocess.run(start_command)
            print(f"Container '{self.container_name}' has been started.")
            return

        gpu_cmd = ["--gpus", self.gpus] if self.gpus else []
        docker_command = [
            "docker", "run", "-d", "--platform", self.platform, "--userns=host",
        ] + gpu_cmd + [
            "--name", self.container_name,
            "--user", "root",
            "-v", f"{self.local_workplace}:{self.docker_workplace}",
            "-w", self.docker_workplace,
            "-p", f"{self.communication_port}:8000",
            "--restart", "unless-stopped",
            self.base_image,
        ]
        print(" ".join(docker_command))
        result = subprocess.run(docker_command, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Failed to start container: {result.stderr}")
        if self.wait_for_container_ready(timeout=timeout):
            print(f"Container '{self.container_name}' has been created and started.")

    def wait_for_container_ready(self, timeout: int = 30):
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", self.container_name],
                capture_output=True, text=True,
            )
            if result.returncode == 0 and "true" in result.stdout.lower():
                try:
                    # 探测端口映射 + tcp_server 进程
                    port_info = _check_container_ports(self.container_name)
                    if port_info:
                        self.communication_port = port_info[0]
                        check = self.run_command("ps aux | grep tcp_server | grep -v grep")
                        if check["status"] == 0:
                            return True
                except Exception as e:
                    print(f"Failed to check container ports: {e}")
            time.sleep(1)
        raise TimeoutError(f"Container {self.container_name} failed to start within {timeout} seconds")

    def stop_container(self):
        result = subprocess.run(["docker", "stop", self.container_name], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Failed to stop container: {result.stderr}")

    def run_command(self, command: str, stream_callback: Optional[Callable[[str], None]] = None) -> Dict:
        """与原版一致：socket 连接容器内命令服务，流式收取 chunk/final。"""
        buffer_size = 4096
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("localhost", self.communication_port))
            s.sendall(command.encode())
            partial_line = ""
            while True:
                chunk = s.recv(buffer_size)
                if not chunk:
                    break
                data = partial_line + chunk.decode("utf-8")
                lines = data.split("\n")
                for line in lines[:-1]:
                    if line:
                        try:
                            response = json.loads(line)
                            if response["type"] == "chunk":
                                if stream_callback:
                                    stream_callback(response["data"])
                            elif response["type"] == "final":
                                return {"status": response["status"], "result": response["result"]}
                        except json.JSONDecodeError:
                            print(f"Invalid JSON: {line}")
                partial_line = lines[-1]
        return {"status": -1, "result": "Connection closed without final response"}

    # ---- 文件操作：容器内 Linux 命令（原版 terminal_tools.py 同款）----
    def read_file(self, path: str) -> str:
        return self.run_command(f"cat {path}")["result"]

    def write_file(self, path: str, content: str, chunk_size: int = 100000) -> str:
        encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        total = math.ceil(len(encoded) / chunk_size)
        for i in range(total):
            chunk = encoded[i * chunk_size:(i + 1) * chunk_size]
            op = ">" if i == 0 else ">>"
            resp = self.run_command(f"echo \"{chunk}\" | base64 -d {op} {path}")
            if resp["status"] != 0:
                return f"Error creating file {path}: {resp['result']}"
        return f"File created at: {path}"

    def list_files(self, path: str) -> str:
        resp = self.run_command(f"ls -1 {path}")
        return resp["result"] if resp["status"] == 0 else f"Error listing files: {resp['result']}"

    def create_directory(self, path: str) -> str:
        resp = self.run_command(f"mkdir -p {path}")
        if resp["status"] != 0:
            return f"Error creating directory: {resp['result']}"
        return f"Directory '{path}' created successfully."

    def gen_tree(self, directory: str) -> str:
        return self.run_command(f"tree {directory}")["result"]

    def run_python_script(
        self,
        code_path: str,
        cwd: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict:
        # 原版 run_python 逻辑：PYTHONPATH + 模块式运行
        if Path(code_path).is_absolute():
            if self.run_command(f"ls {code_path}")["status"] != 0:
                return {"status": -1, "result": f"File {code_path} does not exist"}
            code_abs_path = code_path
        else:
            code_abs_path = f"{self.docker_workplace}/{code_path}"
            if self.run_command(f"ls {code_abs_path}")["status"] != 0:
                return {"status": -1, "result": f"Relative path is from `{self.docker_workplace}`, but `{code_abs_path}` does not exist"}
        if cwd:
            if Path(cwd).is_absolute():
                if self.run_command(f"ls {cwd}")["status"] != 0:
                    return {"status": -1, "result": f"Working directory {cwd} does not exist"}
            else:
                cwd = f"{self.docker_workplace}/{cwd}"
                if self.run_command(f"ls {cwd}")["status"] != 0:
                    return {"status": -1, "result": f"Working directory {cwd} does not exist"}
        else:
            cwd = str(Path(code_abs_path).parent)
        env_str = f"PYTHONPATH={cwd}"
        if env_vars:
            env_str += " " + " ".join(f"{k}={v}" for k, v in env_vars.items())
        try:
            rel_path = Path(code_abs_path).relative_to(cwd)
            module_path = str(rel_path.with_suffix("")).replace(os.sep, ".")
            command = f"cd {cwd} && {env_str} python -m {module_path}"
        except ValueError:
            command = f"cd {cwd} && {env_str} python {code_path}"
        return self.run_command(command, stream_callback)


def _check_container_ports(container_name: str):
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Ports}}"],
        capture_output=True, text=True,
    )
    for mapping in result.stdout.strip().split(","):
        mapping = mapping.strip()
        if "->" in mapping:
            host_part, container_part = mapping.split("->")
            host_port = int(host_part.split(":")[1])
            container_port = int(container_part.split("/")[0])
            return (host_port, container_port)
    return None


# ---------------------------------------------------------------------------
# 全局后端工厂：LANGGRAPH_EXEC_BACKEND=local|docker
# ---------------------------------------------------------------------------
_BACKEND = os.environ.get("LANGGRAPH_EXEC_BACKEND", "local").lower()

_DOCKER_ENV: Optional[DockerEnv] = None
_LOCAL_ENV: Optional[LocalEnv] = None


def get_env() -> Union[LocalEnv, DockerEnv]:
    global _LOCAL_ENV, _DOCKER_ENV
    if _BACKEND == "docker":
        if _DOCKER_ENV is None:
            _DOCKER_ENV = DockerEnv(DockerConfig(
                container_name=os.environ.get("EXEC_CONTAINER_NAME", "langgraph_code_exec"),
                workplace_name="workplace",
                communication_port=int(os.environ.get("EXEC_PORT", "17899")),
            ))
        return _DOCKER_ENV
    if _LOCAL_ENV is None:
        _LOCAL_ENV = LocalEnv()
    return _LOCAL_ENV


def set_backend(backend: str):
    """运行时切换后端（测试/编排用）：set_backend("docker") / set_backend("local")"""
    global _BACKEND, _DOCKER_ENV, _LOCAL_ENV
    backend = backend.lower()
    if backend not in ("local", "docker"):
        raise ValueError(f"未知后端: {backend}")
    _BACKEND = backend
    _DOCKER_ENV = None
    _LOCAL_ENV = None

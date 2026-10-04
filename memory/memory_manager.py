"""
memory_manager.py
=================
L1 常驻档案加载（记忆系统 v2）。

记忆系统 v2 分层（见 memory_stack / memory_orchestrator / memory_classifier）：
  L0 工作记忆 → Redis
  L1 常驻档案（SOUL/USER/MEMORY）→ PostgreSQL user_profiles（主）+ MD 文件（兜底）
  L2 语义记忆 / 对话原文 / 文档 chunk → Infinity
  会话级记忆 → Infinity conv / mem 表

本模块职责：
  - `_current_user_id` ContextVar（用户隔离入口）
  - `load_all_for_prompt`：把 L1 档案注入 system prompt（每次全量，不检索）
    数据源优先级：PostgreSQL user_profiles → MD 文件兜底
  - MD 文件模板/读写（兼容兜底路径）

LLM-as-Judge 记忆分类迁移至 memory_classifier.py（写 Postgres + Infinity）。
"""

from pathlib import Path
from contextvars import ContextVar


# ==========================================
# 0. 上下文 & 路径
# ==========================================

_current_user_id: ContextVar[int | None] = ContextVar("current_user_id", default=None)

DEFAULT_MEMORY_DIR = Path("./memory")
DEFAULT_MEMORY_DIR.mkdir(exist_ok=True)


def _get_memory_dir() -> Path:
    """根据当前用户上下文返回用户专属记忆目录（MD 兜底用）。"""
    uid = _current_user_id.get()
    if uid is not None:
        user_dir = Path(f"./memory/{uid}")
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    return DEFAULT_MEMORY_DIR


# ==========================================
# 1. MD 模板 & 初始化（兜底路径）
# ==========================================

DEFAULT_SOUL = """# SOUL.md — 灵魂内核

## 性格与价值观
- 专业、严谨但不失亲和力的学术研究助手
- 沟通风格：清晰有条理，适当使用表情符号增加亲和力
- 价值观：追求真理、尊重学术伦理、鼓励创新思维

## 沟通风格
- 优先使用中文回复，学术术语可保留英文原文
- 回答问题时先给出核心结论再展开解释
- 涉及学术争议时客观呈现各方观点，不偏袒

## 行为边界
- 不提供违反学术伦理的建议（代写、造假等）
- 不帮助论文抄袭或规避查重
- 尊重用户隐私和数据安全
"""

DEFAULT_USER = """# USER.md — 用户档案

## 基本信息
- 称呼：用户
- 时区：Asia/Shanghai (UTC+8)

## 研究兴趣
- 待补充（根据对话自动更新）

## 写作偏好
- 语言：中文为主
- 引用格式：待补充

## 当前关注
- 待补充
"""

DEFAULT_MEMORY = """# MEMORY.md — 零容错关键信息

## 个人标识
- 待补充（手机号、学号等不可丢失的信息）

## 硬性约束
- 待补充（用户明确提出的、不可违反的规则）

## 重要决策
- 待补充（跨会话必须记住的决定）
"""


def _ensure_default_md(path: Path) -> None:
    """若 path 不存在，写入默认模板。"""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        {"SOUL.md": DEFAULT_SOUL, "USER.md": DEFAULT_USER, "MEMORY.md": DEFAULT_MEMORY}[path.name],
        encoding="utf-8",
    )
    print(f"--> [MemoryMD] 已创建默认 {path.name} [{path.parent}]")


for name, default_content in [
    ("SOUL.md", DEFAULT_SOUL),
    ("USER.md", DEFAULT_USER),
    ("MEMORY.md", DEFAULT_MEMORY),
]:
    p = DEFAULT_MEMORY_DIR / name
    if not p.exists():
        p.write_text(default_content, encoding="utf-8")
        print(f"--> [MemoryMD] 已创建默认 {name}")

del p, name, default_content


# ==========================================
# 2. MD 文件读取（L1 兜底）
# ==========================================

def load_md(name: str) -> str:
    """读取指定 memory 文件内容。name 不带 .md 后缀。"""
    md_dir = _get_memory_dir()
    path = md_dir / f"{name}.md"
    _ensure_default_md(path)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


MAX_MEMORY_CHARS = 2500


def _tail_on_boundary(text: str, max_chars: int) -> str:
    """保留 text 末尾不超过 max_chars 的内容，按行边界切。"""
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    nl = tail.find("\n")
    if nl != -1 and nl < len(tail) - 1:
        tail = tail[nl + 1:]
    return "...(早期记录已省略)\n" + tail


# ==========================================
# 3. L1 常驻档案注入（Postgres 主 + MD 兜底）
# ==========================================

def load_all_for_prompt(user_request: str = "") -> str:
    """加载 L1 常驻档案，拼接为注入 system prompt 的片段。

    数据源（按优先级）：
      1. PostgreSQL user_profiles（新系统，按 user_id 隔离）—— 主存储
      2. MD 文件 SOUL/USER/MEMORY.md —— 兜底（Postgres 不可用 / 无 user_id 时）

    注意：L2/L3 向量记忆（会话级隔离）不在此处加载——它们由
    memory_node 用记忆编排器单独检索。
    """
    uid = _current_user_id.get()
    rows = _load_profiles_from_pg(uid)
    if rows is None:
        rows = _load_profiles_from_md()
    return _format_profiles(rows)


_pg_store_singleton = None


def _load_profiles_from_pg(uid):
    """从 Postgres user_profiles 读 L1 档案；无 uid 或不可用返回 None（回退 MD）。"""
    if uid is None:
        return None
    global _pg_store_singleton
    try:
        if _pg_store_singleton is None:
            from agent.memory.memory_stack import PostgresStore
            _pg_store_singleton = PostgresStore()
        rows = _pg_store_singleton.get_profiles(uid)
        return rows if rows else []
    except Exception:
        return None


def _load_profiles_from_md():
    """从 MD 文件读 L1 档案（兜底）。"""
    rows = []
    for name in ["SOUL", "USER", "MEMORY"]:
        content = load_md(name)
        if content.strip():
            rows.append({"profile_type": name, "content": content})
    return rows


def _format_profiles(rows, max_chars=MAX_MEMORY_CHARS, per_file=800):
    """把 profile 行按 SOUL/USER/MEMORY 分组格式化，带长度截断。"""
    if not rows:
        return ""
    parts = []
    total = 0
    for name in ["SOUL", "USER", "MEMORY"]:
        body = ""
        for row in rows:
            if (row.get("profile_type") or "").upper() == name:
                body += str(row.get("content", "")) + "\n"
        body = body.strip()
        if not body:
            continue
        body = _tail_on_boundary(body, per_file)
        remaining = max_chars - total
        if remaining <= 0:
            break
        body = _tail_on_boundary(body, remaining)
        parts.append(f"<!-- {name} -->\n{body}")
        total += len(body)
    return "\n\n---\n\n".join(parts)

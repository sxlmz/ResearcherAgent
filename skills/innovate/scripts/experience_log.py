#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""innovate · EXPERIENCE.md 经验库读写（确定性操作，运行时不进 LLM 上下文）。

与 ResearcherAgent/innovate/innovation_tools.py 的 _load_experiences / _append_experience
保持同一文件格式（每行一条 JSON：{"time","innovation_id","error","cause","fix"}）。
运行时代理由 fix_and_rerun 工具调用；本脚本提供独立 CLI，供复现/巡检/人工维护。

用法：
    python experience_log.py list [--path project/EXPERIENCE.md] [--limit 20]
    python experience_log.py append '{"error":"...","cause":"...","fix":"..."}' \
        [--id inn_01] [--path project/EXPERIENCE.md] [--keep 20]

仅依赖标准库。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_PATH = Path("project/EXPERIENCE.md")
DEFAULT_KEEP = 20


def load_experiences(path: Path | str, limit: int = DEFAULT_KEEP) -> list[dict]:
    """读取经验库全部条目，按行解析 `- {json}`，返回最近 limit 条。"""
    p = Path(path)
    if not p.exists():
        return []
    entries = []
    try:
        raw = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("- "):
            try:
                d = json.loads(line[2:])
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict):
                entries.append(d)
    return entries[-limit:]


def append_experience(entry: dict, path: Path | str = DEFAULT_PATH,
                      keep: int = DEFAULT_KEEP, innovation_id: str = "") -> None:
    """追加一条经验（时间戳 + 可选创新点 id），文件按 keep 截断。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    entries = load_experiences(p, keep)
    row = {
        "time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "innovation_id": innovation_id,
        **entry,
    }
    entries.append(row)
    entries = entries[-keep:]
    lines = [
        "# 踩坑经验库（fix_and_rerun 自动记录；implement_innovation / fix_and_rerun 会注入最近条数供参考）",
        '# 每行一条 JSON：{"time", "innovation_id", "error", "cause", "fix"}',
        "",
    ]
    lines += [f"- {json.dumps(e, ensure_ascii=False)}" for e in entries]
    p.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EXPERIENCE.md 经验库读写")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="列出最近条目")
    p_list.add_argument("--path", default=str(DEFAULT_PATH))
    p_list.add_argument("--limit", type=int, default=DEFAULT_KEEP)

    p_app = sub.add_parser("append", help="追加一条经验")
    p_app.add_argument("entry", help='JSON 字符串，如 \'{"error":"...","cause":"...","fix":"..."}\'')
    p_app.add_argument("--id", default="", dest="innovation_id")
    p_app.add_argument("--path", default=str(DEFAULT_PATH))
    p_app.add_argument("--keep", type=int, default=DEFAULT_KEEP)

    args = ap.parse_args(argv)
    if args.cmd == "list":
        for e in load_experiences(args.path, args.limit):
            print(json.dumps(e, ensure_ascii=False))
        return 0
    if args.cmd == "append":
        try:
            entry = json.loads(args.entry)
        except json.JSONDecodeError as exc:
            print(f"entry 不是合法 JSON：{exc}", file=sys.stderr)
            return 1
        if not isinstance(entry, dict):
            print("entry 必须是 JSON 对象", file=sys.stderr)
            return 1
        append_experience(entry, args.path, args.keep, args.innovation_id)
        print(f"已追加到 {args.path}（保留最近 {args.keep} 条）")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

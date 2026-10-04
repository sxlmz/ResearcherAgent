"""
arxiv_agent.py
==============
文献综述 Agent —— 独立运行入口（非 Web 模式）。

直接运行:
    python arxiv_agent.py

特性：
  1. 智能路由——简单问题直接回答，不调用 MCP
  2. 对话记忆存入向量数据库（而非论文原文）
  3. 对话按 session 存 JSONL + embedding
  4. Harness 编程——独立评委打分，未通过自动重写
  5. 按用户要求保存综述到本地文件
"""
# --- repo-root bootstrap：保证 `python <本文件>` 在任意 cwd 下都能 import 到包 ---
import sys
from pathlib import Path
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "agent" / "__init__.py").exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
import asyncio
from datetime import datetime
import uuid

# 复用 agent_core 的全部逻辑
from agent.core.agent_core import (
    runtime,
    save_conversation_turn,
    load_conversation_history,
)

# ==========================================
# 主程序
# ==========================================
async def main():
    load_dotenv(Path(".") / ".env")

    # 1. 初始化 Agent（加载模型 + MCP 工具 + 编译图）
    print("⏳ 正在初始化 Agent...")
    await runtime.init()

    session_id = f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # 2. 交互循环
    print("\n" + "=" * 60)
    print("🤖 文献综述 Agent 已就绪")
    print("=" * 60)
    print("支持的命令:")
    print("  · 输入文献综述请求（如: 请写一篇关于 diffusion models 的综述）")
    print("  · 输入简单问题（如: 什么是强化学习？）")
    print("  · 输入 review:<主题> 来指定要查阅的本地文件")
    print("  · 输入 /history 查看当前会话对话历史")
    print("  · 输入 /quit 退出")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("👋 再见！")
            break

        if user_input.lower() == "/history":
            history = load_conversation_history(session_id, last_n=20)
            if history:
                print(f"\n📜 当前会话对话历史:\n{history}")
            else:
                print("\n📜 暂无对话历史。")
            continue

        # 记录用户消息
        save_conversation_turn(session_id, "user", user_input)

        # 运行 Agent
        config = {"configurable": {"thread_id": session_id}}
        inputs = {
            "session_id": session_id,
            "user_request": user_input,
        }

        print("\n" + "─" * 40)
        try:
            async for output in runtime.app.astream(inputs, config=config):
                for node_name, value in output.items():
                    # 打印节点名称
                    from agent.core.agent_core import NODE_LABELS
                    label = NODE_LABELS.get(node_name, f"执行节点 [{node_name}]...")
                    print(f"  {label}")

            # 获取最终结果
            final_state = runtime.app.get_state(config)
            values = final_state.values if final_state else {}
            final_review = values.get("final_review", "")
            score = values.get("review_score", {})
            save_path = values.get("save_path", "")
            rewrites = values.get("rewrite_count", 0)

            if final_review:
                print("\n" + "=" * 50)
                print("📝 最终输出:")
                print("=" * 50)

                # 如果内容较长，截断显示
                if len(final_review) > 2000:
                    print(final_review[:2000])
                    print(f"\n... (共 {len(final_review)} 字符，已截断显示前 2000 字符)")
                else:
                    print(final_review)

                # 打印评分信息
                total_score = score.get("total_score", "N/A")
                feedback = score.get("feedback", "")
                print("\n" + "─" * 40)
                print(f"📊 评委评分: {total_score}/10 | 重写次数: {rewrites}")
                dims = {k: v for k, v in score.items()
                        if k not in ("passed", "total_score", "feedback")}
                if dims:
                    print(f"   各维度: {dims}")
                if save_path:
                    print(f"💾 已保存到: {save_path}")
                if feedback:
                    print(f"💬 评委简评: {feedback[:300]}")
                print("=" * 50)
            else:
                print("\n⚠️ 未能生成内容。")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n❌ 运行出错: {e}")

        print("─" * 40)


if __name__ == "__main__":
    asyncio.run(main())

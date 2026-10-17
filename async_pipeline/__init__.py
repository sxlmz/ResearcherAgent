# -*- coding: utf-8 -*-
"""综述流水线 MQ 异步（方案 v1.2 第 6 节）。

- card_to_json：字段化卡片 → 内容块（第 4 步核心，RabbitMQ 优先 / ThreadPool 降级）
- locate_fallback：关键词失效时 LLM 定位节（复用 graph_nodes._locate_sections_with_llm）
- card_build（MQ1）：论文全文 → 字段化卡片（v1.2.15 制卡流水线，method 缺失转 MQ2 直接提取）

对外接口：
    generate_json_blocks_for_card(card, llm)      # 纯函数：单卡片 → 内容块（测试/复用主路径）
    submit_cards_for_json(cards, llm, ...)        # 批量提交：MQ 优先，自动降级 ThreadPool
    start_json_session / extend_json_session / collect_json_session
                                                  # 流式会话：read_sections 边制卡边送 → write 收集
    start_card_build_session / extend_card_build_session / collect_card_build_session
                                                  # MQ1 制卡流水线会话（read_sections 下载即送）
    submit_locate_fallback(paras, llm, ...)       # 节定位兜底：MQ 优先，自动降级直连
"""
from .mq_client import (
    generate_json_blocks_for_card,
    submit_cards_for_json,
    start_json_session,
    extend_json_session,
    collect_json_session,
    start_card_build_session,
    extend_card_build_session,
    collect_card_build_session,
    submit_locate_fallback,
    MqUnavailable,
)

__all__ = [
    "generate_json_blocks_for_card",
    "submit_cards_for_json",
    "start_json_session",
    "extend_json_session",
    "collect_json_session",
    "start_card_build_session",
    "extend_card_build_session",
    "collect_card_build_session",
    "submit_locate_fallback",
    "MqUnavailable",
]

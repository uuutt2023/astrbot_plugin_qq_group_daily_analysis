"""Jev 预判结果 - 值对象

承载「这批聊天值不值得生成分析报告」这一判定的结论。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JevGateDecision:
    """Jev 预判结论。

    Attributes:
        worth_reporting: 是否值得生成报告。None 表示没判定
            （未启用、调用失败或消息为空），调用方应按「放行」处理。
        probability: Jev 给出的概率，未判定时为 None。
        mood: 聊天氛围标签（Jev 的 choice 答案），未判定时为空串。
        reason: 判定说明，用于日志与错误提示。
    """

    worth_reporting: bool | None = None
    probability: float | None = None
    mood: str = ""
    reason: str = ""

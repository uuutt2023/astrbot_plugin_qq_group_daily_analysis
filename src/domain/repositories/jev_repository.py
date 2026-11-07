"""Jev 预判接口 - 领域层

定义「这批聊天值不值得生成分析报告」这一判定的抽象契约，
把应用层与 Jev 决策网关解耦。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..value_objects.jev_gate import JevGateDecision
    from ..value_objects.unified_message import UnifiedMessage


class IJevGate(ABC):
    """Jev 预判接口"""

    @abstractmethod
    async def evaluate(self, messages: list[UnifiedMessage]) -> JevGateDecision:
        """判断这批聊天是否值得生成分析报告。

        Args:
            messages: 已清洗的统一消息列表。

        Returns:
            JevGateDecision: 未判定时 worth_reporting 为 None，调用方按放行处理。
        """

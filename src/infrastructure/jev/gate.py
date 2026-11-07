"""Jev 预筛：判断这批群聊值不值得生成分析报告。

插件的分析阶段要跑 4~5 次大模型请求，代价不小。消息条数达标的群里也可能
全是刷屏与表情包，这种内容生成出来的报告没有意义。Jev 是廉价的判定模型，
用它在正式分析前先筛一遍。

判定失败（没配网关、网关不可用、返回异常）一律按放行处理：预筛是优化，
不是拦住分析的主路径。
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ...domain.repositories.jev_repository import IJevGate
from ...domain.value_objects.jev_gate import JevGateDecision
from ...utils.logger import logger
from .client import DEFAULT_RETRIES, JevClient, budget_seconds, probability

if TYPE_CHECKING:
    from ...domain.value_objects.unified_message import UnifiedMessage
    from ..config.config_manager import ConfigManager

#: 主判定题的默认提示词。用户在配置里填了「Jev 提示词」就用用户的。
DEFAULT_WORTH_REPORTING_INSTRUCTIONS = (
    "Is this group chat worth generating a daily analysis report for? "
    "Judge from the messages as a whole. It is worth reporting when members "
    "discussed something with substance - questions, decisions, plans, "
    "opinions, disputes, learning, or emotional moments. It is not worth "
    "reporting when the chat is only greetings, small talk, sticker or emoji "
    "flooding, repeated memes, single-character replies, or off-topic noise. "
    "A busy chat full of meaningless messages is still not worth reporting."
)

#: 氛围标签的选项与中文说明
MOOD_LABELS = {
    "deep_discussion": "深度讨论",
    "daily_chatter": "日常闲聊",
    "flooding": "刷屏灌水",
    "sticker_spam": "表情包刷屏",
    "mixed": "混杂",
}


def build_state(
    messages: list[UnifiedMessage],
    max_chars: int = 2000,
    max_messages: int = 120,
) -> str:
    """把群聊记录拼成交给 Jev 的 state。

    形如：
        [Scene] A group chat log for one day. Messages are listed oldest first.
        [Messages]
        [08:31] 张三: 今天群里聊什么
        [08:32] 李四: [图片]
    """
    lines = []
    for message in list(messages or [])[-max(0, int(max_messages)) :]:
        text = _message_line(message)
        if text:
            lines.append(text)
    if not lines:
        return ""

    header = (
        "[Scene] A group chat log for one day. "
        "Messages are listed oldest first."
    )
    state = "\n".join([header, "[Messages]", "\n".join(lines)])
    if max_chars and max_chars > 0 and len(state) > max_chars:
        # 保留头部，裁掉较早的消息
        keep = len(header) + 32
        room = max(0, max_chars - keep)
        trimmed = []
        size = 0
        for line in reversed(lines):
            if size + len(line) > room:
                break
            trimmed.append(line)
            size += len(line)
        state = "\n".join([header, "[Messages]", "\n".join(reversed(trimmed))])
    return state


def _message_line(message: UnifiedMessage) -> str:
    """一条消息的文本行。"""
    try:
        name = message.get_display_name() or message.sender_id
    except Exception:
        name = getattr(message, "sender_id", "")
    text = _message_text(message)
    stamp = ""
    try:
        stamp = _format_time(message.timestamp)
    except Exception:
        stamp = ""
    prefix = f"[{stamp}] " if stamp else ""
    body = text if text else "[非文字消息]"
    return f"{prefix}{name}: {body}"


def _message_text(message: UnifiedMessage) -> str:
    """取消息里的文字部分。"""
    parts = []
    for content in getattr(message, "contents", ()) or ():
        content_type = getattr(content, "type", None)
        text = getattr(content, "text", "") or ""
        if content_type is not None and getattr(content_type, "value", "") == "text":
            if text:
                parts.append(text)
        elif content_type is not None and getattr(content_type, "value", "") == "at":
            at_id = getattr(content, "at_user_id", "") or ""
            if at_id:
                parts.append(f"@[{at_id}]")
    if not parts:
        parts.append(getattr(message, "text_content", "") or "")
    return "".join(parts).strip()


def _format_time(timestamp) -> str:
    """时间戳转 HH:MM。"""
    try:
        from datetime import datetime

        return datetime.fromtimestamp(float(timestamp)).strftime("%H:%M")
    except Exception:
        return ""


def build_questions(prompt: str = "") -> dict:
    """构造判定问题集合。

    主判定题问「值不值得报告」，第二题问「这批聊天是什么氛围」，
    两者在同一次请求里返回，不额外花钱。
    """
    instructions = (prompt or "").strip() or DEFAULT_WORTH_REPORTING_INSTRUCTIONS
    return {
        "worth_reporting": {
            "type": "noul",
            "instructions": instructions,
            "criteria": {
                "true": (
                    "The chat contains something worth a report: real discussion, "
                    "questions, decisions, opinions, or notable moments."
                ),
                "false": (
                    "The chat is only noise: greetings, small talk, sticker or emoji "
                    "flooding, repeated memes, or off-topic chatter."
                ),
            },
        },
        "chat_mood": {
            "type": "choice",
            "instructions": (
                "What best describes this chat? Pick exactly one option."
            ),
            "criteria": {
                "deep_discussion": "Members discussed something with substance.",
                "daily_chatter": "Ordinary small talk without much content.",
                "flooding": "Repeated or meaningless messages flooding the chat.",
                "sticker_spam": "Mostly stickers, emoji and image-only messages.",
                "mixed": "A mix of substantive talk and casual chatter.",
            },
        },
    }


class JevGate(IJevGate):
    """用 Jev 决策网关做分析预筛。"""

    def __init__(self, config_manager: ConfigManager, client: JevClient | None = None):
        """初始化预筛器。

        Args:
            config_manager: 配置管理器。
            client: 网关客户端，缺省自建一个。
        """
        self.config_manager = config_manager
        self.client = client or JevClient()

    def _cfg(self) -> dict:
        """从配置管理器取一份类型安全的 Jev 配置。"""
        cm = self.config_manager
        return {
            "enabled": cm.get_jev_enabled(),
            "proxy_url": cm.get_jev_proxy_url(),
            "proxy_protocol": cm.get_jev_proxy_protocol(),
            "gate_enabled": cm.get_jev_gate_enabled(),
            "threshold": cm.get_jev_threshold(),
            "max_state_chars": cm.get_jev_max_state_chars(),
            "max_messages": cm.get_jev_max_messages(),
            "prompt": cm.get_jev_prompt(),
        }

    def is_enabled(self) -> bool:
        """是否启用 Jev 预筛。"""
        cfg = self._cfg()
        return bool(cfg.get("enabled")) and bool(
            str(cfg.get("proxy_url") or "").strip()
        )

    async def evaluate(self, messages: list[UnifiedMessage]) -> JevGateDecision:
        """判断这批聊天值不值得生成分析报告。"""
        cfg = self._cfg()
        if not self.is_enabled():
            return JevGateDecision(reason="未启用 Jev 预筛")
        if not cfg.get("gate_enabled", True):
            return JevGateDecision(reason="已关闭 Jev 预筛开关")

        state = build_state(
            messages,
            max_chars=int(cfg.get("max_state_chars") or 2000),
            max_messages=int(cfg.get("max_messages") or 120),
        )
        if not state:
            return JevGateDecision(reason="没有可交给 Jev 的消息")

        questions = build_questions(str(cfg.get("prompt") or ""))
        timeout = budget_seconds(cfg)
        deadline = asyncio.get_running_loop().time() + timeout
        try:
            retries = max(0, min(3, int(cfg.get("retries") or DEFAULT_RETRIES)))
        except (TypeError, ValueError):
            retries = DEFAULT_RETRIES

        last_result: dict = {}
        for attempt in range(retries + 1):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                last_result = await asyncio.wait_for(
                    self.client.evaluate(
                        dict(cfg, timeout_sec=timeout), state, questions
                    ),
                    timeout=remaining,
                )
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                return JevGateDecision(reason="Jev 预筛超时")
            if last_result.get("status") == "ready":
                return self._map_answers(last_result.get("answers") or {}, cfg)
            if not last_result.get("retryable"):
                break
            if attempt >= retries:
                break
            wait = float(last_result.get("retry_after") or 0)
            if wait >= deadline - asyncio.get_running_loop().time():
                break
            await asyncio.sleep(min(wait, max(0.0, deadline - asyncio.get_running_loop().time())))

        reason = str(last_result.get("error_message") or last_result.get("status") or "")
        logger.warning(f"[Jev 预筛] 调用失败，按放行处理：{reason}")
        return JevGateDecision(reason=f"Jev 预筛失败：{reason}")

    @staticmethod
    def _map_answers(answers: dict, cfg: dict) -> JevGateDecision:
        """把 Jev 的答案映射成预筛结论。"""
        node = answers.get("worth_reporting")
        prob = probability(node)
        mood = ""
        mood_node = answers.get("chat_mood")
        if isinstance(mood_node, dict):
            mood_key = str(mood_node.get("choice") or "")
            mood = MOOD_LABELS.get(mood_key, mood_key)

        if prob is None:
            return JevGateDecision(
                worth_reporting=None,
                mood=mood,
                reason="Jev 没有给出概率",
            )

        try:
            threshold = float(cfg.get("threshold"))
        except (TypeError, ValueError):
            threshold = 0.5
        worth = prob >= threshold
        return JevGateDecision(
            worth_reporting=worth,
            probability=prob,
            mood=mood,
            reason=f"概率 {round(prob, 3)}（阈值 {threshold}）",
        )

    async def close(self) -> None:
        """关闭网关客户端。"""
        await self.client.close()

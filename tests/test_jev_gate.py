"""Jev 预筛：状态拼装、问题构造、判定映射与服务接入。

覆盖：
- build_state：消息行格式、时间截断、空消息返回空串；
- build_questions：两道题（noul + choice）与自定义提示词；
- JevGate：开关语义、概率映射、调用失败按放行处理；
- 增量分析服务：预筛判定不值得时跳过本批次，且不调用大模型。
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest


class DummyPluginKV:
    """模拟 AstrBot 插件 KV 存储引擎"""

    def __init__(self):
        self._kv: dict = {}

    async def get_kv_data(self, key: str, default=None):
        return self._kv.get(key, default)

    async def put_kv_data(self, key: str, value) -> None:
        if value is None:
            self._kv.pop(key, None)
        else:
            self._kv[key] = value


@pytest.fixture
def dummy_plugin_kv():
    return DummyPluginKV()


def make_message(
    name: str = "张三",
    text: str = "今天群里聊什么",
    sender_id: str = "u1",
    ts: int | None = None,
):
    """构造一条统一消息。"""
    from src.domain.value_objects.unified_message import (
        MessageContent,
        MessageContentType,
        UnifiedMessage,
    )

    return UnifiedMessage(
        message_id=f"m-{sender_id}-{text[:6]}",
        sender_id=sender_id,
        sender_name=name,
        group_id="g1",
        timestamp=ts if ts is not None else int(time.time()),
        text_content=text,
        contents=(
            MessageContent(type=MessageContentType.TEXT, text=text),
        ),
    )


def make_config(**jev_overrides):
    """构造带 Jev 配置的 ConfigManager。"""
    from src.infrastructure.config.config_manager import ConfigManager

    jev = {
        "enabled": True,
        "proxy_url": "http://127.0.0.1:8770",
        "proxy_protocol": "systemone",
        "gate_enabled": True,
        "threshold": 0.5,
        "max_state_chars": 2000,
        "max_messages": 120,
        "prompt": "",
    }
    jev.update(jev_overrides)
    return ConfigManager({"jev": jev})


class TestBuildState:
    """交给 Jev 的状态拼装。"""

    def test_empty_messages(self):
        from src.infrastructure.jev.gate import build_state

        assert build_state([]) == ""

    def test_message_line_format(self):
        from src.infrastructure.jev.gate import build_state

        state = build_state([make_message(name="张三", text="今天群里聊什么")])
        assert "[Scene]" in state
        assert "[Messages]" in state
        assert "张三: 今天群里聊什么" in state

    def test_truncates_old_messages(self):
        from src.infrastructure.jev.gate import build_state

        messages = [
            make_message(name=f"U{i}", text=f"消息内容{i}" * 40, sender_id=str(i))
            for i in range(30)
        ]
        state = build_state(messages, max_chars=600)
        assert len(state) <= 900
        assert "U29" in state

    def test_max_messages_limit(self):
        from src.infrastructure.jev.gate import build_state

        messages = [
            make_message(name=f"U{i}", text=f"消息{i}", sender_id=str(i))
            for i in range(50)
        ]
        state = build_state(messages, max_messages=5)
        assert "U49" in state
        assert "U0 " not in state


class TestBuildQuestions:
    """判定问题构造。"""

    def test_two_questions(self):
        from src.infrastructure.jev.gate import build_questions

        questions = build_questions("")
        assert set(questions) == {"worth_reporting", "chat_mood"}
        assert questions["worth_reporting"]["type"] == "noul"
        assert questions["chat_mood"]["type"] == "choice"
        assert set(questions["chat_mood"]["criteria"]) >= {"deep_discussion", "flooding"}

    def test_custom_prompt_replaces_instructions(self):
        from src.infrastructure.jev.gate import build_questions

        questions = build_questions("只判断有没有人提了问题")
        assert questions["worth_reporting"]["instructions"] == "只判断有没有人提了问题"
        # 其余题保持内置
        assert questions["chat_mood"]["instructions"] != "只判断有没有人提了问题"


class TestJevGate:
    """预筛判定。"""

    def test_disabled_when_not_enabled(self):
        from src.infrastructure.jev.gate import JevGate

        gate = JevGate(make_config(enabled=False))
        assert gate.is_enabled() is False

    def test_disabled_without_proxy(self):
        from src.infrastructure.jev.gate import JevGate

        gate = JevGate(make_config(proxy_url=""))
        assert gate.is_enabled() is False

    @pytest.mark.asyncio
    async def test_not_enabled_returns_none(self):
        from src.infrastructure.jev.gate import JevGate

        gate = JevGate(make_config(enabled=False))
        decision = await gate.evaluate([make_message()])
        assert decision.worth_reporting is None

    @pytest.mark.asyncio
    async def test_probability_below_threshold(self):
        from src.infrastructure.jev.gate import JevGate

        client = MagicMock()
        client.evaluate = AsyncMock(
            return_value={
                "status": "ready",
                "answers": {
                    "worth_reporting": {"type": "noul", "noul": 0.2},
                    "chat_mood": {"type": "choice", "choice": "flooding"},
                },
            }
        )
        gate = JevGate(make_config(), client)
        decision = await gate.evaluate([make_message()])
        assert decision.worth_reporting is False
        assert decision.probability == 0.2
        assert decision.mood == "刷屏灌水"

    @pytest.mark.asyncio
    async def test_probability_above_threshold(self):
        from src.infrastructure.jev.gate import JevGate

        client = MagicMock()
        client.evaluate = AsyncMock(
            return_value={
                "status": "ready",
                "answers": {
                    "worth_reporting": {"type": "noul", "noul": 0.83},
                    "chat_mood": {"type": "choice", "choice": "deep_discussion"},
                },
            }
        )
        gate = JevGate(make_config(), client)
        decision = await gate.evaluate([make_message()])
        assert decision.worth_reporting is True
        assert decision.mood == "深度讨论"

    @pytest.mark.asyncio
    async def test_failure_falls_through(self):
        """网关调用失败要放行，不能拦住分析。"""
        from src.infrastructure.jev.gate import JevGate

        client = MagicMock()
        client.evaluate = AsyncMock(
            return_value={"status": "error", "retryable": False, "error_message": "网关挂了"}
        )
        gate = JevGate(make_config(), client)
        decision = await gate.evaluate([make_message()])
        assert decision.worth_reporting is None
        assert "网关挂了" in decision.reason

    @pytest.mark.asyncio
    async def test_missing_probability_falls_through(self):
        from src.infrastructure.jev.gate import JevGate

        client = MagicMock()
        client.evaluate = AsyncMock(
            return_value={"status": "ready", "answers": {"worth_reporting": {}}}
        )
        gate = JevGate(make_config(), client)
        decision = await gate.evaluate([make_message()])
        assert decision.worth_reporting is None

    @pytest.mark.asyncio
    async def test_request_shape(self):
        """请求体要符合网关约定：state + questions。"""
        from src.infrastructure.jev.gate import JevGate

        captured = {}

        async def fake_evaluate(cfg, state, questions):
            captured["cfg"] = cfg
            captured["state"] = state
            captured["questions"] = questions
            return {
                "status": "ready",
                "answers": {"worth_reporting": {"type": "noul", "noul": 0.9}},
            }

        client = MagicMock()
        client.evaluate = fake_evaluate
        gate = JevGate(make_config(), client)
        await gate.evaluate([make_message()])
        assert captured["state"].startswith("[Scene]")
        assert "worth_reporting" in captured["questions"]
        assert captured["cfg"]["proxy_url"] == "http://127.0.0.1:8770"
        assert captured["cfg"]["proxy_protocol"] == "systemone"


class TestServiceIntegration:
    """服务接入：预筛判定不值得时跳过分析。"""

    @pytest.mark.asyncio
    async def test_run_jev_gate_returns_none_without_port(self):
        from src.application.services.analysis_application_service import (
            AnalysisApplicationService,
        )

        service = AnalysisApplicationService(
            config_manager=MagicMock(),
            bot_manager=MagicMock(),
            history_manager=MagicMock(),
            report_generator=MagicMock(),
            llm_analyzer=MagicMock(),
            statistics_service=MagicMock(),
            analysis_domain_service=MagicMock(),
        )
        assert await service._run_jev_gate([]) is None

    @pytest.mark.asyncio
    async def test_run_jev_gate_swallows_exception(self):
        from src.application.services.analysis_application_service import (
            AnalysisApplicationService,
        )

        gate = MagicMock()
        gate.evaluate = AsyncMock(side_effect=RuntimeError("boom"))
        service = AnalysisApplicationService(
            config_manager=MagicMock(),
            bot_manager=MagicMock(),
            history_manager=MagicMock(),
            report_generator=MagicMock(),
            llm_analyzer=MagicMock(),
            statistics_service=MagicMock(),
            analysis_domain_service=MagicMock(),
            jev_gate=gate,
        )
        assert await service._run_jev_gate([]) is None

    @pytest.mark.asyncio
    async def test_incremental_skips_when_not_worth(self, dummy_plugin_kv):
        """增量路径：预筛判定不值得时跳过本批次，不调用大模型。"""
        from src.application.services.analysis_application_service import (
            AnalysisApplicationService,
        )
        from src.domain.value_objects.jev_gate import JevGateDecision
        from src.infrastructure.persistence.incremental_store import IncrementalStore

        gate = MagicMock()
        gate.evaluate = AsyncMock(
            return_value=JevGateDecision(
                worth_reporting=False, probability=0.1, reason="概率 0.1"
            )
        )
        llm_analyzer = MagicMock()
        llm_analyzer.analyze_incremental_concurrent = AsyncMock()

        config = MagicMock()
        config.get_incremental_min_messages = MagicMock(return_value=1)
        config.get_max_messages = MagicMock(return_value=500)
        config.get_filter_bot_messages = MagicMock(return_value=False)
        config.get_bot_self_ids = MagicMock(return_value=[])
        config.get_group_platform_id = MagicMock(return_value="onebot_test")
        config.get_llm_max_concurrent = MagicMock(return_value=2)
        config.get_analysis_days = MagicMock(return_value=1)

        adapter = AsyncMock()
        adapter.is_group_muted = AsyncMock(return_value=False)
        adapter.fetch_messages = AsyncMock(return_value=[make_message()])
        bot_manager = MagicMock()
        bot_manager.get_adapter = MagicMock(return_value=adapter)

        service = AnalysisApplicationService(
            config_manager=config,
            bot_manager=bot_manager,
            history_manager=MagicMock(),
            report_generator=MagicMock(),
            llm_analyzer=llm_analyzer,
            statistics_service=MagicMock(),
            analysis_domain_service=MagicMock(),
            incremental_store=IncrementalStore(dummy_plugin_kv),
            jev_gate=gate,
        )

        result = await service.execute_incremental_analysis(
            group_id="g1", platform_id="onebot_test"
        )
        assert result["success"] is False
        assert result["reason"] == "not_worth_reporting"
        assert llm_analyzer.analyze_incremental_concurrent.await_count == 0

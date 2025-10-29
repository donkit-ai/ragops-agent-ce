from __future__ import annotations

from ragops_agent_ce.agent.agent import LLMAgent
from ragops_agent_ce.llm.provider_factory import get_provider


def test_provider_factory_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("RAGOPS_LLM_PROVIDER", raising=False)
    prov = get_provider()
    # Should be mock provider by default
    assert prov.name in {"mock", "base"} or prov.__class__.__name__.lower().startswith("mock")


def test_agent_chat_with_mock_provider(monkeypatch):
    monkeypatch.setenv("RAGOPS_LLM_PROVIDER", "mock")
    prov = get_provider()
    agent = LLMAgent(prov)
    result = agent.chat(prompt="hello world")
    assert "mock" in result.lower()
    assert "hello" in result.lower()

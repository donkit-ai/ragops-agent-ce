from dataclasses import dataclass

from ragops_agent_ce.llm.base import LLMProvider


@dataclass
class AgentSettings:
    llm_provider: LLMProvider
    model: str | None

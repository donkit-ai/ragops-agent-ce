from dataclasses import dataclass

from donkit.llm import LLMModelAbstract


@dataclass
class AgentSettings:
    llm_provider: LLMModelAbstract
    model: str | None

import typing as t
from dataclasses import dataclass
from enum import Enum

from _decimal import Decimal

from flow_prompt.responses import AIResponse


class AI_MODELS_PROVIDER(Enum):
    OPENAI = "openai"
    AZURE = "azure"


@dataclass(kw_only=True)
class AIModel:
    max_tokens: int
    tiktoken_encoding: t.Optional[str] = "cl100k_base"
    provider: AI_MODELS_PROVIDER = None
    support_functions: bool = False

    @property
    def name(self) -> str:
        return "undefined_aimodel"

    @property
    def price_per_prompt_1k_tokens(self) -> Decimal:
        raise NotImplementedError

    @property
    def price_per_sample_1k_tokens(self) -> Decimal:
        raise NotImplementedError

    def get_params(self) -> t.Dict[str, t.Any]:
        return {}

    def call(self, *args, **kwargs) -> AIResponse:
        raise NotImplementedError

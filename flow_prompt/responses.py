from decimal import Decimal
import json
import logging
from dataclasses import dataclass, field
import typing as t

logger = logging.getLogger(__name__)


@dataclass
class Prompt:
    messages: t.List[dict] = None
    functions: t.List[dict] = None
    max_tokens: int = 0
    temperature: float = 0
    top_p: float = 0


@dataclass
class Metrics:
    price_of_call: t.Optional[Decimal] = None
    sample_tokens_used: t.Optional[int] = None
    prompt_tokens_used: t.Optional[int] = None
    ai_model_details: t.Optional[dict] = None
    latency: t.Optional[int] = None


@dataclass(kw_only=True)
class AIResponse:
    _response: str = ""
    original_result: object = None
    content: str = ""
    finish_reason: str = ""
    prompt: Prompt = field(default_factory=Prompt)
    metrics: Metrics = field(default_factory=Metrics)

    @property
    def response(self) -> str:
        return self._response

    def get_message_str(self) -> str:
        return json.loads(self.response)

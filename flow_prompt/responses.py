from decimal import Decimal
import json
import logging
from dataclasses import dataclass, field
import typing as t

logger = logging.getLogger(__name__)


@dataclass
class Prompt:
    messages: t.List[dict] = None


@dataclass
class Metrics:
    price_of_call: t.Optional[Decimal] = None


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

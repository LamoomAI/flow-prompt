import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class AIResponse:
    _response: str = ""
    original_result: object = None

    @property
    def response(self) -> str:
        return self._response

    def get_message_str(self) -> str:
        return json.loads(self.response)

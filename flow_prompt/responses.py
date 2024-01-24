import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AIResponse:
    _response: str = ""

    @property
    def response(self) -> str:
        return self._response

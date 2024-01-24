from dataclasses import dataclass
import json
import typing as t
from openai.types.chat import ChatCompletionMessageToolCall as ToolCall
from openai.types.chat import ChatCompletionMessage as Message

FINISH_REASON_LENGTH = "length"
FINISH_REASON_ERROR = "error"
FINISH_REASON_FINISH = "stop"
FINISH_REASON_TOOL_CALLS = "tool_calls"


logger = logging.getLogger(__name__)


@dataclass
class AIResponse:
    response: str = ""


@dataclass
class OpenAIResponse(FlowPromptResponse):
    content: str = ""
    finish_reason: str = ""
    message: Message = None
    first_response_to_user_ts: t.Optional[int] = None
    exception: t.Optional[Exception] = None

    def __init__(self, **kwargs):
        self.content = kwargs.get("content", "")
        self.finish_reason = kwargs.get("finish_reason", "")
        self.message = kwargs.get("message", None)  

    def is_error_response(self) -> bool:
        return self.finish_reason == FINISH_REASON_ERROR

    def set_error(self, exception: Exception):
        self.finish_reason = FINISH_REASON_ERROR
        self.exception = exception

    def is_stop_response(self) -> bool:
        return self.finish_reason in [
            FINISH_REASON_TOOL_CALLS,
            FINISH_REASON_FINISH,
        ]

    @property
    def response(self) -> str:
        return self.content or self.message.content

    def is_function(self) -> bool:
        return self.finish_reason == FINISH_REASON_TOOL_CALLS

    @property
    def tool_calls(self) -> t.List[ToolCall]:
        return self.message.tool_calls

    def get_function_name(self, tool_call: ToolCall) -> t.Optional[str]:
        if tool_call.type != "function":
            logger.error(f"function.type is not function: {tool_call.type}")
            return None
        return tool_call.function.name

    def get_function_args(self, tool_call: ToolCall) -> t.Dict[str, t.Any]:
        if not self.is_function() or not tool_call.function:
            return {}
        arguments = tool_call.function.arguments
        try:
            return json.loads(arguments)
        except json.JSONDecodeError as e:
            logger.info("Failed to parse function arguments", exc_info=e)
            return {}

    def is_reached_limit(self) -> bool:
        return self.finish_reason == FINISH_REASON_LENGTH

    def to_dict(self) -> t.Dict[str, str]:
        return {
            "finish_reason": self.finish_reason,
            "message": self.message,
            "content": self.content,
        }

    def get_message_str(self) -> str:
        return self.message.model_dump_json(indent=2)

    def __str__(self) -> str:
        result = (
            f"finish_reason: {self.finish_reason}\n"
            f"message: {self.get_message_str()}\n"
        )
        return result


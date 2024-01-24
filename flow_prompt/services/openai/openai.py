import json
import logging
import typing as t

import server.settings as settings
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessage as Message
from openai.types.chat import ChatCompletionMessageToolCall as ToolCall
from server.services.openai.keys import AzureOpenAIKey
from server.services.openai.utils import (openai_raise_custom_exception,
                                          raise_openai_rate_limit_exception)

logger = logging.getLogger(__name__)

FINISH_REASON_LENGTH = "length"
FINISH_REASON_ERROR = "error"
FINISH_REASON_FINISH = "stop"
FINISH_REASON_TOOL_CALLS = "tool_calls"


class LamoomResponse:
    content: str = ""

    def __init__(self, **kwargs):
        self.content = kwargs.get("content", "")

    @property
    def response(self) -> str:
        return self.content

    @property
    def message(self) -> str:
        return self.content

    def is_function(self) -> bool:
        return False


class ChatResponse(LamoomResponse):
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

    def get_message_str(self) -> str:
        return self.message.model_dump_json(indent=2)

    def get_content(self) -> str:
        return self.message.content

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

    def __str__(self) -> str:
        result = (
            f"finish_reason: {self.finish_reason}\n"
            f"message: {self.get_message_str()}\n"
        )
        return result


class OpenAIClient:
    _default_model_max: int = 4001
    _max_model: int = 4001
    _connection_params: t.Optional[t.Dict[str, str]] = None

    def get_connection_parameters(self, azure_openai: AzureOpenAIKey):
        if azure_openai:
            api_key = azure_openai.get("key")
            api_base = azure_openai.get("url")

            return {
                "api_type": "azure",
                "api_version": "2023-07-01-preview",
                "api_base": api_base,
                "api_key": api_key,
            }
        return {
            "api_key": settings.OPENAI_API_KEY,
            "organization": settings.OPENAI_ORG,
        }

    @openai_raise_custom_exception("openai_chat_feature")
    def chat_feature(
        self,
        max_tokens: t.Optional[int],
        messages: t.List[t.Dict[str, str]],
        functions: t.List[t.Dict[str, str]] = None,
        provider_params: t.Dict[str, str] = None,
        connection_params: AzureOpenAIKey = None,
        **kwargs,
    ) -> ChatResponse:
        max_tokens = min(max_tokens, self._max_model)
        common_args = {
            "top_p": 1,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }

        connection_params = self.get_connection_parameters(
            azure_openai=connection_params
        )
        kwargs = {
            **{
                "messages": messages,
            },
            **common_args,
            **connection_params,
            **provider_params,
            **kwargs,
        }
        if functions:
            kwargs["tools"] = functions
            kwargs["tool_choice"] = "auto"
        try:
            api_type = kwargs.pop("api_type", "openai")
            if api_type == "azure":
                client = AzureOpenAI(
                    api_version=kwargs.pop("api_version", "2023-07-01-preview"),
                    azure_endpoint=kwargs.pop("api_base", "https://api.openai.com"),
                    api_key=kwargs.pop("api_key", None),
                )
                kwargs["model"] = kwargs.pop("deployment_name", None)
            else:
                client = OpenAI(
                    organization=kwargs.pop("organization", None),
                    api_key=kwargs.pop("api_key", None),
                    base_url=kwargs.pop("api_base", "https://api.openai.com"),
                )
            result = client.chat.completions.create(
                **kwargs,
            )
            return ChatResponse(
                finish_reason=result.choices[0].finish_reason,
                message=result.choices[0].message,
                conent=result.choices[0].message.content,
            )
        except Exception as e:
            logger.exception("[OPENAI] failed to handle chat stream", exc_info=e)
            raise_openai_rate_limit_exception(e)

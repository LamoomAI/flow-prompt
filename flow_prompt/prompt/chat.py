import json
import logging
import typing as t
import uuid
from dataclasses import dataclass
from flow_prompt.exceptions import ValueIsNotResolvedException
from flow_prompt.utils import resolve

logger = logging.getLogger(__name__)


@dataclass
class ValuesCost:
    values: t.List[str]
    cost: int


@dataclass
class ChatCondition:
    if_exists: t.Optional[str] = None
    if_not_exist: t.Optional[str] = None


class ChatMessage:
    role: str
    content: str
    name: t.Optional[str] = None
    tool_calls: t.Dict[str, str]

    def is_not_empty(self):
        return bool(self.content or self.tool_calls)

    def is_empty(self):
        return not self.is_not_empty()

    def not_tool_calls(self):
        return not (self.tool_calls)

    def __init__(self, **kwargs):
        self.role = kwargs.get("role", "user")
        self.content = kwargs["content"]
        self.name = kwargs.get("name")
        self.tool_calls = kwargs.get("tool_calls") or {}

    def to_dict(self):
        result = {
            "role": self.role,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        return result



# can be multiple value
@dataclass
class ChatsEntity:
    content: str = ""
    role: str = "user"
    name: t.Optional[str] = None
    tool_calls: t.Dict[str, str] = None
    priority: int = 0
    required: bool = False
    is_multiple: bool = False
    while_fits: bool = False
    condition: ChatCondition = ChatCondition()
    call_lambda: t.Optional[t.Callable] = None
    add_in_reverse_order: bool = False
    in_one_message: bool = False
    continue_if_doesnt_fit: bool = False
    add_if_fitted: t.List[str] = None
    label: t.Optional[str] = None
    presentation: t.Optional[str] = None
    last_words: t.Optional[str] = None
    ref_name: t.Optional[str] = None
    ref_value: t.Optional[str] = None
    

    def __post_init__(self):
        self._uuid = uuid.uuid4()

    def resolve(self, context: t.Dict[str, t.Any]) -> t.List[ChatMessage]:
        result = []
        content = self.content
        if self.is_multiple:
            # should be just one value like {messages} in prompt
            prompt_value = content.strip().replace("{", "").replace("}", "").strip()
            values = context.get(prompt_value, [])
            if not values:
                return []
            if not isinstance(values, list):
                raise ValueIsNotResolvedException(
                    f"Invalid value {values } for prompt {content}. Should be multiple"
                )
            else:
                # verify that values are json list of ChatMessage
                try:
                    result = [
                        ChatMessage(**({"content": c} if isinstance(c, str) else c))
                        for c in values
                    ]
                except TypeError as e:
                    raise ValueIsNotResolvedException(
                        f"Invalid value { values } for prompt {content}. Error: {e}"
                    )
            return result

        content = resolve(content, context)
        return [
            ChatMessage(
                name=self.name,
                role=self.role,
                content=content,
                tool_calls=self.tool_calls,
                ref_name=self.ref_name,
                ref_value=self.ref_value,
            )
        ]

    def validate(self, values: t.List[ChatMessage], context: t.Dict[str, str]):
        if self.condition.if_exists and not context.get(self.condition.if_exists):
            raise ValueIsNotResolvedException(
                f"If exists condition is not met for {self.prompt}"
            )
        if self.condition.if_not_exist and context.get(self.condition.if_not_exist):
            raise ValueIsNotResolvedException(
                f"If not exists condition is not met for {self.prompt}"
            )

    def get_values(self, context: t.Dict[str, str]) -> t.List[ChatMessage]:
        try:
            values = self.resolve(context)
            if self.call_lambda:
                values = self.call_lambda(values)
            self.validate(values, context)
        except Exception as e:
            logger.error(
                f"Error resolving prompt {self.prompt}, error: {e}", exc_info=True
            )
            return []
        return values

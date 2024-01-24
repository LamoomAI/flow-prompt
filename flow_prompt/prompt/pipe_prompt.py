from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import logging
import typing as t
from flow_prompt.prompt.chat import ChatCondition, ChatsEntity
from flow_prompt.prompt.user_prompt import UserPrompt
from dataclasses import dataclass, field
from flow_prompt.ai_models.attempt_to_call import AttemptToCall
from flow_prompt import PIPE_PROMPTS


logger = logging.getLogger(__name__)

@dataclass(kw_only=True)
class BasePrompt:
    priorities: t.Dict[int, t.List[ChatsEntity]] = field(
        default_factory=lambda: defaultdict(list)
    )
    pipe: t.List[ChatsEntity] = field(default_factory=list)
    functions: t.List[dict] = None

    def add(
        self,
        prompt: str = "",
        role: str = "user",
        name: t.Optional[str] = None,
        content: str = "",
        tool_calls: t.Dict[str, str] = None,
        priority: int = 0,
        required: bool = False,
        if_exists: t.Optional[str] = None,
        if_not_exist: t.Optional[str] = None,
        call_lambda: t.Optional[t.Callable] = None,
        is_multiple: bool = False,
        while_fits: bool = False,
        add_in_reverse_order: bool = False,
        in_one_message: bool = False,
        continue_if_doesnt_fit: bool = False,
        add_if_fitted: t.List[str] = None,
        label: t.Optional[str] = None,
        presentation: t.Optional[str] = None,
        last_words: t.Optional[str] = None,
    ):
        condition = ChatCondition(if_exists, if_not_exist)
        chat_value = ChatsEntity(
            role=role,
            prompt=(prompt or content or ""),
            name=name,
            tool_calls=tool_calls,
            priority=priority,
            required=required,
            condition=condition,
            call_lambda=call_lambda,
            is_multiple=is_multiple,
            while_fits=while_fits,
            add_in_reverse_order=add_in_reverse_order,
            in_one_message=in_one_message,
            continue_if_doesnt_fit=continue_if_doesnt_fit,
            add_if_fitted=add_if_fitted,
            label=label,
            presentation=presentation,
            last_words=last_words,
        )
        self.priorities[priority].append(chat_value)
        self.pipe.append(chat_value)

    def add_function(self, function: dict):
        if not self.functions:
            self.functions = []
        self.functions.append(function)


@dataclass(kw_only=True)
class PipePrompt(BasePrompt):
    '''
    PipePrompt is a class that represents a pipe of chats that will be used to generate a prompt.
    You can add chats with different priorities to the pipe thinking just about the order of chats.
    When you initialize a Prompt, chats will be sorted by priority and then by order of adding.
    '''
    id: str
    max_sample_rokens: int = None
    min_sample_tokens: int = None
    max_prompt_tokens: int = None

    def __post_init__(self):
        PIPE_PROMPTS[self.id] = self


    def create_prompt(self, ai_attempt: AttemptToCall) -> UserPrompt:
        return UserPrompt(
            pipe=deepcopy(self.pipe),
            priorities=deepcopy(self.priorities),
            model_encoding=ai_attempt.model_encoding(),
            model_max_tokens=ai_attempt.model_max_tokens(),
        )
    
 
import logging
import typing as t
from collections import defaultdict
from dataclasses import dataclass, field

import tiktoken

from flow_prompt import settings
from flow_prompt.exceptions import NotEnoughBudgetError
from flow_prompt.prompt.base_prompt import BasePrompt
from flow_prompt.prompt.chat import ChatMessage, ChatsEntity

logger = logging.getLogger(__name__)


@dataclass
class State:
    """
    State of the prompt. left_budget is the budget left for the rest of the prompt.
    fully_fitted_pipitas is the set of labels of chats that were fully fitted in the prompt.
    Pipita references to a small part of pipe, formed with a Spanish ending 'ita' which means a smaller version.
    """

    left_budget: int = 0
    fully_fitted_pipitas: t.Set[str] = field(default_factory=set)
    references: t.Dict[str, t.List[str]] = field(
        default_factory=lambda: defaultdict(list)
    )


@dataclass
class CallingMessages:
    messages: t.List[ChatMessage]
    prompt_budget: int = 0
    left_budget: int = 0
    references: t.Dict[str, t.List[str]] = None
    max_sample_budget: int = 0

    @property
    def calling_messages(self) -> t.List[t.Dict[str, str]]:
        return [m.to_dict() for m in self.messages if not m.is_empty()]

    def get_messages(self) -> t.List[t.Dict[str, str]]:
        result = []
        for m in self.messages:
            if m.is_empty():
                continue
            result.append(m.to_dict())
        return result

    def __str__(self) -> str:
        return "\n".join([str(m.to_dict()) for m in self.messages])


@dataclass(kw_only=True)
class UserPrompt(BasePrompt):
    model_max_tokens: int
    tiktoken_encoding: str
    min_sample_tokens: int
    reserved_tokens_budget_for_sampling: int = None
    safe_gap_tokens: int = settings.SAFE_GAP_TOKENS

    def __post_init__(self):
        self.encoding = tiktoken.get_encoding(self.tiktoken_encoding)

    def resolve(self, context: t.Dict, images: t.Dict[str, t.Any], files: t.Dict[str, t.Any]) -> CallingMessages:
        pipe = {}
        prompt_budget = 0
        ordered_pipe = dict((value, i) for i, value in enumerate(self.pipe))
        state = State()
        state.left_budget = self.left_budget
        for priority in sorted(self.priorities.keys()):
            for chat_value in self.priorities[priority]:
                r = [
                    p in state.fully_fitted_pipitas
                    for p in (chat_value.add_if_fitted_labels or [])
                ]
                if not all(r):
                    continue

                if chat_value.presentation:
                    state.left_budget -= len(
                        self.encoding.encode(chat_value.presentation)
                    )
                if chat_value.last_words:
                    state.left_budget -= len(
                        self.encoding.encode(chat_value.last_words)
                    )

                values = chat_value.get_values(context, images, files)
                logger.debug(f"Got values for {chat_value}: {values}")

                if not values:
                    continue
                if chat_value.in_one_message:
                    messages_budget, messages = self.add_values_in_one_message(
                        values, chat_value, state
                    )
                elif chat_value.while_fits:
                    messages_budget, messages = self.add_values_while_fits(
                        values,
                        chat_value,
                        state,
                    )
                else:
                    messages_budget, messages = self.add_values(values, state)
                    if chat_value.label:
                        state.fully_fitted_pipitas.add(chat_value.label)
                
                if not messages:
                    logger.debug(f"messages is empty for {chat_value}")
                    continue
                if not self.is_enough_budget(state, messages_budget):
                    logger.debug(f"not enough budget for {chat_value}")
                    if chat_value.required:
                        raise NotEnoughBudgetError("Not enough budget")
                    continue
                logger.debug(f"adding {len(messages)} messages for {chat_value}")
                state.left_budget -= messages_budget
                prompt_budget += messages_budget

                if chat_value.presentation:
                    if messages[0].content_type == "text":
                        messages[0].content = chat_value.presentation + messages[0].content
                    else:
                        messages = [ChatMessage(content=chat_value.presentation)] + messages
                if chat_value.last_words:
                    if messages[-1].content_type == "text":
                        messages[-1].content += chat_value.last_words
                    else:
                        messages += [ChatMessage(content=chat_value.last_words)]
                pipe[chat_value._uuid] = messages
                continue

        final_pipe_with_order = [
            pipe.get(chat_id, [])
            for chat_id, _ in sorted(ordered_pipe.items(), key=lambda x: x[1])
        ]
        # skip empty values
        flat_list: t.List[ChatMessage] = [
            item for sublist in final_pipe_with_order for item in sublist if item
        ]
        max_sample_budget = left_budget = state.left_budget + self.min_sample_tokens
        if self.reserved_tokens_budget_for_sampling:
            max_sample_budget = min(
                self.reserved_tokens_budget_for_sampling, left_budget
            )

        return CallingMessages(
            references=state.references,
            messages=flat_list,
            prompt_budget=prompt_budget,
            left_budget=left_budget,
            max_sample_budget=max_sample_budget,
        )

    def add_values_while_fits(
        self,
        values: list[ChatMessage],
        chat_value: ChatsEntity,
        state: State,
    ) -> t.Tuple[int, t.List[ChatMessage]]:
        add_in_reverse_order = chat_value.add_in_reverse_order
        if add_in_reverse_order:
            values = values[::-1]
        values_to_add = []
        messages_budget = 0
        is_fully_fitted = True
        if not values:
            logger.debug(
                f"[{self.task_name}]: values to add is empty {chat_value.content}"
            )
        for i, value in enumerate(values):
            if not self.is_value_not_empty(value):
                continue
            one_budget = self.calculate_budget_for_value(value)

            if not self.is_enough_budget(state, one_budget + messages_budget):
                is_fully_fitted = False
                logger.debug(
                    f"not enough budget:{chat_value.content[:30]} with index {i},"
                    " for while_fits, breaking the loop"
                )
                left_budget = state.left_budget - messages_budget
                if (
                    chat_value.continue_if_doesnt_fit
                    and left_budget > settings.EXPECTED_MIN_BUDGET_FOR_VALUABLE_INPUT
                ):
                    continue
                break
            messages_budget += one_budget
            values_to_add.append(value)
            if value.ref_name and value.ref_value:
                state.references[value.ref_name].append(value.ref_value)
        if is_fully_fitted and chat_value.label:
            state.fully_fitted_pipitas.add(chat_value.label)
        if add_in_reverse_order:
            values_to_add = values_to_add[::-1]
        return messages_budget, values_to_add

    def is_enough_budget(self, state: State, required_budget: int) -> bool:
        return state.left_budget >= required_budget

    def add_values_in_one_message(
        self,
        values: list[ChatMessage],
        chat_value: ChatsEntity,
        state: State,
    ) -> t.Tuple[int, t.List[ChatMessage]]:
        one_message_budget = 0
        one_message: t.List[ChatMessage] = []
        is_fully_fitted = True
        if not values:
            logger.debug(
                f"[{self.task_name}]: values to add is empty {chat_value.content}"
            )

        for i, value in enumerate(values):
            if not self.is_value_not_empty(value):
                continue
            one_budget = self.calculate_budget_for_value(value)
            if not self.is_enough_budget(state, one_budget + one_message_budget):
                is_fully_fitted = False
                logger.debug(
                    f"not enough budget:\n{chat_value.content[:30]} with index {i},"
                    f" for while_fits, breaking the loop."
                    f" Budget required: {one_budget}, "
                    f"left: {state.left_budget - one_message_budget}"
                )

                left_budget = state.left_budget - one_message_budget
                if (
                    chat_value.continue_if_doesnt_fit
                    and left_budget > settings.EXPECTED_MIN_BUDGET_FOR_VALUABLE_INPUT
                ):
                    continue
                break

            one_message_budget += one_budget
            if len(one_message) > 0:
                if one_message[-1].content_type == "text" and value.content_type == "text":
                    one_message[-1].content += "\n" + value.content
                else:
                    one_message.append(value)
            else:
                one_message.append(value)

            if value.ref_name and value.ref_value:
                state.references[value.ref_name].append(value.ref_value)
        if is_fully_fitted and chat_value.label:
            state.fully_fitted_pipitas.add(chat_value.label)
        return one_message_budget, one_message

    @property
    def left_budget(self) -> int:
        return self.model_max_tokens - self.min_sample_tokens - self.safe_gap_tokens

    def calculate_budget_for_value(self, value: ChatMessage) -> int:
        content = 0
        if value.content_type == "text":
            content = len(self.encoding.encode(value.content))
        elif value.content_type == "image":
            # TODO: add tokens calculation for image
            content = 0
        elif value.content_type == "file":
            # TODO: add tokens calculation for file
            content = 0

        role = len(self.encoding.encode(value.role))
        tool_calls = len(self.encoding.encode(value.tool_calls.get("name", "")))
        arguments = len(self.encoding.encode(value.tool_calls.get("arguments", "")))
        return content + role + tool_calls + arguments + settings.SAFE_GAP_PER_MSG

    def is_value_not_empty(self, value: ChatMessage) -> bool:
        if value.content is None:
            return False
        return True

    def add_values(
        self,
        values: t.List[ChatMessage],
        state: State,
    ) -> t.Tuple[int, t.List[ChatMessage]]:
        budget = 0
        result = []

        for value in values:
            if not self.is_value_not_empty(value):
                logger.debug(f"[{self.task_name}]: is_value_not_empty failed {value}")
                continue

            budget += self.calculate_budget_for_value(value)
            result.append(value)
            if value.ref_name and value.ref_value:
                state.references[value.ref_name].append(value.ref_value)

        return budget, result

    def __str__(self) -> str:
        result = ""
        for chat_value in self.pipe:
            result += f"{chat_value}\n"
        return result

    def to_dict(self) -> dict:
        return [chat_value.to_dict() for chat_value in self.pipe]

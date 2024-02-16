import logging
import typing as t
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from flow_prompt import settings
from flow_prompt.ai_models.ai_model import AI_MODELS_PROVIDER, AIModel
from flow_prompt.ai_models.openai.responses import OpenAIResponse
from flow_prompt.exceptions import ProviderNotFoundException

from .utils import raise_openai_exception

C_4K = 4096
C_8K = 8192
C_128K = 127_000
C_16K = 16384
C_32K = 32768
M_DAVINCI = "davinci"

logger = logging.getLogger(__name__)


class FamilyModel(Enum):
    chat = "GPT-3.5"
    gpt4 = "GPT-4"
    instruct_gpt = "InstructGPT"


OPEN_AI_PRICING = {
    FamilyModel.chat.value: {
        C_4K: {
            "price_per_prompt_1k_tokens": Decimal(0.0015),
            "price_per_sample_1k_tokens": Decimal(0.0020),
        },
        C_16K: {
            "price_per_prompt_1k_tokens": Decimal(0.0010),
            "price_per_sample_1k_tokens": Decimal(0.0020),
        },
        C_128K: {
            "price_per_prompt_1k_tokens": Decimal(0.01),
            "price_per_sample_1k_tokens": Decimal(0.03),
        },
    },
    FamilyModel.gpt4.value: {
        C_4K: {
            "price_per_prompt_1k_tokens": Decimal(0.03),
            "price_per_sample_1k_tokens": Decimal(0.06),
        },
        C_32K: {
            "price_per_prompt_1k_tokens": Decimal(0.06),
            "price_per_sample_1k_tokens": Decimal(0.12),
        },
        C_128K: {
            "price_per_prompt_1k_tokens": Decimal(0.01),
            "price_per_sample_1k_tokens": Decimal(0.03),
        },
    },
    FamilyModel.instruct_gpt.value: {
        M_DAVINCI: {
            "price_per_prompt_1k_tokens": Decimal(0.0015),
            "price_per_sample_1k_tokens": Decimal(0.002),
        },
    },
}


@dataclass(kw_only=True)
class OpenAIModel(AIModel):
    model: t.Optional[str]
    support_functions: bool = False
    provider: AI_MODELS_PROVIDER = AI_MODELS_PROVIDER.OPENAI
    family: str = None
    should_verify_client_has_creds: bool = True
    max_sample_budget: int = C_4K

    def __str__(self) -> str:
        return f"openai-{self.model}-{self.family}"

    def __post_init__(self):
        if self.model.startswith("davinci"):
            self.family = FamilyModel.instruct_gpt.value
        elif self.model.startswith("gpt-3"):
            self.family = FamilyModel.chat.value
        elif self.model.startswith(("gpt-4", "gpt")):
            self.family = FamilyModel.gpt4.value
        else:
            logger.warning(
                f"Unknown family for {self.model}. Please add it obviously. Setting as GPT4"
            )
            self.family = FamilyModel.gpt4.value
        if self.should_verify_client_has_creds:
            self.verify_client_has_creds()
        logger.info(f"Initialized OpenAIModel: {self}")

    def verify_client_has_creds(self):
        if self.provider not in settings.AI_CLIENTS:
            raise ProviderNotFoundException(
                f"Provider {self.provider} not found in AI_CLIENTS"
            )

    @property
    def name(self) -> str:
        return self.model

    @property
    def price_per_prompt_1k_tokens(self) -> Decimal:
        return OPEN_AI_PRICING[self.family][self.max_tokens][
            "price_per_prompt_1k_tokens"
        ]

    @property
    def price_per_sample_1k_tokens(self) -> Decimal:
        return OPEN_AI_PRICING[self.family][self.max_tokens][
            "price_per_sample_1k_tokens"
        ]

    def get_params(self) -> t.Dict[str, t.Any]:
        return {
            "model": self.model,
        }

    def call(self, messages, max_tokens, **kwargs) -> OpenAIResponse:
        logger.debug(
            f"Calling {messages} with max_tokens {max_tokens} and kwargs {kwargs}"
        )
        if self.family in [FamilyModel.chat.value, FamilyModel.gpt4.value]:
            return self.call_chat_completion(messages, max_tokens, **kwargs)
        raise NotImplementedError(f"Openai family {self.family} is not implemented")

    def get_client(self):
        return settings.AI_CLIENTS[self.provider]

    def call_chat_completion(
        self,
        messages: t.List[t.Dict[str, str]],
        max_tokens: t.Optional[int],
        functions: t.List[t.Dict[str, str]] = [],
        **kwargs,
    ) -> OpenAIResponse:
        max_tokens = min(max_tokens, self.max_tokens, self.max_sample_budget)
        common_args = {
            "top_p": 1,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        kwargs = {
            **{
                "messages": messages,
            },
            **common_args,
            **self.get_params(),
            **kwargs,
        }
        if functions:
            kwargs["tools"] = functions
        try:
            client = self.get_client()
            result = client.chat.completions.create(
                **kwargs,
            )
            logger.debug(f"Result: {result.choices[0]}")
            return OpenAIResponse(
                finish_reason=result.choices[0].finish_reason,
                message=result.choices[0].message,
                content=result.choices[0].message.content,
                original_result=result,
                prompt_messages=kwargs.get("messages"),
            )
        except Exception as e:
            logger.exception("[OPENAI] failed to handle chat stream", exc_info=e)
            raise_openai_exception(e)

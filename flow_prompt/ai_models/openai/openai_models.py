
from dataclasses import dataclass
from enum import Enum
import logging
from ..ai_model import AIModel, AI_MODELS_PROVIDER
import typing as t
from decimal import Decimal
from openai import AzureOpenAI, OpenAI
from flow_prompt.responses import FlowPromptResponse

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
            "price_per_prompt_1k_tokens": Decimal(0.02),
            "price_per_sample_1k_tokens": Decimal(0.02),
        },
    },
}


@dataclass
class OpenAIModel(AIModel):
    model: t.Optional[str]
    support_functions: bool = False
    provider: AI_MODELS_PROVIDER = AI_MODELS_PROVIDER.OPENAI

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
            "model": self.provider.engine.model,
        }
    
    def call(self, *args, **kwargs) -> FlowPromptResponse:
        if self.family in [FamilyModel.chat.value, FamilyModel.gpt4.value]:
            return self.call_chat_completion(**kwargs)
        raise NotImplementedError(f"family {self.family} is not implemented")

    def get_client(self, **kwargs):
        return OpenAI(
            organization=kwargs.pop("organization", None),
            api_key=kwargs.pop("api_key", None),
            base_url=kwargs.pop("api_base", "https://api.openai.com"),
        )
    
    def call_chat_completion(
        self,
        max_tokens: t.Optional[int],
        messages: t.List[t.Dict[str, str]],
        functions: t.List[t.Dict[str, str]] = None,
        provider_params: t.Dict[str, str] = None,
        **kwargs,
    ) -> FlowPromptResponse:
        max_tokens = min(max_tokens, self._max_model)
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
            **provider_params,
            **kwargs,
        }
        if functions:
            kwargs["tools"] = functions
        try:
            client = self.get_client(**kwargs)
            result = client.chat.completions.create(
                **kwargs,
            )
            return FlowPromptResponse(
                finish_reason=result.choices[0].finish_reason,
                message=result.choices[0].message,
                conent=result.choices[0].message.content,
            )
        except Exception as e:
            logger.exception("[OPENAI] failed to handle chat stream", exc_info=e)
            raise_openai_exception(e)



@dataclass
class AzureAIModel(OpenAIModel):
    realm: t.Optional[str]
    deployment_name: t.Optional[str]
    provider: AI_MODELS_PROVIDER = AI_MODELS_PROVIDER.AZURE

    def name(self) -> str:
        return f'{self.deployment_name}-{self.realm}'
    
    def get_params(self) -> t.Dict[str, t.Any]:
        return {
            "model": self.deployment_name,
        }

    def get_client(self, **kwargs):
        return AzureOpenAI(
            api_version=kwargs.pop("api_version", "2023-07-01-preview"),
            azure_endpoint=kwargs.pop("api_base", "https://api.openai.com"),
            api_key=kwargs.pop("api_key", None),
        )

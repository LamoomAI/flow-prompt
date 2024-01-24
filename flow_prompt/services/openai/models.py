import typing as t
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

C_4K = 4096
C_8K = 8192
C_16K = 16384
C_32K = 32768
M_DAVINCI = "davinci"


class FamilyModel(Enum):
    chat = "GPT-3.5"
    gpt4 = "GPT-4"
    instruct_gpt = "InstructGPT"


OPEN_AI_PRICING = {
    FamilyModel.chat.value: {
        C_4K: {
            "price_per_prompt_1k_tokens": Decimal(0.0015),
            "price_per_sample_1k_tokens": Decimal(0.002),
        },
        C_8K: {
            "price_per_prompt_1k_tokens": Decimal(0.003),
            "price_per_sample_1k_tokens": Decimal(0.004),
        },
        C_16K: {
            "price_per_prompt_1k_tokens": Decimal(0.003),
            "price_per_sample_1k_tokens": Decimal(0.004),
        },
    },
    FamilyModel.gpt4.value: {
        C_8K: {
            "price_per_prompt_1k_tokens": Decimal(0.03),
            "price_per_sample_1k_tokens": Decimal(0.06),
        },
        C_32K: {
            "price_per_prompt_1k_tokens": Decimal(0.06),
            "price_per_sample_1k_tokens": Decimal(0.12),
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
class OpenAIEngine:
    family: str
    max_tokens: int
    model: t.Optional[str] = None
    azure_deployment_name: t.Optional[str] = None
    model_tokenizer: t.Optional[str] = "gpt-3.5-turbo"
    safe_gap_multiplier: int = 1
    support_functions: bool = False

    @property
    def price_per_prompt_1k_tokens(self) -> Decimal:
        return OPEN_AI_PRICING[self.family][self.max_tokens][
            "price_per_prompt_1k_tokens"
        ]

    @property
    def price_per_sample_1k_tokens(self) -> Decimal:
        return OPEN_AI_PRICING[self.family][self.max_tokens][
            "price_per_prompt_1k_tokens"
        ]


class Engines(Enum):
    GPT_3_5_NAMES = OpenAIEngine(
        FamilyModel.chat.value,
        C_4K,
        model="ft:gpt-3.5-turbo-0613:machinet::80qCz2o8",
        model_tokenizer="gpt-3.5-turbo",
    )
    GPT_3_5_TURBO = OpenAIEngine(
        FamilyModel.chat.value,
        C_4K,
        model="gpt-3.5-turbo",
        azure_deployment_name="gpt",
        model_tokenizer="gpt-3.5-turbo",
    )
    GPT_3_5_TURBO_8K = OpenAIEngine(
        FamilyModel.chat.value,
        C_8K,
        model="gpt-3.5-turbo-16k",
        azure_deployment_name="gpt-35-turbo-16k",
        model_tokenizer="gpt-3.5-turbo",
        support_functions=True,
    )
    GPT_3_5_TURBO_16K = OpenAIEngine(
        FamilyModel.chat.value,
        C_16K,
        model="gpt-3.5-turbo-16k",
        azure_deployment_name="gpt-35-turbo-16k",
        support_functions=True,
        model_tokenizer="gpt-3.5-turbo",
    )
    GPT_3_5_TURBO_16K_0613 = OpenAIEngine(
        FamilyModel.chat.value,
        C_16K,
        model="gpt-3.5-turbo-16k-0613",
        azure_deployment_name="gpt-35-turbo-16k",
        support_functions=True,
        model_tokenizer="gpt-3.5-turbo",
    )
    GPT_3_5_TURBO_16K_0613_OPENAI = OpenAIEngine(
        FamilyModel.chat.value,
        C_16K,
        model="gpt-3.5-turbo-16k-0613",
        azure_deployment_name="gpt-3.5-turbo-16k",
        support_functions=True,
        model_tokenizer="gpt-3.5-turbo",
    )
    GPT_3_5_TURBO_AZURE = OpenAIEngine(
        FamilyModel.chat.value,
        C_8K,
        model="gpt-3.5-turbo-16k",
        azure_deployment_name="gpt-35-turbo-16k",
        model_tokenizer="gpt-3.5-turbo",
    )
    # 8k tokens, for 32k tokens pricing is different
    GPT_4 = OpenAIEngine(
        FamilyModel.gpt4.value,
        C_8K,
        model="gpt-4-0613",
        azure_deployment_name="gpt-4",
        model_tokenizer="gpt-3.5-turbo",
        support_functions=True,
    )
    GPT_4_32K = OpenAIEngine(
        FamilyModel.gpt4.value,
        C_32K,
        model="gpt-4-32k",
        azure_deployment_name="gpt-4-32k",
        model_tokenizer="gpt-3.5-turbo",
        support_functions=True,
    )
    TEXT_DAVINCI = OpenAIEngine(
        FamilyModel.instruct_gpt.value,
        C_4K,
        model="text-davinci",
        azure_deployment_name="text-davinci-003",
        model_tokenizer=M_DAVINCI,
    )


def get_engine_by_model(model_name: t.Optional[str] = None) -> OpenAIEngine:
    """
    Returns the OpenAIEngine object for the given model name.
    Returns None if the model name is not found in Engines.
    """
    if model_name == Engines.GPT_3_5_TURBO.value.model:
        return Engines.GPT_3_5_TURBO_AZURE.value

    for engine in Engines:
        if engine.value.model == model_name:
            return engine.value

    return Engines.GPT_3_5_TURBO_AZURE.value

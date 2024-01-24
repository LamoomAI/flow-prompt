from enum import Enum
from flow_prompt.ai_models.ai_model import C_128K, C_32K, C_4K, M_DAVINCI, AIModel, FamilyModel, OpenAIModel


class AI_MODELS(Enum):
    GPT_4_TURBO = OpenAIModel(
        FamilyModel.gpt4.value,
        C_128K,
        model="gpt-4-1106-preview",
        azure_deployment_name="gpt-4-turbo",
        model_tokenizer="gpt-3.5-turbo",
        support_functions=True,
    )
    GPT_4_32K = AIModel(
        FamilyModel.gpt4.value,
        C_32K,
        model="gpt-4-32k",
        azure_deployment_name="gpt-4-32k",
        model_tokenizer="gpt-3.5-turbo",
        support_functions=True,
    )
    TEXT_DAVINCI = AIModel(
        FamilyModel.instruct_gpt.value,
        C_4K,
        model="text-davinci",
        azure_deployment_name="text-davinci",
        model_tokenizer=M_DAVINCI,
    )
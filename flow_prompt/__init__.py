from flow_prompt.prompt.flow_prompt import FlowPrompt
from flow_prompt.ai_models import behaviour
from flow_prompt.secrets import AZURE_KEYS, OPENAI_API_KEY, OPENAI_ORG
from flow_prompt.ai_models.openai.openai_behaviours import (
    OPENAI_GPT4_0125_PREVIEW_BEHAVIOUR, OPENAI_GPT4_1106_PREVIEW_BEHAVIOUR,
    OPENAI_GPT4_1106_VISION_PREVIEW_BEHAVIOUR, OPENAI_GPT4_BEHAVIOUR, 
    OPENAI_GPT4_32K_BEHAVIOUR, OPENAI_GPT3_5_TURBO_0125_BEHAVIOUR, OPENAI_GPT3_5_TURBO_INSTRUCT_BEHAVIOUR
)
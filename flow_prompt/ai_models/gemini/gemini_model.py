from flow_prompt.ai_models.ai_model import AI_MODELS_PROVIDER, AIModel
import logging

from flow_prompt.responses import AIResponse
from decimal import Decimal
from enum import Enum

import typing as t
from dataclasses import dataclass

from flow_prompt.ai_models.gemini.responses import GeminiAIResponse
from flow_prompt.ai_models.gemini.constants import *
from openai.types.chat import ChatCompletionMessage as Message
from flow_prompt.responses import Prompt
from flow_prompt.settings import Secrets
from flow_prompt.exceptions import RetryableCustomException, ConnectionLostException
import google.generativeai as genai 

logger = logging.getLogger(__name__)



C_128K = 127_000

class FamilyModel(Enum):
    flash = "Gemini 1.5 Flash"
    pro = "Gemini 1.5 Pro"

DEFAULT_PRICING = {
    "price_per_prompt_1k_tokens": Decimal(0.00035),
    "price_per_sample_1k_tokens": Decimal(0.00105),
}

GEMINI_AI_PRICING = {
    FamilyModel.flash.value: {
        C_128K: {
            "price_per_prompt_1k_tokens": Decimal(0.00035),
            "price_per_sample_1k_tokens": Decimal(0.00105),
        }
    },
    FamilyModel.pro.value: {
        C_128K: {
            "price_per_prompt_1k_tokens": Decimal(0.0035),
            "price_per_sample_1k_tokens": Decimal(0.0105),
        }
    }
}


@dataclass(kw_only=True)
class GeminiAIModel(AIModel):
    model_name: str
    model: genai.GenerativeModel = None
    api_key: str = None
    provider: AI_MODELS_PROVIDER = AI_MODELS_PROVIDER.GEMINI
    family: str = None
    
    def __post_init__(self):
        if FLASH in self.model_name:
            self.family = FamilyModel.flash.value
        elif PRO in self.model_name:
            self.family = FamilyModel.pro.value
        else:
            logger.warning(
                f"Unknown family for {self.model_name}. Please add it obviously. Setting as Gemini 1.5 Flash"
            )
            self.family = FamilyModel.flash.value

        logger.debug(f"Initialized GeminiAIModel: {self}")
        
        secrets = Secrets()
        
        if self.api_key is None:
            self.api_key = secrets.GEMINI_API_KEY
        
        genai.configure(api_key=self.api_key)
        
        self.model = genai.GenerativeModel(self.model_name)
        

    def call(self, messages: t.List[dict], max_tokens: int, **kwargs) -> AIResponse:
        
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
        
        logger.debug(
            f"Calling {messages} with max_tokens {max_tokens} and kwargs {kwargs}"
        )
        
        stream_function = kwargs.get('stream_function')
        check_connection = kwargs.get('check_connection')
        stream_params = kwargs.get('stream_params')
        
        # Parse only prompt content due to gemini call specifics
        prompts = [obj['content'] for obj in messages if obj['role'] == 'user']
        
        content = ""
        
        try:
            if kwargs.get('stream'):
                for prompt in prompts:
                    response = self.model.generate_content(prompt, stream=False)
                    content = response.text
                    stream_function(content, **stream_params)
            else:
                idx = 0
                for prompt in prompts:
                    if idx % 5 == 0:
                        if not check_connection(**stream_params):
                            raise ConnectionLostException("Connection was lost!")  
                    response = self.model.generate_content(prompt, stream=True)
                    for chunk in response:
                        stream_function(chunk.text, **stream_params)
                        content += chunk.text
                    idx += 1
            
            return GeminiAIResponse(
                message=Message(
                    content=content,
                    role="assistant"
                ),
                content=content,
                prompt=Prompt(
                    messages=kwargs.get("messages"),
                    functions=kwargs.get("tools"),
                    max_tokens=max_tokens,
                    temperature=kwargs.get("temperature"),
                    top_p=kwargs.get("top_p"),
                )
            )
            
        except Exception as e:
            logger.exception("[GEMINIAI] failed to handle chat stream", exc_info=e)
            raise RetryableCustomException(f"Gemini AI call failed!")

    def name(self) -> str:
        return self.model_name

    @property
    def price_per_prompt_1k_tokens(self) -> Decimal:
        return GEMINI_AI_PRICING[self.family].get(self.max_tokens, DEFAULT_PRICING)[
            "price_per_prompt_1k_tokens"
        ]

    @property
    def price_per_sample_1k_tokens(self) -> Decimal:
        return GEMINI_AI_PRICING[self.family].get(self.max_tokens, DEFAULT_PRICING)[
            "price_per_sample_1k_tokens"
        ]

    def get_params(self) -> t.Dict[str, t.Any]:
        return {
            'api_key': self.api_key,
            'model': self.model_name,
            'max_tokens': self.max_tokens,
        }

    def get_metrics_data(self) -> t.Dict[str, t.Any]:
        return {
            'model': self.model_name,
            'max_tokens': self.max_tokens,
        }
        

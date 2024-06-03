from flow_prompt.ai_models.ai_model import AI_MODELS_PROVIDER, AIModel
import logging
import os

from flow_prompt.responses import AIResponse
from decimal import Decimal
from enum import Enum

import typing as t
from dataclasses import dataclass

from flow_prompt.ai_models.claude.responses import ClaudeAIReponse
from openai.types.chat import ChatCompletionMessage as Message
from flow_prompt.responses import Prompt
from flow_prompt.settings import Secrets
from flow_prompt.exceptions import RetryableCustomException
import anthropic 

logger = logging.getLogger(__name__)



C_200K = 200000

class FamilyModel(Enum):
    haiku = "Claude 3 Haiku"
    sonnet = "Claude 3 Sonnet"
    opus = "Claude 3 Opus"

DEFAULT_PRICING = {
    "price_per_prompt_1k_tokens": Decimal(0.01),
    "price_per_sample_1k_tokens": Decimal(0.03),
}

CLAUDE_AI_PRICING = {
    FamilyModel.haiku.value: {
        C_200K: {
            "price_per_prompt_1k_tokens": Decimal(0.00125),
            "price_per_sample_1k_tokens": Decimal(0.00125),
        }
    },
    FamilyModel.sonnet.value: {
        C_200K: {
            "price_per_prompt_1k_tokens": Decimal(0.015),
            "price_per_sample_1k_tokens": Decimal(0.015),
        }
    },
    FamilyModel.opus.value: {
        C_200K: {
            "price_per_prompt_1k_tokens": Decimal(0.075),
            "price_per_sample_1k_tokens": Decimal(0.075),
        }
    },
}


@dataclass(kw_only=True)
class ClaudeAIModel(AIModel):
    model: str
    api_key: str = ""
    provider: AI_MODELS_PROVIDER = AI_MODELS_PROVIDER.CLAUDE
    family: str = None
    
    def __post_init__(self):
        if 'haiku' in self.model:
            self.family = FamilyModel.haiku.value
        elif 'sonnet' in self.model:
            self.family = FamilyModel.sonnet.value
        elif 'opus' in self.model:
            self.family = FamilyModel.opus.value
        else:
            logger.warning(
                f"Unknown family for {self.model}. Please add it obviously. Setting as Claude 3 Haiku"
            )
            self.family = FamilyModel.haiku.value

        logger.debug(f"Initialized ClaudeAIModel: {self}")
        
        secrets = Secrets()
        self.api_key = secrets.CLAUDE_API_KEY
        

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
        
        # Implement the logic to call Claude AI API here
        logger.debug(
            f"Calling {messages} with max_tokens {max_tokens} and kwargs {kwargs}"
        )
        
        stream_function = kwargs.get('stream_function')
        check_connection = kwargs.get('check_connection')
        stream_params = kwargs.get('stream_params')
        
        content = ""
        
        try:
            if kwargs.get('stream'):
                with anthropic.Anthropic(api_key=self.api_key).messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=messages
                ) as stream:
                    idx = 0 
                    for text in stream.text_stream:
                        if idx % 5 == 0:
                            if not check_connection(**stream_params):
                                raise ConnectionError  
                        
                        stream_function(text, **stream_params)
                        content += text
                        idx += 1
            else:
                response = anthropic.Anthropic(api_key=self.api_key).messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=messages
                )
                content = response.content[0].text
                print(content)
            
            
            return ClaudeAIReponse(
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
            logger.exception("[CLAUDEAI] failed to handle chat stream", exc_info=e)
            raise RetryableCustomException(f"Claude AI call failed!")

    def name(self) -> str:
        return self.model

    @property
    def price_per_prompt_1k_tokens(self) -> Decimal:
        return CLAUDE_AI_PRICING[self.family].get(self.max_tokens, DEFAULT_PRICING)[
            "price_per_prompt_1k_tokens"
        ]

    @property
    def price_per_sample_1k_tokens(self) -> Decimal:
        return CLAUDE_AI_PRICING[self.family].get(self.max_tokens, DEFAULT_PRICING)[
            "price_per_sample_1k_tokens"
        ]

    def get_params(self) -> t.Dict[str, t.Any]:
        return {
            'api_key': self.api_key,
            'model': self.model,
            'max_tokens': self.max_tokens,
        }

    def get_metrics_data(self) -> t.Dict[str, t.Any]:
        return {
            'model': self.model,
            'max_tokens': self.max_tokens,
        }
        

from dataclasses import dataclass
from decimal import Decimal
import logging
import typing as t
from flow_prompt import secrets
from flow_prompt import settings, PIPE_PROMPTS
from flow_prompt.ai_models.behaviour import AIModelsBehaviour, PromptAttempts
from flow_prompt.responses import FlowResponse
from flow_prompt.prompt.pipe_prompt import PipePrompt
from flow_prompt.services.flow_prompt import FlowPromptService
from flow_prompt.exceptions import RetryableCustomException

logger = logging.getLogger(__name__)


@dataclass
class FlowPrompt:
    api_token: str = secrets.API_TOKEN
    total_price: Decimal = Decimal(0)
    openai_api_key: str = secrets.OPENAI_API_KEY
    openai_org: str = secrets.OPENAI_ORG
    azure_keys: t.Dict[str, str] = secrets.AZURE_KEYS


    def call(self, prompt_id: str, context: t.Dict[str, str], behaviour: AIModelsBehaviour, 
             params: t.Dict[str, t.Any] = {}, 
             version: str = None,
             count_of_retries: int = None) -> FlowResponse:
        """
        Call flow prompt with context and behaviour
        """
        pipe_prompt = self.get_pipe_prompt(prompt_id, version)
        prompt_attempts = PromptAttempts(behaviour, count_of_retries=count_of_retries)

        while prompt_attempts.initialize_attempt():
            user_prompt = pipe_prompt.create_prompt(context, prompt_attempts.current_attempt.ai_model)
            calling_messages = user_prompt.resolve(context)
            try:
                result = prompt_attempts.current_attempt.ai_model.call(calling_messages, params)
                price_of_call = self.price(
                    self.calculate_budget_for_text(result.get_message_str()), 
                    calling_messages.prompt_budget
                )
                self.total_price += price_of_call
                return result
            except RetryableCustomException as e:
                logger.warning(f"Retryable error: {e}")
                


    def get_pipe_prompt(self, prompt_id: str, version: str = None) -> PipePrompt:
        '''
        if the user has keys:  lib -> service: get_actual_prompt(local_prompt) -> Service:
        generates hash of the prompt;
        check in Redis if that record is the latest; if yes -> return 200, else
        checks if that record exists with that hash;
        if record exists and it's not the last - then we load the latest published prompt; - > return  200 + the last record
        add a new record in storage, and adding that it's the latest published prompt; -> return 200
        update redis with latest record;
        '''
        if settings.USE_API_SERVICE is not None:
            prompt_data = None
            prompt = PIPE_PROMPTS.get(prompt_id)
            if prompt is None:
                prompt_data = prompt.to_dict()
            response = FlowPromptService.get_actual_prompt(self.api_token, prompt_id, prompt_data, version)
            if response.prompt_is_actual:
                return prompt
            else:
                return PipePrompt.from_dict(response.actual_prompt)
        else:
            return PIPE_PROMPTS[prompt_id]

    def calculate_budget_for_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self.user_prompt.encode(text))

    def _decimal(self, value) -> Decimal:
        return Decimal(value).quantize(Decimal(".00001"))

    def get_price(self, tokens_budget) -> Decimal:
        return self._decimal(
            tokens_budget * self.attempt.model_provider.engine.price_per_prompt_1k_tokens / 1000
        )

    def price(self, sample_budget: int, prompt_budget: int) -> Decimal:
        return self.get_price(prompt_budget) + self.get_price(sample_budget)

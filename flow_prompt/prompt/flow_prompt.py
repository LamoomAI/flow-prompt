import logging
import typing as t
from dataclasses import dataclass
from decimal import Decimal

from openai import AzureOpenAI, OpenAI

from flow_prompt import secrets, settings
from flow_prompt.ai_models.ai_model import AI_MODELS_PROVIDER
from flow_prompt.ai_models.attempt_to_call import AttemptToCall
from flow_prompt.ai_models.behaviour import AIModelsBehaviour, PromptAttempts
from flow_prompt.exceptions import RetryableCustomException
from flow_prompt.prompt.pipe_prompt import PipePrompt
from flow_prompt.prompt.user_prompt import UserPrompt
from flow_prompt.responses import AIResponse
from flow_prompt.services.flow_prompt import FlowPromptService
from flow_prompt.utils import current_timestamp_ms

logger = logging.getLogger(__name__)


@dataclass
class FlowPrompt:
    api_token: str = secrets.API_TOKEN
    openai_api_key: str = secrets.OPENAI_API_KEY
    openai_org: str = secrets.OPENAI_ORG
    azure_keys: t.Dict[str, str] = secrets.AZURE_KEYS

    def __post_init__(self):
        if self.openai_api_key:
            openai_client = OpenAI(
                organization=self.openai_org,
                api_key=self.openai_api_key,
            )
            logger.info(f"Initialized OpenAI client: {openai_client}")
            settings.AI_CLIENTS[AI_MODELS_PROVIDER.OPENAI] = openai_client
        if self.azure_keys:
            settings.AI_CLIENTS[AI_MODELS_PROVIDER.AZURE] = {}
            for realm, key_data in self.azure_keys.items():
                if realm in settings.AI_CLIENTS[AI_MODELS_PROVIDER.AZURE]:
                    raise Exception(f"Realm {realm} already exists")
                settings.AI_CLIENTS[AI_MODELS_PROVIDER.AZURE][realm] = AzureOpenAI(
                    api_version=key_data.get("api_version", "2023-07-01-preview"),
                    azure_endpoint=key_data["url"],
                    api_key=key_data["key"],
                )
                logger.info(f"Initialized Azure client for {realm} {key_data['url']}")

    def call(
        self,
        prompt_id: str,
        context: t.Dict[str, str],
        behaviour: AIModelsBehaviour,
        params: t.Dict[str, t.Any] = {},
        version: str = None,
        count_of_retries: int = None,
    ) -> AIResponse:
        """
        Call flow prompt with context and behaviour
        """
        start_time = current_timestamp_ms()
        total_price = Decimal(0)
        pipe_prompt = self.get_pipe_prompt(prompt_id, version)
        prompt_attempts = PromptAttempts(behaviour, count_of_retries=count_of_retries)

        while prompt_attempts.initialize_attempt():
            current_attempt = prompt_attempts.current_attempt
            logger.info(
                f"Calling {prompt_id}. Attempt {prompt_attempts.current_attempt}"
            )
            user_prompt = pipe_prompt.create_prompt(current_attempt)
            calling_messages = user_prompt.resolve(context)
            try:
                result = current_attempt.ai_model.call(
                    calling_messages.get_messages(),
                    calling_messages.max_sample_budget,
                    **params,
                )
                price_of_call = self.get_price(
                    current_attempt,
                    self.calculate_budget_for_text(
                        user_prompt, result.get_message_str()
                    ),
                    calling_messages.prompt_budget,
                )
                self.save_user_interaction(
                    pipe_prompt.dump(),
                    context,
                    result,
                    {
                        "attempt_number": current_attempt.attempt_number,
                        "price": price_of_call,
                        "latency": current_timestamp_ms() - start_time,
                    },
                )
                return result
            except RetryableCustomException as e:
                logger.warning(f"Retryable error: {e}")

    def get_pipe_prompt(self, prompt_id: str, version: str = None) -> PipePrompt:
        """
        if the user has keys:  lib -> service: get_actual_prompt(local_prompt) -> Service:
        generates hash of the prompt;
        check in Redis if that record is the latest; if yes -> return 200, else
        checks if that record exists with that hash;
        if record exists and it's not the last - then we load the latest published prompt; - > return  200 + the last record
        add a new record in storage, and adding that it's the latest published prompt; -> return 200
        update redis with latest record;
        """
        if settings.USE_API_SERVICE is not None:
            prompt_data = None
            prompt = settings.PIPE_PROMPTS.get(prompt_id)
            if prompt is None:
                prompt_data = prompt.dump()
            response = FlowPromptService.get_actual_prompt(
                self.api_token, prompt_id, prompt_data, version
            )
            if response.prompt_is_actual:
                return prompt
            else:
                return PipePrompt.load(response.actual_prompt)
        else:
            return settings.PIPE_PROMPTS[prompt_id]

    def save_user_interaction(
        self,
        prompt_data: dict,
        context: dict[str, t.Any],
        result: AIResponse,
        metrics: dict[str, t.Any] = {},
    ) -> None:
        """
        Save user interaction to flow-prompt
        """
        if settings.USE_API_SERVICE is not None:
            FlowPromptService.save_user_interaction(
                self.api_token, prompt_data, context, result, metrics
            )

    def calculate_budget_for_text(self, user_prompt: UserPrompt, text: str) -> int:
        if not text:
            return 0
        return len(user_prompt.encoding.encode(text))

    def _decimal(self, value) -> Decimal:
        return Decimal(value).quantize(Decimal(".00001"))

    def get_price(
        self, attempt: AttemptToCall, sample_budget: int, prompt_budget: int
    ) -> Decimal:
        return self._decimal(
            prompt_budget * attempt.ai_model.price_per_prompt_1k_tokens / 1000
        ) + self._decimal(
            sample_budget * attempt.ai_model.price_per_sample_1k_tokens / 1000
        )

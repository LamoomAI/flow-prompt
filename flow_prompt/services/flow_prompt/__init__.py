import json
import logging
import typing as t
from dataclasses import dataclass
import requests

from flow_prompt import settings
from flow_prompt.exceptions import NotFoundPromptException
from flow_prompt.utils import DecimalEncoder, current_timestamp_ms

logger = logging.getLogger(__name__)


@dataclass
class FlowPromptServiceResponse:
    prompt_id: str = None
    actual_prompt: dict = None
    hash_key: str = None
    prompt_is_actual: bool = False


class FlowPromptService:
    url: str = "https://api.flow-prompt.com/"
    cached_prompts = {}

    def get_actual_prompt(
        self,
        api_token: str,
        prompt_id: str,
        prompt_data: dict = None,
        version: str = None,
    ) -> FlowPromptServiceResponse:
        """
        Load prompt from flow-prompt
        if the user has keys:  lib -> service: get_actual_prompt(local_prompt) -> Service:
        generates hash of the prompt;
        check in Redis if that record is the latest; if yes -> return 200, else
        checks if that record exists with that hash;
        if record exists and it's not the last - then we load the latest published prompt; - > return  200 + the last record
        add a new record in storage, and adding that it's the latest published prompt; -> return 200
        update redis with latest record;
        """
        logger.debug(f"Getting actual prompt for {prompt_id}")
        cached_prompt = None
        cached_prompt_taken_globally = False
        cached_data = self.get_cached_prompt(prompt_id)
        if cached_data:
            cached_prompt = cached_data.get("prompt")
            cached_prompt_taken_globally = cached_data.get("prompt_taken_globally")
            if cached_prompt:
                logger.debug(f"Prompt {prompt_id} is cached")
                return FlowPromptServiceResponse(
                    prompt_id=prompt_id,
                    actual_prompt=cached_prompt,
                    prompt_is_actual=True,
                )

        url = f"{self.url}lib/prompts"
        headers = {
            "Authorization": f"Token {api_token}",
        }
        data = {
            "prompt": prompt_data,
            "prompt_id": prompt_id,
            "version": version,
            "prompt_taken_globally": cached_prompt_taken_globally,
        }
        json_data = json.dumps(data, cls=DecimalEncoder)
        logger.debug(f"Request to {url} with data: {json_data}")
        response = requests.post(url, headers=headers, data=json_data)
        if response.status_code == 200:
            response_data = response.json()
            prompt_data = response_data.get("prompt", prompt_data)
            prompt_taken_globally = response_data.get("prompt_taken_globally")

            # update cache
            self.cached_prompts[prompt_id] = {
                "prompt": response,
                "timestamp": current_timestamp_ms(),
                "prompt_taken_globally": prompt_taken_globally,
            }
            # returns 200 and the latest published prompt, if the local prompt is the latest, doesn't return the prompt
            return FlowPromptServiceResponse(
                prompt_id=prompt_id,
                actual_prompt=prompt_data,
                hash_key=prompt_data["hash_key"],
                prompt_is_actual=response_data["prompt_is_actual"],
            )
        else:
            raise NotFoundPromptException(response.json())

    def get_cached_prompt(self, prompt_id: str) -> dict:
        cached_data = self.cached_prompts.get(prompt_id)
        if not cached_data:
            return None
        cached_delay = current_timestamp_ms() - cached_data.get("timestamp")
        if cached_delay < settings.CACHE_PROMPT_FOR_EACH_SECONDS * 1000:
            return cached_data
        return None

    @classmethod
    def clear_cache(cls):
        cls.cached_prompts = {}

    @classmethod
    def save_user_interaction(
        cls,
        api_token: str,
        prompt_data: dict[str, t.Any],
        context: dict[str, t.Any],
        response: dict[str, t.Any],
        metrics: dict[str, t.Any] = {},
    ):
        url = f"{cls.url}lib/ai_chronicles"
        headers = {"Authorization": f"Token {api_token}"}
        data = {
            "context": context,
            "prompt": prompt_data,
            "response": response.to_dict(),
            "metrics": metrics,
        }
        json_data = json.dumps(data, cls=DecimalEncoder)

        response = requests.post(url, headers=headers, data=json_data)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(response)

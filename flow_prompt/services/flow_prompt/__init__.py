from dataclasses import dataclass
import logging
from urllib import request
from flow_prompt import settings
import typing as t

from flow_prompt.exceptions import NotFoundPromptException
from flow_prompt.utils import current_timestamp_ms

logger = logging.getLogger(__name__)

@dataclass
class FlowPromptServiceResponse:
    prompt_id: str = None
    actual_prompt: dict = None
    actual_prompt_hash: str = None
    prompt_is_actual: bool = False


class FlowPromptService:
    url: str = "https://flow-prompt.com/api/v1/"
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
        cached_prompt = self.get_cached_prompt(prompt_id)
        if cached_prompt:
            return FlowPromptServiceResponse(
                prompt_id=prompt_id,
                actual_prompt=cached_prompt,
                prompt_is_actual=True,
            )
        url = f"{self.url}prompts/{prompt_id}/"
        if version:
            url += f"?version={version}"
        headers = {"Authorization": f"Token {api_token}"}
        response = request.post(url, headers=headers, data={"prompt": prompt_data})
        if response.status_code == 200:
            response_data = response.json()
            prompt_data = response_data.get("prompt", prompt_data)
            # update cache
            self.cached_prompts[prompt_id] = {
                "prompt": response,
                "timestamp": current_timestamp_ms(),
            }
            # returns 200 and the latest published prompt, if the local prompt is the latest, doesn't return the prompt
            return FlowPromptServiceResponse(
                prompt_id=prompt_id,
                actual_prompt=response,
                actual_prompt_hash=response_data["actual_prompt_hash"],
                prompt_is_actual=response_data["prompt_is_actual"],
            )
        else:
            raise NotFoundPromptException(response.json()["detail"])

    def get_cached_prompt(self, prompt_id: str) -> dict:
        cached_data = self.cached_prompts.get(prompt_id)
        if not cached_data:
            return None
        cached_delay = current_timestamp_ms() - cached_data.get("timestamp")
        if cached_delay < settings.CACHE_PROMPT_FOR_EACH_SECONDS * 1000:
            return cached_data.get("prompt")
        return None
    
    def save_user_interaction(self, 
        api_token: str,
        context: dict[str, t.Any], 
        prompt_data: dict[str, t.Any], 
        response: dict[str, t.Any],
        metrics: dict[str, t.Any] = {}
        ):
        url = f"{self.url}user_interactions/"
        headers = {"Authorization": f"Token {api_token}"}
        data = {
            "context": context,
            "prompt_data": prompt_data,
            "response": response,
            "metrics": metrics,
        }
        response = request.post(url, headers=headers, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise logger.error(response.json()["detail"])

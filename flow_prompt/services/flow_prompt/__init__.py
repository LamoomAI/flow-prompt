from dataclasses import dataclass
from urllib import request
from flow_prompt.exceptions import NotFoundPromptException


@dataclass
class FlowPromptServiceResponse:
    prompt_id: str = None
    actual_prompt: dict = None
    actual_prompt_hash: str = None
    prompt_is_actual: bool = False


class FlowPromptService:
    url: str = "https://flow-prompt.com/api/v1/"

    def get_actual_prompt(self, api_token: str, prompt_id: str, prompt_data: dict = None, version: str = None) -> FlowPromptServiceResponse:
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
        url = f"{self.url}prompts/{prompt_id}/"
        if version:
            url += f"?version={version}"
        headers = {"Authorization": f"Token {api_token}"}
        response = request.post(url, headers=headers, data={"prompt": prompt_data})
        if response.status_code == 200:
            # returns 200 and the latest published prompt, if the local prompt is the latest, doesn't return the prompt
            return FlowPromptServiceResponse(
                prompt_id=prompt_id,
                actual_prompt=response.json().get("actual_prompt", prompt_data),
                actual_prompt_hash=response.json()["actual_prompt_hash"],
                prompt_is_actual=response.json()["prompt_is_actual"],
            )
        else:
            raise NotFoundPromptException(response.json()["detail"])


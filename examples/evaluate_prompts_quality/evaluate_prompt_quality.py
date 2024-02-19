import logging
import random
import requests
import os
from flow_prompt import FlowPrompt, behaviour, AttemptToCall, AzureAIModel, C_128K
import dotenv
from prompt import prompt_to_evaluate_prompt

logger = logging.getLogger(__name__)
dotenv.load_dotenv(dotenv.find_dotenv())
flow_prompt=FlowPrompt()

gpt4_behaviour = behaviour.AIModelsBehaviour(
    attempts=[
        AttemptToCall(
            ai_model=AzureAIModel(
                realm='westus',
                deployment_name="gpt-4-turbo",
                max_tokens=C_128K,
                support_functions=True,
                should_verify_client_has_creds=False,
            ),
            weight=100,
        ),
    ]
)

BEARER_TOKEN = os.getenv('BEARER_TOKEN')
FLOW_PROMPT_URL = 'https://api.flow-prompt.com'



def main():
    for prompt in get_all_prompts():
        prompt_id = prompt['prompt_id']
        prompt_chats = prompt['chats']
        logs = get_logs(prompt_id).get('items')
        if not logs or len(logs) < 5:
            continue
        contexts = []
        responses = []
        for log in random.sample(logs, 5):
            responses.append(log['response']['message'])
            contexts.append(log['context'])
        context = {
            'responses': responses,
            'prompt_data': prompt_chats,
            'prompt_id': prompt_id,
        }
        result = flow_prompt.call(prompt_to_evaluate_prompt.id, context, gpt4_behaviour)
        print(result.content)



def get_all_prompts():
    response = requests.get(f'{FLOW_PROMPT_URL}/prompts', headers={'Authorization': f'Bearer {BEARER_TOKEN}'})
    prompts = response.json()
    return prompts


def get_logs(prompt_id):
    response = requests.get(
        f'{FLOW_PROMPT_URL}/ai_chronicles?prompt_id={prompt_id}&fields=response,context',
        headers={'Authorization': f'Bearer {BEARER_TOKEN}'}
    )
    logs = response.json()
    return logs


if __name__ == '__main__':
    main()
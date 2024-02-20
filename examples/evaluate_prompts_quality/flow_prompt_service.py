

import os
import dotenv
import requests

dotenv.load_dotenv(dotenv.find_dotenv())
FLOW_PROMPT_URL = 'https://api.flow-prompt.com'
BEARER_TOKEN = os.getenv('BEARER_TOKEN')


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


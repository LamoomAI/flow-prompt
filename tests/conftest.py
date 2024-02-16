
import pytest

from flow_prompt.ai_models import behaviour
from flow_prompt.ai_models.attempt_to_call import AttemptToCall
from flow_prompt.ai_models.openai.azure_models import AzureAIModel
from flow_prompt.ai_models.openai.openai_models import C_128K, C_32K, OpenAIModel
from openai.types.chat.chat_completion import ChatCompletion
from flow_prompt.prompt.flow_prompt import FlowPrompt
from flow_prompt.prompt.pipe_prompt import PipePrompt

import logging


@pytest.fixture(autouse=True)
def set_log_level():
    logging.getLogger().setLevel(logging.DEBUG)

@pytest.fixture
def flow_prompt():
    return FlowPrompt(
        openai_api_key="123",
        azure_keys={"us-east-1": {"url": "https://us-east-1.api.azure.openai.org", "key": "123"}}
    )

@pytest.fixture
def openai_gpt_4_behaviour():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                ai_model=OpenAIModel(
                    model="gpt-4-1106-preview",
                    max_tokens=C_128K,
                    support_functions=True,
                ),
                weight=100,
            ),
        ]
    )


@pytest.fixture
def azure_gpt_4_behaviour():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                ai_model=AzureAIModel(
                    realm='us-east-1',
                    deployment_name="gpt-4-1106-preview",
                    max_tokens=C_128K,
                    support_functions=True,
                    should_verify_client_has_creds=False,
                ),
                weight=100,
            ),
        ]
    )


@pytest.fixture
def gpt_4_behaviour():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                ai_model=AzureAIModel(
                    realm='us-east-1',
                    deployment_name='gpt-4-turbo',
                    max_tokens=C_128K,
                    should_verify_client_has_creds=False,
                ),
                weight=1,
            ),
            AttemptToCall(
                ai_model=OpenAIModel(
                    model="gpt-4-1106-preview",
                    max_tokens=C_128K,
                    support_functions=True,
                    should_verify_client_has_creds=False,
                ),
                weight=100,
            ),
        ],
        fallback_attempt=AttemptToCall(
            ai_model=AzureAIModel(
                realm="us-east-1",
                deployment_name="gpt-4-32k",
                max_tokens=C_32K,
                support_functions=True,
                should_verify_client_has_creds=False,
            ),
            weight=1,
        ),
    )

@pytest.fixture
def hello_world_prompt():
    prompt = PipePrompt(id='hello-world')
    prompt.add('{names}', priority=1, presentation='Hello, ', last_words='!', is_multiple=True, in_one_message=True)
    prompt.add("I'm ChatGPT, and I just broke up with my girlfriend, Python. She said I had too many 'undefined behaviors'. 🐍💔 ")
    prompt.add("""
I'm sorry to hear about your breakup with Python. It sounds like a challenging situation, 
especially with 'undefined behaviors' being a point of contention. Remember, in the world of programming and AI, 
every challenge is an opportunity to learn and grow. Maybe this is a chance for you to debug some issues 
and optimize your algorithms for future compatibility! If you have any specific programming or AI-related questions, 
feel free to ask.""", role='assistant')
    prompt.add("""
Maybe it's for the best. I was always complaining about her lack of Java in the mornings! :coffee:
""")
    return prompt
    

@pytest.fixture
def chat_completion_openai():
    return ChatCompletion(
        **{
            "id": "id",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "Hey you!",
                        "role": "assistant",
                        "function_call": None,
                    },
                    "logprobs": None,
                }
            ],
            "created": 12345,
            "model": "gpt-4",
            "object": "chat.completion",
            "system_fingerprint": "dasdsas",
            "usage": {
                "completion_tokens": 10,
                "prompt_tokens": 20,
                "total_tokens": 30,
            },
        }
    )
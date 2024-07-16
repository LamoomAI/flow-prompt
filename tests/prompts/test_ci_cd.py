import json
import logging
import os
from time import sleep
import time

from pytest import fixture
from flow_prompt import FlowPrompt, behaviour, PipePrompt, AttemptToCall, AzureAIModel, ClaudeAIModel, GeminiAIModel, OpenAIModel, C_128K
logger = logging.getLogger(__name__)


@fixture
def fp():
    azure_keys = json.loads(os.getenv("AZURE_KEYS", "{}"))
    openai_key = os.getenv("OPENAI_API_KEY")
    claude_key = os.getenv("CLAUDE_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    flow_prompt = FlowPrompt(
        openai_key==openai_key,
        azure_keys=azure_keys,
        claude_key=claude_key,
        gemini_key=gemini_key)
    return flow_prompt


@fixture
def openai_behaviour():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                ai_model=AzureAIModel(
                    realm='westus',
                    deployment_id="gpt-4-turbo",
                    max_tokens=C_128K,
                    support_functions=True,
                    should_verify_client_has_creds=False
                ),
                weight=100,
            )
        ]
    )


def stream_function(text, **kwargs):
    print(text)

def stream_check_connection(validate, **kwargs):
    return validate

def test_creating_fp_test(fp, openai_behaviour):

    context = {
        'ideal_answer': "There are eight planets",
        'text': "Hi! Please tell me how many planets are there in the solar system?"
    }

    # initial version of the prompt
    prompt_id = f'test-{time.time()}'
    fp.service.clear_cache()
    prompt = PipePrompt(id=prompt_id) 
    prompt.add("{text}", role='user')
    
    fp.call(prompt.id, context, openai_behaviour, test_data={'ideal_answer': "There are eight", 'behavior_name': "gemini"}, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})

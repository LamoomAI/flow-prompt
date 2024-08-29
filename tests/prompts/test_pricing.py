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
        openai_key=openai_key,
        azure_keys=azure_keys,
        claude_key=claude_key,
        gemini_key=gemini_key)
    return flow_prompt


@fixture
def openai_behaviour_4o():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                ai_model=AzureAIModel(
                    realm='useast',
                    deployment_id="gpt-4o",
                    max_tokens=C_128K,
                    support_functions=True,
                ),
                weight=100,
            )
        ]
    )
    
@fixture
def openai_behaviour_4o_mini():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                ai_model=AzureAIModel(
                    realm='useast',
                    deployment_id="gpt-4o-mini",
                    max_tokens=C_128K,
                    support_functions=True,
                ),
                weight=100,
            )
        ]
    )

@fixture
def claude_behaviour_haiku():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                ai_model=ClaudeAIModel(
                    model="claude-3-haiku-20240307",
                    max_tokens=4096                
                ),
                weight=100,
            ),
        ]
    )
    
@fixture
def claude_behaviour_sonnet():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                ai_model=ClaudeAIModel(
                    model="claude-3-sonnet-20240229",
                    max_tokens=4096                
                ),
                weight=100,
            ),
        ]
    )

def stream_function(text, **kwargs):
    print(text)

def stream_check_connection(validate, **kwargs):
    return validate


def test_openai_pricing(fp, openai_behaviour_4o, openai_behaviour_4o_mini):

    context = {
        'ideal_answer': "There are eight planets",
        'text': "Hi! Please tell me how many planets are there in the solar system?"
    }

    # initial version of the prompt
    prompt_id = f'test-{time.time()}'
    fp.service.clear_cache()
    prompt = PipePrompt(id=prompt_id) 
    prompt.add("{text}", role='user')

    result_4o = fp.call(prompt.id, context, openai_behaviour_4o, test_data={'ideal_answer': "There are eight", 'behavior_name': "gemini"}, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    result_4o_mini = fp.call(prompt.id, context, openai_behaviour_4o_mini, test_data={'ideal_answer': "There are eight", 'behavior_name': "gemini"}, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    
    assert result_4o.metrics.price_of_call > result_4o_mini.metrics.price_of_call
    

def test_claude_pricing(fp, claude_behaviour_haiku, claude_behaviour_sonnet):

    context = {
        'ideal_answer': "There are eight planets",
        'text': "Hi! Please tell me how many planets are there in the solar system?"
    }

    # initial version of the prompt
    prompt_id = f'test-{time.time()}'
    fp.service.clear_cache()
    prompt = PipePrompt(id=prompt_id) 
    prompt.add("{text}", role='user')

    fp.call(prompt.id, context, claude_behaviour_haiku, test_data={'ideal_answer': "There are eight", 'behavior_name': "gemini"}, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    fp.call(prompt.id, context, claude_behaviour_sonnet, test_data={'ideal_answer': "There are eight", 'behavior_name': "gemini"}, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
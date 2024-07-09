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
   
@fixture
def claude_behaviour():
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
def gemini_behaviour():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                ai_model=GeminiAIModel(
                    model_name="gemini-1.5-flash",
                    max_tokens=C_128K                
                ),
                weight=100,
            ),
        ]
    )

def stream_function(text, **kwargs):
    print(text)

def stream_check_connection(validate, **kwargs):
    return validate

def test_loading_prompt_from_service(fp, openai_behaviour, azure_behaviour, claude_behaviour, gemini_behaviour):

    context = {
        'messages': ['test1', 'test2'],
        'assistant_response_in_progress': None,
        'files': ['file1', 'file2'],
        'music': ['music1', 'music2'],
        'videos': ['video1', 'video2'],
        'text': "Good morning. Tell me a funny joke!"
    }

    # initial version of the prompt
    prompt_id = f'test-{time.time()}'
    fp.service.clear_cache()
    prompt = PipePrompt(id=prompt_id) 
    # prompt.add('{messages}', is_multiple=True, in_one_message=True, label='messages')
    prompt.add("{text}")
    prompt.add("It's a system message, Hello {name}", role="assistant")
    
    fp.call(prompt.id, context, openai_behaviour, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    fp.call(prompt.id, context, azure_behaviour, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    fp.call(prompt.id, context, claude_behaviour, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    fp.call(prompt.id, context, gemini_behaviour, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    
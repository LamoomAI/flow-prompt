import json
import os
import time

from pytest import fixture
from flow_prompt import FlowPrompt, behaviour, PipePrompt, AttemptToCall, AzureAIModel, C_128K


@fixture
def fp():
    azure_keys = json.loads(os.getenv("AZURE_KEYS", "{}"))
    openai_key = os.getenv("OPENAI_API_KEY")
    api_token = os.getenv("FLOW_PROMPT_API_TOKEN")
    flow_prompt = FlowPrompt(
        openai_key=openai_key,
        azure_keys=azure_keys,
        api_token=api_token)
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
                ),
                weight=100,
            )
        ]
    )


def stream_function(text, **kwargs):
    print(text)

def stream_check_connection(validate, **kwargs):
    return validate


def test_save_user_interaction_return_value(fp, openai_behaviour, caplog):
    prompt_id = f'test-{time.time()}'
    fp.service.clear_cache()
    prompt = PipePrompt(id=prompt_id) 
    prompt.add("{text}", role='user')
    
    context = {
        'ideal_answer': "There are eight planets",
        'text': "Hi! Please tell me how many planets are there in the solar system?"
    }   
    
    with caplog.at_level('ERROR'):
        
        fp.call(
            prompt.id, 
            context, 
            openai_behaviour, 
            stream_function=stream_function, 
            check_connection=stream_check_connection, params={"stream": True}, 
            stream_params={"validate": True, "end": "", "flush": True})


        fp.worker.queue.join()  # This will block until the queue is empty

        error_logs = [record for record in caplog.records if record.levelname == "ERROR"]
        
        assert len(error_logs) == 0

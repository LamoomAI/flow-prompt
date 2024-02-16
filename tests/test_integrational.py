from time import sleep
from uuid import uuid4
import flow_prompt
from flow_prompt import FlowPrompt, behaviour, PipePrompt, AttemptToCall, AzureAIModel, C_128K
import dotenv

from flow_prompt.services.flow_prompt import FlowPromptService
from flow_prompt.settings import PIPE_PROMPTS

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


def _test_loading_prompt_from_service():
    context = {
        'messages': ['test1', 'test2'],
        'assistant_response_in_progress': None,
        'files': ['file1', 'file2'],
        'music': ['music1', 'music2'],
        'videos': ['video1', 'video2'],
    }

    # initial version of the prompt
    prompt_id = f'test-{uuid4().hex[:4]}'
    flow_prompt.service.clear_cache()
    prompt = PipePrompt(id=prompt_id) 
    prompt.add("It's a system message, Hello {name}", role="system")
    prompt.add('{messages}', is_multiple=True, in_one_message=True, label='messages')
    print(flow_prompt.call(prompt.id, context, gpt4_behaviour))

    # updated version of the prompt
    flow_prompt.service.clear_cache()
    prompt = PipePrompt(id=prompt_id) 
    prompt.add("It's a system message, Hello {name}", role="system")
    prompt.add('{music}', is_multiple=True, in_one_message=True, label='music')
    print(flow_prompt.call(prompt.id, context, gpt4_behaviour))

    # call uses outdated version of prompt, should use updated version of the prompt
    sleep(2)
    flow_prompt.service.clear_cache()
    prompt = PipePrompt(id=prompt_id)
    prompt.add("It's a system message, Hello {name}", role="system")
    prompt.add('{messages}', is_multiple=True, in_one_message=True, label='messages')
    result = flow_prompt.call(prompt.id, context, gpt4_behaviour)
    # should call the prompt with music
    assert result.prompt_messages[-1] == {'role': 'user', 'content': 'music1\nmusic2'} 

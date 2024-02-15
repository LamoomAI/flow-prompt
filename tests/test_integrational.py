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
                ),
                weight=100,
            ),
        ]
    )


flow_prompt.service.clear_cache()
prompt = PipePrompt(id='test') 
prompt.add("It's a system message, Hello {name}", role="system")
prompt.add('1')
prompt.add('2')
prompt.add('3')
prompt.add(4)
prompt.add(5)

print(flow_prompt.call(prompt.id, {
    'messages': ['test1', 'test2'],
    'assistant_response_in_progress': None,
    'files': ['file1', 'file2'],
    'music': ['music1', 'music2'],
    'videos': ['video1', 'video2'],
}, gpt4_behaviour))



flow_prompt.service.clear_cache()
prompt = PipePrompt(id='test') 
prompt.add("It's a system message, Hello {name}", role="system")
prompt.add('1')
prompt.add('2')
prompt.add('3')
prompt.add(4)

print(flow_prompt.call(prompt.id, {
    'messages': ['test1', 'test2'],
    'assistant_response_in_progress': None,
    'files': ['file1', 'file2'],
    'music': ['music1', 'music2'],
    'videos': ['video1', 'video2'],
}, gpt4_behaviour))


flow_prompt.service.clear_cache()
prompt = PipePrompt(id='test') 
prompt.add("It's a system message, Hello {name}", role="system")
prompt.add('1')
prompt.add('2')
prompt.add('3')
prompt.add(4)
prompt.add(5)

print(flow_prompt.call(prompt.id, {
    'messages': ['test1', 'test2'],
    'assistant_response_in_progress': None,
    'files': ['file1', 'file2'],
    'music': ['music1', 'music2'],
    'videos': ['video1', 'video2'],
}, gpt4_behaviour)
)

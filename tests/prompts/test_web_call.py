
import json
import logging
from lamoom.ai_models.tools.web_tool import WEB_SEARCH_TOOL
from pytest import fixture
from lamoom import Lamoom, Prompt
logger = logging.getLogger(__name__)


@fixture
def client():
    import dotenv
    dotenv.load_dotenv(dotenv.find_dotenv())
    lamoom = Lamoom()
    return lamoom

def stream_function(text, **kwargs):
    print(text)

def stream_check_connection(validate=True, **kwargs):
    return validate


def test_web_call(client):

    context = {
        'text': f"What's the latest news in the San Francisco and AI?"
    }

    # initial version of the prompt
    prompt_id = 'test-web-search'
    prompt = Prompt(id=prompt_id) 
    prompt.add("", role='system')
    prompt.add("{text}", role='user')
    
    prompt.add_tool(WEB_SEARCH_TOOL)
    print(f'prompt.tool_registry: {prompt.tool_registry}')
    result = client.call(prompt.id, context, "openai/o4-mini")
    assert result.content

    result = client.call(prompt.id, context, "claude/claude-3-7-sonnet-latest")
    assert result.content

    result = client.call(prompt.id, context, "azure/useast/gpt-4.1-mini")
    assert result.content

    result = client.call(prompt.id, context, "custom/nvidia/deepseek-ai/deepseek-r1", 
        stream_function=stream_function, 
        check_connection=stream_check_connection, 
        params={"stream": True}
    )
    assert result.content

    result = client.call(prompt.id, context, "custom/nebius/deepseek-ai/DeepSeek-R1")
    assert result.content

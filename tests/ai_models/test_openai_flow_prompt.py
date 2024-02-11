from unittest.mock import patch
from flow_prompt.ai_models.ai_model import AI_MODELS_PROVIDER
from flow_prompt.ai_models.behaviour import AIModelsBehaviour
from flow_prompt.prompt.flow_prompt import FlowPrompt
from flow_prompt.prompt.pipe_prompt import PipePrompt
from openai.types.chat.chat_completion import ChatCompletion
from flow_prompt.settings import AI_CLIENTS


@patch("openai.OpenAI")
def test_flow_prompt(mock_openai, flow_prompt, chat_completion_openai: ChatCompletion, openai_gpt_4_behaviour: AIModelsBehaviour, hello_world_prompt: PipePrompt):

    mock_openai_instance = mock_openai.return_value
    mock_create = mock_openai_instance.chat.completions.create

    # Mock the model_dump() method
    mock_create.return_value = chat_completion_openai

    flow_prompt = FlowPrompt(openai_api_key="123")
    context = {
        "names": [
            "A Fax Machine",
            "Internet Explorer",
            "A Pager"
        ]
    }
    AI_CLIENTS[AI_MODELS_PROVIDER.OPENAI] = mock_openai_instance

    response = flow_prompt.call(
        hello_world_prompt.id, context, openai_gpt_4_behaviour
    )

    assert  mock_create.call_count == 1
    # called with 4 messages
    assert mock_create.call_args[1]["messages"] == [{'role': 'user', 'content': 'Hello, A Fax Machine\nInternet Explorer\nA Pager!'}, {'role': 'user', 'content': "I'm ChatGPT, and I just broke up with my girlfriend, Python. She said I had too many 'undefined behaviors'. 🐍💔 "}, {'role': 'assistant', 'content': "\nI'm sorry to hear about your breakup with Python. It sounds like a challenging situation, \nespecially with 'undefined behaviors' being a point of contention. Remember, in the world of programming and AI, \nevery challenge is an opportunity to learn and grow. Maybe this is a chance for you to debug some issues \nand optimize your algorithms for future compatibility! If you have any specific programming or AI-related questions, \nfeel free to ask."}, {'role': 'user', 'content': "\nMaybe it's for the best. I was always complaining about her lack of Java in the mornings! :coffee:\n"}]


    # Assert that the response is as expected
    assert response.finish_reason == "stop"
    assert response.response == "Hey you!"
    assert response.original_result.choices[0].finish_reason == "stop"
    assert response.original_result.choices[0].index == 0
    assert response.original_result.choices[0].message.content == "Hey you!"
import logging
import os
from pytest import fixture
from flow_prompt import FlowPrompt
from flow_prompt import FlowPrompt, behaviour, PipePrompt, AttemptToCall, AzureAIModel, ClaudeAIModel, GeminiAIModel, OpenAIModel, C_128K
import json
import time
logger = logging.getLogger(__name__)

@fixture
def fp():
    api_token = os.getenv("FLOW_PROMPT_API_TOKEN")
    gemini_key = os.getenv("GEMINI_API_KEY")
    flow_prompt = FlowPrompt(
        api_token=api_token,
        gemini_key=gemini_key)
    return flow_prompt


@fixture
def gemini_behaviour():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                ai_model=GeminiAIModel(
                    model="gemini-1.5-flash",
                    max_tokens=C_128K                
                ),
                weight=100,
            ),
        ]
    )



def test_feedback_loop(fp, gemini_behaviour):
    prompt_id = f'batterylife-test'
    fp.service.clear_cache()
    prompt = PipePrompt(id=prompt_id) 
    prompt.add("How can I improve the battery life of my smartphone? Provide practical tips.", role='user')
    response = fp.call(prompt.id, {}, gemini_behaviour)
    
    user_feedback = "The tips were helpful, but I expected more details. For example, it would be good to know which specific settings to adjust in the phone's menu to limit background activity. Also, more explanation about how to manage battery-draining apps would have been useful. Please include more step-by-step instructions next time."
    fp.send_feedback(user_feedback, response.id)

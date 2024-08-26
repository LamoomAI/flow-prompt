import logging
import os
import time

from pytest import fixture
from flow_prompt import FlowPrompt, PipePrompt
logger = logging.getLogger(__name__)


@fixture
def fp():
    api_token = os.getenv("FLOW_PROMPT_API_TOKEN")
    flow_prompt = FlowPrompt(
        api_token=api_token)
    return flow_prompt


def test_creating_fp_test(fp):
    context = {
        'ideal_answer': "There are eight planets",
        'text': "Hi! Please tell me how many planets are there in the solar system?"
    }

    # initial version of the prompt
    prompt_id = f'test-{time.time()}'
    fp.service.clear_cache()
    prompt = PipePrompt(id=prompt_id) 
    prompt.add("{text}", role='user')

    fp.create_test(prompt_id, context)
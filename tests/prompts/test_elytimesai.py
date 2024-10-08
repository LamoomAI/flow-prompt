import logging
import os
from pytest import fixture
from flow_prompt import FlowPrompt
logger = logging.getLogger(__name__)

@fixture
def fp():
    api_token = os.getenv("FLOW_PROMPT_API_TOKEN")
    flow_prompt = FlowPrompt(
        api_token=api_token
    )
    return flow_prompt


def test_elytimes(fp):

    user_id = '94187488-9041-7011-ac07-58581cb3f737'
    overview = "I'm handsome"

    # response = fp.get_file_names('/user', user_id)
    # response = fp.update_overview(user_id, overview)
    # response = fp.get_files([
    #     'user/1', 'user/2'
    # ], user_id)
    
    fp.save_files({
        'user/file1.txt': 'Hello world',
        'company/file1.txt': 'Hello company!'
    }, user_id)
    
    # assert 'file_contents' in response
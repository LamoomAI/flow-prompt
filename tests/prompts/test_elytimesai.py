import logging

from pytest import fixture
from flow_prompt import FlowPrompt
logger = logging.getLogger(__name__)

@fixture
def fp():
    flow_prompt = FlowPrompt()
    return flow_prompt


def test_elytimes(fp):

    user_id = '94187488-9041-7011-ac07-58581cb3f737'
    overview = "I'm handsome"

    # response = fp.get_file_names('/user', user_id)
    # response = fp.update_overview(user_id, overview)
    response = fp.get_files([
        'user/1', 'user/2'
    ], user_id)
    
    assert 'file_contents' in response
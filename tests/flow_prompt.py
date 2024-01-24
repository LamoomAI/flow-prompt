from flow_prompt.prompt.flow_prompt import FlowPrompt

def test_flow_prompt():
    flow_prompt = FlowPrompt()
    openai_response = flow_prompt.call(prompt_id, context, gpt_4_behaviour, openai_params)
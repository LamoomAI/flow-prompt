import os
import flow_prompt
from flow_prompt import FlowPrompt, behaviour, PipePrompt, AttemptToCall, AzureAIModel, C_128K

api_token = "g4iV0JjjY/xU9N3+9NbZTmbAIpyfduqZO2XyZxH9r8rDgSTAPP4oYHA5q8I3oyvzHKpgm/VM8Bp6n6IrjRilIDtVdZ8QuEleVnb/V9G9cAluSkU78djVU49WRFZH/xA7cgxhNcltniGDXqsQBZgPrGvSIUpiYtN7MiqnUSvo8DAPNfyi75v6Fy5MWJLhDtybPt8Yg0lLaBRrZ0mTV2gj+6ZLv3g9GjJ8QiPA2TTLEChwKlAyBhIJEERj+bpNE6nk58ZXE3IT+qeGQ3+bnNFu0gz/WYpH4Qd2FCpdrqYIhsYzuS0/9ln0dPnMywTxGBE/6Tnz3/+v526M9BcHaoY4mg=="
flow_prompt.AZURE_OPENAI_KEYS = '{"westus":{"url": "https://westuskate.openai.azure.com/", "key": "406cfc8757b1449688e9e81ca00cd8a3"}}'
flow_prompt=FlowPrompt(
    api_token=api_token,
    azure_keys={"westus":{"url": "https://westuskate.openai.azure.com/", "key": "406cfc8757b1449688e9e81ca00cd8a3"}}
)

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


prompt = PipePrompt(id='merge_code') 
prompt.add("It's a system message, Hello {name}", role="system")
prompt.add('{indexed_context}',
    priority=2, 
    is_multiple=True, while_fits=True, in_one_message=True, continue_if_doesnt_fit=True,
    presentation="The closest indexed context to the user request:",
    label='indexed_context'
)
prompt.add('{messages}', priority=3, 
    while_fits=True, is_multiple=True, add_in_reverse_order=True,
    add_if_fitted_labels=['indexed_context'],
    label='last_messages'
)
prompt.add('{assistant_response_in_progress}',
    role="assistant",
    presentation='Continue the response right after last symbol:'
)
prompt.add('{files}',
    is_multiple=True, while_fits=True, in_one_message=True, continue_if_doesnt_fit=True,
)



print(flow_prompt.call(prompt.id, {
    'messages': ['test1', 'test2'],
    'assistant_response_in_progress': None,
}, gpt4_behaviour))
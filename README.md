# Flow Prompt
The Flow Prompt was born after making another startup with LLM, and understanding that we need code which is working with budgeting, with dynamic data, with seeing metrics, such as latency cost. And looks that there is no such a library, all in one. You need to depend on many libs, but still you can't manage prompt dynamically, without a fear of budget exception.

This library will be valuable for you if you're using LLM for production purposes to work with prompt and distribution of the load across available AI Models. Also for RnD to test quickly different prompt approaches. And to records their results.
That Library will help you with:
- Prompt Development with dynamic data to avoid Budget Exceptions
- Using different LLMs, like OpenAi, Antropic, that list can be extended. Please create an issue or PR :)

With service usage you can get:
- Dynamic Prompt Changes
- Reviewing real-time interactions in the Prod, with request/response
- Testing new prompt online based on your historical data

## To install:
```
pip install flow-prompt
```

## Authentification

### Openai Keys
To add OPENAI_KEYs, you can do that by any of that instructions:
- add `OPENAI_API_KEY` as environment variable
- ```FlowPrompt(openai_api_key={key}, openai_org={org})```
- using as global variable:
```
import flow_prompt
flow_prompt.secrets.OPENAI_API_KEY = None
flow_prompt.secrets.OPENAI_ORG = None
```

### Azure Keys
Because Azure has several realms, and on each independent rate limits (not mentiniong credits), people do deploy on several realms the model. To add Azure keys please use:
- environment variables:
    - `AZURE_OPENAI_API_KEY`
    - `AZURE_OPENAI_ENDPOINT`
- using FlowPrompt
```FlowPrompt(azure_openai_keys={"name_of_realm":{"url": "https://baseurl.azure.com/", "key": "secret"}})```
- using global variable:
```
import flow_prompt
flow_prompt.secrets.AZURE_KEYS = {"uswest":{"url": "https://baseurl.azure.com/", "key": "secret"}}
```

### FlowPrompt Keys
To receive dynamic changes of the prompt, to record LLM interactions, metrics and other features by FlowPrompt, you need to get from https://flow-prompt.com API_TOKEN. To add it, please chose what best works for you:
- as env variable `FLOW_PROMPT_API_TOKEN` as env variable
- ```FlowPrompt(api_token={api_token})```
- using as global variable:
```
import flow_prompt
flow_prompt.secrets.API_TOKEN = None
```
## Usage examples:

1. You can easily create prompt on http://www.flow-prompt.com or in the code:
```
# id of the prompt
prompt = PipePrompt('merge_code') 

prompt.add("It's a system message, Hello {name}", role="system")

# Condition: Be sure that indexed context is prioritized first, add it's added until it fits.
# Start with `The closest indexed context`, if there are no indexed context do not add the "closest indexed context"
prompt.add('{indexed_context}',
    priority=2, 
    is_multiple=True, while_fits=True, in_one_message=True, continue_if_doesnt_fit=True,
    presentation="The closest indexed context to the user request:"
    label='indexed_context'
)
# add if in another msg was fitted fully by priority
prompt.add('{messages}', priority=3, 
    while_fits=True, is_multiple=True, add_in_reverse_order=True,
    add_if_fitted_labels=['indexed_context'],
    label='last_messages'
)
# add assistant response in context if it's in the context
prompt.add('{assistant_response_in_progress}',
    role="assistant",
    presentation='Continue the response right after last symbol:'
)
```

2. Use Behavious:
- use OPENAI_BEHAVIOR
- or add your own Behaviour, you can set max count of attempts, if you have different AI Models, if the first attempt will fail because of retryable error, the second will be called, based on the weights.
```
from flow_prompt import OPENAI_GPT4_0125_PREVIEW_BEHAVIOUR
flow_behaviour = OPENAI_GPT4_0125_PREVIEW_BEHAVIOUR
```
or:
```
from flow_prompt import behaviour
flow_behaviour = behaviour.AIModelsBehaviour(
    attempts=[
        AttemptToCall(
            ai_model=AzureAIModel(
                realm='us-east-1',
                deployment_name="gpt-4-1106-preview",
                max_tokens=C_128K,
                support_functions=True,
            ),
            weight=100,
        ),
        AttemptToCall(
            ai_model=OpenAIModel(
                model="gpt-4-1106-preview",
                max_tokens=C_128K,
                support_functions=True,
            ),
            weight=100,
        ),
    ]
)
```

3. Finally, call AI Model, with flow_prompt:
```
flow = FlowPrompt(
    azure_openai_keys={
        "name_realm":{
            "url": "https://baseurl.azure.com/",
            "key": "secret"
            }
        },
    openai_api_key={key},
    openai_org={org}
)
context = {"name": "World!"}
response = flow.call(
    prompt.id, context, flow_behaviour
)
```

4. Use the response:
Attrs will be set for the response:
```
response.finish_reason: str like (
    FINISH_REASON_LENGTH = "length"
    FINISH_REASON_ERROR = "error"
    FINISH_REASON_FINISH = "stop"
    FINISH_REASON_TOOL_CALLS = "tool_calls"
)
response.message: openai.types.chat.ChatCompletionMessage
response.content : str
response.original_result: openai.types.chat.ChatCompletion | openai.types.chat.AsyncStream[openai.types.chat.ChatCompletionChunk]
```

5. We know, it sounds like overhead. And we agree, but to have a mechanism of running production code you need at leat make a proper setup. And after you can manage your prompt easily:
```
response = flow_prompt.call(
    'production_prompt', context, gpt4_azure_models_with_openai_fallback
)

response = flow_prompt.call(
    'fine_tuned_production_prompt', context, gpt3_behaviour
)

```

### Best Security practices
!For production development we recommend to store secrets in secret storage, and do not use for that environment variables.



## For contributing:

### Rules
- Use f-strings instead .format(). F-strings are more concise and perform better and for our case less prone to bugs.
- Use pre-commit hooks. They re-format code automatically and run linters. Install them with `pre-commit install`.
 

1. **Install Poetry**

Poetry is a tool for dependency management and packaging in Python. Install it using the following command:
For details visit [python-poetry.org](https://python-poetry.org/docs/)
```shell
curl -sSL https://install.python-poetry.org | python3 -
```
3. Install pyenv
Pyenv is a tool for managing multiple Python versions. Install it using the following command:
```shell
curl https://pyenv.run | bash
```
3. **Set Python Version**
Choose the desired Python version (should be higher than 3.11). For example, to use Python 3.11, run:
```shell
pyenv install 3.11.5
pyenv global 3.11.5
```
4. Install all dependencies and activate poetry
```shell
poetry env use 3.11.5
poetry shell
```
If you are on Windows, and the installation failed, this is probably due to uWSGI dependency.
Try to install dependencies w/o it.
```shell
poetry install
```

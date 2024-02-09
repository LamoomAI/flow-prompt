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

## Authentification

### Openai Keys
To add OPENAI_KEYs, you can do that by any of that instructions:
- add `OPENAI_API_KEY`` as environment variable
- ```FlowPrompt(openai_api_key={key}, openai_org={org})```
- using as global variable:
```
import flow_prompt
flow_prompt.OPENAI_API_KEY = None
flow_prompt.OPENAI_ORG = None
```


### Azure Keys
Because Azure has several realms, and on each independent rate limits (not mentiniong credits), people do deploy on several realms the model. To add Azure keys please use:
- environment variables:
    - `AZURE_OPENAI_API_KEY`
    - `AZURE_OPENAI_ENDPOINT`
- using FlowPrompt
```FlowPrompt(azure_openai_keys={"name_realm":{"url": "https://baseurl.azure.com/", "key": "secret"}})```
- using global variable:
```
import flow_prompt
flow_prompt.AZURE_KEYS = {"uswest":{"url": "https://baseurl.azure.com/", "key": "secret"}}
```

### FlowPrompt Keys
To receive dynamic changes of the prompt, to record LLM interactions, metrics and other features by FlowPrompt, you need to get from https://flow-prompt.com API_TOKEN. To add it, please chose what best works for you:
- as env variable `FLOW_PROMPT_API_TOKEN` as env variable
- ```FlowPrompt(api_token={api_token})```
- using as global variable:
```
import flow_prompt
flow_prompt.API_TOKEN = None
```

### Best Security practices
!For production development we recommend to store secrets in secret storage, and do not use for that environment variables.

## Development rules
- Use f-strings instead .format(). F-strings are more concise and perform better and for our case less prone to bugs.
- Use pre-commit hooks. They re-format code automatically and run linters. Install them with `pre-commit install`.

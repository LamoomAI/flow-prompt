# Flow Prompt
The Flow Prompt was born after making another startup with LLM, and understanding that we need code which is working with budgeting, with dynamic data, with seeing metrics of latency, budget and cost in another product, because looks that there is no such a library.

So that library will be valuable for you if you're using LLM for production purposes to work with prompt and distribution of the load across available AI Models. Also for RnD to test quickly different promp
That Library will help you with:
- Prompt Development with dynamic data to avoid Budget Exceptions
- Using different LLMs, like OpenAi, Antropic, that list can be extended. Please create an issue or PR :)

Using the service you can get:
- Dynamic Prompt Changes
- Reviewing real-time interactions in the Prod, with request/response
- Testing new prompt online based on the historical data

## Authentification

### Openai Keys
To add OPENAI_KEYs, you can:
- add `OPENAI_API_KEY`` as environment variable
- ```FlowPrompt(openai_api_key={key}, openai_org={org})```
- using as global variable:
```
import flow_prompt
flow_prompt.OPENAI_API_KEY = None
flow_prompt.OPENAI_ORG = None
```


### Azure Keys
Because Azure has several realms, and on each independent rate limits (not mentiniong credits), people do deploy on several realms the model. To add Azure keys please:
- environment variables:
    - `AZURE_OPENAI_API_KEY`
    - `OPENAI_API_VERSION`
    - `AZURE_OPENAI_ENDPOINT`
- using FlowPrompt
```FlowPrompt(azure_openai_keys={"uswest":{"url": "https://baseurl.azure.com/", "key": "secret"}})```
- using global variable:
```
import flow_prompt
flow_prompt.AZURE_KEYS = {"uswest":{"url": "https://baseurl.azure.com/", "key": "secret"}}
```

### FlowPrompt Keys
To receive dynamic changes of the prompt, to record LLM interactions, metrics and other features by FlowPrompt, you need to get from https://flow-prompt.com API_TOKEN. And use it in any way:
- as env variable `FLOW_PROMPT_API_TOKEN` as env variable
- ```FlowPrompt(api_token={api_token})```
- using as global variable:
```
import flow_prompt
flow_prompt.API_TOKEN = None
```

### Best Security practices
For production development we recommend to store secrets in secret storage, and do not use for that environment variables.


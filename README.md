# Flow Prompt

## Introduction

Flow Prompt is a dynamic, all-in-one library designed for managing and optimizing prompts for large language models (LLMs) in production and R&D settings. It facilitates budget-aware operations, dynamic data integration, latency and cost metrics visibility, and efficient load distribution across multiple AI models.

## Features

- **Dynamic Prompt Development**: Avoid budget exceptions with dynamic data.
- **Multi-Model Support**: Seamlessly integrate with various LLMs like OpenAI, Anthropic, and more.
- **Real-Time Insights**: Monitor interactions, request/response metrics in production.
- **Prompt Testing and Evolution**: Quickly test and iterate on prompts using historical data.

## Installation

Install Flow Prompt using pip:

```bash
pip install flow-prompt
```

## Authentication
### OpenAI Keys
```python
# setting as os.env
os.setenv('OPENAI_API_KEY', 'your_key_here')
# or creating flow_prompt obj
FlowPrompt(openai_key="your_key", openai_org="your_org")
```

### Azure Keys
Add Azure keys to accommodate multiple realms:
```python
# setting as os.env
os.setenv('AZURE_KEYS', '{"name_realm":{"url": "https://baseurl.azure.com/","key": "secret"}}')
# or creating flow_prompt obj
FlowPrompt(azure_keys={"realm_name":{"url": "https://baseurl.azure.com/", "key": "your_secret"}})
```

### FlowPrompt Keys
Obtain an API token from Flow Prompt and add it:

```python
# As an environment variable:
os.setenv('FLOW_PROMPT_API_TOKEN', 'your_token_here')
# Via code: 
FlowPrompt(api_token='your_api_token')
```

Usage Examples:

```python
from flow_prompt import FlowPrompt, PipePrompt

# Initialize and configure FlowPrompt
flow = FlowPrompt(openai_key='your_api_key', openai_org='your_org')

# Create a prompt
prompt = PipePrompt('greet_user')
prompt.add("Hello {name}", role="system")

# Call AI model with FlowPrompt
context = {"name": "John Doe"}
response = flow.call(prompt.id, context)
print(response.content)
```
For more examples, visit Flow Prompt Usage Documentation.

## Best Security Practices
For production environments, it is recommended to store secrets securely and not directly in your codebase. Consider using a secret management service or encrypted environment variables.

## Contributing
We welcome contributions! Please see our Contribution Guidelines for more information on how to get involved.

## License
This project is licensed under the Apache2.0 License - see the [LICENSE](LICENSE.txt) file for details.

## Contact
For support or contributions, please contact us via GitHub Issues.
from flow_prompt import C_16K
from flow_prompt.ai_models import behaviour
from flow_prompt.ai_models.claude.claude_model import ClaudeAIModel
from flow_prompt.ai_models.gemini.gemini_model import GeminiAIModel
from flow_prompt.ai_models.openai.azure_models import AzureAIModel
from flow_prompt.ai_models.openai.openai_models import OpenAIModel
from flow_prompt.prompt.base_prompt import ImagePromptContent
from flow_prompt.prompt.flow_prompt import FlowPrompt
from flow_prompt.prompt.pipe_prompt import AttemptToCall, PipePrompt

fp = FlowPrompt()

prompt = PipePrompt("picture_description")

image = ImagePromptContent.load_from_url("your_image_url", mime_type="your_image_mime_type")
prompt.add(content_type=image.content_type, content=image.dump(), role="user")
prompt.add("Describe the picture", role="user")

def_behaviour = behaviour.AIModelsBehaviour(
    attempts=[
        AttemptToCall(
            ai_model=ClaudeAIModel(
                max_tokens=C_16K, 
                model="claude-3-5-opus"
            ),
            weight=100,
        ),
        AttemptToCall(
            ai_model=GeminiAIModel(
                max_tokens=C_16K, 
                model="gemini-1.5-flash"
            ),
            weight=100,
        ),
        AttemptToCall(
            ai_model=OpenAIModel(
                max_tokens=C_16K, 
                model="gpt-4o-mini"
            ),
            weight=100,
        ),
        AttemptToCall(
            ai_model=AzureAIModel(
                realm='useast',
                deployment_id='gpt-4o-mini',
                max_tokens=C_16K,
            ),
            weight=100
        ),
    ]
)

response = fp.call(prompt.id, {}, def_behaviour)

print(response.content)

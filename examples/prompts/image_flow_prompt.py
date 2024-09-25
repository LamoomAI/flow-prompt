from flow_prompt import C_16K
from flow_prompt.ai_models import behaviour
from flow_prompt.ai_models.claude.claude_model import ClaudeAIModel
from flow_prompt.ai_models.gemini.gemini_model import GeminiAIModel
from flow_prompt.ai_models.openai.azure_models import AzureAIModel
from flow_prompt.ai_models.openai.openai_models import OpenAIModel
from flow_prompt.prompt.chat import ImagePromptContent
from flow_prompt.prompt.flow_prompt import FlowPrompt
from flow_prompt.prompt.pipe_prompt import AttemptToCall, PipePrompt
from flow_prompt.utils import parse_mime_type

fp = FlowPrompt()

prompt = PipePrompt("image_description")

prompt.add("What is movie character in the picture?", role="user")
prompt.add(content_type="image", ref_image="character")

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
                realm='us-east-1',
                deployment_id='gpt-4o-mini',
                max_tokens=C_16K,
            ),
            weight=100
        ),
    ]
)

image_url = "your_image_url"
response = fp.call(prompt.id, 
                   {},
                   def_behaviour, 
                   images={"character": ImagePromptContent.load_from_url(image_url, parse_mime_type(image_url))}
                )

print(response.content)

from flow_prompt.prompt.base_prompt import ImagePromptContent
from flow_prompt.prompt.pipe_prompt import PipePrompt

prompt = PipePrompt("image_prompt")

image = ImagePromptContent(base64_image="your_base64_encoded_image", mime_type="image_mime_type")
prompt.add(content=image.dump(), content_type=image.content_type)

image = ImagePromptContent.load_from_url(url="your_image_url", mime_type="image_mime_type")
prompt.add(content_type=image.content_type, content=image.dump())

image = ImagePromptContent.load_from_path(path="path_to_image",mime_type="image_mime_type")
prompt.add(content=image.dump(), content_type=image.content_type)




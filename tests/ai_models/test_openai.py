from flow_prompt.ai_models.constants import C_16K
from flow_prompt.ai_models.openai.openai_models import OpenAIModel

def test_openai_model_prepare_text_message():
    model_name = "gpt-4o-mini"
    model = OpenAIModel(model=model_name, max_tokens=C_16K)

    msg = model.prepare_message({"role": "user", "type": "text", "content": "Hello"})
    assert msg["role"] == "user"
    assert msg["content"] == [{"type": "text", "text": "Hello"}]

def test_openai_model_prepare_image_message():
    model_name = "gpt-4o-mini"
    model = OpenAIModel(model=model_name, max_tokens=C_16K)

    msg = model.prepare_message({"role": "user", "type": "image", "content": {
        "image": "base64_str", "mime_type":"image/jpeg"}})
    assert msg["role"] == "user"
    assert msg["content"] == [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,base64_str" }}]

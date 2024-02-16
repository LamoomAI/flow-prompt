from flow_prompt.ai_models.attempt_to_call import AttemptToCall
from flow_prompt.ai_models.behaviour import AIModelsBehaviour
from flow_prompt.ai_models.openai.openai_models import (
    C_128K,
    C_4K,
    OpenAIModel,
    C_16K,
    C_32K,
)


OPENAI_GPT4_0125_PREVIEW_BEHAVIOUR = AIModelsBehaviour(
    attempts=[
        AttemptToCall(
            ai_model=OpenAIModel(
                model="gpt-4-0125-preview",
                max_tokens=C_128K,
                support_functions=True,
                should_verify_client_has_creds=False,
            ),
            weight=100,
        ),
    ]
)

OPENAI_GPT4_1106_PREVIEW_BEHAVIOUR = AIModelsBehaviour(
    attempts=[
        AttemptToCall(
            ai_model=OpenAIModel(
                model="gpt-4-1106-preview",
                max_tokens=C_128K,
                support_functions=True,
                should_verify_client_has_creds=False,
            ),
            weight=100,
        ),
    ]
)

OPENAI_GPT4_1106_VISION_PREVIEW_BEHAVIOUR = AIModelsBehaviour(
    attempts=[
        AttemptToCall(
            ai_model=OpenAIModel(
                model="gpt-4-1106-vision-preview",
                max_tokens=C_128K,
                support_functions=True,
                should_verify_client_has_creds=False,
            ),
            weight=100,
        ),
    ]
)

OPENAI_GPT4_BEHAVIOUR = AIModelsBehaviour(
    attempts=[
        AttemptToCall(
            ai_model=OpenAIModel(
                model="gpt-4",
                max_tokens=C_4K,
                support_functions=True,
                should_verify_client_has_creds=False,
            ),
            weight=100,
        ),
    ]
)

OPENAI_GPT4_32K_BEHAVIOUR = AIModelsBehaviour(
    attempts=[
        AttemptToCall(
            ai_model=OpenAIModel(
                model="gpt-4-32k",
                max_tokens=C_32K,
                support_functions=True,
                should_verify_client_has_creds=False,
            ),
            weight=100,
        ),
    ]
)

OPENAI_GPT3_5_TURBO_0125_BEHAVIOUR = AIModelsBehaviour(
    attempts=[
        AttemptToCall(
            ai_model=OpenAIModel(
                model="gpt-3.5-turbo-0125",
                max_tokens=C_16K,
                support_functions=True,
                should_verify_client_has_creds=False,
            ),
            weight=100,
        ),
    ]
)

OPENAI_GPT3_5_TURBO_INSTRUCT_BEHAVIOUR = AIModelsBehaviour(
    attempts=[
        AttemptToCall(
            ai_model=OpenAIModel(
                model="gpt-3.5-turbo-instruct",
                max_tokens=C_4K,
                support_functions=True,
                should_verify_client_has_creds=False,
            ),
            weight=100,
        ),
    ]
)

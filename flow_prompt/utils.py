import logging
import os


from flow_prompt import settings

from time import time
import typing as t

logger = logging.getLogger(__name__)

def curr_timestamp_in_ms():
    return int(time() * 1000)


def resolve(prompt: str, context: t.Dict[str, str]) -> str:
    if not prompt or "{" not in prompt:
        return prompt
    # TODO: monitor how many values were not resolved and what values
    for key in context:
        prompt = prompt.replace(f"{{{key}}}", str(context[key]))
    return prompt

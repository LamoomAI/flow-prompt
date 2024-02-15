from dataclasses import dataclass
import json
import os

@dataclass
class Secrets:
    API_TOKEN = os.getenv("FLOW_PROMPT_API_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_ORG = os.getenv("OPENAI_ORG")
    AZURE_OPENAI_KEYS = json.loads(os.getenv("AZURE_OPENAI_KEYS"))

secrets = Secrets()

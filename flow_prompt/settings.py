from dataclasses import dataclass, field
import json
import os

from flow_prompt.utils import parse_bool


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_SCRIPTS_DIR = os.environ.get(
    "FLOW_PROMPT_TEMP_SCRIPTS_DIR", os.path.join(BASE_DIR, "temp_scripts")
)
SAVE_PROMPTS_LOCALLY = os.environ.get("FLOW_PROMPT_SAVE_PROMPTS_LOCALLY", False)
ENVIRONMENT = os.environ.get("FLOW_PROMPT_ENVIRONMENT", "prod")

DEFAULT_MAX_BUDGET = os.environ.get("FLOW_PROMPT_DEFAULT_MAX_BUDGET", 16000)
DEFAULT_SAMPLE_MIN_BUDGET = os.environ.get("FLOW_PROMPT_DEFAULT_ANSWER_BUDGET", 3000)
DEFAULT_PROMPT_BUDGET = os.environ.get(
    "FLOW_PROMPT_DEFAULT_PROMPT_BUDGET", DEFAULT_MAX_BUDGET - DEFAULT_SAMPLE_MIN_BUDGET
)

EXPECTED_MIN_BUDGET_FOR_VALUABLE_INPUT = os.environ.get(
    "FLOW_PROMPT_EXPECTED_MIN_BUDGET_FOR_VALUABLE_INPUT", 100
)

SAFE_GAP_TOKENS: int = os.environ.get("FLOW_PROMPT_SAFE_GAP_TOKENS", 100)
SAFE_GAP_PER_MSG: int = os.environ.get("FLOW_PROMPT_SAFE_GAP_PER_MSG", 4)
DEFAULT_ENCODING = "cl100k_base"

USE_API_SERVICE = parse_bool(os.environ.get("FLOW_PROMPT_USE_API_SERVICE", True))
FLOW_PROMPT_API_URI = os.environ.get(
    "FLOW_PROMPT_API_URI", "https://api.flow-prompt.com/"
)

CACHE_PROMPT_FOR_EACH_SECONDS = int(
    os.environ.get("FLOW_PROMPT_CACHE_PROMPT_FOR_EACH_SECONDS", 5 * 60)
)  # 5 minutes by default
RECEIVE_PROMPT_FROM_SERVER = parse_bool(
    os.environ.get("FLOW_PROMPT_RECEIVE_PROMPT_FROM_SERVER", True)
)
PIPE_PROMPTS = {}


@dataclass
class Secrets:
    API_TOKEN: str = field(default_factory=lambda: os.getenv("FLOW_PROMPT_API_TOKEN"))
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    CLAUDE_API_KEY: str = field(default_factory=lambda: os.getenv("CLAUDE_API_KEY"))
    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    OPENAI_ORG: str = field(default_factory=lambda: os.getenv("OPENAI_ORG"))
    azure_keys: dict = field(
        default_factory=lambda: json.loads(os.getenv("azure_keys", "{}"))
    )

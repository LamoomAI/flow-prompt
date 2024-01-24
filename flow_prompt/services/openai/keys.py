import typing as t
from dataclasses import dataclass


class BaseOpenAIKey:
    api_type: str = "openai"

    def is_openai(self) -> bool:
        return self.api_type == "openai"

    def is_azure(self) -> bool:
        return self.api_type == "azure"


@dataclass
class OpenAiKey(BaseOpenAIKey):
    org: str
    key: str


@dataclass
class AzureOpenAIKey(BaseOpenAIKey):
    api_base: str
    api_key: str
    api_type: str = "azure"
    api_version: str = "2023-05-15"
    deployment_name: t.Optional[str] = None

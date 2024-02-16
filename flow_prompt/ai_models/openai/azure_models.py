import logging
import typing as t
from dataclasses import dataclass

from openai import AzureOpenAI

from flow_prompt import settings
from flow_prompt.ai_models.ai_model import AI_MODELS_PROVIDER
from flow_prompt.ai_models.openai.openai_models import FamilyModel, OpenAIModel
from flow_prompt.exceptions import ProviderNotFoundException

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class AzureAIModel(OpenAIModel):
    realm: t.Optional[str]
    deployment_name: t.Optional[str]
    provider: AI_MODELS_PROVIDER = AI_MODELS_PROVIDER.AZURE
    model: t.Optional[str] = None

    def __str__(self) -> str:
        return f"{self.realm}-{self.deployment_name}-{self.family}"

    def _define_family(self):
        if self.deployment_name.startswith("davinci"):
            self.family = FamilyModel.instruct_gpt.value
        elif self.deployment_name.startswith(("gpt3", "gpt-3")):
            self.family = FamilyModel.chat.value
        elif self.deployment_name.startswith(("gpt4", "gpt-4", "gpt")):
            self.family = FamilyModel.gpt4.value
        else:
            logger.warning(
                f"Unknown family for {self.deployment_name}. Please add it obviously. Setting as GPT4"
            )
            self.family = FamilyModel.gpt4.value

    def _define_tiktoken_encoding(self):
        if self.family in (FamilyModel.chat.value, FamilyModel.gpt4.value):
            self.tiktoken_encoding = "cl100k_base"
        elif self.family == FamilyModel.instruct_gpt.value:
            self.tiktoken_encoding = ""
        else:
            logger.warning(
                f"Unknown realm for {self.deployment_name}. Please add it obviously. Setting as cl100k_base"
            )
            self.tiktoken_encoding = "cl100k_base"

    def __post_init__(self):
        if not self.family:
            if self.deployment_name.startswith("davinci"):
                self.family = FamilyModel.instruct_gpt.value
            elif self.deployment_name.startswith(("gpt3", "gpt-3")):
                self.family = FamilyModel.chat.value
            elif self.deployment_name.startswith(("gpt4", "gpt-4", "gpt")):
                self.family = FamilyModel.gpt4.value
            else:
                logger.warning(
                    f"Unknown family for {self.deployment_name}. Please add it obviously. Setting as GPT4"
                )
                self.family = FamilyModel.gpt4.value
        if self.should_verify_client_has_creds:
            self.verify_client_has_creds()
        logger.info(f"Initialized AzureAIModel: {self}")

    def verify_client_has_creds(self):
        if self.realm not in settings.AI_CLIENTS[self.provider]:
            raise ProviderNotFoundException(
                f"Realm {self.realm} not found in AI_CLIENTS"
            )

    @property
    def name(self) -> str:
        return f"{self.deployment_name}-{self.realm}"

    def get_params(self) -> t.Dict[str, t.Any]:
        return {
            "model": self.deployment_name,
        }

    def get_client(self):
        return settings.AI_CLIENTS[self.provider][self.realm]

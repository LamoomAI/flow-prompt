class FlowPromptError(Exception):
    pass


class RetryableCustomError(FlowPromptError):
    pass


class FlowPromptIsnotFoundError(FlowPromptError):
    pass


class BehaviourIsNotDefined(FlowPromptError):
    pass


class ConnectionLostError(FlowPromptError):
    pass


class ValueIsNotResolvedError(FlowPromptError):
    pass


class NotEnoughBudgetError(FlowPromptError):
    pass


class NotFoundPromptError(FlowPromptError):
    pass


class ProviderNotFoundError(FlowPromptError):
    pass


class NotParsedResponseException(FlowPromptError):
    pass

class APITokenNotProvided(FlowPromptError):
    pass
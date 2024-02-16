class FlowPromptException(Exception):
    pass


class RetryableCustomException(FlowPromptException):
    pass


class FlowPromptIsnotFoundException(FlowPromptException):
    pass


class BehaviourIsNotDefined(FlowPromptException):
    pass


class ValueIsNotResolvedException(FlowPromptException):
    pass


class NotEnoughBudgetException(FlowPromptException):
    pass


class NotFoundPromptException(FlowPromptException):
    pass


class ProviderNotFoundException(FlowPromptException):
    pass

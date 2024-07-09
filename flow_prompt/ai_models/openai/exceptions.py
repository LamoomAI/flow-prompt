from flow_prompt.exceptions import FlowPromptError, RetryableCustomError


class OpenAIChunkedEncodingError(RetryableCustomError):
    pass


class OpenAITimeoutError(RetryableCustomError):
    pass


class OpenAIResponseWasFilteredError(RetryableCustomError):
    pass


class OpenAIAuthenticationError(RetryableCustomError):
    pass


class OpenAIInternalError(RetryableCustomError):
    pass


class OpenAiRateLimitError(RetryableCustomError):
    pass


class OpenAiPermissionDeniedError(RetryableCustomError):
    pass


class OpenAIUnknownError(RetryableCustomError):
    pass


### Non-retryable Errors ###
class OpenAIInvalidRequestError(FlowPromptError):
    pass


class OpenAIBadRequestError(FlowPromptError):
    pass


class ConnectionCheckError(FlowPromptError):
    pass

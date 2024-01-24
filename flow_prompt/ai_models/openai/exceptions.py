from flow_prompt.exceptions import RetryableCustomException, FlowPromptException


class OpenAIChunkedEncodingError(RetryableCustomException):
    pass


class OpenAITimeoutException(RetryableCustomException):
    pass


class OpenAIResponseWasFilteredError(RetryableCustomException):
    pass


class OpenAIAuthenticationError(RetryableCustomException):
    pass


class OpenAIInternalException(RetryableCustomException):
    pass


class OpenAiRateLimitException(RetryableCustomException):
    pass

class OpenAiPermissionDeniedError(RetryableCustomException):
    pass

class OpenAIUnknownException(RetryableCustomException):
    pass

### Not retryable exceptions ###
class OpenAIInvalidRequestError(FlowPromptException):
    pass

class OpenAIBadRequestException(FlowPromptException):
    pass
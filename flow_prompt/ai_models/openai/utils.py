
import request
import openai
import logging
from urllib import request
from .exceptions import (
    OpenAIUnknownException,
    OpenAiRateLimitException,
    OpenAIInternalException,
    OpenAIInvalidRequestError,
    OpenAIAuthenticationError,
    OpenAIResponseWasFilteredError,
    OpenAITimeoutException,
    OpenAIChunkedEncodingError,
    OpenAiPermissionDeniedError,
    OpenAIBadRequestException,
)

logger = logging.getLogger(__name__)


def raise_openai_exception(
    exc: Exception,
) -> None:
    if isinstance(exc, request.exceptions.ChunkedEncodingError):
        raise OpenAIChunkedEncodingError()

    if isinstance(exc, openai.APITimeoutError):
        raise OpenAITimeoutException()

    if isinstance(exc, openai.BadRequestError):
        if "response was filtered" in str(exc):
            raise OpenAIResponseWasFilteredError()
        if "Too many inputs" in str(exc):
            raise OpenAiRateLimitException()
        raise OpenAIInvalidRequestError()
    if isinstance(exc, openai.RateLimitError):
        raise OpenAiRateLimitException()
    
    if isinstance(exc, openai.AuthenticationError):
        raise OpenAIAuthenticationError()
    
    if isinstance(exc, openai.InternalServerError):
        raise OpenAIInternalException()
    
    if isinstance(exc, openai.PermissionDeniedError):
        raise OpenAiPermissionDeniedError()
    
    if isinstance(exc, openai.APIStatusError):
        raise OpenAIBadRequestException()

    logger.error(
        "Unknown OPENAI error, please add it in raise_openai_rate_limit_exception",
        exc_info=exc,
    )
    raise OpenAIUnknownException()
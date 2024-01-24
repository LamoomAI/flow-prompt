import logging

import openai
import requests

from flow_prompt.exceptions import (
    OpenAIUnknownException,
    OpenAiOverloadedException,
    OpenAiRateLimitException,
    OpenAIWrongAuth,
    OpenAIInternalException,
    OpenAIInvalidRequestError,
    OpenAIResponseWasFilteredError,
    OpenAITimeoutException,
    OpenAIChunkedEncodingError,
    CustomException,
)
from server.utils import curr_timestamp_in_ms

logger = logging.getLogger(__name__)


def raise_openai_rate_limit_exception(
    exc: Exception,
) -> None:
    if isinstance(exc, CustomException):
        raise exc

    if isinstance(exc, requests.exceptions.ChunkedEncodingError):
        raise OpenAIChunkedEncodingError()

    if isinstance(exc, openai.APITimeoutError):
        raise OpenAITimeoutException()

    if isinstance(exc, openai.BadRequestError):
        if "response was filtered" in str(exc):
            raise OpenAIResponseWasFilteredError()
        if "Too many inputs" in str(exc):
            raise OpenAiRateLimitException()
        raise OpenAIInvalidRequestError()

    if isinstance(exc, openai.APIError) and not isinstance(exc, openai.RateLimitError):
        logger.error(
            f"APIError error: {exc}",
            exc_info=exc,
        )
        raise OpenAIInternalException()

    if isinstance(exc, openai.AuthenticationError):
        raise OpenAIWrongAuth()

    if not isinstance(exc, openai.RateLimitError):
        logger.error(
            "Unknown OPENAI error, please add it in raise_openai_rate_limit_exception",
            exc_info=exc,
        )
        raise OpenAIUnknownException()

    if "overloaded" in str(exc):
        raise OpenAiOverloadedException()
    else:
        raise OpenAiRateLimitException()


# decorator to send latency metric of function in CloudWatch
def openai_raise_custom_exception(function_name):
    def decorator(function):
        def wrapper(*args, **kwargs):
            start_time = curr_timestamp_in_ms()
            with_azure = kwargs.get("with_azure", False)
            raised = True
            result = None
            try:
                try:
                    result = function(*args, **kwargs)
                    raised = False
                    if result.exception:
                        raise result.exception
                except Exception as e:
                    raise_openai_rate_limit_exception(e)
            except CustomException as e:
                timing = curr_timestamp_in_ms() - start_time
                logger.info(f"[Latency][{function_name}] = {timing} failed: {e}")
                if not raised:
                    return result
                raise e
            except Exception as e:
                logger.exception("Unknown exception in calling openai")
                value = str(e)
                first_sentence = value.split(".")[0]
                timing = (curr_timestamp_in_ms() - start_time) / 1000.0
                logger.info(f"[Latency][{function_name}] = {timing} failed: {e}")
                if not raised:
                    return result
                raise e

            timing = curr_timestamp_in_ms() - start_time
            logger.info(f"[Latency][{function_name}] = {timing} passed")
            return result

        return wrapper

    return decorator

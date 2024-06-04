from typing import List
import os
import time
import json
import logging
import openai
import asyncio
from requests.exceptions import SSLError
import httpx
from loguru import logger
from models.models import Source
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from functools import lru_cache, wraps
from openai import AsyncOpenAI, APIError, RateLimitError, Timeout
import requests

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = AsyncOpenAI(api_key=OPENAI_API_KEY, max_retries=0)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIError(Exception):
    """Custom exception for API errors."""
    pass

class MaxRetriesExceeded(Exception):
    """Custom exception for exceeding maximum retries."""
    pass

def rate_limit(wait_time: float):
    """Rate limit decorator that pauses execution to limit the number of calls per minute."""
    def decorator(func):
        last_called = [0.0]

        @wraps(func)
        async def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait = max(0, wait_time - elapsed)
            if wait > 0:
                await asyncio.sleep(wait)
            result = await func(*args, **kwargs)
            last_called[0] = time.time()
            return result

        return wrapper
    return decorator


def is_retriable_exception(exception):
    if isinstance(exception, (APIError, RateLimitError, Timeout, httpx.RequestError, SSLError)):
        return True
    elif hasattr(exception, 'code') and exception.code == 'insufficient_quota':
        return True
    elif isinstance(exception, httpx.HTTPStatusError) and exception.response.status_code == 429:
        return True
    return False

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
)
@rate_limit(20)  # 3 requests per minute (20 seconds wait time)
async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Embed texts using OpenAI's ada model.

    Args:
        texts: The list of texts to embed.

    Returns:
        A list of embeddings, each of which is a list of floats.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    try:
        deployment = os.environ.get("OPENAI_EMBEDDINGMODEL_DEPLOYMENTID")

        response = {}
        if deployment is None:
            response = await client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        else:
            response = await client.embeddings.create(input=texts, deployment_id=deployment)

        data = response.data  # type: ignore

        return [result.embedding for result in data]
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded: {e}")
        raise
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429 and 'insufficient_quota' in str(e):
            logger.error("Insufficient quota. Please check your OpenAI plan and billing details.")
            raise RuntimeError("Insufficient quota. Please check your OpenAI plan and billing details.")
        else:
            logger.error(f"HTTP error occurred: {e}")
            raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
)
@rate_limit(20)  # 3 requests per minute (20 seconds wait time)
async def get_chat_completion(
    messages,
    model="gpt-3.5-turbo",
    deployment_id=None,
) -> str:
    """
    Generate a chat completion using OpenAI's chat completion API.

    Args:
        messages: The list of messages in the chat history.
        model: The name of the model to use for the completion. Default is gpt-3.5-turbo, which is a fast, cheap and versatile model. Use gpt-4 for higher quality but slower results.

    Returns:
        A string containing the chat completion.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    response = {}
    if deployment_id is None:
        response = await client.chat.completions.create(model=model, messages=messages, max_tokens= 128)
    else:
        response = await client.chat.completions.create(deployment_id=deployment_id, messages=messages, max_tokens= 128)

    choices = response.choices  # type: ignore
    completion = choices[0].message.content.strip()
    logger.info(f"Completion: {completion}")
    return completion

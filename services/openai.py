from typing import List
from openai import OpenAI

client = OpenAI()
import os
from loguru import logger

from tenacity import retry, wait_exponential, stop_after_attempt

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")


@retry(wait=wait_exponential(multiplier = 2,min=20, max=20000), stop=stop_after_attempt(2000))
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Embed texts using OpenAI's ada model.

    Args:
        texts: The list of texts to embed.

    Returns:
        A list of embeddings, each of which is a list of floats.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    # Call the OpenAI API to get the embeddings
    # NOTE: Azure Open AI requires deployment id
    deployment = os.environ.get("OPENAI_EMBEDDINGMODEL_DEPLOYMENTID")

    response = {}
    if deployment is None:
        response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    else:
        response = client.embeddings.create(input=texts, deployment_id=deployment)

    # Extract the embedding data from the response
    data = response.data  # type: ignore

    # Return the embeddings as a list of lists of floats
    return [result["embedding"] for result in data]


@retry(wait=wait_exponential(multiplier = 2,min=20, max=20000), stop=stop_after_attempt(2000))
def get_chat_completion(
    messages,
    model="gpt-3.5-turbo",  # use "gpt-4" for better results
    deployment_id=None,
):
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
    # call the OpenAI chat completion API with the given messages
    # Note: Azure Open AI requires deployment id
    response = {}
    if deployment_id == None:
        response = client.chat.completions.create(model=model,
        messages=messages)
    else:
        response = client.chat.completions.create(deployment_id=deployment_id,
        messages=messages)

    choices = response.choices  # type: ignore
    completion = choices[0].message.content.strip()
    logger.info(f"Completion: {completion}")
    return completion

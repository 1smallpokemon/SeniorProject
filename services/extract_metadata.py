from models.models import Source
from services.openai import get_chat_completion
import json
from typing import Dict
import os
from loguru import logger
from tenacity import retry, wait_exponential, stop_after_attempt


async def extract_metadata_from_document(text: str) -> Dict[str, str]:
    sources = Source.__members__.keys()
    sources_string = ", ".join(sources)
    messages = [
        {
            "role": "system",
            "content": f"""
            Given a document from a user, try to extract the following metadata:
            - source: string, one of {sources_string}
            - url: string or don't specify
            - created_at: string or don't specify
            - author: string or don't specify

            Respond with a JSON containing the extracted metadata in key value pairs. If you don't find a metadata field, don't specify it.
            """,
        },
        {"role": "user", "content": text},
    ]

    completion = await get_chat_completion(messages, "gpt-4")
    logger.info(f"completion: {completion}")

    try:
        metadata = json.loads(completion)
    except Exception as e:
        logger.error(f"Error parsing completion: {e}")
        metadata = {}

    return metadata
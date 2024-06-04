from typing import Dict, List, Optional, Tuple
import uuid
import os
from models.models import Document, DocumentChunk, DocumentChunkMetadata
import time
from loguru import logger
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import tiktoken

from services.openai import get_embeddings, is_retriable_exception
from functools import lru_cache
import asyncio
from functools import wraps

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Execution time for {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper

async def retry_with_backoff(func, *args, max_retries=3, initial_wait=1, max_wait=30, **kwargs):
    """Retry function with exponential backoff."""
    retries = 0
    wait = initial_wait
    while retries < max_retries:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if not is_retriable_exception(e):
                raise
            retries += 1
            await asyncio.sleep(wait)
            wait = min(wait * 2, max_wait)
            logger.warning(f"Retrying {func.__name__} due to {e} ({retries}/{max_retries})")
    raise Exception(f"Exceeded maximum retries for {func.__name__}")

@lru_cache(maxsize=100)
def cached_tokenizer(text: str) -> List[int]:
    return tokenizer.encode(text, disallowed_special=())
# Global variables
# --------------------------------------

tokenizer = tiktoken.get_encoding(
    "cl100k_base"
)  # The encoding scheme to use for tokenization

# Constants
CHUNK_SIZE = 50  # The target size of each text chunk in tokens
MIN_CHUNK_SIZE_CHARS = 50  # The minimum size of each text chunk in characters
MIN_CHUNK_LENGTH_TO_EMBED = 5  # Discard chunks shorter than this
EMBEDDINGS_BATCH_SIZE = int(
    os.environ.get("OPENAI_EMBEDDING_BATCH_SIZE", 3)
)  # The number of embeddings to request at a time
MAX_NUM_CHUNKS = 10000  # The maximum number of chunks to generate from a text



def parallel_get_text_chunks(documents: List[Document], chunk_token_size: Optional[int]) -> Dict[str, List[DocumentChunk]]:
    """
    Split a list of documents into chunks of ~CHUNK_SIZE tokens, based on punctuation and newline boundaries.

    Args:
        documents: The list of documents to split into chunks.
        chunk_token_size: The target size of each chunk in tokens, or None to use the default CHUNK_SIZE.

    Returns:
        A dictionary mapping each document id to a list of text chunks, each of which is a string of ~CHUNK_SIZE tokens.
    """
    # Check if the document text is empty or whitespace
    if not documents:
        return {}

    # Get the number of available CPU cores
    num_cpus = multiprocessing.cpu_count()

    # Create a ProcessPoolExecutor with the number of available CPU cores
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        # Submit a task to create document chunks for each document
        futures = {executor.submit(create_document_chunks, doc, chunk_token_size): doc.id for doc in documents}

        # Collect the results of the tasks
        chunks = {}
        for future in as_completed(futures):
            doc_id = futures[future]
            try:
                doc_chunks, _ = future.result()
                chunks[doc_id] = doc_chunks
            except Exception as exc:
                logger.error(f"Document id {doc_id} generated an exception: {exc}")

    # Return the dictionary of document chunks
    return chunks


@log_execution_time
def get_text_chunks(text: str, chunk_token_size: Optional[int]) -> List[str]:
    """
    Split a text into chunks of ~CHUNK_SIZE tokens, based on punctuation and newline boundaries.

    Args:
        text: The text to split into chunks.
        chunk_token_size: The target size of each chunk in tokens, or None to use the default CHUNK_SIZE.

    Returns:
        A list of text chunks, each of which is a string of ~CHUNK_SIZE tokens.
    """
    # Return an empty list if the text is empty or whitespace
    if not text or text.isspace():
        return []

    # Tokenize the text
    tokens = cached_tokenizer(text)

    # Initialize an empty list of chunks
    chunks = []

    # Use the provided chunk token size or the default one
    chunk_size = chunk_token_size or CHUNK_SIZE

    # Initialize a counter for the number of chunks
    num_chunks = 0

    # Loop until all tokens are consumed
    while tokens and num_chunks < MAX_NUM_CHUNKS:
        # Take the first chunk_size tokens as a chunk
        chunk = tokens[:chunk_size]

        # Decode the chunk into text
        chunk_text = tokenizer.decode(chunk)

        # Skip the chunk if it is empty or whitespace
        if not chunk_text or chunk_text.isspace():
            # Remove the tokens corresponding to the chunk text from the remaining tokens
            tokens = tokens[len(chunk) :]
            # Continue to the next iteration of the loop
            continue

        # Find the last period or punctuation mark in the chunk
        last_punctuation = max(
            chunk_text.rfind("."),
            chunk_text.rfind("?"),
            chunk_text.rfind("!"),
            chunk_text.rfind("\n"),
        )

        # If there is a punctuation mark, and the last punctuation index is before MIN_CHUNK_SIZE_CHARS
        if last_punctuation != -1 and last_punctuation > MIN_CHUNK_SIZE_CHARS:
            # Truncate the chunk text at the punctuation mark
            chunk_text = chunk_text[: last_punctuation + 1]

        # Remove any newline characters and strip any leading or trailing whitespace
        chunk_text_to_append = chunk_text.replace("\n", " ").strip()

        if len(chunk_text_to_append) > MIN_CHUNK_LENGTH_TO_EMBED:
            # Append the chunk text to the list of chunks
            chunks.append(chunk_text_to_append)

        # Remove the tokens corresponding to the chunk text from the remaining tokens
        tokens = tokens[len(cached_tokenizer(chunk_text)) :]

        # Increment the number of chunks
        num_chunks += 1

    # Handle the remaining tokens
    if tokens:
        remaining_text = tokenizer.decode(tokens).replace("\n", " ").strip()
        if len(remaining_text) > MIN_CHUNK_LENGTH_TO_EMBED:
            chunks.append(remaining_text)

    return chunks

def parallel_create_document_chunks(documents: List[Document], chunk_token_size: Optional[int]) -> Dict[str, List[DocumentChunk]]:
    """
    Create a list of document chunks from a list of document objects and return a dictionary mapping document ids to lists of document chunks.

    Args:
        documents: The list of document objects to create chunks from. Each document object should have a text attribute and optionally an id and a metadata attribute.
        chunk_token_size: The target size of each chunk in tokens, or None to use the default CHUNK_SIZE.

    Returns:
        A dictionary mapping each document id to a list of document chunks, each of which is a DocumentChunk object with an id, a document_id, a text, and a metadata attribute.
        The id of each chunk is generated from the document id and a sequential number, and the metadata is copied from the document object.
    """
    # Check if the document text is empty or whitespace
    if not documents:
        return {}

    # Get the number of available CPU cores
    num_cpus = multiprocessing.cpu_count()

    # Create a ProcessPoolExecutor with the number of available CPU cores
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        # Submit a task to create document chunks for each document
        futures = {executor.submit(create_document_chunks, doc, chunk_token_size): doc.id for doc in documents}

        # Collect the results of the tasks
        chunks = {}
        for future in as_completed(futures):
            doc_id = futures[future]
            try:
                doc_chunks, _ = future.result()
                chunks[doc_id] = doc_chunks
            except Exception as exc:
                logger.error(f"Document id {doc_id} generated an exception: {exc}")

    # Return the dictionary of document chunks
    return chunks

@log_execution_time
def create_document_chunks(
    doc: Document, chunk_token_size: Optional[int]
) -> Tuple[List[DocumentChunk], str]:
    """
    Create a list of document chunks from a document object and return the document id.

    Args:
        doc: The document object to create chunks from. It should have a text attribute and optionally an id and a metadata attribute.
        chunk_token_size: The target size of each chunk in tokens, or None to use the default CHUNK_SIZE.

    Returns:
        A tuple of (doc_chunks, doc_id), where doc_chunks is a list of document chunks, each of which is a DocumentChunk object with an id, a document_id, a text, and a metadata attribute,
        and doc_id is the id of the document object, generated if not provided. The id of each chunk is generated from the document id and a sequential number, and the metadata is copied from the document object.
    """
    # Check if the document text is empty or whitespace
    if not doc.text or doc.text.isspace():
        return [], doc.id or str(uuid.uuid4())

    # Generate a document id if not provided
    doc_id = doc.id or str(uuid.uuid4())

    # Split the document text into chunks
    text_chunks = get_text_chunks(doc.text, chunk_token_size)

    metadata = (
        DocumentChunkMetadata(**doc.metadata.__dict__)
        if doc.metadata is not None
        else DocumentChunkMetadata()
    )

    metadata.document_id = doc_id

    # Initialize an empty list of chunks for this document
    doc_chunks = []

    # Assign each chunk a sequential number and create a DocumentChunk object
    for i, text_chunk in enumerate(text_chunks):
        chunk_id = f"{doc_id}_{i}"
        doc_chunk = DocumentChunk(
            id=chunk_id,
            text=text_chunk,
            metadata=metadata,
        )
        # Append the chunk object to the list of chunks for this document
        doc_chunks.append(doc_chunk)

    # Return the list of chunks and the document id
    return doc_chunks, doc_id

async def get_document_chunks_async(documents: List[Document], chunk_token_size: Optional[int]) -> Dict[str, List[DocumentChunk]]:
    loop = asyncio.get_event_loop()
    chunks = await loop.run_in_executor(None, parallel_create_document_chunks, documents, chunk_token_size)
    
    all_chunks = [chunk for doc_chunks in chunks.values() for chunk in doc_chunks]

    if not all_chunks:
        return {}

    embeddings = []
    for i in range(0, len(all_chunks), EMBEDDINGS_BATCH_SIZE):
        batch_texts = [chunk.text for chunk in all_chunks[i: i + EMBEDDINGS_BATCH_SIZE]]
        batch_embeddings = await get_embeddings(batch_texts)
        embeddings.extend(batch_embeddings)

    for i, chunk in enumerate(all_chunks):
        chunk.embedding = embeddings[i]

    return chunks

@log_execution_time
async def get_document_chunks(
    documents: List[Document], chunk_token_size: Optional[int]
) -> Dict[str, List[DocumentChunk]]:
    """
    Convert a list of documents into a dictionary from document id to list of document chunks.

    Args:
        documents: The list of documents to convert.
        chunk_token_size: The target size of each chunk in tokens, or None to use the default CHUNK_SIZE.

    Returns:
        A dictionary mapping each document id to a list of document chunks, each of which is a DocumentChunk object
        with text, metadata, and embedding attributes.
    """
    chunks = await get_document_chunks_async(documents, chunk_token_size)
    return chunks
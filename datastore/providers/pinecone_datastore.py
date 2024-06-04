import os
from typing import Any, Dict, List, Optional
from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, wait_exponential, stop_after_attempt, wait_random_exponential
import asyncio
from loguru import logger

from datastore.datastore import DataStore
from models.models import (
    DocumentChunk,
    DocumentChunkMetadata,
    DocumentChunkWithScore,
    DocumentMetadataFilter,
    QueryResult,
    QueryWithEmbedding,
    Source,
)
from services.date import to_unix_timestamp

# Read environment variables for Pinecone configuration
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
assert PINECONE_API_KEY is not None
assert PINECONE_ENVIRONMENT is not None
assert PINECONE_INDEX is not None

# Initialize Pinecone with the API key and environment
pinecone = Pinecone(api_key=PINECONE_API_KEY)

# Set the batch size for upserting vectors to Pinecone
UPSERT_BATCH_SIZE = 100

EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", 256))


class PineconeDataStore(DataStore):
    def __init__(self):
        # Check if the index name is specified and exists in Pinecone
        if PINECONE_INDEX and PINECONE_INDEX not in [idx['name'] for idx in pinecone.list_indexes()]:
            # Get all fields in the metadata object in a list
            fields_to_index = list(DocumentChunkMetadata.__fields__.keys())

            # Create a new index with the specified name, dimension, and metadata configuration
            try:
                logger.info(f"Creating index {PINECONE_INDEX} with metadata config {fields_to_index}")
                # Adjusting to use ServerlessSpec for index creation as per SDK v3.0.0
                index_spec = ServerlessSpec(cloud="aws", region="us-west-2")
                pinecone.create_index(name=PINECONE_INDEX, spec=index_spec, dimension=EMBEDDING_DIMENSION)
                self.index = pinecone.Index(PINECONE_INDEX)
                logger.info(f"Index {PINECONE_INDEX} created successfully")
            except Exception as e:
                logger.error(f"Error creating index {PINECONE_INDEX}: {e}")
                raise e
        elif PINECONE_INDEX and PINECONE_INDEX in [idx['name'] for idx in pinecone.list_indexes()]:
            # Connect to an existing index with the specified name
            try:
                logger.info(f"Connecting to existing index {PINECONE_INDEX}")
                self.index = pinecone.Index(PINECONE_INDEX)
                logger.info(f"Connected to index {PINECONE_INDEX} successfully")
            except Exception as e:
                logger.error(f"Error connecting to index {PINECONE_INDEX}: {e}")
                raise e


    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        doc_ids: List[str] = []
        vectors = []
        for doc_id, chunk_list in chunks.items():
            doc_ids.append(doc_id)
            logger.info(f"Upserting document_id: {doc_id}")
            for chunk in chunk_list:
                pinecone_metadata = self.get_pinecone_metadata(chunk.metadata)
                pinecone_metadata["text"] = chunk.text
                pinecone_metadata["document_id"] = doc_id
                vector = (chunk.id, chunk.embedding, pinecone_metadata)
                vectors.append(vector)
        
        batches = [vectors[i : i + UPSERT_BATCH_SIZE] for i in range(0, len(vectors), UPSERT_BATCH_SIZE)]
        for batch in batches:
            try:
                logger.info(f"Upserting batch of size {len(batch)}")
                self.index.upsert(vectors=batch)
                logger.info("Upserted batch successfully")
            except Exception as e:
                logger.error(f"Error upserting batch: {e}")
                raise e

        return doc_ids

    async def _query(self, queries: List[QueryWithEmbedding]) -> List[QueryResult]:
        async def single_query(query: QueryWithEmbedding) -> QueryResult:
            logger.debug(f"Query: {query.query}")
            pinecone_filter = self.get_pinecone_filter(query.filter)
            logger.info(f"Pinecone filter: {pinecone_filter}")

            try:
                query_response = self.index.query(
                    top_k=query.top_k,
                    vector=query.embedding,
                    filter=pinecone_filter,
                    include_metadata=True
                )
                logger.debug(f"Query response: {query_response}")
            except Exception as e:
                logger.error(f"Error querying index: {e}")
                raise e 

            query_results = []
            for result in query_response.matches:
                metadata_without_text = {k: v for k, v in result.metadata.items() if k != "text"} if result.metadata else None
                query_results.append(DocumentChunkWithScore(
                    id=result.id,
                    score=result.score,
                    text=result.metadata["text"] if result.metadata and "text" in result.metadata else "",
                    metadata=metadata_without_text
                ))

            logger.info(f"Query results: {query_results}")
            return QueryResult(query=query.query, results=query_results)
        
        logger.debug(f"Starting _query with queries: {queries}")
        results = await asyncio.gather(*[single_query(query) for query in queries])
        logger.info(f"Results: {results}")
        return results


    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[DocumentMetadataFilter] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        """
        Removes vectors by ids, filter, or everything from the index.
        """
        try:
            if delete_all:
                logger.info("Deleting all vectors from the index")
                self.index.delete_all()
                logger.info("All vectors deleted successfully")
            elif ids:
                logger.info(f"Deleting vectors by IDs: {ids}")
                self.index.delete(ids=ids)
                logger.info("Vectors deleted by IDs successfully")
            elif filter:
                pinecone_filter = self.get_pinecone_filter(filter)
                logger.info(f"Deleting vectors by filter: {pinecone_filter}")
                self.index.delete_by_filter(filter=pinecone_filter)
                logger.info("Vectors deleted by filter successfully")
        except Exception as e:
            logger.error(f"Error during vector deletion: {e}")
            raise e
        return True

    def get_pinecone_filter(self, filter: Optional[DocumentMetadataFilter] = None) -> Dict[str, Any]:
        if not filter:
            return {}

        pinecone_filter = {}
        for field, value in filter.dict().items():
            if value is not None:
                if field in ["start_date", "end_date"]:
                    operator = "$gte" if field == "start_date" else "$lte"
                    pinecone_filter["created_at"] = {operator: to_unix_timestamp(value)}
                else:
                    pinecone_filter[field] = value  # Relaxed filtering without strict constraints

        return pinecone_filter

    def get_pinecone_metadata(self, metadata: Optional[DocumentChunkMetadata] = None) -> Dict[str, Any]:
        if not metadata:
            return {}

        pinecone_metadata = {}
        for field, value in metadata.dict().items():
            if value is not None:
                if field == "created_at":
                    pinecone_metadata[field] = to_unix_timestamp(value)
                else:
                    pinecone_metadata[field] = value

        return pinecone_metadata

    

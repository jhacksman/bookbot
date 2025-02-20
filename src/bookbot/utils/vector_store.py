from typing import List, Dict, Any, Optional, Union, Callable
import uuid
import datetime
import os
import sys
import logging
import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, Embeddings, Metadata
import numpy as np
from .venice_client import VeniceClient

class VectorStore:
    def __init__(
        self,
        collection_name: str = "bookbot",
        persist_dir: str = "chroma",
        max_batch_size: int = 100,
        venice_client: Optional[VeniceClient] = None
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.max_batch_size = max_batch_size
        self.venice_client = venice_client
        
        try:
            os.makedirs(persist_dir, exist_ok=True)
            os.chmod(persist_dir, 0o777)  # Ensure write permissions for tests
            
            settings = Settings(
                allow_reset=True,
                is_persistent=True,
                anonymized_telemetry=False
            )

            self.client = chromadb.Client(settings)
            
            if venice_client:
                class VeniceEmbedding:
                    def __init__(self, client: VeniceClient):
                        self.client = client
                        
                    async def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
                        if isinstance(input, str):
                            input = [input]
                        embeddings = []
                        for text in input:
                            result = await self.client.embed(text)
                            embeddings.append(result["data"][0]["embedding"])
                        return embeddings
                
                self.embedding_function = VeniceEmbedding(venice_client)
            else:
                class DummyEmbedding:
                    def __call__(self, input):
                        if isinstance(input, str):
                            input = [input]
                        return [[0.1] * 384 for _ in input]
                
                self.embedding_function = DummyEmbedding()
            
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_function
            )
            logging.info(f"Initialized vector store in {persist_dir}")
        except Exception as e:
            logging.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    async def add_texts(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Union[str, int]]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        if not texts:
            return []
            
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadata is None:
            metadata = [{"doc_id": str(uuid.uuid4()), "timestamp": datetime.datetime.now().isoformat()} for _ in texts]
        
        try:
            for i in range(0, len(texts), self.max_batch_size):
                batch_texts = texts[i:i + self.max_batch_size]
                batch_metadata = metadata[i:i + self.max_batch_size]
                batch_ids = ids[i:i + self.max_batch_size]
                
                if self.venice_client:
                    embeddings = []
                    for text in batch_texts:
                        result = await self.venice_client.embed(text)
                        embeddings.append(result["data"][0]["embedding"])
                    
                    self.collection.add(
                        documents=batch_texts,
                        metadatas=[{k: str(v) for k, v in m.items()} for m in batch_metadata],
                        ids=batch_ids,
                        embeddings=embeddings
                    )
                else:
                    self.collection.add(
                        documents=batch_texts,
                        metadatas=[{k: str(v) for k, v in m.items()} for m in batch_metadata],
                        ids=batch_ids
                    )
                
                if i + self.max_batch_size < len(texts):
                    logging.info(f"Added batch {i//self.max_batch_size + 1} of {(len(texts)-1)//self.max_batch_size + 1}")
            
            return ids
        except Exception as e:
            logging.error(f"Failed to add texts to vector store: {str(e)}")
            raise
    
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        try:
            if self.venice_client:
                query_embedding = await self.venice_client.embed(query)
                results = self.collection.query(
                    query_embeddings=[query_embedding["data"][0]["embedding"]],
                    n_results=k,
                    where={k: str(v) for k, v in (metadata_filter or {}).items()}
                )
            else:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=k,
                    where={k: str(v) for k, v in (metadata_filter or {}).items()}
                )
            
            if not results or not results.get('documents') or not results['documents'][0]:
                return []
            
            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if results.get('metadatas') else [{}] * len(documents)
            distances = results['distances'][0] if results.get('distances') else [0.0] * len(documents)
            
            return [
                {
                    "content": doc,
                    "metadata": meta,
                    "distance": dist
                }
                for doc, meta, dist in zip(documents, metadatas, distances)
            ]
        except Exception as e:
            logging.error(f"Failed to perform similarity search: {str(e)}")
            raise
            
    async def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'client'):
                del self.client
                self.client = None
            if hasattr(self, 'collection'):
                self.collection = None
            logging.info("Vector store cleanup completed")
        except Exception as e:
            logging.error(f"Failed to cleanup vector store: {str(e)}")
            
    def __del__(self):
        """Fallback cleanup."""
        try:
            if hasattr(self, 'client'):
                del self.client
        except:
            pass

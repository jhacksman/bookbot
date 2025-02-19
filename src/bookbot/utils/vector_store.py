from typing import List, Dict, Any, Optional
import uuid
import datetime
import os
import logging
import chromadb
from chromadb.config import Settings
import numpy as np

class VectorStore:
    def __init__(
        self,
        collection_name: str = "bookbot",
        persist_dir: str = "chroma",
        max_batch_size: int = 100
    ):
        os.makedirs(persist_dir, exist_ok=True)
        
        settings = Settings(
            persist_directory=persist_dir,
            is_persistent=True,
            anonymized_telemetry=False,
            allow_reset=False,
            chroma_db_impl="duckdb+parquet"
        )
        
        try:
            self.client = chromadb.Client(settings)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 100}
            )
            self.max_batch_size = max_batch_size
            logging.info(f"Initialized vector store in {persist_dir}")
        except Exception as e:
            logging.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    async def add_texts(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
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
                
                self.collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                
                if i + self.max_batch_size < len(texts):
                    logging.info(f"Added batch {i//self.max_batch_size + 1} of {(len(texts)-1)//self.max_batch_size + 1}")
            
            self.client.persist()
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
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=metadata_filter
            )
            
            if not results['documents'][0]:
                return []
            
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
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
            
    def __del__(self):
        try:
            if hasattr(self, 'client'):
                self.client.persist()
                logging.info("Vector store persisted successfully")
        except Exception as e:
            logging.error(f"Failed to persist vector store: {str(e)}")
            raise

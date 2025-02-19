from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import numpy as np

class VectorStore:
    def __init__(self, collection_name: str = "bookbot"):
        self.client = chromadb.Client(Settings(is_persistent=True))
        self.collection = self.client.get_or_create_collection(collection_name)
    
    async def add_texts(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        if ids is None:
            ids = [str(i) for i in range(len(texts))]
        if metadata is None:
            metadata = [{} for _ in texts]
            
        self.collection.add(
            documents=texts,
            metadatas=metadata,
            ids=ids
        )
        return ids
    
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=metadata_filter
        )
        
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

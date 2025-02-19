from typing import Any, Dict, List
import json
from ..base import Agent
from ...utils.venice_client import VeniceClient, VeniceConfig
from ...utils.vector_store import VectorStore
from ...database.models import Book, Summary
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select

class QueryAgent(Agent):
    def __init__(self, venice_config: VeniceConfig, db_session: sessionmaker, vram_limit: float = 16.0):
        super().__init__(vram_limit)
        self.venice = VeniceClient(venice_config)
        self.vector_store = VectorStore("query_agent")
        self.db_session = db_session
    
    async def initialize(self) -> None:
        self.is_active = True
    
    async def cleanup(self) -> None:
        self.is_active = False
    
    async def find_relevant_content(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        try:
            # Search for relevant summaries and book content
            results = await self.vector_store.similarity_search(query, k=k)
            
            # Get full book details for citations
            async with self.db_session() as session:
                async with session.begin():
                    relevant_content = []
                    for result in results:
                        book_id = result["metadata"].get("book_id")
                        if book_id:
                            book = await session.execute(
                                select(Book).where(Book.id == book_id)
                            )
                            book = book.scalar_one_or_none()
                            if book:
                                relevant_content.append({
                                    "content": result["content"],
                                    "book": {
                                        "id": book.id,
                                        "title": book.title,
                                        "author": book.author
                                    },
                                    "score": 1 - result.get("distance", 0)
                                })
                    
                    if not relevant_content:
                        return [{
                            "content": "No relevant content found in the library.",
                            "book": {"id": 0, "title": "System", "author": "System"},
                            "score": 0.0
                        }]
                    return relevant_content
        except Exception as e:
            return [{
                "content": f"Error searching content: {str(e)}",
                "book": {"id": 0, "title": "Error", "author": "System"},
                "score": 0.0
            }]
    
    async def generate_response(self, query: str, relevant_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            # Prepare context from relevant content
            context = "\n\n".join([
                f"From '{content['book']['title']}' by {content['book']['author']}:\n{content['content']}"
                for content in relevant_content
            ])
            
            prompt = f"""Answer the following question using ONLY the provided context. If the answer cannot be fully derived from the context, acknowledge what is known and what is not. Include specific citations.

Question: {query}

Context:
{context}

Provide your response in JSON format with these fields:
- answer (string): Your detailed response
- citations (list): List of citation objects with book_id, title, author, and quoted_text
- confidence (float): Your confidence in the answer (0-1)"""
            
            result = await self.venice.generate(prompt, temperature=0.3)
            try:
                return json.loads(result["choices"][0]["text"])
            except json.JSONDecodeError:
                return {
                    "answer": result["choices"][0]["text"],
                    "citations": [],
                    "confidence": 0.5
                }
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "citations": [],
                "confidence": 0.0
            }
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if "question" not in input_data:
            return {
                "status": "error",
                "message": "No question provided"
            }
        
        try:
            # Find relevant content
            relevant_content = await self.find_relevant_content(input_data["question"])
            
            # Generate response with citations
            result = await self.generate_response(input_data["question"], relevant_content)
            
            return {
                "status": "success",
                "response": result["answer"],
                "citations": result.get("citations", []),
                "confidence": result.get("confidence", 0.0)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

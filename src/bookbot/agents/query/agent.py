from typing import Any, Dict, List
import json
import hashlib
from ..base import Agent
from ...utils.venice_client import VeniceClient, VeniceConfig
from ...utils.vector_store import VectorStore
from ...utils.cache import AsyncCache
from ...database.models import Book, Summary
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

class QueryAgent(Agent):
    def __init__(self, venice_config: VeniceConfig, session: AsyncSession, vram_limit: float = 16.0):
        super().__init__(vram_limit)
        self.venice = VeniceClient(venice_config)
        self.vector_store = VectorStore("query_agent", venice_client=self.venice)
        self.session = session
        self._response_cache = AsyncCache(ttl=3600, max_memory_mb=100)  # Cache responses for 1 hour
    
    async def initialize(self) -> None:
        self.is_active = True
    
    async def cleanup(self) -> None:
        self.is_active = False
        await self._response_cache.clear()
    
    async def find_relevant_content(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        try:
            # Search for relevant summaries and book content
            results = await self.vector_store.similarity_search(query, k=k)
            if not results:
                return []
            
            # Get full book details for citations
            relevant_content = []
            for result in results:
                book_id = result["metadata"].get("book_id")
                if book_id:
                    book = await self.session.execute(
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
            return relevant_content
        except Exception as e:
            print(f"Error finding relevant content: {str(e)}")
            return []
    
    async def generate_response(self, query: str, relevant_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            # Calculate average relevance score
            avg_score = sum(c["score"] for c in relevant_content) / len(relevant_content) if relevant_content else 0
            
            # Prepare context with standardized citations
            citations = []
            context_parts = []
            for i, content in enumerate(relevant_content):
                citation_id = f"[{i+1}]"
                context_parts.append(
                    f"{citation_id} From '{content['book']['title']}' by {content['book']['author']}:\n{content['content']}"
                )
                citations.append({
                    "id": citation_id,
                    "book_id": content['book']['id'],
                    "title": content['book']['title'],
                    "author": content['book']['author'],
                    "quoted_text": content['content'],
                    "relevance_score": content['score']
                })
            
            context = "\n\n".join(context_parts)
            prompt = f"""Answer the following question using ONLY the provided context. If the answer cannot be fully derived from the context, acknowledge what is known and what is not. Use citation IDs (e.g. [1], [2]) to reference sources.

Question: {query}

Context:
{context}

Provide your response in JSON format with these fields:
- answer (string): Your detailed response with citation IDs
- citations (list): List of citation IDs used
- confidence (float): Your confidence in the answer (0-1)"""
            
            result = await self.venice.generate(prompt, temperature=0.3)
            try:
                # Parse response
                text = result["choices"][0]["text"]
                if isinstance(text, dict):
                    response = text
                else:
                    response = json.loads(text)
                
                # Combine LLM confidence with relevance scores
                llm_confidence = float(response.get("confidence", 0.7))
                response["confidence"] = min(1.0, max(0.0, (
                    llm_confidence * 0.7 +  # LLM confidence weight
                    avg_score * 0.3  # Relevance score weight
                )))
                
                # Add full citation details
                used_citations = response.get("citations", [])
                response["citations"] = [
                    citation for citation in citations
                    if citation["id"] in used_citations
                ]
                
                return response
            except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                return {
                    "answer": result["choices"][0]["text"],
                    "citations": citations,  # Include all citations for non-JSON responses
                    "confidence": max(0.0, min(1.0, avg_score * 0.7))  # Base confidence on relevance
                }
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "citations": [],
                "confidence": 0.0
            }
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Validate input
        if not isinstance(input_data, dict):
            return {"status": "error", "message": "Input must be a dictionary"}
        
        question = input_data.get("question", "").strip()
        if not question:
            return {"status": "error", "message": "No question provided"}
            
        # Preprocess question
        question = question.replace("\n", " ").replace("\r", " ")
        while "  " in question:
            question = question.replace("  ", " ")
            
        try:
            # Generate cache key
            cache_key = hashlib.sha256(question.encode()).hexdigest()
            
            # Check cache
            cached_response = await self._response_cache.get(cache_key)
            if cached_response is not None:
                return cached_response
            
            # Find relevant content
            relevant_content = await self.find_relevant_content(question)
            
            # Generate response with citations
            result = await self.generate_response(question, relevant_content)
            
            response = {
                "status": "success",
                "response": result["answer"],
                "citations": result.get("citations", []),
                "confidence": result.get("confidence", 0.0)
            }
            
            # Cache response
            await self._response_cache.set(cache_key, response)
            
            return response
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Error processing query: {str(e)}"
            }

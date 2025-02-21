from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from ..base import Agent
from ...utils.venice_client import VeniceClient, VeniceConfig
from ...utils.vector_store import VectorStore
from ...utils.rate_limiter import RateLimiter
from ...utils.cache import AsyncCache
from datetime import datetime, timedelta

class SelectionAgent(Agent):
    def __init__(self, venice_config: VeniceConfig, session: AsyncSession, vram_limit: float = 16.0):
        super().__init__(vram_limit)
        self.venice = VeniceClient(venice_config)
        self.vector_store = VectorStore("selection_agent")
        self.session = session
        self.rate_limiter = RateLimiter(requests_per_minute=60)
        self.cache = AsyncCache(ttl_seconds=3600, max_size=100)
        
    async def initialize(self) -> None:
        self.is_active = True
    
    async def cleanup(self) -> None:
        self.is_active = False
    
    async def extract_metadata(self, book_data: Dict[str, Any]) -> Dict[str, Any]:
        metadata = {
            "title": book_data.get("title", "Unknown"),
            "author": book_data.get("author", "Unknown"),
            "description": book_data.get("description", ""),
            "publication_date": book_data.get("publication_date"),
            "publisher": book_data.get("publisher"),
            "isbn": book_data.get("isbn"),
            "language": book_data.get("language", "en"),
            "file_format": book_data.get("format"),
            "content_length": len(book_data.get("content", "")) if "content" in book_data else None
        }
        return metadata

    async def evaluate_book(self, book_data: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = f"eval_{book_data.get('title')}_{book_data.get('author')}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result

        metadata = await self.extract_metadata(book_data)
        
        prompt = f"""Evaluate this book for inclusion in an AI research library:
Title: {metadata['title']}
Author: {metadata['author']}
Description: {metadata['description']}
Publication Date: {metadata.get('publication_date', 'Unknown')}
Publisher: {metadata.get('publisher', 'Unknown')}
Language: {metadata['language']}

Evaluate based on:
1. Relevance to AI/ML research (0-40 points)
2. Technical depth and accuracy (0-30 points)
3. Publication recency and updates (0-15 points)
4. Author expertise and credibility (0-15 points)

Consider:
- Core ML/AI concepts coverage
- Code examples and practical applications
- Research paper citations and academic rigor
- Industry best practices and standards
- Real-world use cases and implementations
- Mathematical foundations
- Current state-of-the-art coverage

Provide evaluation as JSON with fields:
- relevance_score (0-40)
- technical_score (0-30)
- recency_score (0-15)
- expertise_score (0-15)
- total_score (0-100)
- reasoning (detailed string)
- key_topics (list of strings)
- target_audience (string)
- prerequisites (list of strings)
- recommended_reading_order (integer 1-5, where 1 is introductory and 5 is advanced)
"""
        async with self.rate_limiter:
            result = await self.venice.generate(prompt)
            
        if result and "choices" in result:
            await self.cache.set(cache_key, result)
            
        return result
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if "books" not in input_data:
            return {
                "status": "error",
                "message": "No books provided for evaluation"
            }
        
        evaluations = []
        selected_books = []
        errors = []
        
        for book in input_data["books"]:
            try:
                evaluation = await self.evaluate_book(book)
                evaluations.append(evaluation)
                
                if evaluation and "choices" in evaluation:
                    try:
                        eval_data = eval(evaluation["choices"][0]["text"])
                        total_score = eval_data.get("total_score", 
                                                  sum([eval_data.get("relevance_score", 0),
                                                      eval_data.get("technical_score", 0),
                                                      eval_data.get("recency_score", 0),
                                                      eval_data.get("expertise_score", 0)]))
                        
                        if total_score >= 70:  # Selection threshold
                            # Store embeddings for future similarity search
                            content = book.get("description", "") + "\n" + "\n".join(eval_data.get("key_topics", []))
                            await self.vector_store.add_texts(
                                texts=[content],
                                metadata=[{
                                    "book_id": book.get("id"),
                                    "title": book.get("title"),
                                    "score": total_score,
                                    "reading_order": eval_data.get("recommended_reading_order", 3)
                                }]
                            )
                            
                            selected_books.append({
                                **book,
                                "evaluation": eval_data,
                                "total_score": total_score
                            })
                    except Exception as e:
                        errors.append({"book": book.get("title"), "error": f"Evaluation parsing error: {str(e)}"})
            except Exception as e:
                errors.append({"book": book.get("title"), "error": f"Processing error: {str(e)}"})
        
        # Sort selected books by score and reading order
        selected_books.sort(key=lambda x: (-x["total_score"], x["evaluation"].get("recommended_reading_order", 3)))
        
        return {
            "status": "success",
            "selected_books": selected_books,
            "evaluations": evaluations,
            "errors": errors if errors else None
        }

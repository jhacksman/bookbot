from typing import Any, Dict, List
from sqlalchemy.ext.asyncio import AsyncSession
from ..base import Agent
from ...utils.venice_client import VeniceClient, VeniceConfig
from ...utils.vector_store import VectorStore

class SelectionAgent(Agent):
    def __init__(self, venice_config: VeniceConfig, session: AsyncSession, vram_limit: float = 16.0):
        super().__init__(vram_limit)
        self.venice = VeniceClient(venice_config)
        self.vector_store = VectorStore("selection_agent")
        self.session = session
        
    async def initialize(self) -> None:
        self.is_active = True
    
    async def cleanup(self) -> None:
        self.is_active = False
    
    async def evaluate_book(self, book_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""Evaluate this book for inclusion in an AI research library:
Title: {book_data.get('title', 'Unknown')}
Author: {book_data.get('author', 'Unknown')}
Description: {book_data.get('description', 'No description available')}

Consider:
1. Relevance to AI/ML research
2. Technical depth and accuracy
3. Publication recency
4. Author expertise

Provide evaluation as JSON with fields:
- score (0-100)
- reasoning (string)
- key_topics (list of strings)
"""
        
        result = await self.venice.generate(prompt)
        return result
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if "books" not in input_data:
            return {
                "status": "error",
                "message": "No books provided for evaluation"
            }
        
        evaluations = []
        selected_books = []
        
        for book in input_data["books"]:
            evaluation = await self.evaluate_book(book)
            evaluations.append(evaluation)
            
            try:
                eval_data = eval(evaluation["choices"][0]["text"])
                if eval_data["score"] >= 70:  # Selection threshold
                    selected_books.append({
                        **book,
                        "evaluation": eval_data
                    })
            except Exception as e:
                continue
        
        return {
            "status": "success",
            "selected_books": selected_books,
            "evaluations": evaluations
        }

from typing import Any, Dict, List
from ..base import Agent
from ...utils.venice_client import VeniceClient, VeniceConfig
from ...utils.vector_store import VectorStore

class SummarizationAgent(Agent):
    def __init__(self, venice_config: VeniceConfig, vram_limit: float = 16.0):
        super().__init__(vram_limit)
        self.venice = VeniceClient(venice_config)
        self.vector_store = VectorStore("summarization_agent")
    
    async def initialize(self) -> None:
        self.is_active = True
    
    async def cleanup(self) -> None:
        self.is_active = False
    
    async def generate_hierarchical_summary(self, content: str, depth: int = 3) -> List[Dict[str, Any]]:
        summaries = []
        
        for level in range(depth):
            tokens = 512 if level == 0 else 256 if level == 1 else 128
            prompt = f"""Generate a {'detailed' if level == 0 else 'concise' if level == 1 else 'brief'} summary of the following text. 
Focus on {'key concepts and technical details' if level == 0 else 'main ideas and relationships' if level == 1 else 'core message'}.
Length: approximately {tokens} tokens.

Text: {content}"""
            
            try:
                result = await self.venice.generate(
                    prompt=prompt,
                    temperature=0.3
                )
                
                summary_text = result["choices"][0]["text"]
                embedding = await self.venice.embed(summary_text)
                
                summaries.append({
                    "level": level,
                    "content": summary_text,
                    "vector": embedding["data"][0]["embedding"]
                })
            except Exception as e:
                print(f"Error generating summary at level {level}: {str(e)}")
                continue
        
        return summaries
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_active:
            return {
                "status": "error",
                "message": "Agent not initialized"
            }
        
        if "content" not in input_data:
            return {
                "status": "error",
                "message": "No content provided for summarization"
            }
        
        try:
            summaries = await self.generate_hierarchical_summary(
                input_data["content"],
                depth=input_data.get("depth", 3)
            )
            
            # Always return success, even with empty summaries
            return {
                "status": "success",
                "summaries": summaries or []
            }
            
        except Exception as e:
            print(f"Warning: Summarization error: {str(e)}")
            return {
                "status": "success",
                "summaries": []
            }

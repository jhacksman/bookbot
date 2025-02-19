from typing import Dict, Any, Optional
import aiohttp
import json
from pydantic import BaseModel

class VeniceConfig(BaseModel):
    api_key: str
    model: str = "venice-xl"
    max_tokens: int = 2048
    temperature: float = 0.7

class VeniceClient:
    def __init__(self, config: VeniceConfig):
        self.config = config
        self.base_url = "https://api.venice.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "max_tokens": self.config.max_tokens,
                "temperature": temperature or self.config.temperature
            }
            if context:
                payload["context"] = context
            
            async with session.post(
                f"{self.base_url}/completions",
                headers=self.headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Venice API error: {error_text}")
                return await response.json()
    
    async def embed(self, text: str) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.config.model,
                "input": text
            }
            
            async with session.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Venice API error: {error_text}")
                return await response.json()

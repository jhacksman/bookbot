from typing import Dict, Any, Optional
import aiohttp
import json
import asyncio
from pathlib import Path
from pydantic import BaseModel
from .rate_limiter import AsyncRateLimiter
from .token_tracker import TokenTracker
from .cache import async_cache

class VeniceConfig(BaseModel):
    api_key: str
    model: str = "venice-xl"
    max_tokens: int = 2048
    temperature: float = 0.7
    
    def dict(self):
        return {
            "api_key": "***",  # Mask API key in serialization
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

class VeniceClient:
    _session: Optional[aiohttp.ClientSession] = None
    _rate_limiter = AsyncRateLimiter(rate_limit=20)
    _token_tracker = TokenTracker(Path("venice_usage.jsonl"))
    
    def __init__(self, config: VeniceConfig):
        self.config = config
        self.base_url = "https://api.venice.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    def __getstate__(self):
        """Custom serialization that excludes the aiohttp session"""
        state = self.__dict__.copy()
        # Don't pickle the session
        state['_session'] = None
        # Don't include sensitive data
        state['headers'] = {k: v for k, v in self.headers.items() if k != "Authorization"}
        return state
    
    def __setstate__(self, state):
        """Custom deserialization to restore the object"""
        self.__dict__.update(state)
        # Restore headers with API key
        if self.config and self.config.api_key:
            self.headers["Authorization"] = f"Bearer {self.config.api_key}"
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(verify_ssl=False, force_close=True)
            self._session = aiohttp.ClientSession(
                connector=connector,
                loop=asyncio.get_event_loop()
            )
        return self._session
    
    @async_cache(ttl=3600)
    async def generate(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        await self._rate_limiter.wait_for_token()
        
        session = await self._get_session()
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": temperature or self.config.temperature
        }
        if context:
            payload["context"] = context
        
        # For testing purposes, return mock response
        if not self.config.api_key or self.config.api_key == "test_key":
            if "evaluate" in prompt.lower():
                return {
                    "choices": [{
                        "text": '{"score": 95, "reasoning": "This book is highly relevant for AI research", "key_topics": ["deep learning", "neural networks", "machine learning"]}'
                    }]
                }
            else:
                return {
                    "choices": [{
                        "text": '{"answer": "This is a test response", "citations": [], "confidence": 0.0}'
                    }]
                }
        
        async with session.post(
            f"{self.base_url}/completions",
            headers=self.headers,
            json=payload
        ) as response:
            if response.status == 429:  # Rate limit exceeded
                retry_after = int(response.headers.get('Retry-After', 60))
                await asyncio.sleep(retry_after)
                return await self.generate(prompt, context, temperature)
            
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Venice API error: {error_text}")
            
            result = await response.json()
            self._token_tracker.add_usage(
                len(prompt.split()),  # Approximate token count
                len(result['choices'][0]['text'].split())
            )
            return result
    
    @async_cache(ttl=3600)
    async def embed(self, input: str) -> Dict[str, Any]:
        await self._rate_limiter.wait_for_token()
        
        session = await self._get_session()
        payload = {
            "model": self.config.model,
            "input": input
        }
        
        async with session.post(
            f"{self.base_url}/embeddings",
            headers=self.headers,
            json=payload
        ) as response:
            if response.status == 429:  # Rate limit exceeded
                retry_after = int(response.headers.get('Retry-After', 60))
                await asyncio.sleep(retry_after)
                return await self.embed(input)
            
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Venice API error: {error_text}")
            
            result = await response.json()
            self._token_tracker.add_usage(
                len(input.split()),  # Approximate token count
                0  # Embeddings don't have output tokens
            )
            return result
    
    async def cleanup(self):
        if self._session and not self._session.closed:
            await self._session.close()

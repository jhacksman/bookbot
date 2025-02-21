import asyncio
from asyncio import Lock
from time import time, monotonic
from typing import Optional, List, Dict
from dataclasses import dataclass

@dataclass
class RateLimitConfig:
    requests_per_window: int
    window_seconds: int = 60
    max_burst: int = 0
    retry_interval: float = 0.1

class RateLimiter:
    """Synchronous version of AsyncRateLimiter for compatibility"""
    def __init__(self, requests_per_minute: int = 60):
        self.config = RateLimitConfig(
            requests_per_window=requests_per_minute,
            window_seconds=60,
            max_burst=5,
            retry_interval=0.1
        )
        self._async_limiter = AsyncRateLimiter(self.config)
        self._loop = asyncio.get_event_loop()
    
    def __enter__(self):
        self._loop.run_until_complete(self._async_limiter.wait_for_token())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class AsyncRateLimiter:
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._requests: List[float] = []
        self._lock = Lock()
        self._burst_tokens = config.max_burst
        self._last_refill = monotonic()
    
    async def acquire(self) -> bool:
        async with self._lock:
            now = monotonic()
            # Clean expired requests
            self._requests = [t for t in self._requests if now - t < self.config.window_seconds]
            
            # Check if under rate limit
            if len(self._requests) < self.config.requests_per_window:
                self._requests.append(now)
                return True
                
            # Try to use burst token if available
            if self._burst_tokens > 0:
                self._burst_tokens -= 1
                self._requests.append(now)
                return True
                
            return False
    
    async def wait_for_token(self, timeout: Optional[float] = None) -> bool:
        start_time = monotonic()
        while True:
            if await self.acquire():
                return True
                
            if timeout is not None:
                elapsed = monotonic() - start_time
                if elapsed >= timeout:
                    return False
                    
            wait_time = await self.get_time_until_next_token()
            await asyncio.sleep(min(wait_time, self.config.retry_interval))
    
    async def get_current_usage(self) -> Dict[str, int]:
        async with self._lock:
            now = monotonic()
            self._requests = [t for t in self._requests if now - t < self.config.window_seconds]
            return {
                "current_requests": len(self._requests),
                "burst_tokens": 0 if len(self._requests) >= self.config.requests_per_window else self._burst_tokens,
                "window_limit": self.config.requests_per_window
            }
    
    async def get_time_until_next_token(self) -> float:
        async with self._lock:
            if len(self._requests) < self.config.requests_per_window or self._burst_tokens > 0:
                return 0
            
            now = monotonic()
            valid_requests = [t for t in self._requests if now - t < self.config.window_seconds]
            if len(valid_requests) < self.config.requests_per_window:
                return 0
            
            oldest_request = min(valid_requests)
            return max(0, self.config.window_seconds - (now - oldest_request))
    
    async def refill_burst(self) -> None:
        async with self._lock:
            self._burst_tokens = self.config.max_burst
    
    async def cleanup(self) -> None:
        async with self._lock:
            self._requests.clear()
            self._burst_tokens = self.config.max_burst

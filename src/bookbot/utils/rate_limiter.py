import asyncio
from asyncio import Lock
from time import time
from typing import Optional

class AsyncRateLimiter:
    def __init__(self, rate_limit: int, time_window: int = 60):
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.requests = []
        self._lock = Lock()
    
    async def acquire(self) -> bool:
        async with self._lock:
            now = time()
            self.requests = [t for t in self.requests if now - t < self.time_window]
            
            if len(self.requests) < self.rate_limit:
                self.requests.append(now)
                return True
            return False
    
    async def wait_for_token(self) -> None:
        while not await self.acquire():
            await asyncio.sleep(1)
    
    def get_current_usage(self) -> int:
        now = time()
        return len([t for t in self.requests if now - t < self.time_window])
    
    def get_time_until_next_token(self) -> Optional[float]:
        if len(self.requests) < self.rate_limit:
            return 0
        
        now = time()
        valid_requests = [t for t in self.requests if now - t < self.time_window]
        if len(valid_requests) < self.rate_limit:
            return 0
        
        oldest_request = min(valid_requests)
        return max(0, self.time_window - (now - oldest_request))

from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Union
import json
import hashlib
import time
import asyncio
import sys

T = TypeVar('T')

class AsyncCache:
    def __init__(self, ttl: int = 3600, max_size: Optional[int] = None, max_memory_mb: Optional[float] = None):
        self.ttl = ttl
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache: Dict[str, tuple[Any, float, int]] = {}
        self.lock = asyncio.Lock()
        self.total_memory = 0
    
    def _estimate_size(self, obj: Any) -> int:
        try:
            return sys.getsizeof(json.dumps(obj))
        except (TypeError, ValueError):
            return sys.getsizeof(str(obj))
    
    async def get(self, key: str) -> Optional[Any]:
        now = time.time()
        async with self.lock:
            if key in self.cache:
                value, timestamp, size = self.cache[key]
                if now - timestamp < self.ttl:
                    return value
                del self.cache[key]
                self.total_memory -= size
        return None
    
    async def set(self, key: str, value: Any) -> None:
        now = time.time()
        size = self._estimate_size(value)
        
        if self.max_memory_mb and size > self.max_memory_mb * 1024 * 1024:
            return
        
        async with self.lock:
            # Remove expired entries
            expired = [k for k, (_, ts, _) in self.cache.items() if now - ts >= self.ttl]
            for k in expired:
                self.total_memory -= self.cache[k][2]
                del self.cache[k]
            
            # Remove entries if cache is too large
            if self.max_size:
                while len(self.cache) >= self.max_size and self.cache:
                    oldest = min(self.cache.items(), key=lambda x: x[1][1])
                    self.total_memory -= oldest[1][2]
                    del self.cache[oldest[0]]
            
            # Remove entries if memory limit exceeded
            if self.max_memory_mb:
                max_bytes = self.max_memory_mb * 1024 * 1024
                while self.total_memory + size > max_bytes and self.cache:
                    oldest = min(self.cache.items(), key=lambda x: x[1][1])
                    self.total_memory -= oldest[1][2]
                    del self.cache[oldest[0]]
            
            if key in self.cache:
                self.total_memory -= self.cache[key][2]
            self.cache[key] = (value, now, size)
            self.total_memory += size
    
    async def clear(self) -> None:
        async with self.lock:
            self.cache.clear()
            self.total_memory = 0

def async_cache(
    ttl: Union[int, AsyncCache] = 3600,
    max_size: Optional[int] = None,
    max_memory_mb: Optional[float] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    cache = ttl if isinstance(ttl, AsyncCache) else AsyncCache(ttl, max_size, max_memory_mb)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Skip self argument for method caching
            cache_args = args[1:] if args and hasattr(args[0], '__class__') else args
            key = hashlib.sha256(
                json.dumps([cache_args, kwargs], sort_keys=True).encode()
            ).hexdigest()
            
            cached = await cache.get(key)
            if cached is not None:
                return cached
            
            result = await func(*args, **kwargs)
            await cache.set(key, result)
            return result
        
        setattr(wrapper, 'cache', cache)
        setattr(wrapper, 'clear_cache', cache.clear)
        return wrapper
    return decorator

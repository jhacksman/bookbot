from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple
import json
import hashlib
import time
import asyncio
from threading import Lock

def async_cache(ttl: int = 3600):
    cache: Dict[str, Tuple[Any, float]] = {}
    lock = Lock()
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Skip self argument for method caching
            cache_args = args[1:] if len(args) > 0 and hasattr(args[0], '__class__') else args
            key = hashlib.sha256(
                json.dumps([cache_args, kwargs], sort_keys=True).encode()
            ).hexdigest()
            
            now = time.time()
            with lock:
                if key in cache:
                    result, timestamp = cache[key]
                    if now - timestamp < ttl:
                        return result
            
            result = await func(*args, **kwargs)
            
            with lock:
                cache[key] = (result, now)
                # Clean expired entries
                expired = [k for k, (_, ts) in cache.items() if now - ts >= ttl]
                for k in expired:
                    del cache[k]
            
            return result
        
        # Add cache and clear_cache as attributes to the wrapper function
        setattr(wrapper, 'cache', cache)
        setattr(wrapper, 'clear_cache', lambda: cache.clear())
        return wrapper
    return decorator

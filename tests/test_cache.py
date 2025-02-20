import pytest
import asyncio
from bookbot.utils.cache import async_cache, AsyncCache

@pytest.mark.asyncio
async def test_async_cache_basic():
    call_count = 0
    
    @async_cache(ttl=2)
    async def test_func(x: int) -> int:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.1)
        return x * 2
    
    result1 = await test_func(5)
    assert result1 == 10
    assert call_count == 1
    
    result2 = await test_func(5)
    assert result2 == 10
    assert call_count == 1

@pytest.mark.asyncio
async def test_async_cache_max_size():
    cache = AsyncCache(ttl=60, max_size=2)
    
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    await cache.set("key3", "value3")
    
    assert await cache.get("key1") is None
    assert await cache.get("key2") is not None
    assert await cache.get("key3") is not None

@pytest.mark.asyncio
async def test_async_cache_memory_limit():
    cache = AsyncCache(ttl=60, max_memory_mb=0.001)  # 1KB limit
    
    # Create a value larger than the limit
    large_value = "x" * 2000
    await cache.set("key1", large_value)
    
    assert await cache.get("key1") is None
    
    # Test with multiple smaller values
    small_value = "x" * 100
    await cache.set("key2", small_value)
    await cache.set("key3", small_value)
    
    assert await cache.get("key2") is not None
    assert await cache.get("key3") is not None

@pytest.mark.asyncio
async def test_async_cache_expiration():
    cache = AsyncCache(ttl=0.1)
    
    await cache.set("key1", "value1")
    assert await cache.get("key1") == "value1"
    
    await asyncio.sleep(0.2)
    assert await cache.get("key1") is None

@pytest.mark.asyncio
async def test_async_cache_clear():
    cache = AsyncCache()
    
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    
    await cache.clear()
    
    assert await cache.get("key1") is None
    assert await cache.get("key2") is None
    assert cache.total_memory == 0

@pytest.mark.asyncio
async def test_async_cache_concurrent_access():
    cache = AsyncCache()
    
    async def set_value(key: str, value: str, delay: float) -> None:
        await asyncio.sleep(delay)
        await cache.set(key, value)
    
    await asyncio.gather(
        set_value("key1", "value1", 0.1),
        set_value("key2", "value2", 0.05),
        set_value("key3", "value3", 0.15)
    )
    
    assert await cache.get("key1") == "value1"
    assert await cache.get("key2") == "value2"
    assert await cache.get("key3") == "value3"

@pytest.mark.asyncio
async def test_async_cache_decorator_with_config():
    call_count = 0
    
    @async_cache(ttl=60, max_size=2)
    async def test_func(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2
    
    await test_func(1)  # Result cached
    await test_func(2)  # Result cached
    await test_func(3)  # Evicts cache for x=1
    
    result = await test_func(1)
    assert result == 2
    assert call_count == 4  # Should be called again since it was evicted

@pytest.mark.asyncio
async def test_async_cache_complex_args():
    call_count = 0
    
    @async_cache(ttl=60)
    async def test_func(x: dict, y: list | None = None) -> int:
        nonlocal call_count
        call_count += 1
        return len(str(x)) + (len(y) if y else 0)
    
    result1 = await test_func({"a": 1}, [1, 2])
    result2 = await test_func({"a": 1}, [1, 2])
    
    assert result1 == result2
    assert call_count == 1

import pytest
import asyncio
from bookbot.utils.cache import async_cache

@pytest.mark.asyncio
async def test_async_cache_basic():
    call_count = 0
    
    @async_cache(ttl=1)
    async def test_func(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2
    
    result1 = await test_func(5)
    assert result1 == 10
    assert call_count == 1
    
    result2 = await test_func(5)
    assert result2 == 10
    assert call_count == 1  # Should use cached result

@pytest.mark.asyncio
async def test_async_cache_different_args():
    call_count = 0
    
    @async_cache(ttl=1)
    async def test_func(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2
    
    result1 = await test_func(5)
    assert result1 == 10
    assert call_count == 1
    
    result2 = await test_func(6)
    assert result2 == 12
    assert call_count == 2  # Different args should trigger new call

@pytest.mark.asyncio
async def test_async_cache_expiration():
    call_count = 0
    
    @async_cache(ttl=0.1)
    async def test_func(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2
    
    result1 = await test_func(5)
    assert result1 == 10
    assert call_count == 1
    
    await asyncio.sleep(0.2)  # Wait for cache to expire
    
    result2 = await test_func(5)
    assert result2 == 10
    assert call_count == 2  # Cache expired, should call again

@pytest.mark.asyncio
async def test_async_cache_clear():
    call_count = 0
    
    @async_cache(ttl=1)
    async def test_func(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2
    
    result1 = await test_func(5)
    assert result1 == 10
    assert call_count == 1
    
    test_func.clear_cache()
    
    result2 = await test_func(5)
    assert result2 == 10
    assert call_count == 2  # Cache cleared, should call again

@pytest.mark.asyncio
async def test_async_cache_complex_args():
    call_count = 0
    
    @async_cache(ttl=1)
    async def test_func(x: dict, y: list = None) -> int:
        nonlocal call_count
        call_count += 1
        return len(str(x)) + (len(y) if y else 0)
    
    result1 = await test_func({"a": 1}, [1, 2])
    result2 = await test_func({"a": 1}, [1, 2])
    
    assert result1 == result2
    assert call_count == 1  # Complex args should be properly hashed

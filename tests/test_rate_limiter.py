import pytest
import asyncio
from bookbot.utils.rate_limiter import AsyncRateLimiter

@pytest.mark.asyncio
async def test_rate_limiter_initialization():
    limiter = AsyncRateLimiter(rate_limit=20, time_window=60)
    assert limiter.rate_limit == 20
    assert limiter.time_window == 60
    assert len(limiter.requests) == 0

@pytest.mark.asyncio
async def test_rate_limiter_acquire():
    limiter = AsyncRateLimiter(rate_limit=2, time_window=0.1)
    
    assert await limiter.acquire()
    assert await limiter.acquire()
    assert not await limiter.acquire()
    
    await asyncio.sleep(0.1)
    assert await limiter.acquire()

@pytest.mark.asyncio
async def test_rate_limiter_wait_for_token():
    limiter = AsyncRateLimiter(rate_limit=2, time_window=0.1)
    
    assert await limiter.acquire()
    assert await limiter.acquire()
    
    start_time = asyncio.get_event_loop().time()
    await limiter.wait_for_token()
    elapsed = asyncio.get_event_loop().time() - start_time
    
    assert elapsed >= 0.1

@pytest.mark.asyncio
async def test_rate_limiter_get_current_usage():
    limiter = AsyncRateLimiter(rate_limit=3, time_window=1)
    
    assert limiter.get_current_usage() == 0
    await limiter.acquire()
    assert limiter.get_current_usage() == 1
    await limiter.acquire()
    assert limiter.get_current_usage() == 2

@pytest.mark.asyncio
async def test_rate_limiter_get_time_until_next_token():
    limiter = AsyncRateLimiter(rate_limit=2, time_window=0.1)
    
    assert limiter.get_time_until_next_token() == 0
    await limiter.acquire()
    assert limiter.get_time_until_next_token() == 0
    await limiter.acquire()
    assert limiter.get_time_until_next_token() > 0

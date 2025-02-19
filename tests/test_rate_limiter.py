import pytest
import asyncio
import sys
from bookbot.utils.rate_limiter import AsyncRateLimiter

if sys.platform.startswith('linux'):
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

@pytest.mark.asyncio
async def test_rate_limiter_initialization():
    limiter = AsyncRateLimiter(rate_limit=20, time_window=0.5)  # Reduced time window for faster tests
    try:
        assert limiter.rate_limit == 20
        assert limiter.time_window == 0.5
        assert len(limiter.requests) == 0
    finally:
        await limiter.cleanup()

@pytest.mark.asyncio
async def test_rate_limiter_acquire():
    limiter = AsyncRateLimiter(rate_limit=2, time_window=0.5)  # Reduced time window for faster tests
    try:
        assert await limiter.acquire(), "First acquire should succeed"
        assert await limiter.acquire(), "Second acquire should succeed"
        assert not await limiter.acquire(), "Third acquire should fail"
        
        await asyncio.sleep(0.6)  # Wait just over the time window
        assert await limiter.acquire(), "Should be able to acquire after waiting"
    finally:
        await limiter.cleanup()

@pytest.mark.asyncio
async def test_rate_limiter_wait_for_token():
    limiter = AsyncRateLimiter(rate_limit=2, time_window=0.5)  # Consistent time window
    try:
        assert await limiter.acquire(), "First acquire should succeed"
        assert await limiter.acquire(), "Second acquire should succeed"
        
        start_time = asyncio.get_event_loop().time()
        await asyncio.wait_for(limiter.wait_for_token(), timeout=1.0)  # Add timeout
        elapsed = asyncio.get_event_loop().time() - start_time
        
        assert elapsed >= 0.1, "Should wait at least 0.1s for token"
        assert elapsed <= 0.7, "Should not wait too long for token"  # Allow some buffer
    finally:
        await limiter.cleanup()

@pytest.mark.asyncio
async def test_rate_limiter_get_current_usage():
    limiter = AsyncRateLimiter(rate_limit=3, time_window=1)
    try:
        assert limiter.get_current_usage() == 0
        await limiter.acquire()
        assert limiter.get_current_usage() == 1
        await limiter.acquire()
        assert limiter.get_current_usage() == 2
    finally:
        await limiter.cleanup()

@pytest.mark.asyncio
async def test_rate_limiter_get_time_until_next_token():
    limiter = AsyncRateLimiter(rate_limit=2, time_window=0.5)  # Consistent time window
    try:
        assert limiter.get_time_until_next_token() == 0, "Should have no wait initially"
        await limiter.acquire()
        assert limiter.get_time_until_next_token() == 0, "Should have no wait after first acquire"
        await limiter.acquire()
        wait_time = limiter.get_time_until_next_token()
        assert 0 < wait_time <= 0.5, f"Wait time {wait_time} should be between 0 and 0.5s"
    finally:
        await limiter.cleanup()

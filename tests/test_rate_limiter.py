import pytest
import asyncio
import sys
from bookbot.utils.rate_limiter import AsyncRateLimiter, RateLimitConfig

if sys.platform.startswith('linux'):
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

@pytest.mark.asyncio
async def test_rate_limiter_initialization():
    config = RateLimitConfig(requests_per_window=20, window_seconds=0.5)
    limiter = AsyncRateLimiter(config)
    try:
        usage = await limiter.get_current_usage()
        assert usage["window_limit"] == 20
        assert usage["current_requests"] == 0
        assert usage["burst_tokens"] == 0
    finally:
        await limiter.cleanup()

@pytest.mark.asyncio
async def test_rate_limiter_acquire():
    config = RateLimitConfig(requests_per_window=2, window_seconds=0.5)
    limiter = AsyncRateLimiter(config)
    try:
        assert await limiter.acquire(), "First acquire should succeed"
        assert await limiter.acquire(), "Second acquire should succeed"
        assert not await limiter.acquire(), "Third acquire should fail"
        
        await asyncio.sleep(0.6)  # Wait just over the window
        assert await limiter.acquire(), "Should be able to acquire after waiting"
    finally:
        await limiter.cleanup()

@pytest.mark.asyncio
async def test_rate_limiter_burst():
    config = RateLimitConfig(requests_per_window=2, window_seconds=0.5, max_burst=1)
    limiter = AsyncRateLimiter(config)
    try:
        assert await limiter.acquire(), "First acquire should succeed"
        assert await limiter.acquire(), "Second acquire should succeed"
        assert await limiter.acquire(), "Burst token should allow third acquire"
        assert not await limiter.acquire(), "Fourth acquire should fail"
        
        await limiter.refill_burst()
        assert await limiter.acquire(), "Should be able to acquire after burst refill"
    finally:
        await limiter.cleanup()

@pytest.mark.asyncio
async def test_rate_limiter_wait_for_token():
    config = RateLimitConfig(requests_per_window=2, window_seconds=0.5)
    limiter = AsyncRateLimiter(config)
    try:
        assert await limiter.acquire(), "First acquire should succeed"
        assert await limiter.acquire(), "Second acquire should succeed"
        
        start_time = asyncio.get_event_loop().time()
        assert await limiter.wait_for_token(timeout=1.0), "Wait should succeed within timeout"
        elapsed = asyncio.get_event_loop().time() - start_time
        
        assert elapsed >= 0.1, "Should wait at least 0.1s for token"
        assert elapsed <= 0.7, "Should not wait too long for token"
    finally:
        await limiter.cleanup()

@pytest.mark.asyncio
async def test_rate_limiter_wait_timeout():
    config = RateLimitConfig(requests_per_window=1, window_seconds=1.0)
    limiter = AsyncRateLimiter(config)
    try:
        assert await limiter.acquire(), "First acquire should succeed"
        assert not await limiter.wait_for_token(timeout=0.1), "Wait should timeout"
    finally:
        await limiter.cleanup()

@pytest.mark.asyncio
async def test_rate_limiter_get_current_usage():
    config = RateLimitConfig(requests_per_window=3, window_seconds=1, max_burst=1)
    limiter = AsyncRateLimiter(config)
    try:
        usage = await limiter.get_current_usage()
        assert usage["current_requests"] == 0
        assert usage["burst_tokens"] == 1
        
        await limiter.acquire()
        usage = await limiter.get_current_usage()
        assert usage["current_requests"] == 1
        
        await limiter.acquire()
        await limiter.acquire()  # Uses burst token
        usage = await limiter.get_current_usage()
        assert usage["current_requests"] == 3
        assert usage["burst_tokens"] == 0
    finally:
        await limiter.cleanup()

@pytest.mark.asyncio
async def test_rate_limiter_get_time_until_next_token():
    config = RateLimitConfig(requests_per_window=2, window_seconds=0.5)
    limiter = AsyncRateLimiter(config)
    try:
        wait_time = await limiter.get_time_until_next_token()
        assert wait_time == 0, "Should have no wait initially"
        
        await limiter.acquire()
        wait_time = await limiter.get_time_until_next_token()
        assert wait_time == 0, "Should have no wait after first acquire"
        
        await limiter.acquire()
        wait_time = await limiter.get_time_until_next_token()
        assert 0 < wait_time <= 0.5, f"Wait time {wait_time} should be between 0 and 0.5s"
    finally:
        await limiter.cleanup()

import pytest
import asyncio
from pathlib import Path
from bookbot.utils.venice_client import VeniceClient, VeniceConfig

@pytest.mark.asyncio
async def test_venice_client_initialization():
    config = VeniceConfig(
        api_key="test_key",
        model="venice-xl",
        max_tokens=2048,
        temperature=0.7
    )
    client = VeniceClient(config)
    assert client.config.api_key == "test_key"
    assert client.config.model == "venice-xl"
    assert client.config.max_tokens == 2048
    assert client.config.temperature == 0.7

@pytest.mark.asyncio
async def test_venice_client_session_management():
    config = VeniceConfig(api_key="test_key")
    client = VeniceClient(config)
    
    session1 = await client._get_session()
    session2 = await client._get_session()
    assert session1 is session2
    
    await client.cleanup()
    assert client._session.closed
    
    session3 = await client._get_session()
    assert session3 is not session1

@pytest.mark.asyncio
async def test_venice_client_rate_limiting():
    config = VeniceConfig(api_key="test_key")
    client = VeniceClient(config)
    
    start_time = asyncio.get_event_loop().time()
    
    # First request should go through immediately
    result1 = await client.generate("test prompt 1")
    assert result1["choices"][0]["text"]
    
    # Make 20 more requests to hit rate limit
    for i in range(20):
        await client.generate(f"test prompt {i+2}")
    
    # Next request should be delayed
    result2 = await client.generate("test prompt 22")
    elapsed = asyncio.get_event_loop().time() - start_time
    assert elapsed >= 1.0
    assert result2["choices"][0]["text"]
    
    await client.cleanup()

@pytest.mark.asyncio
async def test_venice_client_caching():
    config = VeniceConfig(api_key="test_key")
    client = VeniceClient(config)
    
    # First request
    result1 = await client.generate("test prompt", temperature=0.7)
    
    # Same request should use cache
    result2 = await client.generate("test prompt", temperature=0.7)
    assert result1 == result2
    
    # Different parameters should bypass cache
    result3 = await client.generate("test prompt", temperature=0.8)
    assert result3 != result1
    
    await client.cleanup()

@pytest.mark.asyncio
async def test_venice_client_token_tracking():
    config = VeniceConfig(api_key="test_key")
    client = VeniceClient(config)
    
    initial_usage = client._token_tracker.get_usage()
    await client.generate("test prompt")
    final_usage = client._token_tracker.get_usage()
    
    assert final_usage.input_tokens > initial_usage.input_tokens
    assert final_usage.output_tokens > initial_usage.output_tokens
    assert final_usage.cost > initial_usage.cost
    
    await client.cleanup()

@pytest.mark.asyncio
async def test_venice_client_error_handling():
    config = VeniceConfig(api_key="invalid_key")
    client = VeniceClient(config)
    
    with pytest.raises(RuntimeError) as exc_info:
        await client.generate("test prompt")
    assert "Venice API error" in str(exc_info.value)
    
    await client.cleanup()

@pytest.mark.asyncio
async def test_venice_client_embed():
    config = VeniceConfig(api_key="test_key")
    client = VeniceClient(config)
    
    initial_usage = client._token_tracker.get_usage()
    result = await client.embed("test input")
    final_usage = client._token_tracker.get_usage()
    
    assert final_usage.input_tokens > initial_usage.input_tokens
    assert final_usage.output_tokens == initial_usage.output_tokens
    
    await client.cleanup()

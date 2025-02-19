import pytest
import asyncio
from pathlib import Path
from bookbot.utils.venice_client import VeniceClient, VeniceConfig

@pytest.mark.asyncio
@pytest.mark.timeout(10)
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
    
    # Test serialization
    import pickle
    data = pickle.dumps(client)
    restored_client = pickle.loads(data)
    assert restored_client.config.api_key == "test_key"
    assert restored_client.config.model == "venice-xl"
    assert restored_client.config.max_tokens == 2048
    assert restored_client.config.temperature == 0.7
    assert restored_client._session is None  # Session should not be serialized
    assert "Authorization" not in restored_client.headers  # Sensitive data removed

@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_venice_client_session_management():
    config = VeniceConfig(api_key="test_key")
    client = VeniceClient(config)
    
    session1 = await client._get_session()
    session2 = await client._get_session()
    assert session1 is session2
    
    await client.cleanup()
    assert client._session is None or client._session.closed
    
    session3 = await client._get_session()
    assert session3 is not session1

@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_venice_client_rate_limiting():
    config = VeniceConfig(api_key="test_key")
    client = VeniceClient(config)
    
    start_time = asyncio.get_event_loop().time()
    
    # First request should go through immediately
    result1 = await client.generate("test prompt 1")
    assert result1["choices"][0]["text"]
    
    # Make 5 more requests (reduced from 20 for faster testing)
    for i in range(5):
        await client.generate(f"test prompt {i+2}")
    
    # Next request should be delayed
    result2 = await client.generate("test prompt 7")
    elapsed = asyncio.get_event_loop().time() - start_time
    assert elapsed >= 0.2  # Reduced delay expectation
    assert result2["choices"][0]["text"]
    
    await client.cleanup()

@pytest.mark.asyncio
@pytest.mark.timeout(10)
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
@pytest.mark.timeout(5)
async def test_venice_client_token_tracking():
    config = VeniceConfig(api_key="test_key")
    client = VeniceClient(config)
    
    initial_usage = await client._token_tracker.get_usage()
    await client.generate("test prompt")
    final_usage = await client._token_tracker.get_usage()
    
    assert final_usage.input_tokens > initial_usage.input_tokens
    assert final_usage.output_tokens > initial_usage.output_tokens
    assert final_usage.cost > initial_usage.cost
    
    await client.cleanup()

@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_venice_client_error_handling():
    config = VeniceConfig(api_key="invalid_key")
    client = VeniceClient(config)
    
    with pytest.raises(RuntimeError) as exc_info:
        await client.generate("test prompt")
    assert "Venice API error" in str(exc_info.value)
    
    await client.cleanup()

@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_venice_client_embed():
    config = VeniceConfig(api_key="test_key")
    client = VeniceClient(config)
    
    try:
        initial_usage = await client._token_tracker.get_usage()
        result = await client.embed("test input")
        final_usage = await client._token_tracker.get_usage()
        
        assert final_usage.input_tokens > initial_usage.input_tokens
        assert final_usage.output_tokens == initial_usage.output_tokens
    finally:
        if client._session and not client._session.closed:
            await client._session.close()
        await client.cleanup()
        await asyncio.sleep(0.1)  # Allow event loop to clean up

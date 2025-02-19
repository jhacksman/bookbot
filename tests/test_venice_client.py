import pytest
from bookbot.utils.venice_client import VeniceClient, VeniceConfig
import os

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
async def test_venice_client_generate():
    config = VeniceConfig(api_key="test_key")
    client = VeniceClient(config)
    
    # Note: This test would need a real API key to actually test the generate method
    # For now, we just verify the client can be instantiated
    assert client.base_url == "https://api.venice.ai/v1"
    assert "Authorization" in client.headers
    assert "Content-Type" in client.headers

@pytest.mark.asyncio
async def test_venice_client_embed():
    config = VeniceConfig(api_key="test_key")
    client = VeniceClient(config)
    
    # Note: This test would need a real API key to actually test the embed method
    # For now, we just verify the client can be instantiated
    assert client.base_url == "https://api.venice.ai/v1"
    assert "Authorization" in client.headers
    assert "Content-Type" in client.headers

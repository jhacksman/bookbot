import pytest
from unittest.mock import AsyncMock
from bookbot.utils.resource_manager import VRAMManager
from bookbot.utils.venice_client import VeniceClient, VeniceConfig
from typing import Dict, Any

@pytest.fixture
def mock_venice_client(monkeypatch):
    class MockVeniceClient:
        def __init__(self, *args, **kwargs):
            self.config = kwargs.get("config")
            if not self.config and args:
                self.config = args[0]
            if not self.config:
                self.config = VeniceConfig(api_key="test_key")

        async def generate(self, prompt: str, *args, **kwargs):
            try:
                if "hierarchical" in prompt.lower():
                    return {
                        "choices": [{
                            "text": "A detailed summary of the content at the specified level."
                        }]
                    }
                elif "evaluate" in prompt.lower():
                    return {
                        "choices": [{
                            "text": '{"score": 95, "reasoning": "This book is highly relevant for AI research", "key_topics": ["deep learning", "neural networks", "machine learning"]}'
                        }]
                    }
                else:
                    return {
                        "choices": [{
                            "text": '{"answer": "A detailed response", "citations": [], "confidence": 0.9}'
                        }]
                    }
            except Exception as e:
                return {
                    "choices": [{
                        "text": str(e)
                    }]
                }

        async def embed(self, texts: list, *args, **kwargs):
            return {"data": [{"embedding": [0.1, 0.2, 0.3] * 128} for _ in range(len(texts) if isinstance(texts, list) else 1)]}

    # Patch VeniceClient in all modules
    for module in [
        "bookbot.utils.venice_client",
        "bookbot.agents.selection.agent",
        "bookbot.agents.summarization.agent",
        "bookbot.agents.librarian.agent",
        "bookbot.agents.query.agent"
    ]:
        monkeypatch.setattr(f"{module}.VeniceClient", MockVeniceClient)
    
    return MockVeniceClient()

@pytest.fixture
def venice_config(mock_venice_client):
    return VeniceConfig(api_key="test_key")

@pytest.fixture
def vram_manager():
    return VRAMManager(total_vram=64.0)

@pytest.fixture
def test_book_data() -> Dict[str, Any]:
    return {
        "title": "Test Book",
        "author": "Test Author",
        "content": "Test content for verification",
        "metadata": {
            "format": "pdf",
            "pages": 100,
            "language": "en"
        }
    }

@pytest.fixture
def test_summary_data() -> Dict[str, Any]:
    return {
        "book_id": 1,
        "level": 0,
        "content": "Test summary content",
        "vector_id": "test_vector_123"
    }

@pytest.fixture
async def initialized_vram_manager(vram_manager):
    yield vram_manager
    # Cleanup any remaining allocations
    vram_manager.allocated_vram = 0.0
    vram_manager.allocations.clear()

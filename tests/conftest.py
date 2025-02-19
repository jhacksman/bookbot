import pytest
from bookbot.utils.resource_manager import VRAMManager
from typing import Dict, Any

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

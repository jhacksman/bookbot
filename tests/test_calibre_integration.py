import pytest
import json
from pathlib import Path
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from bookbot.utils.venice_client import VeniceConfig
from bookbot.agents.librarian.agent import LibrarianAgent
from bookbot.utils.calibre_connector import CalibreConnector

@pytest.fixture
async def mock_calibre_db(tmp_path):
    db_path = str(tmp_path / "metadata.db")
    calibre = CalibreConnector(Path(db_path))
    try:
        await calibre.initialize()
        
        # Add test books
        await calibre.add_book({
        "title": "Test Book 1",
        "author": "Test Author 1",
        "format": "EPUB",
        "identifiers": {"isbn": "1234567890"},
        "tags": ["test", "fiction"],
        "series": "Test Series",
        "series_index": 1,
        "last_modified": datetime.now()
    })
    
    await calibre.add_book({
        "title": "Test Book 2",
        "author": "Test Author 2",
        "format": "PDF",
        "identifiers": {"doi": "10.1234/test"},
        "tags": ["test", "non-fiction"],
        "last_modified": datetime.now()
    })
    
        return str(db_path)
    finally:
        await calibre.cleanup()

@pytest.mark.asyncio
async def test_librarian_calibre_initialization(venice_config, mock_calibre_db):
    agent = LibrarianAgent(
        venice_config=venice_config,
        calibre_path=mock_calibre_db
    )
    await agent.initialize()
    assert hasattr(agent, 'calibre')
    assert agent.calibre is not None
    await agent.cleanup()

@pytest.mark.asyncio
async def test_calibre_sync_success(venice_config, mock_calibre_db):
    agent = LibrarianAgent(
        venice_config=venice_config,
        calibre_path=mock_calibre_db
    )
    await agent.initialize()
    
    result = await agent.process({"action": "sync_calibre"})
    assert result["status"] == "success"
    assert result["books_synced"] == 2
    assert result["total_books"] == 2
    
    # Verify books were added to database
    book1 = await agent.get_book(1)
    assert book1 is not None
    assert book1["title"] == "Test Book 1"
    assert book1["author"] == "Test Author 1"
    metadata1 = json.loads(book1["metadata"])
    assert metadata1["calibre_id"] is not None
    assert metadata1["format"] == "EPUB"
    assert metadata1["tags"] == ["test", "fiction"]
    assert metadata1["series"] == "Test Series"
    
    book2 = await agent.get_book(2)
    assert book2 is not None
    assert book2["title"] == "Test Book 2"
    assert book2["author"] == "Test Author 2"
    metadata2 = json.loads(book2["metadata"])
    assert metadata2["calibre_id"] is not None
    assert metadata2["format"] == "PDF"
    assert metadata2["tags"] == ["test", "non-fiction"]
    
    await agent.cleanup()

@pytest.mark.asyncio
async def test_calibre_sync_no_calibre_configured(venice_config):
    agent = LibrarianAgent(venice_config=venice_config)
    await agent.initialize()
    
    result = await agent.process({"action": "sync_calibre"})
    assert result["status"] == "error"
    assert "Calibre integration not configured" in result["message"]
    
    await agent.cleanup()

@pytest.mark.asyncio
async def test_calibre_sync_update_existing(venice_config, mock_calibre_db):
    agent = LibrarianAgent(
        venice_config=venice_config,
        calibre_path=mock_calibre_db
    )
    await agent.initialize()
    
    # First sync
    await agent.process({"action": "sync_calibre"})
    
    # Update a book in Calibre
    calibre = CalibreConnector(mock_calibre_db)
    await calibre.initialize()
    await calibre.update_book(1, {
        "tags": ["test", "fiction", "updated"],
        "series": "Updated Series"
    })
    
    # Second sync
    result = await agent.process({"action": "sync_calibre"})
    assert result["status"] == "success"
    assert result["books_synced"] == 2
    
    # Verify update
    book1 = await agent.get_book(1)
    metadata1 = json.loads(book1["metadata"])
    assert "updated" in metadata1["tags"]
    assert metadata1["series"] == "Updated Series"
    
    await agent.cleanup()
    await calibre.cleanup()

import pytest
import json
from pathlib import Path
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from bookbot.utils.venice_client import VeniceConfig
from bookbot.agents.librarian.agent import LibrarianAgent
from bookbot.utils.calibre_connector import CalibreConnector
from bookbot.database.models import Base

@pytest.fixture
async def async_session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    session_maker = async_sessionmaker(engine, expire_on_commit=False)
    async with session_maker() as session:
        yield session
    
    await engine.dispose()

@pytest.fixture
async def mock_calibre_db(tmp_path):
    db_path = tmp_path / "metadata.db"
    calibre = CalibreConnector(db_path)
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
    
    db_path_str = str(db_path)
    yield db_path_str
    await calibre.cleanup()

@pytest.mark.asyncio
async def test_librarian_calibre_initialization(venice_config, mock_calibre_db, async_session):
    db_path_str = await mock_calibre_db
    db_path = Path(db_path_str)
    agent = LibrarianAgent(
        venice_config=venice_config,
        session=async_session,
        db_url="sqlite+aiosqlite:///:memory:",
        calibre_path=db_path,
        vram_limit=16.0
    )
    await agent.initialize()
    assert hasattr(agent, 'calibre')
    assert agent.calibre is not None
    await agent.cleanup()

@pytest.mark.asyncio
async def test_calibre_sync_success(venice_config, mock_calibre_db, async_session):
    db_path_str = await mock_calibre_db
    db_path = Path(db_path_str)
    agent = LibrarianAgent(
        venice_config=venice_config,
        session=async_session,
        db_url="sqlite+aiosqlite:///:memory:",
        calibre_path=db_path,
        vram_limit=16.0
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
async def test_calibre_sync_no_calibre_configured(venice_config, async_session):
    agent = LibrarianAgent(venice_config=venice_config, session=async_session, db_url="sqlite+aiosqlite:///:memory:", calibre_path=None, vram_limit=16.0)
    await agent.initialize()
    
    result = await agent.process({"action": "sync_calibre"})
    assert result["status"] == "error"
    assert "Calibre integration not configured" in result["message"]
    
    await agent.cleanup()

@pytest.mark.asyncio
async def test_calibre_sync_update_existing(venice_config, mock_calibre_db, async_session):
    db_path_str = await mock_calibre_db
    db_path = Path(db_path_str)
    agent = LibrarianAgent(
        venice_config=venice_config,
        session=async_session,
        db_url="sqlite+aiosqlite:///:memory:",
        calibre_path=db_path,
        vram_limit=16.0
    )
    await agent.initialize()
    
    # First sync
    await agent.process({"action": "sync_calibre"})
    
    # Update a book in Calibre
    calibre = CalibreConnector(db_path)
    await calibre.initialize()
    
    # Get current book data
    books = await calibre.get_books()
    book_id = books[0]["id"] if books else 1
    
    # Update book
    result = await calibre.tag_book(book_id, "updated")
    assert result["status"] == "success"
    
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

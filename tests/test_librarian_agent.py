import pytest
import asyncio
from ebooklib import epub
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from bookbot.agents.librarian.agent import LibrarianAgent
from bookbot.utils.venice_client import VeniceConfig
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

@pytest.mark.asyncio
async def test_librarian_agent_initialization(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = LibrarianAgent(venice_config=config, session=async_session, db_url="sqlite+aiosqlite:///:memory:", calibre_path=None, vram_limit=16.0)
    await agent.initialize()
    assert agent.is_active
    await agent.cleanup()
    assert not agent.is_active

@pytest.mark.asyncio
async def test_librarian_agent_add_book(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = LibrarianAgent(venice_config=config, session=async_session, db_url="sqlite+aiosqlite:///:memory:", calibre_path=None, vram_limit=16.0)
    await agent.initialize()
    
    test_book = {
        "title": "Test Book",
        "author": "Test Author",
        "content_hash": "abc123",
        "metadata": {"format": "pdf"},
        "vector_id": "vec123"
    }
    
    result = await agent.process({
        "action": "add_book",
        "book": test_book
    })
    
    assert result["status"] == "success"
    assert "book_id" in result
    
    # Verify book was added
    book = await agent.get_book(result["book_id"])
    assert book is not None
    assert book["title"] == test_book["title"]
    assert book["author"] == test_book["author"]
    
    await agent.cleanup()

@pytest.mark.asyncio
async def test_librarian_agent_add_summary(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = LibrarianAgent(venice_config=config, session=async_session, db_url="sqlite+aiosqlite:///:memory:", calibre_path=None, vram_limit=16.0)
    await agent.initialize()
    
    # First add a book
    book_result = await agent.process({
        "action": "add_book",
        "book": {"title": "Test Book"}
    })
    
    test_summary = {
        "book_id": book_result["book_id"],
        "level": 0,
        "content": "Test summary content",
        "vector_id": "vec456"
    }
    
    result = await agent.process({
        "action": "add_summary",
        "summary": test_summary
    })
    
    assert result["status"] == "success"
    assert "summary_id" in result
    
    await agent.cleanup()

@pytest.mark.asyncio
async def test_librarian_agent_invalid_action(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = LibrarianAgent(venice_config=config, session=async_session, db_url="sqlite+aiosqlite:///:memory:", calibre_path=None, vram_limit=16.0)
    await agent.initialize()
    
    result = await agent.process({
        "action": "invalid_action"
    })
    
    assert result["status"] == "error"
    assert "message" in result
    
    await agent.cleanup()

@pytest.fixture
def test_epub_path(tmp_path):
    from ebooklib import epub
    from ebooklib import epub
    book = epub.EpubBook()
    book.set_identifier('test123')
    book.set_title('Test Book')
    book.set_language('en')
    book.add_author('Test Author')
    
    c1 = epub.EpubHtml(title='Chapter 1', file_name='chap_01.xhtml', lang='en')
    c1.content = '<h1>Chapter 1</h1><p>This is a test chapter.</p>'
    c1.id = 'chapter1'
    book.add_item(c1)
    
    nav = epub.EpubNav()
    nav.id = 'nav'
    book.add_item(nav)
    
    # Add navigation files
    nav = epub.EpubNav()
    nav.id = 'nav'
    book.add_item(nav)
    
    # Create spine with nav first
    book.spine = ['nav', c1.id]
    book.toc = [(epub.Section('Test Book'), [c1])]
    
    # Add NCX
    book.add_item(epub.EpubNcx())
    
    epub_path = tmp_path / "test.epub"
    epub.write_epub(str(epub_path), book)
    return str(epub_path)

@pytest.mark.asyncio
async def test_librarian_agent_process_epub(test_epub_path, async_session):
    from ebooklib import epub
    config = VeniceConfig(api_key="test_key")
    agent = LibrarianAgent(venice_config=config, session=async_session, db_url="sqlite+aiosqlite:///:memory:", calibre_path=None, vram_limit=16.0)
    await agent.initialize()
    
    try:
        result = await agent.process({
            "action": "process_epub",
            "file_path": str(test_epub_path)
        })
        
        assert result["status"] == "success"
        assert "book_id" in result
        assert "vector_ids" in result
        assert len(result["vector_ids"]) > 0
        
        # Verify book was added
        book_result = await agent.process({
            "action": "get_book",
            "book_id": result["book_id"]
        })
        
        assert book_result["status"] == "success"
        assert book_result["book"]["title"] == "Test Book"
        assert book_result["book"]["author"] == "Test Author"
        assert "content_hash" in book_result["book"]
        assert "vector_id" in book_result["book"]
    finally:
        await agent.cleanup()

@pytest.mark.asyncio
async def test_librarian_agent_process_epub_invalid_file(async_session, tmp_path):
    config = VeniceConfig(api_key="test_key")
    agent = LibrarianAgent(venice_config=config, session=async_session, db_url="sqlite+aiosqlite:///:memory:", calibre_path=None, vram_limit=16.0)
    await agent.initialize()
    
    invalid_path = tmp_path / "invalid.epub"
    with open(invalid_path, 'w') as f:
        f.write("Not an EPUB file")
    
    try:
        result = await agent.process({
            "action": "process_epub",
            "file_path": str(invalid_path)
        })
        
        assert result["status"] == "error"
        assert "message" in result
    finally:
        await agent.cleanup()

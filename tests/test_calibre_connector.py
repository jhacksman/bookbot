import pytest
import sqlite3
from pathlib import Path
from datetime import datetime
import asyncio
from bookbot.utils.calibre_connector import CalibreConnector, LibraryWatcher

@pytest.fixture
def mock_calibre_db(tmp_path):
    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create tables
    cursor.executescript("""
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            author_sort TEXT,
            timestamp TIMESTAMP,
            pubdate TIMESTAMP,
            series_index REAL,
            isbn TEXT,
            path TEXT
        );
        
        CREATE TABLE data (
            id INTEGER PRIMARY KEY,
            book INTEGER NOT NULL,
            format TEXT NOT NULL,
            name TEXT NOT NULL,
            FOREIGN KEY(book) REFERENCES books(id)
        );
        
        CREATE TABLE tags (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        );
        
        CREATE TABLE books_tags_link (
            book INTEGER NOT NULL,
            tag INTEGER NOT NULL,
            PRIMARY KEY(book, tag),
            FOREIGN KEY(book) REFERENCES books(id),
            FOREIGN KEY(tag) REFERENCES tags(id)
        );
        
        CREATE TABLE identifiers (
            id INTEGER PRIMARY KEY,
            book INTEGER NOT NULL,
            type TEXT NOT NULL,
            val TEXT NOT NULL,
            FOREIGN KEY(book) REFERENCES books(id)
        );
        
        CREATE TABLE comments (
            id INTEGER PRIMARY KEY,
            book INTEGER NOT NULL,
            text TEXT NOT NULL,
            FOREIGN KEY(book) REFERENCES books(id)
        );
    """)
    
    # Add test data
    cursor.execute("""
        INSERT INTO books (title, author_sort, isbn, path)
        VALUES (?, ?, ?, ?)
    """, ('Test Book', 'Test Author', '1234567890', 'Test Book (1)'))
    book_id = cursor.lastrowid
    
    cursor.execute("INSERT INTO tags (name) VALUES (?)", ('test-tag',))
    tag_id = cursor.lastrowid
    
    cursor.execute("INSERT INTO books_tags_link (book, tag) VALUES (?, ?)",
                  (book_id, tag_id))
    
    cursor.execute("""
        INSERT INTO data (book, format, name)
        VALUES (?, ?, ?)
    """, (book_id, 'EPUB', 'Test Book - Test Author.epub'))
    
    cursor.execute("""
        INSERT INTO identifiers (book, type, val)
        VALUES (?, ?, ?)
    """, (book_id, 'isbn', '1234567890'))
    
    cursor.execute("""
        INSERT INTO comments (book, text)
        VALUES (?, ?)
    """, (book_id, 'Test book description'))
    
    conn.commit()
    conn.close()
    return tmp_path

@pytest.mark.asyncio
async def test_get_books(mock_calibre_db):
    connector = CalibreConnector(mock_calibre_db)
    books = await connector.get_books()
    
    assert len(books) == 1
    book = books[0]
    assert book['title'] == 'Test Book'
    assert book['author_sort'] == 'Test Author'
    assert book['isbn'] == '1234567890'
    assert book['formats'] == ['EPUB']
    assert book['tags'] == ['test-tag']
    assert book['identifiers'] == {'isbn': '1234567890'}
    assert book['description'] == 'Test book description'

@pytest.mark.asyncio
async def test_get_book_files(mock_calibre_db):
    connector = CalibreConnector(mock_calibre_db)
    files = await connector.get_book_files(1)
    
    assert len(files) == 1
    assert files[0]['format'] == 'EPUB'
    assert files[0]['path'].name == 'Test Book - Test Author.epub'

@pytest.mark.asyncio
async def test_last_sync_time(mock_calibre_db):
    connector = CalibreConnector(mock_calibre_db)
    assert connector.last_sync_time is None
    
    await connector.get_books()
    assert isinstance(connector.last_sync_time, datetime)

@pytest.mark.asyncio
async def test_tag_book(mock_calibre_db):
    connector = CalibreConnector(mock_calibre_db)
    
    # Add new tag
    await connector.tag_book(1, "test-tag-2")
    tags = await connector.get_book_tags(1)
    assert "test-tag-2" in tags
    assert "test-tag" in tags  # Original tag still present
    
    # Add same tag again (should not error)
    await connector.tag_book(1, "test-tag-2")
    tags = await connector.get_book_tags(1)
    assert len([t for t in tags if t == "test-tag-2"]) == 1  # No duplicates

@pytest.mark.asyncio
async def test_get_book_tags(mock_calibre_db):
    connector = CalibreConnector(mock_calibre_db)
    tags = await connector.get_book_tags(1)
    assert tags == ["test-tag"]
    
    # Test non-existent book
    tags = await connector.get_book_tags(999)
    assert tags == []

@pytest.mark.asyncio
async def test_library_watcher(mock_calibre_db):
    connector = CalibreConnector(mock_calibre_db)
    
    # Start watching
    observer = await connector.watch_library()
    try:
        # Trigger a change
        with open(mock_calibre_db / "metadata.db", "a") as f:
            f.write(" ")  # Modify the file
        
        # Wait for callback
        await asyncio.sleep(1)
        
        # Verify last_sync_time was updated
        assert connector.last_sync_time is not None
    finally:
        observer.stop()
        observer.join()

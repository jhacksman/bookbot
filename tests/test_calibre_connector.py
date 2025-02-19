import pytest
import sqlite3
import os
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
async def test_library_watcher(mock_calibre_db, caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    
    connector = CalibreConnector(mock_calibre_db)
    callback_count = 0
    
    # Create an event to track callback completion
    callback_completed = asyncio.Event()
    
    async def wrapped_callback():
        nonlocal callback_count
        callback_count += 1
        print(f"DEBUG: Callback triggered (count: {callback_count})")
        try:
            await connector.get_books()
            callback_completed.set()
            print("DEBUG: Callback completed")
        except Exception as e:
            print(f"DEBUG: Callback error: {e}")
            raise
    
    # Set up the callback before starting the observer
    connector._on_library_change = wrapped_callback
    
    # Start watching
    observer = await connector.watch_library()
    try:
        # Give the observer time to start watching and settle
        await asyncio.sleep(1.0)
        print("DEBUG: Starting file modification")
        
        # Trigger multiple changes to increase chance of detection
        for i in range(3):
            # Open file in binary mode to ensure consistent writes
            with open(mock_calibre_db / "metadata.db", "ab") as f:
                f.write(b" ")
                f.flush()
                os.fsync(f.fileno())
            # Give the file system events time to propagate
            await asyncio.sleep(1.0)
            print(f"DEBUG: File modified (attempt {i+1})")
            
            # Check if callback was triggered after each modification
            if callback_count > 0:
                print("DEBUG: Callback was triggered, breaking loop")
                break
        
        print("DEBUG: Waiting for callback")
        try:
            # Wait for callback to complete
            await asyncio.wait_for(callback_completed.wait(), timeout=5.0)
            assert callback_count > 0, "Callback was never triggered"
            assert connector.last_sync_time is not None, "last_sync_time was not updated"
        except asyncio.TimeoutError:
            print("DEBUG: Timeout waiting for callback")
            print(f"DEBUG: Current callback count: {callback_count}")
            print(f"DEBUG: Last sync time: {connector.last_sync_time}")
            pytest.fail(f"Library watcher callback did not complete in time. Callback count: {callback_count}")
    finally:
        print("DEBUG: Stopping observer")
        # Clean up the event processor first
        event_handler.cleanup()
        # Then stop the observer
        observer.stop()
        # Give the observer thread time to clean up
        await asyncio.sleep(1.0)
        observer.join(timeout=2.0)
        if observer.is_alive():
            print("DEBUG: First join attempt failed, trying again")
            observer.join(timeout=1.0)
            if observer.is_alive():
                pytest.fail("Observer thread did not stop properly")

from typing import Dict, List, Optional, Any, Callable, Awaitable
import sqlite3
import json
import asyncio
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

class LibraryWatcher(FileSystemEventHandler):
    def __init__(self, callback: Callable[[], Awaitable[None]]):
        self.callback = callback
        self._loop = asyncio.get_event_loop()
        self._queue = asyncio.Queue()
        self._task = None
        self._running = True
        self._shutdown = False
        
    async def _process_events(self):
        try:
            while not self._shutdown:
                try:
                    event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                    if event is None:  # Shutdown signal
                        break
                    print(f"DEBUG: Processing event: {event}")
                    try:
                        if not self._shutdown:
                            await self.callback()
                            print("DEBUG: Callback completed successfully")
                    except Exception as e:
                        print(f"DEBUG: Callback error: {e}")
                    finally:
                        self._queue.task_done()
                except asyncio.TimeoutError:
                    if self._shutdown:
                        break
                    continue
                except RuntimeError as e:
                    if "Event loop is closed" in str(e) or self._shutdown:
                        break
        except asyncio.CancelledError:
            print("DEBUG: Event processor cancelled")
        finally:
            self._running = False
            # Drain the queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except asyncio.QueueEmpty:
                    break
    
    def on_modified(self, event: FileModifiedEvent) -> None:
        if not event.src_path.endswith("metadata.db"):
            return
            
        print(f"DEBUG: File modification detected: {event.src_path}")
        try:
            # Put the event in the queue
            future = asyncio.run_coroutine_threadsafe(
                self._queue.put(event), self._loop
            )
            future.result(timeout=1.0)  # Wait for queue put to complete
            
            # Start the event processor if not running
            if self._task is None or self._task.done():
                self._task = asyncio.run_coroutine_threadsafe(
                    self._process_events(), self._loop
                )
                print("DEBUG: Started event processor task")
        except Exception as e:
            print(f"DEBUG: Error in on_modified: {e}")
            
    def cleanup(self):
        """Clean up the event processor task."""
        self._shutdown = True
        if self._task and not self._task.done():
            try:
                # Signal shutdown
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._queue.put(None), self._loop
                    )
                    future.result(timeout=0.1)
                except Exception:
                    pass
                
                # Cancel the task
                self._loop.call_soon_threadsafe(self._task.cancel)
                
                # Wait briefly for cancellation
                try:
                    self._task.result(timeout=0.1)
                except Exception:
                    pass
            finally:
                self._task = None
                # Create a new queue to prevent any lingering tasks
                self._queue = asyncio.Queue()

class CalibreConnector:
    def __init__(self, library_path: Path):
        self.library_path = Path(library_path)
        self.db_path = self.library_path / "metadata.db"
        self._lock = asyncio.Lock()
        self._last_sync = None
        
    async def initialize(self) -> None:
        """Initialize the Calibre connector."""
        if not self.library_path.exists():
            self.library_path.mkdir(parents=True, exist_ok=True)
            
        # Create initial database if it doesn't exist
        if not self.db_path.exists():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create basic schema
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS books (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    author_sort TEXT,
                    path TEXT,
                    has_cover BOOL DEFAULT 0,
                    series_index REAL DEFAULT 1.0,
                    timestamp REAL DEFAULT 0.0,
                    pubdate REAL DEFAULT 0.0,
                    isbn TEXT DEFAULT NULL
                );
                
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE
                );
                
                CREATE TABLE IF NOT EXISTS books_tags_link (
                    book INTEGER NOT NULL,
                    tag INTEGER NOT NULL,
                    PRIMARY KEY (book, tag),
                    FOREIGN KEY (book) REFERENCES books(id),
                    FOREIGN KEY (tag) REFERENCES tags(id)
                );
                
                CREATE TABLE IF NOT EXISTS identifiers (
                    id INTEGER PRIMARY KEY,
                    book INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    val TEXT NOT NULL,
                    FOREIGN KEY (book) REFERENCES books(id)
                );
                
                CREATE TABLE IF NOT EXISTS data (
                    id INTEGER PRIMARY KEY,
                    book INTEGER NOT NULL,
                    format TEXT NOT NULL,
                    name TEXT NOT NULL,
                    FOREIGN KEY (book) REFERENCES books(id)
                );
                
                CREATE TABLE IF NOT EXISTS comments (
                    id INTEGER PRIMARY KEY,
                    book INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY (book) REFERENCES books(id)
                );
                
                CREATE TABLE IF NOT EXISTS series (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE
                );
                
                CREATE TABLE IF NOT EXISTS books_series_link (
                    book INTEGER NOT NULL,
                    series INTEGER NOT NULL,
                    PRIMARY KEY (book, series),
                    FOREIGN KEY (book) REFERENCES books(id),
                    FOREIGN KEY (series) REFERENCES series(id)
                );
            """)
            conn.commit()
            conn.close()
            
    async def add_book(self, book_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a book to the Calibre database."""
        async with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                
                # Add to books table
                cursor.execute("""
                    INSERT INTO books (title, author_sort, path, has_cover, series_index, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    book_data["title"],
                    book_data.get("author", "Unknown"),
                    book_data.get("path", ""),
                    0,  # has_cover
                    book_data.get("series_index", 1.0),
                    book_data.get("last_modified", datetime.now()).timestamp()
                ))
                book_id = cursor.lastrowid
                
                # Add identifiers
                if "identifiers" in book_data:
                    for id_type, id_val in book_data["identifiers"].items():
                        cursor.execute("""
                            INSERT INTO identifiers (book, type, val)
                            VALUES (?, ?, ?)
                        """, (book_id, id_type, id_val))
                
                # Add tags
                if "tags" in book_data:
                    for tag in book_data["tags"]:
                        cursor.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
                        cursor.execute("SELECT id FROM tags WHERE name = ?", (tag,))
                        tag_id = cursor.fetchone()[0]
                        cursor.execute("""
                            INSERT INTO books_tags_link (book, tag)
                            VALUES (?, ?)
                        """, (book_id, tag_id))
                
                # Add series
                if "series" in book_data:
                    cursor.execute("INSERT OR IGNORE INTO series (name) VALUES (?)", (book_data["series"],))
                    cursor.execute("SELECT id FROM series WHERE name = ?", (book_data["series"],))
                    series_id = cursor.fetchone()[0]
                    cursor.execute("""
                        INSERT INTO books_series_link (book, series)
                        VALUES (?, ?)
                    """, (book_id, series_id))
                
                conn.commit()
                return {"status": "success", "book_id": book_id}
                
            except Exception as e:
                conn.rollback()
                return {"status": "error", "message": str(e)}
            finally:
                conn.close()
    
    async def get_books(self) -> List[Dict[str, Any]]:
        async with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        books.id,
                        books.title,
                        books.author_sort,
                        books.timestamp,
                        books.pubdate,
                        books.series_index,
                        books.isbn,
                        GROUP_CONCAT(DISTINCT data.format) as formats,
                        GROUP_CONCAT(DISTINCT tags.name) as tags,
                        GROUP_CONCAT(DISTINCT identifiers.type || ':' || identifiers.val) as identifiers,
                        books.path,
                        comments.text as description
                    FROM books
                    LEFT JOIN data ON books.id = data.book
                    LEFT JOIN books_tags_link ON books.id = books_tags_link.book
                    LEFT JOIN tags ON books_tags_link.tag = tags.id
                    LEFT JOIN identifiers ON books.id = identifiers.book
                    LEFT JOIN comments ON books.id = comments.book
                    GROUP BY books.id
                """)
                
                columns = [description[0] for description in cursor.description]
                books = []
                
                for row in cursor.fetchall():
                    book = dict(zip(columns, row))
                    
                    # Parse formats and tags from concatenated strings
                    book['formats'] = book['formats'].split(',') if book['formats'] else []
                    book['tags'] = book['tags'].split(',') if book['tags'] else []
                    
                    # Parse identifiers into a dictionary
                    identifiers = {}
                    if book['identifiers']:
                        for identifier in book['identifiers'].split(','):
                            try:
                                id_type, id_val = identifier.split(':', 1)
                                identifiers[id_type] = id_val
                            except ValueError:
                                continue
                    book['identifiers'] = identifiers
                    
                    books.append(book)
                
                self._last_sync = datetime.now()
                return books
            finally:
                conn.close()
    
    async def get_book_files(self, book_id: int) -> List[Dict[str, str]]:
        async with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT format, name
                    FROM data
                    WHERE book = ?
                """, (book_id,))
                
                return [
                    {
                        'format': row[0],
                        'path': (self.library_path / row[1]).resolve()
                    }
                    for row in cursor.fetchall()
                ]
            finally:
                conn.close()
    
    @property
    def last_sync_time(self) -> Optional[datetime]:
        return self._last_sync
        
    async def _on_library_change(self) -> None:
        await self.get_books()
    
    async def watch_library(self) -> tuple[Observer, LibraryWatcher]:
        observer = Observer()
        event_handler = LibraryWatcher(self._on_library_change)
        observer.schedule(event_handler, str(self.library_path), recursive=False)
        observer.start()
        print("DEBUG: Started watching library")
        return observer, event_handler
    
    async def tag_book(self, book_id: int, tag: str) -> Dict[str, Any]:
        async with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                
                # Get or create tag
                cursor.execute("SELECT id FROM tags WHERE name = ?", (tag,))
                tag_result = cursor.fetchone()
                if not tag_result:
                    cursor.execute("INSERT INTO tags (name) VALUES (?)", (tag,))
                    tag_id = cursor.lastrowid
                else:
                    tag_id = tag_result[0]
                
                # Add tag to book
                cursor.execute("""
                    INSERT OR IGNORE INTO books_tags_link (book, tag)
                    VALUES (?, ?)
                """, (book_id, tag_id))
                
                conn.commit()
                return {"status": "success"}
            except Exception as e:
                conn.rollback()
                return {"status": "error", "message": str(e)}
            finally:
                conn.close()
                
    async def get_book_tags(self, book_id: int) -> List[str]:
        async with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT tags.name
                    FROM tags
                    JOIN books_tags_link ON tags.id = books_tags_link.tag
                    WHERE books_tags_link.book = ?
                """, (book_id,))
                return [row[0] for row in cursor.fetchall()]
            finally:
                conn.close()
                
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass  # No resources to clean up currently

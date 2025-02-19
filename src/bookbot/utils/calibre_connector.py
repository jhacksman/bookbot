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
        
    async def _process_events(self):
        try:
            while True:
                event = await self._queue.get()
                if event is None:  # Shutdown signal
                    break
                print(f"DEBUG: Processing event: {event}")
                try:
                    await self.callback()
                    print("DEBUG: Callback completed successfully")
                except Exception as e:
                    print(f"DEBUG: Callback error: {e}")
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            print("DEBUG: Event processor cancelled")
            raise
    
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
        if self._task and not self._task.done():
            try:
                # Signal shutdown and wait for completion
                future = asyncio.run_coroutine_threadsafe(
                    self._queue.put(None), self._loop
                )
                future.result(timeout=1.0)
                self._task.result(timeout=2.0)
            except Exception as e:
                print(f"DEBUG: Error during cleanup: {e}")
                self._task.cancel()

class CalibreConnector:
    def __init__(self, library_path: Path):
        self.library_path = library_path
        self.db_path = library_path / "metadata.db"
        self._lock = asyncio.Lock()
        self._last_sync = None
    
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
    
    async def watch_library(self) -> Observer:
        observer = Observer()
        event_handler = LibraryWatcher(self._on_library_change)
        observer.schedule(event_handler, str(self.library_path), recursive=False)
        observer.start()
        print("DEBUG: Started watching library")
        return observer
    
    async def tag_book(self, book_id: int, tag: str) -> None:
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

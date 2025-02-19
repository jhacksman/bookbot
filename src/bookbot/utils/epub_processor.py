from typing import Dict, Any, List
import ebooklib
from ebooklib import epub
import html2text
import hashlib
import json
import logging

class EPUBProcessor:
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_images = True
        self.html_converter.ignore_tables = True
    
    def _safe_get_metadata(self, book: epub.EpubBook, metadata_type: str, default: Any = None) -> Any:
        try:
            metadata = book.get_metadata('DC', metadata_type)
            return metadata[0][0] if metadata else default
        except Exception:
            return default

    async def process_file(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise RuntimeError(f"EPUB file not found: {file_path}")

        try:
            book = epub.read_epub(file_path)
            if not book:
                raise RuntimeError("Failed to read EPUB file: empty book")
        
        # Ensure spine and items are properly set
        items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        if not items:
            raise RuntimeError("Invalid EPUB file: no document items found")
            
        # Ensure spine is properly set
        if not book.spine:
            book.spine = []
            for item in items:
                if not hasattr(item, 'id'):
                    item.id = f'item_{len(book.spine)}'
                book.spine.append(item)

        # Extract metadata
        metadata = {
            "title": self._safe_get_metadata(book, 'title', "Unknown"),
            "author": self._safe_get_metadata(book, 'creator', "Unknown"),
            "language": self._safe_get_metadata(book, 'language', "en"),
            "format": "epub",
            "version": getattr(book, 'version', '2.0')
        }

        # Process content from HTML items
        content = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            if hasattr(item, 'content') and item.content:
                try:
                    text = self.html_converter.handle(item.content.decode('utf-8'))
                    content.append(text)
                except Exception:
                    continue

        if not content:
            raise RuntimeError("Invalid EPUB file: no content found")
        
        full_content = "\n\n".join(content)
        content_hash = hashlib.sha256(full_content.encode()).hexdigest()
        
        chunks = self._chunk_content(full_content)
        
        return {
            "metadata": metadata,
            "content": full_content,
            "content_hash": content_hash,
            "chunks": chunks
        }
    
    def _chunk_content(self, content: str) -> List[str]:
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in content.split('\n'):
            line_size = len(line.split())
            if current_size + line_size > self.max_chunk_size:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

from typing import Dict, Any, List
import os
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
            
            # Get all document items
            items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
            if not items:
                raise RuntimeError("Invalid EPUB file: no document items found")
            
            # Ensure each item has an ID
            for item in items:
                if not hasattr(item, 'id'):
                    item.id = f'item_{len(items)}'
                book.add_item(item)

            # Add navigation files if they don't exist
            if not book.get_item_with_id('nav'):
                nav = epub.EpubNav()
                nav.id = 'nav'
                book.add_item(nav)
            if not book.get_item_with_id('ncx'):
                ncx = epub.EpubNcx()
                ncx.id = 'ncx'
                book.add_item(ncx)

            # Add items to spine in correct order
            spine_items = []
            for item in items:
                if not hasattr(item, 'id') or not item.id:
                    item.id = f'item_{len(spine_items)}'
                if hasattr(item, 'file_name'):
                    item.file_name = item.file_name.replace('\\', '/')
                if not book.get_item_with_id(item.id):
                    book.add_item(item)
                spine_items.append(item.id)  # Use just the ID for spine entries
            
            # Set spine with nav first and build TOC
            book.spine = ['nav'] + spine_items
            book.toc = [(epub.Section('Contents'), items)]
            
            # Ensure all items are properly added
            for item in items:
                if not book.get_item_with_id(item.id):
                    book.add_item(item)
            
            # Ensure metadata exists to prevent nsmap issues
            if not hasattr(book, 'metadata') or not book.metadata:
                book.metadata = {'nsmap': {'dc': 'http://purl.org/dc/elements/1.1/'}}
            
            # Add metadata if missing
            if not book.get_metadata('DC', 'title'):
                book.add_metadata('DC', 'title', 'Unknown')
            if not book.get_metadata('DC', 'creator'):
                book.add_metadata('DC', 'creator', 'Unknown')
            if not book.get_metadata('DC', 'language'):
                book.add_metadata('DC', 'language', 'en')
            
            # Ensure all paths in the book use forward slashes
            for item in book.get_items():
                if hasattr(item, 'file_name'):
                    item.file_name = item.file_name.replace('\\', '/')
                if hasattr(item, 'href'):
                    item.href = item.href.replace('\\', '/')
            
            # Add NCX and Nav if missing
            if not book.get_item_with_id('nav'):
                book.add_item(epub.EpubNav())
            if not book.get_item_with_id('ncx'):
                book.add_item(epub.EpubNcx())
        except epub.EpubException as e:
            raise RuntimeError(f"Invalid EPUB file: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to process EPUB file: {str(e)}")

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

from typing import Dict, Any, List
import ebooklib
from ebooklib import epub
import html2text
import hashlib
import json

class EPUBProcessor:
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_images = True
        self.html_converter.ignore_tables = True
    
    async def process_file(self, file_path: str) -> Dict[str, Any]:
        try:
            book = epub.read_epub(file_path, options={'ignore_ncx': True})
            if not book.spine:
                raise RuntimeError("Invalid EPUB file: missing spine")
            
            metadata = {
                "title": book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else "Unknown",
                "author": book.get_metadata('DC', 'creator')[0][0] if book.get_metadata('DC', 'creator') else None,
                "language": book.get_metadata('DC', 'language')[0][0] if book.get_metadata('DC', 'language') else None,
                "format": "epub",
                "version": book.version
            }
            
            content = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                text = self.html_converter.handle(item.content.decode('utf-8'))
                content.append(text)
            
            full_content = "\n\n".join(content)
            content_hash = hashlib.sha256(full_content.encode()).hexdigest()
            
            chunks = self._chunk_content(full_content)
            
            return {
                "metadata": metadata,
                "content": full_content,
                "content_hash": content_hash,
                "chunks": chunks
            }
        except Exception as e:
            raise RuntimeError(f"Failed to process EPUB file: {str(e)}")
    
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

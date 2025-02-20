import pytest
import asyncio
import os
from bookbot.utils.epub_processor import EPUBProcessor

@pytest.fixture
def test_epub_path(tmp_path):
    from ebooklib import epub
    book = epub.EpubBook()
    book.set_identifier('test123')
    book.set_title('Test Book')
    book.set_language('en')
    book.add_author('Test Author')
    
    c1 = epub.EpubHtml(title='Chapter 1', file_name='chap_01.xhtml', lang='en')
    c1.id = 'chapter1'
    c1.content = '<h1>Chapter 1</h1><p>This is a test chapter.</p>'
    book.add_item(c1)
    
    nav = epub.EpubNav()
    nav.id = 'nav'
    book.add_item(nav)
    
    # Create spine and TOC
    book.spine = ['nav', 'chapter1']
    book.toc = [(epub.Section('Test Book'), [c1])]
    
    # Add NCX and Nav files
    book.add_item(epub.EpubNcx())
    book.add_item(nav)
    
    epub_path = tmp_path / "test.epub"
    epub.write_epub(str(epub_path), book, {"spine": ["nav", "chapter1"]})
    return str(epub_path)

@pytest.mark.asyncio
async def test_epub_processor_metadata(test_epub_path):
    processor = EPUBProcessor()
    result = await processor.process_file(test_epub_path)
    
    assert result["metadata"]["title"] == "Test Book"
    assert result["metadata"]["author"] == "Test Author"
    assert result["metadata"]["language"] == "en"
    assert result["metadata"]["format"] == "epub"

@pytest.mark.asyncio
async def test_epub_processor_content(test_epub_path):
    processor = EPUBProcessor()
    result = await processor.process_file(test_epub_path)
    
    assert result["content"]
    assert result["content_hash"]
    assert isinstance(result["chunks"], list)
    assert len(result["chunks"]) > 0

@pytest.mark.asyncio
async def test_epub_processor_invalid_file(tmp_path):
    invalid_path = tmp_path / "invalid.epub"
    with open(invalid_path, 'w') as f:
        f.write("Not an EPUB file")
    
    processor = EPUBProcessor()
    with pytest.raises(RuntimeError):
        await processor.process_file(str(invalid_path))

@pytest.mark.asyncio
async def test_epub_processor_chunking(test_epub_path):
    processor = EPUBProcessor(max_chunk_size=5)
    result = await processor.process_file(test_epub_path)
    
    assert len(result["chunks"]) > 1
    for chunk in result["chunks"]:
        words = chunk.split()
        assert len(words) <= 5

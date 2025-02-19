import pytest
import asyncio
import os
import sys
import shutil
from bookbot.utils.vector_store import VectorStore

if sys.platform.startswith('linux'):
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

@pytest.fixture
def test_persist_dir():
    persist_dir = "test_chroma"
    yield persist_dir
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

@pytest.mark.asyncio
async def test_vector_store_initialization(test_persist_dir):
    store = VectorStore("test_collection", persist_dir=test_persist_dir)
    try:
        assert store.collection.name == "test_collection"
        assert store.max_batch_size > 0
    finally:
        await store.cleanup()

@pytest.mark.asyncio
async def test_add_texts(test_persist_dir):
    store = VectorStore("test_collection", persist_dir=test_persist_dir)
    try:
        texts = ["This is a test document", "This is another test document"]
        metadata = [{"source": "test1"}, {"source": "test2"}]
        ids = ["1", "2"]
        
        result_ids = await store.add_texts(texts, metadata, ids)
        assert result_ids == ids
        
        # Test persistence by creating a new instance
        await store.cleanup()
        del store
        
        new_store = VectorStore("test_collection", persist_dir=test_persist_dir)
        try:
            results = await new_store.similarity_search("test document", k=2)
            assert len(results) == 2
        finally:
            await new_store.cleanup()
    finally:
        if 'store' in locals():
            await store.cleanup()

@pytest.mark.asyncio
async def test_similarity_search(test_persist_dir):
    store = VectorStore("test_collection", persist_dir=test_persist_dir)
    try:
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "A quick brown dog jumps over the lazy fox"
        ]
        await store.add_texts(texts)
        
        results = await store.similarity_search("quick fox", k=2)
        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)
        assert all("content" in r for r in results)
        assert all("metadata" in r for r in results)
        assert all("distance" in r for r in results)
    finally:
        await store.cleanup()

@pytest.mark.asyncio
async def test_batching(test_persist_dir):
    store = VectorStore("test_batch", persist_dir=test_persist_dir, max_batch_size=2)
    try:
        texts = ["doc1", "doc2", "doc3"]
        ids = await store.add_texts(texts)
        assert len(ids) == 3
        
        results = await store.similarity_search("doc", k=3)
        assert len(results) == 3
    finally:
        await store.cleanup()

@pytest.mark.asyncio
async def test_empty_operations(test_persist_dir):
    store = VectorStore("test_empty", persist_dir=test_persist_dir)
    try:
        # Test empty add
        ids = await store.add_texts([])
        assert len(ids) == 0
        
        # Test empty search
        results = await store.similarity_search("nonexistent", k=1)
        assert len(results) == 0
    finally:
        await store.cleanup()

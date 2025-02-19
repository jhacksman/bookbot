import pytest
from bookbot.utils.vector_store import VectorStore

@pytest.mark.asyncio
async def test_vector_store_initialization():
    store = VectorStore("test_collection")
    assert store.collection.name == "test_collection"

@pytest.mark.asyncio
async def test_add_texts():
    store = VectorStore("test_collection")
    texts = ["This is a test document", "This is another test document"]
    metadata = [{"source": "test1"}, {"source": "test2"}]
    ids = ["1", "2"]
    
    result_ids = await store.add_texts(texts, metadata, ids)
    assert result_ids == ids

@pytest.mark.asyncio
async def test_similarity_search():
    store = VectorStore("test_collection")
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

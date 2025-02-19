import pytest
from bookbot.agents.selection.agent import SelectionAgent
from bookbot.agents.summarization.agent import SummarizationAgent
from bookbot.agents.librarian.agent import LibrarianAgent
from bookbot.agents.query.agent import QueryAgent

@pytest.mark.asyncio
async def test_selection_agent():
    agent = SelectionAgent()
    await agent.initialize()
    assert agent.is_active
    result = await agent.process({"query": "test"})
    assert "status" in result
    assert "selected_books" in result
    await agent.cleanup()
    assert not agent.is_active

@pytest.mark.asyncio
async def test_summarization_agent():
    agent = SummarizationAgent()
    await agent.initialize()
    assert agent.is_active
    result = await agent.process({"text": "test content"})
    assert "status" in result
    assert "summaries" in result
    await agent.cleanup()
    assert not agent.is_active

@pytest.mark.asyncio
async def test_librarian_agent():
    agent = LibrarianAgent()
    await agent.initialize()
    assert agent.is_active
    result = await agent.process({"action": "add_book", "book": {}})
    assert "status" in result
    assert "library_updates" in result
    await agent.cleanup()
    assert not agent.is_active

@pytest.mark.asyncio
async def test_query_agent():
    agent = QueryAgent()
    await agent.initialize()
    assert agent.is_active
    result = await agent.process({"question": "test question"})
    assert "status" in result
    assert "response" in result
    assert "citations" in result
    await agent.cleanup()
    assert not agent.is_active

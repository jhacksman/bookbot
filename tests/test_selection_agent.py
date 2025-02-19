import pytest
from bookbot.agents.selection.agent import SelectionAgent
from bookbot.utils.venice_client import VeniceConfig

@pytest.mark.asyncio
async def test_selection_agent_initialization():
    config = VeniceConfig(api_key="test_key")
    agent = SelectionAgent(config)
    assert agent.vram_limit == 16.0
    assert not agent.is_active

@pytest.mark.asyncio
async def test_selection_agent_process_empty_input():
    config = VeniceConfig(api_key="test_key")
    agent = SelectionAgent(config)
    await agent.initialize()
    
    result = await agent.process({})
    assert result["status"] == "error"
    assert "message" in result

@pytest.mark.asyncio
async def test_selection_agent_process():
    config = VeniceConfig(api_key="test_key")
    agent = SelectionAgent(config)
    await agent.initialize()
    
    test_books = [{
        "title": "Deep Learning",
        "author": "Ian Goodfellow",
        "description": "Comprehensive guide to deep learning"
    }]
    
    result = await agent.process({"books": test_books})
    assert result["status"] == "success"
    assert "selected_books" in result
    assert "evaluations" in result

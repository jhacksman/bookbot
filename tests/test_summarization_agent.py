import pytest
import asyncio
from bookbot.agents.summarization.agent import SummarizationAgent
from bookbot.utils.venice_client import VeniceConfig

@pytest.mark.asyncio
async def test_summarization_agent_initialization():
    config = VeniceConfig(api_key="test_key")
    agent = SummarizationAgent(config)
    assert agent.vram_limit == 16.0
    assert not agent.is_active

@pytest.mark.asyncio
async def test_summarization_agent_process_empty_input():
    config = VeniceConfig(api_key="test_key")
    agent = SummarizationAgent(config)
    await agent.initialize()
    
    result = await agent.process({})
    assert result["status"] == "error"
    assert "message" in result

@pytest.mark.asyncio
async def test_summarization_agent_process():
    config = VeniceConfig(api_key="test_key")
    agent = SummarizationAgent(config)
    await agent.initialize()
    
    test_content = """
    Deep learning is a subset of machine learning that uses neural networks with multiple layers.
    These networks can automatically learn representations from data without explicit feature engineering.
    The depth allows the model to learn hierarchical representations, with each layer building upon the previous ones.
    """
    
    result = await agent.process({
        "content": test_content,
        "book_id": "test123",
        "title": "Test Book"
    })
    
    assert result["status"] == "success"
    assert "summaries" in result
    assert len(result["summaries"]) == 3  # Default depth
    assert all("level" in s for s in result["summaries"])
    assert all("content" in s for s in result["summaries"])
    assert all("vector" in s for s in result["summaries"])

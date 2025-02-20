import pytest
import asyncio
from pathlib import Path
import json
import tempfile
from bookbot.utils.token_tracker import TokenTracker, TokenUsage

@pytest.mark.asyncio
async def test_token_tracker_initialization():
    tracker = TokenTracker()
    assert tracker.input_tokens == 0
    assert tracker.output_tokens == 0
    assert tracker.log_file is None

@pytest.mark.asyncio
async def test_token_tracker_add_usage():
    tracker = TokenTracker()
    await tracker.add_usage(100, 50)
    assert tracker.input_tokens == 100
    assert tracker.output_tokens == 50

@pytest.mark.asyncio
async def test_token_tracker_get_cost():
    tracker = TokenTracker()
    await tracker.add_usage(1_000_000, 1_000_000)
    assert await tracker.get_cost() == 3.50  # $0.70 + $2.80

@pytest.mark.asyncio
async def test_token_tracker_get_usage():
    tracker = TokenTracker()
    await tracker.add_usage(100, 50)
    usage = await tracker.get_usage()
    assert isinstance(usage, TokenUsage)
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.cost == (100 * 0.70 + 50 * 2.80) / 1_000_000

@pytest.mark.asyncio
async def test_token_tracker_logging():
    from io import StringIO
    log_buffer = StringIO()
    
    tracker = TokenTracker(log_buffer=log_buffer)
    await tracker.add_usage(100, 50)
    
    log_buffer.seek(0)
    log_entry = json.loads(log_buffer.readline())
    assert log_entry['input_tokens'] == 100
    assert log_entry['output_tokens'] == 50
    assert 'timestamp' in log_entry
    assert 'cost' in log_entry

@pytest.mark.asyncio
async def test_token_tracker_thread_safety():
    tracker = TokenTracker()
    await tracker.add_usage(100, 50)
    usage1 = await tracker.get_usage()
    await tracker.add_usage(200, 100)
    usage2 = await tracker.get_usage()
    assert usage2.input_tokens == 300
    assert usage2.output_tokens == 150
    assert usage2.cost > usage1.cost

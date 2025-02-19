import pytest
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
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        log_path = Path(f.name)
        try:
            tracker = TokenTracker(log_file=log_path)
            await tracker.add_usage(100, 50)
            
            f.seek(0)
            log_entry = json.loads(f.readline())
            assert log_entry['input_tokens'] == 100
            assert log_entry['output_tokens'] == 50
            assert 'timestamp' in log_entry
            assert 'cost' in log_entry
        finally:
            log_path.unlink()

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

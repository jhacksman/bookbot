import pytest
import asyncio
from bookbot.utils.resource_manager import VRAMManager

@pytest.mark.asyncio
async def test_vram_allocation(vram_manager):
    async with vram_manager.allocate("test_agent", 32.0):
        assert await vram_manager.get_available_vram() == 32.0
        allocations = await vram_manager.get_allocations()
        assert allocations["test_agent"] == 32.0

@pytest.mark.asyncio
async def test_vram_overflow(vram_manager):
    with pytest.raises(RuntimeError):
        async with vram_manager.allocate("test_agent", 70.0):
            pass

@pytest.mark.asyncio
async def test_multiple_allocations(vram_manager):
    async with vram_manager.allocate("agent1", 20.0):
        async with vram_manager.allocate("agent2", 20.0):
            allocations = await vram_manager.get_allocations()
            assert len(allocations) == 2
            assert allocations["agent1"] == 20.0
            assert allocations["agent2"] == 20.0
            assert await vram_manager.get_available_vram() == 24.0

@pytest.mark.asyncio
async def test_nested_allocations(vram_manager):
    async with vram_manager.allocate("parent", 32.0):
        async with vram_manager.allocate("child1", 16.0):
            async with vram_manager.allocate("child2", 8.0):
                allocations = await vram_manager.get_allocations()
                assert len(allocations) == 3
                assert allocations["parent"] == 32.0
                assert allocations["child1"] == 16.0
                assert allocations["child2"] == 8.0
                assert await vram_manager.get_available_vram() == 8.0

@pytest.mark.asyncio
async def test_concurrent_allocations(vram_manager):
    async def allocate_and_wait(name: str, vram: float, delay: float):
        async with vram_manager.allocate(name, vram):
            await asyncio.sleep(delay)
    
    tasks = [
        allocate_and_wait("agent1", 16.0, 0.1),
        allocate_and_wait("agent2", 16.0, 0.2),
        allocate_and_wait("agent3", 16.0, 0.1)
    ]
    await asyncio.gather(*tasks)
    assert await vram_manager.get_available_vram() == 64.0
    assert len(await vram_manager.get_allocations()) == 0

@pytest.mark.asyncio
async def test_allocation_cleanup(vram_manager):
    try:
        async with vram_manager.allocate("test", 32.0):
            raise RuntimeError("Test error")
    except RuntimeError:
        pass
    
    assert await vram_manager.get_available_vram() == 64.0
    assert len(await vram_manager.get_allocations()) == 0

@pytest.mark.asyncio
async def test_agent_vram_limits():
    vram_manager = VRAMManager(total_vram=64.0)
    selection_alloc = vram_manager.allocate("selection", 16.0)
    summarization_alloc = vram_manager.allocate("summarization", 16.0)
    librarian_alloc = vram_manager.allocate("librarian", 16.0)
    query_alloc = vram_manager.allocate("query", 16.0)
    
    async with selection_alloc as _:
        async with summarization_alloc as _:
            async with librarian_alloc as _:
                async with query_alloc as _:
                    assert await vram_manager.get_available_vram() == 0.0
                    with pytest.raises(RuntimeError):
                        async with vram_manager.allocate("extra", 1.0):
                            pass

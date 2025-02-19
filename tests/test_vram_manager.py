import pytest
from bookbot.utils.resource_manager import VRAMManager

async def test_vram_allocation(vram_manager):
    async with vram_manager.allocate("test_agent", 32.0):
        assert await vram_manager.get_available_vram() == 32.0
        allocations = await vram_manager.get_allocations()
        assert allocations["test_agent"] == 32.0

async def test_vram_overflow(vram_manager):
    with pytest.raises(RuntimeError):
        async with vram_manager.allocate("test_agent", 70.0):
            pass

async def test_multiple_allocations(vram_manager):
    async with vram_manager.allocate("agent1", 20.0):
        async with vram_manager.allocate("agent2", 20.0):
            allocations = await vram_manager.get_allocations()
            assert len(allocations) == 2
            assert allocations["agent1"] == 20.0
            assert allocations["agent2"] == 20.0
            assert await vram_manager.get_available_vram() == 24.0

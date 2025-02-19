from typing import Dict
import asyncio
from contextlib import asynccontextmanager

class VRAMManager:
    def __init__(self, total_vram: float = 64.0):
        self.total_vram = total_vram
        self.allocated_vram = 0.0
        self.lock = asyncio.Lock()
        self.allocations: Dict[str, float] = {}
    
    @asynccontextmanager
    async def allocate(self, agent_id: str, amount: float):
        async with self.lock:
            if self.allocated_vram + amount > self.total_vram:
                raise RuntimeError(f"VRAM allocation exceeded: {amount}GB requested")
            self.allocated_vram += amount
            self.allocations[agent_id] = amount
        try:
            yield
        finally:
            async with self.lock:
                self.allocated_vram -= amount
                del self.allocations[agent_id]
    
    async def get_available_vram(self) -> float:
        async with self.lock:
            return self.total_vram - self.allocated_vram
    
    async def get_allocations(self) -> Dict[str, float]:
        async with self.lock:
            return self.allocations.copy()

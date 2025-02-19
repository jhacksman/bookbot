from abc import ABC, abstractmethod
from typing import Any, Dict

class Agent(ABC):
    def __init__(self, vram_limit: float = 16.0):
        self.vram_limit = vram_limit
        self.is_active = False
    
    @abstractmethod
    async def initialize(self) -> None:
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        pass
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

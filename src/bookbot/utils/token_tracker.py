from dataclasses import dataclass
from typing import Dict, Optional
import json
from pathlib import Path
from time import time
import asyncio

@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cost: float

class TokenTracker:
    def __init__(self, log_file: Optional[Path] = None):
        self.input_tokens = 0
        self.output_tokens = 0
        self.log_file = log_file
        self._lock = asyncio.Lock()
    
    async def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        async with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            if self.log_file:
                await self._log_usage(input_tokens, output_tokens)
    
    def _calculate_cost(self) -> float:
        return (self.input_tokens * 0.70 + self.output_tokens * 2.80) / 1_000_000
    
    async def get_cost(self) -> float:
        async with self._lock:
            return self._calculate_cost()
    
    async def get_usage(self) -> TokenUsage:
        async with self._lock:
            cost = self._calculate_cost()
            return TokenUsage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                cost=cost
            )
    
    async def _log_usage(self, input_tokens: int, output_tokens: int) -> None:
        if self.log_file is None:
            return
        async with self._lock:
            with open(str(self.log_file), 'a') as f:
                json.dump({
                    'timestamp': time(),
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'cost': (input_tokens * 0.70 + output_tokens * 2.80) / 1_000_000
                }, f)
                f.write('\n')
                f.flush()  # Ensure writes are flushed to disk

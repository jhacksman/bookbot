from dataclasses import dataclass
from typing import Dict, Optional, TextIO
import json
from pathlib import Path
from time import time
import asyncio
from io import StringIO

@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cost: float

class TokenTracker:
    def __init__(self, log_file: Optional[Path] = None, log_buffer: Optional[TextIO] = None):
        self.input_tokens = 0
        self.output_tokens = 0
        self.log_file = log_file
        self.log_buffer = log_buffer
        self._lock = asyncio.Lock()
        self._error = None
    
    async def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        if self._error:
            raise RuntimeError(f"TokenTracker encountered an error: {self._error}")
        async with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            await self._log_usage(input_tokens, output_tokens)
    
    def _calculate_cost(self) -> float:
        return (self.input_tokens * 0.70 + self.output_tokens * 2.80) / 1_000_000
    
    async def get_cost(self) -> float:
        if self._error:
            raise RuntimeError(f"TokenTracker encountered an error: {self._error}")
        async with self._lock:
            return self._calculate_cost()
    
    async def get_usage(self) -> TokenUsage:
        if self._error:
            raise RuntimeError(f"TokenTracker encountered an error: {self._error}")
        async with self._lock:
            cost = self._calculate_cost()
            return TokenUsage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                cost=cost
            )
    
    async def _log_usage(self, input_tokens: int, output_tokens: int) -> None:
        if not (self.log_file or self.log_buffer):
            return
            
        log_entry = {
            'timestamp': time(),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': (input_tokens * 0.70 + output_tokens * 2.80) / 1_000_000
        }
        
        try:
            if self.log_buffer:
                json.dump(log_entry, self.log_buffer)
                self.log_buffer.write('\n')
                self.log_buffer.flush()
            if self.log_file:
                with open(str(self.log_file), 'a') as f:
                    json.dump(log_entry, f)
                    f.write('\n')
                    f.flush()
        except Exception as e:
            self._error = e
            raise
    
    async def _flush_logs(self) -> None:
        try:
            if self.log_buffer:
                self.log_buffer.flush()
            if self.log_file and self.log_file.exists():
                with open(str(self.log_file), 'a') as f:
                    f.flush()
        except Exception as e:
            self._error = e
            raise

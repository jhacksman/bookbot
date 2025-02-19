from dataclasses import dataclass
from typing import Dict, Optional
import json
from pathlib import Path
from time import time
from threading import Lock

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
        self._lock = Lock()
    
    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            if self.log_file:
                self._log_usage(input_tokens, output_tokens)
    
    def get_cost(self) -> float:
        with self._lock:
            return (self.input_tokens * 0.70 + self.output_tokens * 2.80) / 1_000_000
    
    def get_usage(self) -> TokenUsage:
        with self._lock:
            return TokenUsage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                cost=self.get_cost()
            )
    
    def _log_usage(self, input_tokens: int, output_tokens: int) -> None:
        if self.log_file is None:
            return
        with open(str(self.log_file), 'a') as f:
            json.dump({
                'timestamp': time(),
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cost': (input_tokens * 0.70 + output_tokens * 2.80) / 1_000_000
            }, f)
            f.write('\n')

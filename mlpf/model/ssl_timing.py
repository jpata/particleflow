"""Sample CPU wall and CUDA-stream phase timings without intermediate barriers."""

from contextlib import contextmanager
from time import perf_counter

import torch


class StepTimings:
    def __init__(self, device, enabled=True):
        self.device, self.enabled = device, enabled
        self.wall, self.events = {}, {}

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        start = perf_counter()
        if self.device.type == "cuda":
            first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            first.record()
        try:
            yield
        finally:
            self.wall[name] = self.wall.get(name, 0) + perf_counter() - start
            if self.device.type == "cuda":
                last.record()
                self.events.setdefault(name, []).append((first, last))

    def results(self):
        result = {"cpu_wall_seconds": self.wall}
        if self.events:
            # One barrier after the complete step, never between phases.
            torch.cuda.synchronize(self.device)
            result["cuda_stream_seconds"] = {name: sum(a.elapsed_time(b) for a, b in pairs) / 1000 for name, pairs in self.events.items()}
        return result

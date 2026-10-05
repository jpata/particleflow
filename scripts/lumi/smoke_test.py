"""Check GPU execution and an RCCL all-reduce under torchrun."""

import datetime
import os

from comet_ml import OfflineExperiment  # noqa: F401
import torch
import torch.distributed as dist

import mlpf.pipeline  # noqa: F401
from flash_attn import flash_attn_varlen_func

rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert torch.version.hip, "Expected ROCm PyTorch"
assert torch.cuda.device_count() >= world_size
print(f"rank={rank} torch={torch.__version__} hip={torch.version.hip} devices={torch.cuda.device_count()}", flush=True)
torch.cuda.set_device(rank)
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
value = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
dist.all_reduce(value)
assert value.item() == world_size * (world_size + 1) / 2
matrix = torch.randn(128, 128, device=f"cuda:{rank}", dtype=torch.bfloat16)
assert torch.isfinite(matrix @ matrix).all()
print(f"rank={rank} RCCL all-reduce and bfloat16 matmul passed", flush=True)
q = torch.randn(32, 4, 32, device=f"cuda:{rank}", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
lengths = torch.tensor([0, 16, 32], device=f"cuda:{rank}", dtype=torch.int32)
attention = flash_attn_varlen_func(q, k, v, lengths, lengths, 16, 16)
attention.float().square().mean().backward()
assert torch.isfinite(attention).all() and torch.isfinite(q.grad).all()
print(f"rank={rank} pipeline import and Flash Attention forward/backward passed", flush=True)
dist.destroy_process_group()

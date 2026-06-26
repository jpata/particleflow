import torch
import time
from mlpf.conf import MLPFConfig
from mlpf.model.mlpf import MLPF


def main():
    B, S, D = 1, 10000, 41
    X = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(B, S, device="cuda", dtype=torch.bool)

    conv_types = ["attention", "gnnlsh", "hept", "heptv2"]

    def get_loss(obj):
        if isinstance(obj, dict):
            return sum(get_loss(v) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            return sum(get_loss(v) for v in obj)
        elif isinstance(obj, torch.Tensor) and obj.is_floating_point():
            return obj.sum()
        return 0.0

    for conv in conv_types:
        config_dict = {
            "dataset": "cms",
            "data_dir": "/tmp",
            "conv_type": conv,
            "model": {
                "type": conv,
                "hept": {
                    "block_size": 100,
                },
                "heptv2": {
                    "block_size": 100,
                },
            },
        }

        try:
            config = MLPFConfig(**config_dict)
            # Need to manually set elemtypes_nonzero and num_classes for cms
            config.input_dim = 41
            config.num_classes = 8
            config.elemtypes_nonzero = [1, 2, 3, 4, 5, 6, 7]

            mdl = MLPF(config).cuda().bfloat16()
            mdl = torch.compile(mdl)
            mdl.train()
        except Exception as e:
            print(f"Failed to instantiate model {conv}: {e}")
            continue

        # Warmup
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for _ in range(10):
                    out = mdl(X, mask)  # noqa: F821
                    loss = get_loss(out)
                    loss.backward()
                    mdl.zero_grad(set_to_none=True)  # noqa: F821
        except Exception as e:
            print(f"Failed warmup for {conv}: {e}")
            del mdl  # noqa: F821
            torch.cuda.empty_cache()
            continue

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        num_iters = 5

        fwd_times = []
        bwd_times = []

        for _ in range(num_iters):
            # Forward pass
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = mdl(X, mask)  # noqa: F821
            torch.cuda.synchronize()
            fwd_times.append(time.time() - t0)

            # Dummy loss
            loss = get_loss(out)

            # Backward pass
            torch.cuda.synchronize()
            t0 = time.time()
            loss.backward()
            torch.cuda.synchronize()
            bwd_times.append(time.time() - t0)

            mdl.zero_grad(set_to_none=True)  # noqa: F821

        avg_fwd_time = sum(fwd_times) / num_iters
        avg_bwd_time = sum(bwd_times) / num_iters
        max_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"{conv:<10} : Fwd {avg_fwd_time:.4f}s | Bwd {avg_bwd_time:.4f}s | Peak Train Mem: {max_mem_gb:.4f} GB")

        del mdl  # noqa: F821
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

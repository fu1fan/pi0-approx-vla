import time

import torch


def synchronize(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def benchmark(fn, repeat=50, warmup=10, device="cpu"):
    for _ in range(warmup):
        fn()
    synchronize(device)

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        synchronize(device)
        times.append((time.perf_counter() - start) * 1000.0)

    t = torch.tensor(times, dtype=torch.float64)
    return {
        "latency_mean_ms": t.mean().item(),
        "latency_std_ms": t.std(unbiased=False).item() if len(times) > 1 else 0.0,
    }

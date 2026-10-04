"""Synthetic E186 GPU/CPU parity and synchronized timing; no competition data."""
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from e186_batch_ensemble import BatchEnsembleNet, individual_mse


def synchronize(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is unavailable; CPU fallback is forbidden")
    devices = [{"index": i, "name": torch.cuda.get_device_name(i),
                "vram_bytes": torch.cuda.get_device_properties(i).total_memory}
               for i in range(torch.cuda.device_count())]
    selected = max(devices, key=lambda x: x["vram_bytes"])
    device = f"cuda:{selected['index']}"
    torch.cuda.set_device(device)
    model = BatchEnsembleNet(272, [20] * 14).eval()
    gpu = copy.deepcopy(model).to(device)
    xn, xc, y = torch.randn(16, 272), torch.randint(20, (16, 14)), torch.randn(16)
    reference, actual = model(xn, xc), gpu(xn.to(device), xc.to(device))
    torch.testing.assert_close(actual.cpu(), reference, rtol=3e-4, atol=3e-5)
    individual_mse(reference, y).backward()
    individual_mse(actual, y.to(device)).backward()
    max_grad_error = 0.0
    for cp, gp in zip(model.parameters(), gpu.parameters()):
        torch.testing.assert_close(gp.grad.cpu(), cp.grad, rtol=1e-3, atol=3e-5)
        max_grad_error = max(max_grad_error, (gp.grad.cpu() - cp.grad).abs().max().item())
    result = {"torch": torch.__version__, "hip": torch.version.hip,
              "devices": devices, "selected_device": device, "cpu_threads": 4,
              "batch_size": 512, "numeric": 272, "category_cardinalities": [20] * 14,
              "max_forward_error": (actual.cpu() - reference).abs().max().item(),
              "max_gradient_error": max_grad_error, "timing": {}}
    for name, target in [("cpu", "cpu"), ("gpu", device)]:
        net = copy.deepcopy(model).to(target).train()
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=1e-5)
        x = torch.randn(512, 272, device=target)
        cat = torch.randint(20, (512, 14), device=target)
        labels = torch.randn(512, device=target)
        times = []
        for step in range(13):
            synchronize(target)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = individual_mse(net(x, cat), labels)
            loss.backward()
            optimizer.step()
            synchronize(target)
            elapsed = time.perf_counter() - started
            if step >= 3:
                times.append(elapsed)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite synthetic loss")
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in net.parameters()):
            raise RuntimeError("Nonfinite or missing gradient")
        result["timing"][name] = {"median_seconds": statistics.median(times),
                                  "mean_seconds": statistics.mean(times), "samples": times}
    result["speedup_cpu4_vs_gpu"] = result["timing"]["cpu"]["median_seconds"] / result["timing"]["gpu"]["median_seconds"]
    result["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    result["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), Path(__file__).resolve().parents[1] / "src/e186_batch_ensemble.py"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""E186: a TabM-style BatchEnsemble version of the E100/E181a backbone.

Independent implementation of the rank-one BatchEnsemble equation, not the official
TabM package. References: https://arxiv.org/abs/2002.06715 and
https://github.com/yandex-research/tabm . E100's widths, activations and shared
category embeddings are retained. No numerical embeddings or extra dependencies.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F

from e186_runtime import (backend_provenance, configure_backend, device_memory,
                          select_device, synchronize)


class BatchEnsembleLinear(nn.Module):
    def __init__(self, n_in: int, n_out: int, k: int, *, first: bool = False,
                 chunks: list[int] | None = None):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.empty(n_out, n_in))
        self.r = nn.Parameter(torch.ones(k, n_in))
        self.s = nn.Parameter(torch.ones(k, n_out))
        self.bias = nn.Parameter(torch.empty(k, n_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad():
            shared_bias = torch.empty(n_out).uniform_(-1 / math.sqrt(n_in), 1 / math.sqrt(n_in))
            self.bias.copy_(shared_bias.expand(k, -1))
            if first:
                chunks = chunks or [1] * n_in
                if sum(chunks) != n_in or min(chunks) < 1:
                    raise ValueError("Invalid feature chunks")
                self.r.copy_(torch.repeat_interleave(torch.randn(k, len(chunks)), torch.tensor(chunks), dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.k, self.weight.shape[1]):
            raise ValueError("Expected (batch, k, n_in)")
        return F.linear(x * self.r, self.weight) * self.s + self.bias


class BatchEnsembleNet(nn.Module):
    def __init__(self, n_num: int, card: list[int], k: int = 16,
                 widths: tuple[int, int, int] = (512, 512, 256)):
        super().__init__()
        self.k = k
        self.emb = nn.ModuleList(nn.Embedding(c, min(16, (c + 1) // 2 + 1)) for c in card)
        chunks = [1] * n_num + [e.embedding_dim for e in self.emb]
        d = sum(chunks)
        blocks = []
        for i, width in enumerate(widths):
            blocks.extend([BatchEnsembleLinear(d, width, k, first=i == 0, chunks=chunks if i == 0 else None), nn.SiLU()])
            if i < len(widths) - 1:
                blocks.append(nn.Dropout(0.1))
            d = width
        self.body = nn.Sequential(*blocks)
        self.head_weight = nn.Parameter(torch.empty(k, d))
        self.head_bias = nn.Parameter(torch.empty(k))
        nn.init.uniform_(self.head_weight, -1 / math.sqrt(d), 1 / math.sqrt(d))
        nn.init.uniform_(self.head_bias, -1 / math.sqrt(d), 1 / math.sqrt(d))

    def forward(self, xn: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
        x = torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.emb)], dim=1)
        x = self.body(x[:, None, :].expand(-1, self.k, -1))
        return (x * self.head_weight).sum(dim=-1) + self.head_bias


def individual_mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 2 or y.ndim != 1 or len(pred) != len(y):
        raise ValueError("Expected predictions (batch, k) and targets (batch,)")
    return (pred - y[:, None]).square().mean()


def smoke(device: str, batch_size: int = 512) -> dict:
    """Synthetic forward/backward timing and CPU/device parity; never reads PRC labels."""
    torch.set_num_threads(2)
    torch.manual_seed(0)
    device = select_device(device)
    configure_backend(device, max_device_gib=8)
    model = BatchEnsembleNet(272, [20] * 14).eval()
    xn, xc = torch.randn(16, 272), torch.randint(20, (16, 14))
    with torch.no_grad():
        reference = model(xn, xc)
        model = model.to(device)
        actual = model(xn.to(device), xc.to(device)).cpu()
    torch.testing.assert_close(actual, reference, rtol=3e-4, atol=3e-5)
    model.train()
    xn = torch.randn(batch_size, 272, device=device)
    xc = torch.randint(20, (batch_size, 14), device=device)
    y = torch.randn(batch_size, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    times = []
    for i in range(8):
        synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = individual_mse(model(xn, xc), y)
        loss.backward()
        for p in model.parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError("Nonfinite/missing synthetic gradient")
        optimizer.step()
        synchronize(device)
        if i >= 3:
            times.append(time.perf_counter() - start)
    return {"device": str(device), "runtime": backend_provenance(device), "batch_size": batch_size, "parameters": sum(p.numel() for p in model.parameters()),
            "max_cpu_device_error": float((actual - reference).abs().max()),
            "seconds_per_step": sum(times) / len(times), "median_seconds_per_step": statistics.median(times), "loss": float(loss.detach().cpu()),
            "device_memory": device_memory(device)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:<index>")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(smoke(args.device, args.batch_size), indent=2))

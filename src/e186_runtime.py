"""Device selection, fixed FP32 execution and resource checks for E186.

ROCm uses PyTorch's CUDA API. Only explicit synthetic checks may use CPU;
the CV entry point requires an accelerator and never falls back to CPU.
"""
from __future__ import annotations

import math
import os
import platform
import resource
import sys

import torch


def select_device(requested: str = "auto", *, require_accelerator: bool = False) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            raise RuntimeError("E186 needs CUDA/ROCm or MPS; CPU fallback is forbidden (synthetic smoke: --device cpu)")
    if requested == "cpu":
        if require_accelerator:
            raise RuntimeError("E186 CV forbids CPU execution and CPU fallback")
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS is unavailable; no backend fallback")
        return torch.device("mps")
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA/ROCm is unavailable; no backend fallback")
        count = torch.cuda.device_count()
        if requested == "cuda":
            # Prefer the dedicated card when ROCm also exposes an integrated GPU.
            index = max(range(count), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
        else:
            try:
                index = int(requested.split(":", 1)[1])
            except ValueError as e:
                raise ValueError("CUDA device must be cuda or cuda:<index>") from e
        if not 0 <= index < count:
            raise ValueError(f"CUDA/ROCm device index {index} is outside available range 0..{count - 1}")
        return torch.device("cuda", index)
    raise ValueError("Device must be auto, cpu, mps, cuda, or cuda:<index>")


def configure_backend(device: torch.device, *, max_device_gib: float) -> None:
    if not math.isfinite(max_device_gib) or max_device_gib <= 0:
        raise ValueError("Device memory cap must be finite and positive")
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    # No AMP, TF32 or reduced-precision matmul on CUDA; this is also the ROCm API.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, max_device_gib * 1024 ** 3 / total), device)
    elif device.type == "mps":
        total = torch.mps.recommended_max_memory()
        torch.mps.set_per_process_memory_fraction(min(1.0, max_device_gib * 1024 ** 3 / total))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def device_memory(device: torch.device) -> dict:
    if device.type == "cuda":
        return {"allocated_bytes": torch.cuda.memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device)}
    if device.type == "mps":
        return {"allocated_bytes": torch.mps.current_allocated_memory(),
                "reserved_bytes": torch.mps.driver_allocated_memory()}
    return {"allocated_bytes": 0, "reserved_bytes": 0}


def memory_guard(device: torch.device, *, max_rss_gib: float, max_device_gib: float, phase: str) -> dict:
    if not all(math.isfinite(v) and v > 0 for v in (max_rss_gib, max_device_gib)):
        raise ValueError("Memory caps must be finite and positive")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gib = rss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)
    if gib > max_rss_gib:
        raise MemoryError(f"E186 {phase}: peak RSS {gib:.2f} GiB exceeds {max_rss_gib:g} GiB limit")
    memory = device_memory(device)
    if max(memory.values()) > max_device_gib * 1024 ** 3:
        raise MemoryError(f"E186 {phase}: device allocation/reservation exceeds {max_device_gib:g} GiB limit: {memory}")
    return {"peak_rss_gib": gib, **memory}


def backend_provenance(device: torch.device) -> dict:
    hardware = {"machine": platform.machine(), "processor": platform.processor(),
                "cpu_count": os.cpu_count(), "system": platform.platform()}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        hardware.update(name=props.name, total_memory_bytes=props.total_memory,
                        capability=list(torch.cuda.get_device_capability(device)),
                        gcn_arch=getattr(props, "gcnArchName", None))
    elif device.type == "mps":
        hardware.update(name="Apple MPS", recommended_memory_bytes=torch.mps.recommended_max_memory())
    backend = "rocm" if device.type == "cuda" and torch.version.hip else device.type
    return {"device": str(device), "backend": backend, "hardware": hardware,
            "torch": str(torch.__version__), "hip": torch.version.hip, "cuda": torch.version.cuda,
            "python": platform.python_version(), "dtype": "float32", "amp": False,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}

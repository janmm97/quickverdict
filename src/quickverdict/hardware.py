"""Find the GPU(s) and pick the biggest model that fits, so a bigger card gets a smarter judge."""

import importlib.util

import torch

GIB = 1024**3

# Best first. Apache-2.0 Qwen2.5 Instruct models only (3B and 72B have other licences).
# GiB = weights in memory: bf16 is 2 bytes/param; nf4 is 0.5625 bytes/param (4 bits + a float32 scale per 64
# weights, measured with bitsandbytes 0.50) and keeps embeddings and the head in bf16.
TIERS = [
    ("Qwen/Qwen2.5-32B-Instruct", False, 61.0),
    ("Qwen/Qwen2.5-32B-Instruct", True, 19.2),
    ("Qwen/Qwen2.5-14B-Instruct", False, 27.4),
    ("Qwen/Qwen2.5-14B-Instruct", True, 9.8),
    ("Qwen/Qwen2.5-7B-Instruct", False, 14.2),
    ("Qwen/Qwen2.5-7B-Instruct", True, 5.5),
]
SMALLEST = "Qwen/Qwen2.5-1.5B-Instruct"  # 2.9 GiB in bf16; also the CPU default
DOWNLOAD_GB = {"Qwen/Qwen2.5-1.5B-Instruct": 3, "Qwen/Qwen2.5-7B-Instruct": 15, "Qwen/Qwen2.5-14B-Instruct": 30,
               "Qwen/Qwen2.5-32B-Instruct": 66}  # first-run download (bf16 safetensors), so it is no surprise


def pick_model(vram_gib: float, can_4bit: bool) -> tuple[str, bool]:
    """(model id, load in 4-bit) for this much GPU memory, leaving room for activations."""
    budget = vram_gib - max(1.0, 0.15 * vram_gib)
    for model, nf4, need in TIERS:
        if need <= budget and (can_4bit or not nf4):
            return model, nf4
    return SMALLEST, False


def detect(device: str | None = None) -> tuple[str, float, str]:
    """(device, GPU memory in GiB, name). Several NVIDIA/AMD GPUs are pooled with device "auto"."""
    if device in (None, "auto") and torch.cuda.is_available():
        n = torch.cuda.device_count()
        props = [torch.cuda.get_device_properties(i) for i in range(n)]
        pool = n > 1 and importlib.util.find_spec("accelerate") is not None
        props = props if pool else props[:1]
        name = " + ".join(p.name for p in props)
        return ("auto" if pool else "cuda"), sum(p.total_memory for p in props) / GIB, name
    if device and device.startswith("cuda") and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(torch.device(device))
        return device, p.total_memory / GIB, p.name
    if device in (None, "auto") and hasattr(torch, "xpu") and torch.xpu.is_available():
        p = torch.xpu.get_device_properties(0)
        return "xpu", p.total_memory / GIB, p.name
    if device in (None, "auto") and torch.backends.mps.is_available():
        return "mps", torch.mps.recommended_max_memory() / GIB, "Apple Silicon"
    if device is None or device == "cpu":
        return "cpu", 0.0, "cpu"
    return device, 0.0, device  # an explicit device we don't size: smallest model


def can_load_4bit(device: str) -> bool:
    """bitsandbytes nf4 needs an NVIDIA CUDA build of torch."""
    return (device in ("cuda", "auto") or device.startswith("cuda:")) and torch.version.hip is None \
        and importlib.util.find_spec("bitsandbytes") is not None


def auto_model(device: str | None = None) -> tuple[str, str, bool, str]:
    """(model id, device, load in 4-bit, one-line explanation) for `--model auto`."""
    dev, gib, name = detect(device)
    if dev == "cpu" or gib == 0:
        return SMALLEST, dev, False, f"auto: {name}, using {SMALLEST} (first run downloads ~{DOWNLOAD_GB[SMALLEST]} GB)"
    model, nf4 = pick_model(gib, can_load_4bit(dev))
    return (model, dev, nf4, f"auto: {name} ({gib:.0f} GiB), using {model}" + (" in 4-bit" if nf4 else "")
            + f" (first run downloads ~{DOWNLOAD_GB[model]} GB)")

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


class NoamScheduler:
    """Transformer learning-rate schedule from Vaswani et al."""

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup: int, factor: float = 1.0) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup
        self.factor = factor
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        lr = self.rate()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def rate(self) -> float:
        step = max(1, self.step_num)
        return self.factor * (self.d_model ** -0.5) * min(step ** -0.5, step * self.warmup ** -1.5)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def safe_exp(value: float) -> float:
    if value > 20:
        return float("inf")
    return math.exp(value)


def cpu_count_for_workers() -> int:
    if os.name == "nt":
        return 0
    return min(4, os.cpu_count() or 0)

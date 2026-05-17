"""Training helpers."""

from __future__ import annotations

import random

import torch


def set_seed(seed: int) -> None:
    """Set Python and PyTorch RNG seeds."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Return number of trainable parameters."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path: str, model: torch.nn.Module, extra: dict) -> None:
    """Save model checkpoint.

    Args:
        path: Output path.
        model: Model to save.
        extra: Extra metadata, such as config and tokenizer vocab.
    """

    torch.save({
        "model_state_dict": model.state_dict(),
        **extra,
    }, path)

    # raise NotImplementedError("TODO: torch.save state_dict and metadata")


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> dict:
    """Load checkpoint dictionary."""

    checkpoint = torch.load(path, map_location)
    return checkpoint

    # raise NotImplementedError("TODO: torch.load checkpoint")


def evaluate(model, dataloader):
    """Evaluate the model"""

    raise NotImplementedError()
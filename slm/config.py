"""Configuration dataclasses for the SLM homework project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


PosEncodingName = Literal["none", "sinusoidal", "learned_absolute", "alibi", "rope"]


@dataclass
class DataConfig:
    """Dataset configuration.

    Args:
        task: Arithmetic task name. Start with "add"; later try "sub" or "mul".
        max_digits: Maximum number of digits per operand.
        fixed_width: If True, pad numbers with leading zeros.
        train_size: Number of generated training samples.
        val_size: Number of generated validation samples.
        max_seq_len: Maximum token length after BOS/EOS.
    """

    task: str = "add"
    max_digits: int = 2
    fixed_width: bool = False
    train_size: int = 20000
    val_size: int = 1000
    max_seq_len: int = 16


@dataclass
class ModelConfig:
    """Transformer model configuration.

    Args:
        vocab_size: Size of tokenizer vocabulary. Fill this after tokenizer creation.
        d_model: Hidden size.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden size.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length supported by positional encodings.
        pos_encoding: One of "none", "sinusoidal", "learned_absolute", "alibi", "rope".
    """

    vocab_size: int | None = None
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 128
    pos_encoding: PosEncodingName = "rope"


@dataclass
class TrainConfig:
    """Training configuration.

    Args:
        batch_size: Number of examples per batch.
        lr: AdamW learning rate.
        steps: Number of optimizer steps.
        eval_every: Validation interval in steps.
        out_dir: Directory for checkpoints.
    """

    batch_size: int = 64
    lr: float = 3e-4
    steps: int = 2_000
    eval_every: int = 200
    out_dir: str = "runs"


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    seed: int = 0
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from YAML.

    Args:
        path: Path to a YAML config file.

    Returns:
        ExperimentConfig with nested dataclasses.

    TODO:
        Implement YAML loading and nested dataclass construction.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    
    return ExperimentConfig(
        seed=raw.get("seed", 0),
        device=raw.get("device", 0),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
    )

    # raise NotImplementedError("TODO: parse YAML into ExperimentConfig")

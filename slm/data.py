"""Arithmetic dataset generation and batching."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer


ArithmeticTask = Literal["add", "sub", "mul"]


def format_problem(
    a: int,
    b: int,
    task: ArithmeticTask,
    fixed_width: bool,
    width: int,
) -> str:
    """Format one arithmetic problem and answer.

    Args:
        a: First operand.
        b: Second operand.
        task: Operation name.
        fixed_width: Whether to zero-pad operands and result.
        width: Operand width when fixed_width is True.

    Returns:
        A string such as "12+7=19" or "012+008=020".

    TODO:
        Implement add/sub/mul formatting. For subtraction, decide whether to
        avoid negative answers or include a '-' sign.
    """

    raise NotImplementedError


@dataclass
class ArithmeticExample:
    """One generated arithmetic example."""

    text: str
    input_ids: list[int]
    target_ids: list[int]


class ArithmeticDataset(Dataset):
    """Generated character-level arithmetic dataset.

    Each item should train next-token prediction:

    ```text
    full ids:   [BOS, "1", "2", "+", "7", "=", "1", "9", EOS]
    input_ids:  full ids[:-1]
    target_ids: full ids[1:]
    ```
    """

    def __init__(
        self,
        tokenizer: CharTokenizer,
        size: int,
        task: ArithmeticTask = "add",
        max_digits: int = 2,
        fixed_width: bool = False,
        max_seq_len: int = 32,
        seed: int = 0,
    ):
        """Create a deterministic generated dataset.

        Args:
            tokenizer: Character tokenizer.
            size: Number of examples.
            task: Arithmetic operation.
            max_digits: Maximum digits per operand.
            fixed_width: Whether to zero-pad operands.
            max_seq_len: Pad/truncate length for model input.
            seed: RNG seed for reproducible samples.
        """

        self.tokenizer = tokenizer
        self.size = size
        self.task = task
        self.max_digits = max_digits
        self.fixed_width = fixed_width
        self.max_seq_len = max_seq_len
        self.rng = random.Random(seed)

        # TODO: generate and store examples in self.examples.
        self.examples: list[ArithmeticExample] = []
        raise NotImplementedError

    def __len__(self) -> int:
        """Return number of examples."""

        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a padded training example.

        Args:
            idx: Example index.

        Returns:
            Dictionary with:
                input_ids: LongTensor [max_seq_len]
                target_ids: LongTensor [max_seq_len]
                loss_mask: FloatTensor [max_seq_len], 1 for real tokens, 0 for PAD.

        TODO:
            Convert stored lists to tensors and pad them to max_seq_len.
        """

        raise NotImplementedError


def make_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate dataset samples into a batch.

    Args:
        samples: List of examples from ArithmeticDataset.

    Returns:
        Batched dict with tensors:
            input_ids: [batch, seq_len]
            target_ids: [batch, seq_len]
            loss_mask: [batch, seq_len]
    """

    raise NotImplementedError

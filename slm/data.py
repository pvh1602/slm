"""Arithmetic dataset generation and batching."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer, SpecialTokens


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


    # raise NotImplementedError
    if task == "add":
        answer = a + b
        text = f"{a}+{b}={answer}"
    elif task == "sub":
        answer = a - b
        text = f"{a}-{b}={answer}"
    elif task == "mul":
        answer = a * b
        text = f"{a}*{b}={answer}"
    else:   
        raise ValueError(f"Invalid task: {task}")   
    
    if fixed_width and task == "add":
        a_str = str(a).zfill(width)
        b_str = str(b).zfill(width)
        answer_str = str(answer).zfill(width + 1)
        text = f"{a_str}+{b_str}={answer_str}"
    return text


@dataclass
class ArithmeticExample:
    """One generated arithmetic example."""

    text: str
    input_ids: list[int]
    target_ids: list[int]
    loss_mask: list[int]


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
        # raise NotImplementedError
        for _ in range(size):
            a = self.rng.randint(0, 10**max_digits -1)
            b = self.rng.randint(0, 10**max_digits -1)
            text = format_problem(a, b, task, fixed_width, max_digits)
            full_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            target_ids = full_ids[1:]
            input_ids = full_ids[:-1]

            # if the input_ids is longer than max_seq_len, skip the example
            if len(input_ids) > max_seq_len:
                continue
            else:
                loss_mask = [1] * len(input_ids) + [0] * (max_seq_len - len(input_ids))
                input_ids = input_ids + [self.tokenizer.stoi[SpecialTokens.pad]] * (max_seq_len - len(input_ids))
                target_ids = target_ids + [self.tokenizer.stoi[SpecialTokens.pad]] * (max_seq_len - len(target_ids))
            self.examples.append(ArithmeticExample(text, input_ids, target_ids, loss_mask))

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

        example = self.examples[idx]
        input_ids = torch.tensor(example.input_ids, dtype=torch.long)
        target_ids = torch.tensor(example.target_ids, dtype=torch.long)
        loss_mask = torch.tensor(example.loss_mask, dtype=torch.float16)
        # text = example.text
        return {
            # "text": text,
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask
        }
        # raise NotImplementedError


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
    input_ids = torch.stack([sample["input_ids"] for sample in samples])
    target_ids = torch.stack([sample["target_ids"] for sample in samples])
    loss_mask = torch.stack([sample["loss_mask"] for sample in samples])
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "loss_mask": loss_mask
    }


if __name__ == "__main__":
    tokenizer = CharTokenizer()
    dataset = ArithmeticDataset(tokenizer, size=10, task="add", max_digits=2, fixed_width=False, max_seq_len=16, seed=0)
    print(dataset[0])
    batch = make_batch(dataset)
    print(batch)
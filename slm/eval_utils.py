"""Evaluation helpers for arithmetic generation."""

from __future__ import annotations


def parse_arithmetic_answer(text: str) -> str | None:
    """Extract answer substring after '='.

    Args:
        text: Decoded generated text.

    Returns:
        Answer string if parseable, otherwise None.

    TODO:
        Implement simple parsing, stopping at EOS or first non-answer char.
    """

    raise NotImplementedError


def exact_match(pred: str, target: str) -> bool:
    """Return whether predicted answer exactly matches target answer."""

    return pred == target

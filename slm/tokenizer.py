"""Character tokenizer for arithmetic language modeling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialTokens:
    """Special tokens used by the character tokenizer."""

    pad: str = "<pad>"
    bos: str = "<bos>"
    eos: str = "<eos>"


class CharTokenizer:
    """Simple character-level tokenizer.

    The tokenizer should support arithmetic strings such as "12+7=19".

    Attributes:
        stoi: Mapping from token string to integer id.
        itos: Reverse mapping from id to token string.
        pad_id: Padding token id.
        bos_id: Beginning-of-sequence token id.
        eos_id: End-of-sequence token id.
    """

    def __init__(self, extra_chars: str = "0123456789+-*= "):
        """Create a tokenizer.

        Args:
            extra_chars: Non-special single-character tokens to include.

        TODO:
            Build stoi/itos. Put special tokens first for stable ids.
        """

        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Return the number of tokens in the vocabulary."""

        raise NotImplementedError

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        """Convert a string into token ids.

        Args:
            text: Input arithmetic string.
            add_bos: Whether to prepend BOS.
            add_eos: Whether to append EOS.

        Returns:
            List of integer token ids.
        """

        raise NotImplementedError

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Convert token ids back into a string.

        Args:
            ids: Token ids.
            skip_special: If True, omit BOS/EOS/PAD from output.

        Returns:
            Decoded string.
        """

        raise NotImplementedError

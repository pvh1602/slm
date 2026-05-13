#!/usr/bin/env python3
"""Generate from a trained tiny transformer."""

from __future__ import annotations

import argparse


def main() -> None:
    """Generation entry point.

    Expected steps:
        1. Load checkpoint/config/tokenizer.
        2. Encode prompt.
        3. Run model.generate(...).
        4. Decode and print result.
        5. Try both --use-cache and --no-use-cache and compare outputs.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()
    raise NotImplementedError("TODO: implement generation")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inspect KV-cache behavior during generation."""

from __future__ import annotations

import argparse


def main() -> None:
    """KV-cache inspection entry point.

    Suggested experiments:
        1. Print cache length after each generated token.
        2. Compare no pruning vs keep-last-N pruning.
        3. Confirm logits match between cached and uncached generation before pruning.
        4. Show when pruning changes predictions.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="12+7=")
    parser.add_argument("--keep-last", type=int, default=0)
    args = parser.parse_args()
    raise NotImplementedError("TODO: implement KV-cache diagnostics")


if __name__ == "__main__":
    main()

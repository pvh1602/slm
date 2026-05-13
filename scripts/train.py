#!/usr/bin/env python3
"""Train the tiny transformer on arithmetic data.

Homework:
    Fill in the TODOs in slm/*.py, then complete this script.
"""

from __future__ import annotations

import argparse


def main() -> None:
    """Train entry point.

    Expected steps:
        1. Load YAML config.
        2. Build tokenizer and set cfg.model.vocab_size.
        3. Build train/val datasets and DataLoaders.
        4. Create TinyTransformerLM.
        5. Run AdamW training loop.
        6. Periodically evaluate validation loss.
        7. Save checkpoint.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise NotImplementedError("TODO: implement training loop")


if __name__ == "__main__":
    main()

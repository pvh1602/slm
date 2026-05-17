#!/usr/bin/env python3
"""Train the tiny transformer on arithmetic data.

Homework:
    Fill in the TODOs in slm/*.py, then complete this script.
"""

from __future__ import annotations

import torch

import argparse

from torch.utils.data import DataLoader
from slm.config import load_config
from slm.slm.model import TinyTransformerLM
from slm.tokenizer import CharTokenizer
from slm.data import ArithmeticDataset, make_batch

from slm.train_utils import set_seed, load_checkpoint, save_checkpoint, evaluate

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
    parser.add_argument("--valid_step", type=int, default=-1)
    args = parser.parse_args()
    # raise NotImplementedError("TODO: implement training loop")

    cfg = load_config(args.config) # Read config path
    seed = cfg.seed
    device = cfg.device

    set_seed(seed)

    tokenizer = CharTokenizer()
    cfg.model.vocab_size = tokenizer.vocab_size


    # Loading datasets
    train_ds = ArithmeticDataset(
        tokenizer,
        size=cfg.data.train_size,
        task=cfg.data.task,
        max_digits=cfg.data.max_digits,
        fixed_width=cfg.data.fixed_width,
        max_seq_len=cfg.data.max_seq_len,
        seed=seed
    )
    test_ds = ArithmeticDataset(
        tokenizer,
        size=cfg.data.test_size,
        task=cfg.data.task,
        max_digits=cfg.data.max_digits,
        fixed_width=cfg.data.fixed_width,
        max_seq_len=cfg.data.max_seq_len,
        seed=seed
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.train.batch_size,
        collate_fn=make_batch
    )

    test_loader = DataLoader(
        test_ds, 
        batch_size=cfg.train.batch_size,
        collate_fn=make_batch
    )
    
    model = TinyTransformerLM(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    
    for step in range(cfg.train.steps):
        batch = next(iter(train_loader))
        out = model(
            batch["input_ids"],
            targets=batch["target_ids"],
            loss_mask=batch["loss_mask"]
        )
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        # Validate
        if args.valid_step != -1 and step % args.valid_step == 0:
            model.eval()
            with torch.no_grad():
                metrics = evaluate(model, test_ds)
            model.train()



if __name__ == "__main__":
    main()

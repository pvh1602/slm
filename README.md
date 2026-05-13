# SLM Homework: Tiny Transformer + KV Cache

This folder is a learning scaffold for implementing a small language model from
scratch on arithmetic sequence tasks.

The goal is not to hide the important work behind a framework. The files define
classes, function signatures, docstrings, expected tensor shapes, and TODOs.
You fill in the implementations.

## Learning Goals

1. Build a character-level arithmetic dataset.
2. Implement tokenization and batching.
3. Implement transformer attention, MLP, residual blocks, and causal masking.
4. Compare positional encodings:
   - sinusoidal,
   - learned absolute,
   - ALiBi,
   - RoPE.
5. Implement autoregressive generation with and without KV cache.
6. Experiment with simple KV-cache optimizations.

## Suggested Order

1. `slm/tokenizer.py`
2. `slm/data.py`
3. `slm/position.py`
4. `slm/attention.py`
5. `slm/model.py`
6. `scripts/train.py`
7. `scripts/generate.py`
8. `slm/kv_cache.py`
9. `scripts/inspect_kv_cache.py`

## Arithmetic Task Examples

The dataset should produce strings like:

```text
12+7=19
034+008=042
9*8=72
```

For language modeling, each full string is tokenized and the model is trained to
predict the next character.

## Expected Usage After You Implement TODOs

```bash
cd /dcs/pg24/u5627327/Code/kvcache/slm
python scripts/train.py --config configs/tiny_rope.yaml
python scripts/generate.py --checkpoint runs/tiny_rope.pt --prompt "12+7="
python scripts/inspect_kv_cache.py --checkpoint runs/tiny_rope.pt
```

## Project Map

```text
slm/
  README.md
  pyproject.toml
  configs/
    tiny_sinusoidal.yaml
    tiny_rope.yaml
  slm/
    __init__.py
    config.py
    tokenizer.py
    data.py
    position.py
    attention.py
    kv_cache.py
    model.py
    train_utils.py
    eval_utils.py
  scripts/
    train.py
    generate.py
    inspect_kv_cache.py
  tests/
    test_shapes.py
```

## Homework Milestones

### Milestone 1: Tokenizer

Implement a character tokenizer with:

- digits,
- operators,
- equals sign,
- padding token,
- BOS/EOS tokens.

### Milestone 2: Dataset

Generate arithmetic examples and return:

```python
input_ids:  LongTensor [seq_len]
target_ids: LongTensor [seq_len]
```

where `target_ids[t]` is the next token after `input_ids[t]`.

### Milestone 3: Transformer Forward

Implement:

- token embeddings,
- positional encoding,
- causal self-attention,
- MLP,
- final logits.

### Milestone 4: KV Cache

Implement incremental decoding:

```text
prefill prompt once
then generate one token at a time using cached K/V
```

Compare speed and outputs with non-cache generation.

### Milestone 5: KV Cache Optimization

Try small policies:

- keep all tokens,
- keep last N tokens,
- keep BOS + last N tokens,
- keep every k-th old token + last N tokens.

This prepares you to understand more advanced methods such as TriAttention.

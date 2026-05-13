"""Shape tests for your implementation.

Run with:

```bash
cd /dcs/pg24/u5627327/Code/kvcache/slm
pytest
```
"""

import pytest
import torch


def test_placeholder():
    """Replace this with real shape tests as you implement modules."""

    assert torch.tensor([1]).item() == 1


@pytest.mark.skip(reason="Enable after implementing tokenizer and model")
def test_model_forward_shapes():
    """Expected shape test.

    TODO:
        Build tokenizer, config, model, fake input batch.
        Assert logits shape is [batch, seq_len, vocab_size].
    """

    raise NotImplementedError

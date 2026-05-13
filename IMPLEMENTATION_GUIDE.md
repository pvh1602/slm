# SLM Implementation Guide

This guide walks through the TODOs in the `slm/` homework project. Follow it in
order if you want to build the tiny transformer from first principles.

The project goal is to train a decoder-only transformer on character-level
arithmetic examples such as:

```text
12+7=19
034+008=042
9*8=72
```

The model learns next-token prediction:

```text
given:  <bos> 1 2 + 7 =
predict:      1 2 + 7 = 1
```

More formally, for token ids:

```text
x = [x_0, x_1, ..., x_{T-1}]
y = [x_1, x_2, ..., x_T]
```

the model is trained to predict `y_t` from the prefix `x_{\le t}`.

## 1. Recommended Implementation Order

Implement files in this order:

```text
1. slm/tokenizer.py
2. slm/data.py
3. slm/position.py
4. slm/kv_cache.py
5. slm/attention.py
6. slm/model.py
7. slm/config.py
8. slm/train_utils.py
9. scripts/train.py
10. scripts/generate.py
11. scripts/inspect_kv_cache.py
12. tests/test_shapes.py
```

You can run syntax checks at any time:

```bash
cd /dcs/pg24/u5627327/Code/kvcache
python -m py_compile slm/slm/*.py slm/scripts/*.py
```

After implementing tests:

```bash
cd /dcs/pg24/u5627327/Code/kvcache/slm
pytest
```

## 2. Tokenizer

File:

```text
slm/tokenizer.py
```

Implement `CharTokenizer`.

### 2.1 Vocabulary

Use special tokens first:

```text
0: <pad>
1: <bos>
2: <eos>
```

Then add characters:

```text
0 1 2 3 4 5 6 7 8 9 + - * = space
```

Make sure each character appears only once. A good pattern:

```python
tokens = [pad, bos, eos]
for ch in extra_chars:
    if ch not in tokens:
        tokens.append(ch)
```

Then:

```python
self.stoi = {tok: i for i, tok in enumerate(tokens)}
self.itos = {i: tok for tok, i in self.stoi.items()}
```

### 2.2 Encoding

For text:

```text
12+7=19
```

with BOS/EOS:

```text
[<bos>, 1, 2, +, 7, =, 1, 9, <eos>]
```

Return integer ids.

### 2.3 Decoding

Convert ids back to strings. If `skip_special=True`, omit:

```text
<pad>, <bos>, <eos>
```

This is useful for reading generations.

## 3. Arithmetic Dataset

File:

```text
slm/data.py
```

Implement:

```python
format_problem(...)
ArithmeticDataset
make_batch(...)
```

### 3.1 Formatting Arithmetic Problems

For addition:

```python
answer = a + b
text = f"{a}+{b}={answer}"
```

For multiplication:

```python
answer = a * b
text = f"{a}*{b}={answer}"
```

For subtraction, start simple by avoiding negative answers:

```python
if b > a:
    a, b = b, a
answer = a - b
text = f"{a}-{b}={answer}"
```

If `fixed_width=True`, zero-pad operands and answer. For width `w`:

```python
a_str = str(a).zfill(w)
b_str = str(b).zfill(w)
```

For addition, the answer may need `w + 1` digits:

```python
ans_width = w + 1
answer_str = str(answer).zfill(ans_width)
```

Example:

```text
w = 3
a = 34
b = 8
text = "034+008=0042"
```

Fixed width makes the task easier because sequence lengths and digit positions
are more regular.

### 3.2 Dataset Generation

For `max_digits = d`, sample:

```python
max_value = 10**d - 1
a = rng.randint(0, max_value)
b = rng.randint(0, max_value)
```

Then format, tokenize, and create:

```python
full_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
input_ids = full_ids[:-1]
target_ids = full_ids[1:]
```

Skip or regenerate examples longer than `max_seq_len`.

### 3.3 Padding

Each dataset item should return tensors with fixed length `max_seq_len`.

If the unpadded length is `L`, then:

```text
input_ids[:L] = real input ids
input_ids[L:] = pad_id

target_ids[:L] = real target ids
target_ids[L:] = pad_id

loss_mask[:L] = 1
loss_mask[L:] = 0
```

Shape:

```text
input_ids:  [max_seq_len]
target_ids: [max_seq_len]
loss_mask:  [max_seq_len]
```

### 3.4 Collation

`make_batch(samples)` should stack each key:

```python
batch = {
    "input_ids": torch.stack([s["input_ids"] for s in samples]),
    "target_ids": torch.stack([s["target_ids"] for s in samples]),
    "loss_mask": torch.stack([s["loss_mask"] for s in samples]),
}
```

Final shapes:

```text
input_ids:  [batch, seq_len]
target_ids: [batch, seq_len]
loss_mask:  [batch, seq_len]
```

## 4. Positional Encoding

File:

```text
slm/position.py
```

You will implement four options:

```text
sinusoidal
learned_absolute
alibi
rope
```

### 4.1 Sinusoidal Positional Encoding

For position `p` and channel index `i`, define:

```text
PE[p, 2i]   = sin(p / 10000^{2i / d_model})
PE[p, 2i+1] = cos(p / 10000^{2i / d_model})
```

In code:

```python
position = torch.arange(max_seq_len).unsqueeze(1)
div_term = torch.exp(
    torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
)
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

Store:

```python
self.register_buffer("pe", pe.unsqueeze(0))
```

Forward:

```python
x = x + self.pe[:, start_pos:start_pos + seq_len]
```

### 4.2 Learned Absolute Positional Encoding

Create:

```python
self.pos_emb = nn.Embedding(max_seq_len, d_model)
```

Forward:

```python
positions = torch.arange(start_pos, start_pos + seq_len, device=x.device)
x = x + self.pos_emb(positions).unsqueeze(0)
```

### 4.3 ALiBi

ALiBi does not add position vectors to embeddings. It adds a head-specific bias
to attention logits.

Attention logits before softmax:

```text
S = Q K^T / sqrt(d_head)
```

ALiBi modifies this:

```text
S'_{h,i,j} = S_{h,i,j} - m_h * distance(i, j)
```

where:

```text
h = attention head
i = query position
j = key position
m_h = positive slope for head h
```

For causal attention, a simple distance is:

```text
distance(i, j) = query_absolute_position_i - key_absolute_position_j
```

For a full training sequence with no cache:

```text
query position i = i
key position j   = j
distance = i - j
```

Allowed attention has `j <= i`, so distance is non-negative.

Return a bias tensor:

```text
[1, n_heads, q_len, k_len]
```

with values:

```text
bias[h, i, j] = -slope[h] * distance(i, j)
```

A simple slope choice for homework:

```python
slopes = torch.tensor([2 ** (-8 * h / n_heads) for h in range(n_heads)])
```

This is not the exact ALiBi paper slope schedule, but it is enough for learning.
You can later replace it with the official schedule.

### 4.4 RoPE

RoPE rotates query and key vectors in each head.

For each frequency pair, treat two dimensions as a 2D vector:

```text
x_pair = [x_even, x_odd]
```

At position `p`, rotate by angle:

```text
theta_{p,f} = p * omega_f
```

where:

```text
omega_f = 1 / base^{2f / d_head}
```

The rotation is:

```text
x'_even = x_even * cos(theta) - x_odd * sin(theta)
x'_odd  = x_even * sin(theta) + x_odd * cos(theta)
```

Implement:

```python
def precompute_rope_frequencies(head_dim, max_seq_len, theta=10000):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, inv_freq)
    return torch.cos(angles), torch.sin(angles)
```

For `apply_rope(x, cos, sin, start_pos)`:

```text
x shape: [batch, n_heads, seq_len, head_dim]
cos/sin: [max_seq_len, head_dim // 2]
```

Use positions:

```python
cos_t = cos[start_pos:start_pos + seq_len]
sin_t = sin[start_pos:start_pos + seq_len]
```

Broadcast to:

```text
[1, 1, seq_len, head_dim // 2]
```

Then rotate even/odd pairs.

## 5. KV Cache

File:

```text
slm/kv_cache.py
```

The cache stores K and V tensors for each layer.

For one layer:

```text
key:       [batch, n_heads, cache_len, head_dim]
value:     [batch, n_heads, cache_len, head_dim]
positions: [cache_len]
```

### 5.1 Appending

When decoding, each new token produces:

```text
new_key:   [batch, n_heads, q_len, head_dim]
new_value: [batch, n_heads, q_len, head_dim]
```

Usually `q_len = 1` during generation.

Append on cache dimension:

```python
all_key = torch.cat([old_key, new_key], dim=2)
all_value = torch.cat([old_value, new_value], dim=2)
all_positions = torch.cat([old_positions, new_positions], dim=0)
```

If the cache is empty, just set it to the new tensors.

### 5.2 Keep-Last Pruning

Keep only the last `N` cache entries:

```python
layer_cache.key = layer_cache.key[:, :, -N:, :]
layer_cache.value = layer_cache.value[:, :, -N:, :]
layer_cache.positions = layer_cache.positions[-N:]
```

### 5.3 BOS Plus Last-N Pruning

Keep slot `0` and the most recent `N` slots:

```python
keep = sorted(set([0] + list(range(cache_len - N, cache_len))))
```

Then gather:

```python
idx = torch.tensor(keep, device=key.device)
key = key.index_select(2, idx)
value = value.index_select(2, idx)
positions = positions.index_select(0, idx)
```

### 5.4 Strided Plus Last-N Pruning

Keep every `stride`-th old token plus recent tokens:

```python
old = list(range(0, max(0, cache_len - keep_last), stride))
recent = list(range(max(0, cache_len - keep_last), cache_len))
keep = sorted(set(old + recent))
```

This gives a crude sparse long-range memory.

## 6. Causal Self-Attention

File:

```text
slm/attention.py
```

Input:

```text
x: [batch, q_len, d_model]
```

Projection:

```text
Q = x W_Q
K = x W_K
V = x W_V
```

Shapes after splitting heads:

```text
Q: [batch, n_heads, q_len, head_dim]
K: [batch, n_heads, q_len, head_dim]
V: [batch, n_heads, q_len, head_dim]
```

where:

```text
head_dim = d_model / n_heads
```

### 6.1 RoPE In Attention

If `pos_encoding == "rope"`:

```python
Q = apply_rope(Q, cos, sin, start_pos)
K = apply_rope(K, cos, sin, start_pos)
```

Do this before appending K to cache.

### 6.2 KV Cache In Attention

If `use_cache=True`, append new K/V to the layer cache:

```python
positions = torch.arange(start_pos, start_pos + q_len, device=x.device)
K_all, V_all = append_to_cache(layer_cache, K, V, positions)
```

Then attention uses:

```text
Q current tokens
K all cached tokens
V all cached tokens
```

So:

```text
Q:     [batch, n_heads, q_len, head_dim]
K_all: [batch, n_heads, k_len, head_dim]
V_all: [batch, n_heads, k_len, head_dim]
```

### 6.3 Attention Formula

Scaled dot-product attention:

```text
S = Q K^T / sqrt(head_dim)
```

Shape:

```text
S: [batch, n_heads, q_len, k_len]
```

Add causal mask:

```text
S = S + mask
```

Add ALiBi bias if enabled:

```text
S = S + alibi_bias
```

Then:

```text
A = softmax(S, dim=-1)
O = A V
```

Shapes:

```text
A: [batch, n_heads, q_len, k_len]
O: [batch, n_heads, q_len, head_dim]
```

Merge heads:

```text
O -> [batch, q_len, d_model]
```

Then output projection:

```text
y = O W_O
```

### 6.4 Causal Mask

For full training, `q_len = k_len = T`.

Allow token `i` to attend only to keys `j <= i`.

Mask:

```text
mask[i, j] = 0      if j <= i
mask[i, j] = -inf   if j > i
```

In PyTorch, use a large negative value:

```python
neg_inf = torch.finfo(dtype).min
```

For decoding with cache:

```text
past_len = k_len - q_len
```

Query row `i` in the current chunk corresponds to absolute local index:

```text
past_len + i
```

Allow:

```text
j <= past_len + i
```

This handles both:

```text
q_len = 1 incremental decoding
q_len > 1 prefill/chunk decoding
```

## 7. Transformer Block

File:

```text
slm/model.py
```

Use a pre-layernorm block:

```text
x = x + Attention(LN1(x))
x = x + MLP(LN2(x))
```

Feed-forward network:

```text
MLP(x) = W_2 GELU(W_1 x)
```

Shapes:

```text
x:      [batch, seq_len, d_model]
W_1 x:  [batch, seq_len, d_ff]
MLP(x): [batch, seq_len, d_model]
```

## 8. TinyTransformerLM

File:

```text
slm/model.py
```

### 8.1 Model Components

Implement:

```python
self.token_emb = nn.Embedding(vocab_size, d_model)
self.blocks = nn.ModuleList([...])
self.ln_f = nn.LayerNorm(d_model)
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
```

For sinusoidal or learned absolute positional encoding, create:

```python
self.pos_emb = SinusoidalPositionalEncoding(...)
```

or:

```python
self.pos_emb = LearnedAbsolutePositionalEncoding(...)
```

For RoPE and ALiBi, do not add embedding-level position vectors:

```text
RoPE is applied inside attention to Q/K.
ALiBi is added inside attention to logits.
```

### 8.2 Forward Pass

```python
x = self.token_emb(input_ids)
```

If embedding-level position encoding is active:

```python
x = self.pos_emb(x, start_pos=start_pos)
```

Then:

```python
for layer_idx, block in enumerate(self.blocks):
    layer_cache = kv_cache.layers[layer_idx] if kv_cache is not None else None
    x = block(x, layer_cache=layer_cache, start_pos=start_pos, use_cache=use_cache)
```

Final logits:

```python
x = self.ln_f(x)
logits = self.lm_head(x)
```

Shape:

```text
logits: [batch, seq_len, vocab_size]
```

### 8.3 Language Modeling Loss

Cross entropy per token:

```text
loss_{b,t} = -log softmax(logits_{b,t})[target_{b,t}]
```

In PyTorch:

```python
loss_flat = F.cross_entropy(
    logits.view(batch * seq_len, vocab_size),
    targets.view(batch * seq_len),
    reduction="none",
)
loss = loss_flat.view(batch, seq_len)
```

Apply mask:

```python
loss = (loss * loss_mask).sum() / loss_mask.sum().clamp_min(1)
```

This avoids training on PAD positions.

## 9. Generation

File:

```text
slm/model.py
```

Implement:

```python
TinyTransformerLM.generate(...)
```

### 9.1 No-Cache Generation

Simplest path:

```python
ids = input_ids
for _ in range(max_new_tokens):
    out = self(ids, use_cache=False)
    next_logits = out["logits"][:, -1, :]
    next_id = sample(next_logits)
    ids = torch.cat([ids, next_id[:, None]], dim=1)
return ids
```

This recomputes all previous tokens every step.

Cost per step grows with sequence length.

### 9.2 Cache Generation

With KV cache:

```python
cache = KVCache(n_layers)
ids = input_ids

# prefill
out = self(input_ids, kv_cache=cache, start_pos=0, use_cache=True)
next_id = sample(out["logits"][:, -1, :])
ids = torch.cat([ids, next_id[:, None]], dim=1)

# decode one token at a time
for step in range(1, max_new_tokens):
    start_pos = input_ids.shape[1] + step - 1
    out = self(next_id[:, None], kv_cache=cache, start_pos=start_pos, use_cache=True)
    next_id = sample(out["logits"][:, -1, :])
    ids = torch.cat([ids, next_id[:, None]], dim=1)
```

In cached decoding, the model only receives the newest token after prefill, but
attention sees all previous keys/values through the cache.

### 9.3 Sampling

Greedy:

```python
next_id = logits.argmax(dim=-1)
```

Temperature sampling:

```python
probs = torch.softmax(logits / temperature, dim=-1)
next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
```

For arithmetic, greedy decoding is usually easier to debug.

## 10. Config Loading

File:

```text
slm/config.py
```

Implement `load_config(path)`.

YAML loading:

```python
with open(path, "r") as f:
    raw = yaml.safe_load(f)
```

Then:

```python
return ExperimentConfig(
    seed=raw.get("seed", 0),
    device=raw.get("device", "cuda"),
    data=DataConfig(**raw.get("data", {})),
    model=ModelConfig(**raw.get("model", {})),
    train=TrainConfig(**raw.get("train", {})),
)
```

Remember that YAML may contain:

```yaml
vocab_size: null
```

Set `cfg.model.vocab_size = tokenizer.vocab_size` after tokenizer creation.

## 11. Training Script

File:

```text
scripts/train.py
```

Expected steps:

```python
cfg = load_config(args.config)
set_seed(cfg.seed)

tokenizer = CharTokenizer()
cfg.model.vocab_size = tokenizer.vocab_size

train_ds = ArithmeticDataset(...)
val_ds = ArithmeticDataset(...)

train_loader = DataLoader(train_ds, batch_size=..., collate_fn=make_batch)

model = TinyTransformerLM(cfg.model).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

for step in range(cfg.train.steps):
    batch = next_batch(...)
    out = model(
        batch["input_ids"],
        targets=batch["target_ids"],
        loss_mask=batch["loss_mask"],
    )
    loss = out["loss"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

Validation:

```python
model.eval()
with torch.no_grad():
    compute average val loss
model.train()
```

Save checkpoint:

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "config": cfg,
    "tokenizer_stoi": tokenizer.stoi,
}, path)
```

## 12. Cache Inspection Script

File:

```text
scripts/inspect_kv_cache.py
```

This is where you learn KV-cache behavior.

Suggested experiments:

### 12.1 Verify Cache vs No Cache

Before any pruning, cached and non-cached logits should match closely.

Procedure:

```text
1. Run full model on prompt.
2. Run prefill with cache on same prompt.
3. Compare final logits.
```

Expected:

```text
max_abs_diff < 1e-4 or similar
```

Small numerical differences are normal.

### 12.2 Print Cache Lengths

After each generated token:

```python
for layer_idx, layer_cache in enumerate(cache.layers):
    print(layer_idx, layer_cache.key.shape[2])
```

Without pruning, cache length should grow by one each step.

### 12.3 Apply Toy Pruning

Call:

```python
prune_cache_keep_last(layer_cache, keep_last=N)
```

after each decode step or every few steps.

Then observe:

```text
cache length stops growing beyond N
generation quality may degrade
```

This gives intuition for why smarter KV-selection methods are useful.

## 13. Tests To Add

File:

```text
tests/test_shapes.py
```

Add small tests as you implement.

### 13.1 Tokenizer Roundtrip

```python
tok = CharTokenizer()
text = "12+7=19"
ids = tok.encode(text)
assert tok.decode(ids) == text
```

### 13.2 Dataset Shapes

```python
sample = ds[0]
assert sample["input_ids"].shape == (max_seq_len,)
assert sample["target_ids"].shape == (max_seq_len,)
assert sample["loss_mask"].shape == (max_seq_len,)
```

### 13.3 Attention Shapes

```python
attn = CausalSelfAttention(...)
x = torch.randn(batch, seq_len, d_model)
y = attn(x)
assert y.shape == x.shape
```

### 13.4 Model Shapes

```python
out = model(input_ids)
assert out["logits"].shape == (batch, seq_len, vocab_size)
```

### 13.5 Cache Growth

```python
cache = KVCache(n_layers)
model(prompt, kv_cache=cache, use_cache=True)
assert cache.layers[0].key.shape[2] == prompt_len

model(next_token, kv_cache=cache, start_pos=prompt_len, use_cache=True)
assert cache.layers[0].key.shape[2] == prompt_len + 1
```

## 14. Common Bugs

### 14.1 Wrong Attention Mask With Cache

If cached generation differs from full generation before pruning, check your
causal mask.

For query row `i`, allowed keys are:

```text
j <= past_len + i
```

not just:

```text
j <= i
```

### 14.2 RoPE Shape Broadcasting

For RoPE:

```text
x:   [batch, heads, seq_len, head_dim]
cos: [seq_len, head_dim // 2]
```

You likely need:

```python
cos = cos[None, None, :, :]
sin = sin[None, None, :, :]
```

### 14.3 Using Absolute Embeddings During Cached Decode

For learned/sinusoidal absolute embeddings, the first decoded token after a
prompt of length `P` must use:

```text
start_pos = P
```

not `0`.

### 14.4 Forgetting To Mask PAD Loss

If you train on PAD targets, the model may learn to emit PAD too often.

Use:

```python
loss = (loss * loss_mask).sum() / loss_mask.sum()
```

### 14.5 Non-Contiguous Tensor View

After transposes, use:

```python
x = x.contiguous().view(...)
```

or prefer:

```python
x = x.reshape(...)
```

## 15. Stretch Goals

After the basic model works:

1. Compare positional encodings on fixed-width addition.
2. Train on 2-digit addition, test on 3-digit addition.
3. Add multiplication.
4. Add scratchpad data, e.g. `12+7=12+07=19`.
5. Implement top-k attention-token pruning based on recent attention weights.
6. Implement a toy TriAttention-like scorer using RoPE phases.
7. Plot cache length vs accuracy for different pruning policies.

## 16. Connection To KV-Cache Optimization

Full KV cache keeps:

```text
K_0, V_0, K_1, V_1, ..., K_t, V_t
```

At decode step `t+1`, attention uses all previous keys:

```text
softmax(q_{t+1} K_{\le t}^T / sqrt(d)) V_{\le t}
```

KV-cache pruning replaces `K_{\le t}, V_{\le t}` with a subset:

```text
K_S, V_S where S subset {0, ..., t}
```

Then attention becomes:

```text
softmax(q_{t+1} K_S^T / sqrt(d)) V_S
```

The model no longer has access to pruned tokens. They are not zeroed; they are
removed.

This project starts with simple policies like:

```text
S = last N tokens
S = BOS + last N tokens
S = every kth old token + last N tokens
```

Advanced methods like TriAttention try to choose `S` by predicting which cached
keys will matter for future queries.

## 17. Completion Checklist

Use this checklist:

```text
[ ] tokenizer roundtrip works
[ ] dataset returns padded input/target/mask
[ ] sinusoidal position encoding works
[ ] learned absolute position encoding works
[ ] ALiBi bias shape is correct
[ ] RoPE apply function preserves shape
[ ] attention forward works without cache
[ ] causal mask prevents future attention
[ ] transformer forward returns logits and loss
[ ] training loss decreases
[ ] generation produces valid-looking arithmetic strings
[ ] KV cache generation matches no-cache generation before pruning
[ ] keep-last pruning caps cache length
[ ] cache inspection script prints useful diagnostics
```

When all boxes are checked, you will have implemented the foundations needed to
understand transformer inference and KV-cache compression methods.

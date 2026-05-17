"""Tiny decoder-only transformer language model."""

from __future__ import annotations

from sympy.integrals.transforms import SineCosineTypeTransform
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .kv_cache import KVCache
from .position import LearnedAbsolutePositionalEncoding, SinusoidalPositionalEncoding
from .attention import CausalSelfAttention


class FeedForward(nn.Module):
    """Transformer MLP block."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # TODO: Linear d_model -> d_ff, GELU, Linear d_ff -> d_model, Dropout.
        # raise NotImplementedError
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network.

        Args:
            x: Tensor [batch, seq_len, d_model].

        Returns:
            Tensor [batch, seq_len, d_model].
        """

        x = self.linear(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear2(x)

        return x



class TransformerBlock(nn.Module):
    """Pre-LN decoder transformer block."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # TODO: LayerNorm, CausalSelfAttention, LayerNorm, FeedForward.
        # raise NotImplementedError
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            max_seq_len=cfg.max_seq_len,
            pos_encoding=cfg.pos_encoding,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
        )


    def forward(
        self,
        x: torch.Tensor,
        layer_cache=None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Run one transformer block.

        Args:
            x: Hidden states [batch, seq_len, d_model].
            layer_cache: Optional LayerKVCache for this block.
            start_pos: Absolute position of first token in x.
            use_cache: Whether to append/use KV cache.

        Returns:
            Updated hidden states [batch, seq_len, d_model].
        """
        x = x + self.attn(
                        self.ln1(x), 
                        layer_cache=layer_cache,
                        start_pos=start_pos,
                        use_cache=use_cache
                    )

        x = x + self.ffn(self.ln2(x))
        
        return x

        # raise NotImplementedError


class TinyTransformerLM(nn.Module):
    """Small decoder-only Transformer language model."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.vocab_size is None:
            raise ValueError("cfg.vocab_size must be set before constructing model")
        self.cfg = cfg

        # TODO:
        # - token embedding
        # - optional sinusoidal or learned absolute pos embedding
        # - transformer blocks
        # - final layer norm
        # - lm_head
        # raise NotImplementedError
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_seq_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for i in range(cfg.n_layers)]
        )
        self.ln = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)


    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run the language model.

        Args:
            input_ids: Token ids [batch, seq_len].
            targets: Optional next-token targets [batch, seq_len].
            loss_mask: Optional float mask [batch, seq_len], 1 for valid loss positions.
            kv_cache: Optional KVCache used during generation.
            start_pos: Absolute position of input_ids[:, 0].
            use_cache: If True, blocks append/use KV cache.

        Returns:
            Dict containing:
                logits: [batch, seq_len, vocab_size]
                loss: scalar tensor if targets are provided, else absent or None.

        TODO:
            Implement embeddings, blocks, logits, and masked CE loss.
        """

        x = self.token_emb(input_ids) # x's shape [batch, d_model, seq_len]
        x = self.pos_emb(x, start_pos=start_pos)
        for layer_idx in range(self.cfg.n_layers):
            layer_cache = kv_cache.layers[layer_idx] if kv_cache is not None else None
            x = self.blocks[layer_idx](
                x, 
                layer_cahce=layer_cache, 
                start_pos=start_pos,
                use_cache=use_cache
            )
        
        x = self.ln(x)
        logits = self.lm_head(x) # logits' shape [batch, seq_len, vocab_size]

        # Compute CE loss
        loss = self._compute_loss(logits, targets, loss_mask)

        return {"logits": x, "loss": loss}


    def _compute_loss(
        self,
        logits: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ Compute the Cross Entropy loss per token
        
        Args:
            logits: Output logits [batch, seq_len, vocab_size]
            targets: Optional next-token targets [batch, seq_len].
            loss_mask: Optional float mask [batch, seq_len], 1 for valid loss positions.

        Returns:
            Loss value
        """
        (batch, seq_len, vocab_size) = logits.shape

        loss_flat = F.cross_entropy(
            logits.view(batch * seq_len, vocab_size),
            targets.view(batch * seq_len),
            reduction="none"
        )

        loss = loss_flat * loss_mask.view(batch * seq_len)
        loss = loss.sum() / loss_mask.sum().clamp_min(1)

        return loss


    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: int = 0
    ) -> torch.Tensor:

        """
        Samples the next token ID from the model's logits.
    
        Args:
            logits: Shape [batch_size, vocab_size]
                                representing the last token's predictions.
            temperature: Controls randomness. Lower = deterministic, Higher = random.
            top_k: Keeps only the top K highest probability tokens.
            top_p: Keeps the top tokens whose cumulative probability <= top_p.
        """
        if temperature > 0:
            logits = logits / temperature
        else:
            return torch.argmax(logits, dim=-1, keepdim=True)

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return next_id
        # TODO: implement top-K and top-p filtering

        # raise NotImplementedError


    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively generate token ids.

        Args:
            input_ids: Prompt ids [batch, prompt_len].
            max_new_tokens: Number of new tokens to sample.
            temperature: Sampling temperature. Use 0 or greedy path if you prefer.
            use_cache: Whether to use incremental KV cache.

        Returns:
            Full generated ids [batch, prompt_len + max_new_tokens].

        TODO:
            Implement two paths:
            - no cache: feed full prefix every step,
            - with cache: prefill once, then feed only last token.
        """

        ids = input_ids

        if not use_cache:
            for _ in range(max_new_tokens):
                out = self.forward(ids, use_cache=False) 
                next_logits = out["logits"][:, -1, :] # [batch, vocab_size]
                next_id = self._sample_next_token(next_logits, temperature) # [batch]
                ids = torch.cat([ids, next_id[:, None]])
        
        if use_cache:
            kv_cache = KVCache(self.cfg.n_layers)

            # Prefilling: process full prompts once
            # kv is stored in the forward pass of attn layers
            out = self.forward(ids, kv_cache=kv_cache, start_pos=0, use_cache=True)
            next_logits = out["logits"][:, -1, :] 
            next_id = self._sample_next_token(next_logits, temperature)
            ids = torch.cat([ids, next_id[:, None]]) 

            # Decoding: process each tokens at a time, add new token's kv to cache
            for pos in range(ids.shape[1], ids.shape[1] + max_new_tokens - 1):
                out = self.forward(
                    ids[:, -1, :],      # Only feed the last generated token
                    kv_cache=kv_cache, 
                    start_pos=pos,      # stride the start pos to the current pos
                    use_cache=True
                    )
                next_logits = out["logits"][:, -1, :]
                next_id = self._sample_next_token(next_logits, temperature)
                ids = torch.cat([ids, next_id[:, None]])

        return ids

        # raise NotImplementedError


if __name__ == "__main__":
    pass
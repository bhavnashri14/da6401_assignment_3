"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import spacy
from collections import Counter
from datasets import load_dataset
import gdown 
import os
# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_w = torch.softmax(scores, dim=-1)

    output = torch.matmul(attn_w, V)

    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    src_mask = (src == pad_idx) #Find pad position
    src_mask = src_mask.unsqueeze(1).unsqueeze(2) # change shape to [batch, 1, 1, src_len]

    return src_mask


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    batch_size, tgt_len = tgt.shape

    # padding mask
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2) # shape [batch,1,1,tgt_len]


    causal_mask = torch.triu(  
        torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool),
        diagonal=1
    )     #mask future positions

    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
    # shape[1,1,tgt_len,tgt_len]

    tgt_mask = pad_mask | causal_mask # combine masks with OR

    return tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.use_scale = True

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert:
            [batch, seq_len, d_model]
        to:
            [batch, num_heads, seq_len, d_k]
        """

        batch_size, seq_len, _ = x.shape

        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert:
            [batch, num_heads, seq_len, d_k]
        to:
            [batch, seq_len, d_model]
        """

        batch_size, _, seq_len, _ = x.shape

        x = x.transpose(1, 2).contiguous()

        return x.view(batch_size, seq_len, self.d_model)


    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """
        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        d_k = Q.size(-1)
        scale = math.sqrt(d_k) if self.use_scale else 1.0
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output  = torch.matmul(attn_weights, V)

        concat_output = self.combine_heads(attn_output)
        output = self.W_o(concat_output)
        output = self.dropout(output)
        return output


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term) # sin for even indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos for odd indices
        pe = pe.unsqueeze(0) # reshape to [1, max_len, d_model]

        self.register_buffer('pe', pe)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  

        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]  # add positional encoding
        return self.dropout(x)

# Add LearnedPositionalEncoding class to model.py:
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout    = nn.Dropout(p=dropout)
        self.pos_embed  = nn.Embedding(max_len, d_model)
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pos_embed(positions))


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        attn_output = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask=src_mask
        )
        x = self.norm1(x + self.dropout(attn_output))  #Post LN
        # x + sublayer(norm(x))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))  #Post LN

        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """

        attn_output = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask=tgt_mask
        )

        x = self.norm1(x + self.dropout(attn_output))

        cross_output = self.cross_attn(
            query=x,
            key=memory,
            value=memory,
            mask=src_mask
        )

        x = self.norm2(x + self.dropout(cross_output))

        ffn_output = self.ffn(x)

        x = self.norm3(x + self.dropout(ffn_output))

        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(N)]
        )
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        return x


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(N)]
        )

        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        x = self.norm(x)

        return x


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
    self,
    src_vocab_size: int = 10000,
    tgt_vocab_size: int = 10000,
    d_model: int = 512,
    N: int = 6,
    num_heads: int = 8,
    d_ff: int = 2048,
    dropout: float = 0.1,
    pos_encoding_type="sinusoidal",
    checkpoint_path: str = None,
) -> None:
        super().__init__()
        self.src_vocab = None
        self.tgt_vocab = None
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size


        # embeddings
        self.src_embed = nn.Embedding(self.src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(self.tgt_vocab_size, d_model)

        import subprocess
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            import sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "de_core_news_sm"], check=True)
            self.spacy_de = spacy.load("de_core_news_sm")

        # positional encoding
        if pos_encoding_type == "learned":
            self.pos_encoding = LearnedPositionalEncoding(d_model, dropout)
        else:
            self.pos_encoding = PositionalEncoding(d_model, dropout)

        # encoder / decoder
        self.encoder = Encoder(
            EncoderLayer(d_model, num_heads, d_ff, dropout),
            N
        )

        self.decoder = Decoder(
            DecoderLayer(d_model, num_heads, d_ff, dropout),
            N
        )

        # output
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

        # init weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.checkpoint_path = "model.pt"
        gdrive_id = "1YcArQGpScCQdKbkK0eaCbBRB78bBQILQ"

        if not os.path.exists(self.checkpoint_path):
            gdown.download(
                f"https://drive.google.com/uc?id={gdrive_id}",
                self.checkpoint_path,
                quiet=False
            )

        if os.path.exists(self.checkpoint_path):
          state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
          self.load_state_dict(state, strict=False)

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
    
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        memory = self.encoder(x, src_mask)
        return memory

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.decoder(
            x,
            memory,
            src_mask,
            tgt_mask
        )
        logits = self.fc_out(x)
        return logits

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        logits = self.decode(
            memory,
            src_mask,
            tgt,
            tgt_mask
        )
        return logits


    def infer(self, src_sentence: str) -> str:
      self.eval()
      device = next(self.parameters()).device

      tokens = ["<sos>"] + [t.text.lower() for t in self.spacy_de.tokenizer(src_sentence.strip())] + ["<eos>"]

      src_indices = [
          self.src_vocab.stoi.get(tok, self.src_vocab.stoi["<unk>"])
          for tok in tokens
      ]

      src = torch.tensor(src_indices, dtype=torch.long, device=device).unsqueeze(0)
      src_mask = make_src_mask(src).to(device)

      memory = self.encode(src, src_mask)

      ys = torch.tensor(
          [[self.tgt_vocab.stoi["<sos>"]]],
          dtype=torch.long,
          device=device
      )

      max_len = 100

      for _ in range(max_len):
          tgt_mask = make_tgt_mask(ys).to(device)

          out = self.decode(memory, src_mask, ys, tgt_mask)

          prob = out[:, -1, :]
          next_word = torch.argmax(prob, dim=-1).item()

          ys = torch.cat(
              [ys, torch.tensor([[next_word]], device=device)],
              dim=1
          )

          if next_word == self.tgt_vocab.stoi["<eos>"]:
              break

      # detokenize
      output_tokens = ys.squeeze(0).tolist()

      words = []
      for idx in output_tokens:
          token = self.tgt_vocab.itos[idx]

          if token in ["<sos>", "<pad>"]:
              continue
          if token == "<eos>":
              break
          words.append(token)

      return " ".join(words)
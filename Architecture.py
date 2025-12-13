import torch
import math
from torch import nn
import numpy as np
from typing import Optional
import torch.nn.functional as F



def generate_causal_mask(seq_len: int, device: Optional[torch.device] = None):
    """
    Additive mask (T, T) with 0 for allowed, -1e9 for blocked future positions.
    Suitable to add to attention logits.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    mask = mask.float().masked_fill(mask, float("-1e9"))
    return mask.to(device) if device is not None else mask

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, max_len=2048):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(input_dim, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # For cross-attn
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # RoPE module
        self.rope = RotaryEmbedding(self.head_dim, max_len=max_len)

    def _apply_rope(self, q, k):
        # q, k: (B, H, T, Dh)
        return self.rope(q), self.rope(k)

    def _attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores + mask.to(scores.device)

        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        qkv = self.qkv(x)  # (B, T, 3D)
        qkv = qkv.view(B, T, self.num_heads, 3*self.head_dim).permute(0,2,1,3)
        q, k, v = qkv.chunk(3, dim=-1)

        # Apply RoPE
        q, k = self._apply_rope(q, k)

        out = self._attention(q, k, v, mask)
        out = out.permute(0,2,1,3).contiguous().view(B, T, -1)
        return self.out_proj(out)

    def forward_cross_attention(self, q_in, k_in, v_in, mask=None):
        B, Tq, _ = q_in.shape
        _, Tk, _ = k_in.shape

        q = self.q_linear(q_in).view(B, Tq, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = self.k_linear(k_in).view(B, Tk, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = self.v_linear(v_in).view(B, Tk, self.num_heads, self.head_dim).permute(0,2,1,3)

        # Apply RoPE to cross-attn q,k
        q, k = self._apply_rope(q, k)

        out = self._attention(q, k, v, mask)
        out = out.permute(0,2,1,3).contiguous().view(B, Tq, -1)
        return self.out_proj(out)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_len=2048):
        super().__init__()
        self.head_dim = head_dim

        # Compute base frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # Precompute sin/cos tables
        t = torch.arange(max_len)
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)  # interleave
        self.register_buffer("cos_emb", emb.cos())
        self.register_buffer("sin_emb", emb.sin())

    def forward(self, x):
        """
        x: (B, H, T, D)
        apply RoPE on last dimension D
        """
        B, H, T, D = x.shape
        cos = self.cos_emb[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin_emb[:T].unsqueeze(0).unsqueeze(0)

        x1 = x[..., : D//2]
        x2 = x[..., D//2 :]

        # rotate
        x_rot = torch.cat([-x2, x1], dim=-1)
        return (x * cos) + (x_rot * sin)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()

    def forward(self, x):
        return x   # RoPE replaces absolute positional embeddings

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn)
        self.w2 = nn.Linear(d_model, d_ffn)
        self.w3 = nn.Linear(d_ffn, d_model)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class EncodingLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForwardNetwork(d_model, d_ffn)

    def forward(self, x, src_mask: Optional[torch.Tensor] = None):
        # Pre-norm: LN -> MHA -> resid
        x = x + self.mha(self.ln1(x), mask=src_mask)
        # FFN block
        x = x + self.ff(self.ln2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ffn: int, max_len: int = 2048):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([EncodingLayer(d_model, num_heads, d_ffn) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.LongTensor, src_mask: Optional[torch.Tensor] = None):
        """
        input_ids: (B, T)
        returns (B, T, d_model)
        """
        x = self.token_emb(input_ids)  # (B, T, d_model)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.ln(x)

class DecodingLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_att = MultiHeadAttention(d_model, d_model, num_heads)

        self.ln2 = nn.LayerNorm(d_model)
        self.cross_att = MultiHeadAttention(d_model, d_model, num_heads)

        self.ln3 = nn.LayerNorm(d_model)
        self.ff = FeedForwardNetwork(d_model, d_ffn)

    def forward(self, x: torch.Tensor, enc_output: Optional[torch.Tensor] = None, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None):
        # masked self-attention (pre-norm)
        x = x + self.self_att(self.ln1(x), mask=tgt_mask)

        # cross-attention only if encoder output is provided
        if enc_output is not None:
            x = x + self.cross_att.forward_cross_attention(self.ln2(x), enc_output, enc_output, mask=src_mask)

        # feed-forward
        x = x + self.ff(self.ln3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ffn: int, max_len: int = 2048):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([DecodingLayer(d_model, num_heads, d_ffn) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.LongTensor, enc_output: Optional[torch.Tensor] = None, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None):
        B, T = input_ids.shape
        x = self.token_emb(input_ids)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, enc_output=enc_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.ln(x)

class EncoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ffn: int, max_len: int = 2048):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, num_layers, num_heads, d_ffn, max_len=max_len)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.LongTensor, src_mask: Optional[torch.Tensor] = None):
        enc = self.encoder(input_ids, src_mask=src_mask)
        enc = self.ln(enc)
        logits = self.head(enc)  # (B, T, V)
        return logits

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ffn: int, max_len: int = 2048):
        super().__init__()
        # build a decoder; token embedding sized to vocab
        self.decoder = Decoder(vocab_size, d_model, num_layers, num_heads, d_ffn, max_len=max_len)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.LongTensor):
        """
        input_ids: (B, T)
        returns logits: (B, T, V)
        """
        B, T = input_ids.shape
        tgt_mask = generate_causal_mask(T, device=input_ids.device)
        dec = self.decoder(input_ids, enc_output=None, src_mask=None, tgt_mask=tgt_mask)
        dec = self.ln(dec)
        logits = self.head(dec)
        return logits





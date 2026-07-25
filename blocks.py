import copy
import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import PositionwiseFeedForward, LayerNorm, ResidualConnection


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.res_conn_1 = ResidualConnection(d_model=d_model, dropout=dropout)
        self.res_conn_2 = ResidualConnection(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(q=x, k=x, v=x, mask=src_mask)
        x = self.res_conn_1(x, attn_out)

        ffn_out = self.ffn(x)
        x = self.res_conn_2(x, ffn_out)

        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int = 6, d_model: int = 512):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.res_conn_1 = ResidualConnection(d_model=d_model, dropout=dropout)
        self.res_conn_2 = ResidualConnection(d_model=d_model, dropout=dropout)
        self.res_conn_3 = ResidualConnection(d_model=d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        self_attn_out, _ = self.self_attn(q=x, k=x, v=x, mask=tgt_mask)
        x = self.res_conn_1(x, self_attn_out)

        cross_attn_out, _ = self.cross_attn(q=x, k=enc_out, v=enc_out, mask=src_mask)
        x = self.res_conn_2(x, cross_attn_out)

        ffn_out = self.ffn(x)
        x = self.res_conn_3(x, ffn_out)

        return x


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int = 6, d_model: int = 512):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)
import torch
import torch.nn as nn
from embeddings import TransformerEmbedding
from blocks import EncoderLayer, Encoder, DecoderLayer, Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.src_embedding = TransformerEmbedding(src_vocab_size, d_model, max_len, dropout)
        self.tgt_embedding = TransformerEmbedding(tgt_vocab_size, d_model, max_len, dropout)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, num_encoder_layers, d_model)

        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(dec_layer, num_decoder_layers, d_model)

        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt_len = tgt.size(1)

        pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)

        causal_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), device=tgt.device)
        ).bool()

        tgt_mask = pad_mask & causal_mask
        return tgt_mask

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src_embedded = self.src_embedding(src)
        return self.encoder(src_embedded, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        tgt_embedded = self.tgt_embedding(tgt)
        return self.decoder(tgt_embedded, enc_out, src_mask, tgt_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)

        logits = self.generator(dec_out)
        return logits
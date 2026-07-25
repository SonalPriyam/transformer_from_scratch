import torch
from blocks import EncoderLayer, Encoder, DecoderLayer, Decoder


def run_test():
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    d_model = 512

    src_embedded = torch.randn(batch_size, src_seq_len, d_model)
    tgt_embedded = torch.randn(batch_size, tgt_seq_len, d_model)

    enc_layer = EncoderLayer(d_model=d_model, num_heads=8, d_ff=2048, dropout=0.1)
    encoder = Encoder(layer=enc_layer, N=6, d_model=d_model)

    dec_layer = DecoderLayer(d_model=d_model, num_heads=8, d_ff=2048, dropout=0.1)
    decoder = Decoder(layer=dec_layer, N=6, d_model=d_model)

    enc_out = encoder(src_embedded)

    dec_out = decoder(x=tgt_embedded, enc_out=enc_out)

    
    print("1. Encoder Input Shape:  ", src_embedded.shape)
    print("2. Encoder Output Shape: ", enc_out.shape)
    print("3. Decoder Input Shape:  ", tgt_embedded.shape)
    print("4. Decoder Output Shape: ", dec_out.shape)
    print("   Expected Enc Output Shape: torch.Size([2, 10, 512])")
    print("   Expected Dec Output Shape: torch.Size([2, 8, 512])")

    assert enc_out.shape == (2, 10, 512), "Encoder Output shape mismatch!"
    assert dec_out.shape == (2, 8, 512), "Decoder Output shape mismatch!"

    print("\nSuccess! N=6 Encoder and Decoder stacks successfully created and verified.")


if __name__ == "__main__":
    run_test()
import torch
from feedforward import PositionwiseFeedForward, LayerNorm, ResidualConnection


def run_test():
    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048

    x = torch.randn(batch_size, seq_len, d_model)

    ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.1)
    norm = LayerNorm(d_model=d_model)
    res_conn = ResidualConnection(d_model=d_model, dropout=0.1)

    ffn_out = ffn(x)

    res_out = res_conn(x, ffn_out)

    
    print("1. Input Tensor Shape:      ", x.shape)
    print("2. FFN Output Tensor Shape: ", ffn_out.shape)
    print("3. Add & Norm Output Shape: ", res_out.shape)
    print("   Expected Shape:          torch.Size([2, 10, 512])")

    assert ffn_out.shape == (2, 10, 512), "FFN output shape mismatch!"
    assert res_out.shape == (2, 10, 512), "Residual Connection shape mismatch!"

    normed_test = norm(x)
    mean_val = normed_test.mean(dim=-1).abs().max().item()
    print(f"\nMax deviation from 0 mean after LayerNorm: {mean_val:.6f}")
    assert mean_val < 1e-4, "LayerNorm mean calculation check failed!"

    print("\nSuccess! Position-wise Feed-Forward Network and LayerNorm residual connections are working properly.")


if __name__ == "__main__":
    run_test()
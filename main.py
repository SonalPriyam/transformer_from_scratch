import torch
from tokenizer import BPETokenizer
from transformer import Transformer


def run_full_pipeline():
    print("--- Phase 6: End-to-End Transformer Assembly ---")

    corpus = "attention is all you need. transformers execute end to end text generation."
    tokenizer = BPETokenizer(vocab_size=280)
    tokenizer.train(corpus)

    src_text = "attention is all you need"
    tgt_text = "transformers execute generation"

    src_tokens = tokenizer.encode(src_text, add_special_tokens=True)
    tgt_tokens = tokenizer.encode(tgt_text, add_special_tokens=True)

    max_src_len = 10
    max_tgt_len = 8
    padded_src = tokenizer.pad_sequence(src_tokens, max_src_len)
    padded_tgt = tokenizer.pad_sequence(tgt_tokens, max_tgt_len)

    src_tensor = torch.tensor([padded_src], dtype=torch.long)
    tgt_tensor = torch.tensor([padded_tgt], dtype=torch.long)

    print("Source Tensor Shape:", src_tensor.shape)
    print("Target Tensor Shape:", tgt_tensor.shape)

    vocab_size = tokenizer.vocab_size
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_pad_idx=tokenizer.PAD_IDX,
        tgt_pad_idx=tokenizer.PAD_IDX,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1
    )

    logits = model(src_tensor, tgt_tensor)

    print("\nModel Output Logits Shape:", logits.shape)
    print(f"Expected Shape: torch.Size([1, {max_tgt_len}, {vocab_size}])")

    assert logits.shape == (1, max_tgt_len, vocab_size), "Output logits shape mismatch!"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters in Model: {total_params:,}")

    print("\n SUCCESS! Complete Transformer built from scratch and verified end-to-end!")


if __name__ == "__main__":
    run_full_pipeline()
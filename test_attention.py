import torch
from tokenizer import BPETokenizer
from embeddings import TransformerEmbedding
from attention import MultiHeadAttention


def run_test():
    corpus = "attention is all you need. multi-head attention splits d_model into parallel heads."
    tokenizer = BPETokenizer(vocab_size=280)
    tokenizer.train(corpus)

    text = "attention is all you need"
    tokens = tokenizer.encode(text, add_special_tokens=True)
    padded_tokens = tokenizer.pad_sequence(tokens, max_length=10)

    input_tensor = torch.tensor([padded_tokens, padded_tokens], dtype=torch.long)

    embedding_layer = TransformerEmbedding(vocab_size=tokenizer.vocab_size, d_model=512)
    embedded_vectors = embedding_layer(input_tensor)

    mha = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.1)

    output, attn_weights = mha(
        q=embedded_vectors,
        k=embedded_vectors,
        v=embedded_vectors
    )

    
    print("1. Input Embedded Vector Shape:", embedded_vectors.shape)
    print("2. MHA Output Tensor Shape:   ", output.shape)
    print("3. Attention Weights Shape:   ", attn_weights.shape)
    print("   Expected MHA Output Shape: torch.Size([2, 10, 512])")
    print("   Expected Weights Shape:    torch.Size([2, 8, 10, 10])")

    assert output.shape == (2, 10, 512), "MHA Output shape mismatch!"
    assert attn_weights.shape == (2, 8, 10, 10), "Attention Weights shape mismatch!"

    print("\nSuccess! Multi-Head Attention split d_model into 8 heads (64 dimensions each) and computed self-attention successfully.")


if __name__ == "__main__":
    run_test()
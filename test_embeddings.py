import torch
from tokenizer import BPETokenizer
from embeddings import TransformerEmbedding


def run_test():
    corpus = "attention is all you need. transformers rely on positional encodings."
    tokenizer = BPETokenizer(vocab_size=280)
    tokenizer.train(corpus)

    text = "attention is all you need"
    tokens = tokenizer.encode(text, add_special_tokens=True)
    padded_tokens = tokenizer.pad_sequence(tokens, max_length=10)

    input_tensor = torch.tensor([padded_tokens], dtype=torch.long)
    print("1. Tokenizer Batch Tensor Shape:", input_tensor.shape)

    embedding_layer = TransformerEmbedding(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        max_len=100,
        dropout=0.1
    )

    embedded_vectors = embedding_layer(input_tensor)

    print("2. Embedded Output Shape:", embedded_vectors.shape)
    print("   Expected Shape: torch.Size([1, 10, 512])")

    assert embedded_vectors.shape == (1, 10, 512), "Embedding shape mismatch!"
    print("\nSuccess! Token IDs successfully mapped to d_model=512 dense vectors with positional encodings.")


if __name__ == "__main__":
    run_test()
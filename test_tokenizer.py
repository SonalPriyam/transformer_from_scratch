import torch
from tokenizer import BPETokenizer


def run_test():
    corpus = (
        "attention is all you need. "
        "transformers use self-attention mechanisms for natural language tasks."
    )

    tokenizer = BPETokenizer(vocab_size=280)
    tokenizer.train(corpus)

    text1 = "attention is all you need"
    text2 = "transformers perform tokenization"

    tokens1 = tokenizer.encode(text1, add_special_tokens=True)
    tokens2 = tokenizer.encode(text2, add_special_tokens=True)

    max_len = 12
    padded1 = tokenizer.pad_sequence(tokens1, max_len)
    padded2 = tokenizer.pad_sequence(tokens2, max_len)

    batch_tensor = torch.tensor([padded1, padded2], dtype=torch.long)

    print("Batch Tensor Shape:", batch_tensor.shape)
    print("\nBatch Tensor Matrix:\n", batch_tensor)
    print("\nDecoded Output 1:", tokenizer.decode(padded1))
    print("Decoded Output 2:", tokenizer.decode(padded2))


if __name__ == "__main__":
    run_test()
import torch
import torch.nn as nn
from tokenizer import BPETokenizer
from transformer import Transformer


def train_model():
    print("--- Training Transformer Model Locally ---")

    corpus = """
    attention is all you need.
    transformers rely on self attention mechanisms to process language.
    byte pair encoding tokenizes text into integer tokens efficiently.
    multi head attention splits vector representations into parallel heads.
    positional encodings inject order information into token embeddings.
    encoder decoder stacks process source and target sequences autoregressively.
    """

    vocab_size = 280
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(corpus)

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_pad_idx=tokenizer.PAD_IDX,
        tgt_pad_idx=tokenizer.PAD_IDX,
        d_model=256,
        num_heads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=512,
        dropout=0.1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_IDX)
    model.train()

    src_text = "attention is all you need"
    tgt_text = "transformers rely on self attention"

    src_tokens = tokenizer.encode(src_text, add_special_tokens=True)
    tgt_tokens = tokenizer.encode(tgt_text, add_special_tokens=True)

    src_tensor = torch.tensor([tokenizer.pad_sequence(src_tokens, 12)], dtype=torch.long)
    tgt_input = torch.tensor([tokenizer.pad_sequence(tgt_tokens[:-1], 12)], dtype=torch.long)
    tgt_label = torch.tensor([tokenizer.pad_sequence(tgt_tokens[1:], 12)], dtype=torch.long)

    epochs = 400
    for epoch in range(epochs):
        optimizer.zero_grad()

        logits = model(src_tensor, tgt_input)

        loss = criterion(logits.view(-1, vocab_size), tgt_label.view(-1))

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "transformer_model.pth")
    print("\nTraining Complete! Model weights successfully saved to 'transformer_model.pth'.")


if __name__ == "__main__":
    train_model()
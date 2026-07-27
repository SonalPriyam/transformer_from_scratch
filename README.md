# Encoder-Decoder Transformer from Scratch (PyTorch)

A modular, lightweight implementation of the original Sequence-to-Sequence Transformer architecture as introduced in Vaswani et al. (2017), *"Attention Is All You Need"*. 

Built strictly from scratch using core PyTorch primitives without relying on high-level `torch.nn.Transformer` or pre-built library modules. Includes a custom Byte-Pair Encoding (BPE) tokenizer and causal and padding masking pipelines.

---

## Key Features

* **Custom Byte-Pair Encoding (BPE):** Implemented an in-memory BPE tokenizer handling raw UTF-8 byte mappings and special control tokens (`<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`).
* **Multi-Head Attention (MHA):** Parallelized Scaled Dot-Product Attention projecting Query, Key, and Value representations across h=8 heads (d_k = 64).
* **Positional Encodings:** Static sinusoidal wave functions injected into d_model = 512 embedding spaces to provide spatial order context without recurrent operations.
* **Add & Norm Sublayers:** Layer Normalization built from scratch with learnable scale (gamma) and shift (beta) parameters, wrapped around residual connections.
* **Masking Pipeline:** Integrated target causal masking (lower-triangular boolean matrices) to prevent future-token information leakage during autoregressive decoding, alongside source padding masks.

---

## Repository Structure

```text
transformer_from_scratch/
│
├── tokenizer.py        # BPE Tokenizer class & special token handling
├── embeddings.py       # Token Embeddings + Sinusoidal Positional Encodings
├── attention.py        # Scaled Dot-Product & Multi-Head Attention
├── feedforward.py      # Position-wise FFN & Layer Normalization
├── blocks.py           # EncoderLayer, DecoderLayer, Encoder, and Decoder stacks
├── transformer.py      # Unified Seq2Seq Model & Causal Masking logic
├── train.py            # Local training script
├── app.py              # Gradio web interface for Hugging Face Spaces
├── main.py             # End-to-end integration & verification test
└── requirements.txt    # Project dependencies
import streamlit as st
import torch
from tokenizer import BPETokenizer
from transformer import Transformer

st.set_page_config(
    page_title="Transformer from Scratch",
    page_icon="🤖",
    layout="wide"
)

VOCAB_SIZE = 280
D_MODEL = 256
MAX_LEN = 12

CORPUS = """
attention is all you need.
transformers rely on self attention mechanisms to process language.
byte pair encoding tokenizes text into integer tokens efficiently.
multi head attention splits vector representations into parallel heads.
positional encodings inject order information into token embeddings.
encoder decoder stacks process source and target sequences autoregressively.
"""


@st.cache_resource
def load_system():
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.train(CORPUS)

    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        src_pad_idx=tokenizer.PAD_IDX,
        tgt_pad_idx=tokenizer.PAD_IDX,
        d_model=D_MODEL,
        num_heads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=512,
        dropout=0.1,
    )

    weights_loaded = False
    try:
        model.load_state_dict(
            torch.load("transformer_model.pth", map_location=torch.device("cpu"))
        )
        weights_loaded = True
    except Exception as e:
        pass

    model.eval()
    return tokenizer, model, weights_loaded


tokenizer, model, weights_loaded = load_system()

st.sidebar.title("Model Architecture")
st.sidebar.markdown("**Paper:** *Attention Is All You Need* (2017)")
st.sidebar.write("---")
st.sidebar.markdown(f"- **Embedding Dim ($d_{{model}}$):** {D_MODEL}")
st.sidebar.markdown("- **Attention Heads ($h$):** 4")
st.sidebar.markdown("- **Head Dim ($d_k$):** 64")
st.sidebar.markdown("- **Encoder Layers:** 3")
st.sidebar.markdown("- **Decoder Layers:** 3")
st.sidebar.markdown("- **Feed-Forward Dim ($d_{{ff}}$):** 512")
st.sidebar.markdown(f"- **Vocab Size:** {VOCAB_SIZE}")
st.sidebar.write("---")

if weights_loaded:
    st.sidebar.success("Model weights loaded from `transformer_model.pth`!")
else:
    st.sidebar.warning("Running with initial weights (`transformer_model.pth` not found).")

st.title("Encoder-Decoder Transformer from Scratch 🤖")
st.markdown(
    "A full PyTorch implementation of the sequence-to-sequence Transformer architecture, "
    "featuring custom BPE tokenization, positional encodings, multi-head attention, "
    "and causal masking."
)

st.write("---")

input_text = st.text_input(
    "Enter Source Text Prompt:",
    value="attention is all you need",
    help="Type a sentence to run through the custom Tokenizer and Transformer pipeline."
)

col1, col2 = st.columns(2)

if st.button("Generate Output", type="primary"):
    if not input_text.strip():
        st.error("Please enter a valid prompt.")
    else:
        src_tokens = tokenizer.encode(input_text, add_special_tokens=True)
        padded_src = tokenizer.pad_sequence(src_tokens, MAX_LEN)
        src_tensor = torch.tensor([padded_src], dtype=torch.long)

        tgt_tokens = [tokenizer.SOS_IDX]

        with torch.no_grad():
            src_mask = model.make_src_mask(src_tensor)
            enc_out = model.encode(src_tensor, src_mask)

            for _ in range(MAX_LEN - 1):
                tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long)
                tgt_mask = model.make_tgt_mask(tgt_tensor)

                dec_out = model.decode(tgt_tensor, enc_out, src_mask, tgt_mask)
                logits = model.generator(dec_out)

                next_token = logits[0, -1, :].argmax(dim=-1).item()

                if next_token == tokenizer.EOS_IDX:
                    break

                tgt_tokens.append(next_token)

        decoded_output = tokenizer.decode(tgt_tokens)

        with col1:
            st.subheader("Output Text")
            st.info(f"**{decoded_output}**")

        with col2:
            st.subheader("Internal Token IDs")
            st.json({
                "Source Input Tokens": padded_src,
                "Generated Target Tokens": tgt_tokens
            })
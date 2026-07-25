import gradio as gr
import torch
from tokenizer import BPETokenizer
from transformer import Transformer

VOCAB_SIZE = 280
D_MODEL = 256
MAX_LEN = 12

corpus = """
attention is all you need.
transformers rely on self attention mechanisms to process language.
byte pair encoding tokenizes text into integer tokens efficiently.
multi head attention splits vector representations into parallel heads.
positional encodings inject order information into token embeddings.
encoder decoder stacks process source and target sequences autoregressively.
"""

tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
tokenizer.train(corpus)

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
    dropout=0.1
)

try:
    model.load_state_dict(torch.load("transformer_model.pth", map_location=torch.device("cpu")))
    print("Successfully loaded model weights.")
except Exception as e:
    print("Running with initialized weights (model file not found):", e)

model.eval()

def generate_translation(input_text):
    if not input_text.strip():
        return "Please enter a valid text prompt.", "[]"
    
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
            
    decoded_text = tokenizer.decode(tgt_tokens)
    return decoded_text, str(tgt_tokens)

demo = gr.Interface(
    fn=generate_translation,
    inputs=gr.Textbox(lines=2, placeholder="Enter prompt e.g., attention is all you need", label="Input Text"),
    outputs=[
        gr.Textbox(label="Generated Output Text"),
        gr.Textbox(label="Generated Token IDs")
    ],
    title="Transformer From Scratch 🤖",
    description="Custom PyTorch Seq2Seq Transformer model built from scratch as per the 'Attention Is All You Need' research paper."
)

if __name__ == "__main__":
    demo.launch()
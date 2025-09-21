# app.py
import os
import math
import json
import streamlit as st
import torch
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer

# --------- SETTINGS: update if your train hyperparams differ ----------
# If you saved model_hyperparams.json, the code will try to read it.
HP_JSON = "model_hyperparams.json"
if os.path.exists(HP_JSON):
    with open(HP_JSON, "r") as f:
        hp = json.load(f)
    EMBEDDING_DIM = hp.get("EMBEDDING_DIM", 256)
    ENC_HID_DIM = hp.get("ENC_HID_DIM", 256)
    DEC_HID_DIM = hp.get("DEC_HID_DIM", ENC_HID_DIM*2)
    ENC_LAYERS = hp.get("ENC_LAYERS", 2)
    DEC_LAYERS = hp.get("DEC_LAYERS", 4)
    MAX_LEN_SRC = hp.get("MAX_LEN_SRC", 50)
    MAX_LEN_TGT = hp.get("MAX_LEN_TGT", 50)
else:
    # Default values (change to the ones you trained with if different)
    EMBEDDING_DIM = 256
    ENC_HID_DIM = 256
    DEC_HID_DIM = ENC_HID_DIM * 2
    ENC_LAYERS = 2
    DEC_LAYERS = 4
    MAX_LEN_SRC = 50
    MAX_LEN_TGT = 50

MODEL_PATH = "best_model.pt"
TOKENIZER_ROMAN_VOCAB = "tokenizers/roman/vocab.json"
TOKENIZER_ROMAN_MERGES = "tokenizers/roman/merges.txt"
TOKENIZER_URDU_VOCAB = "tokenizers/urdu/vocab.json"
TOKENIZER_URDU_MERGES = "tokenizers/urdu/merges.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Model classes (must match your training architecture) ----------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, enc_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, enc_hid_dim, num_layers=enc_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if enc_layers>1 else 0.0)
    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, dec_hid_dim, dec_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, dec_hid_dim, num_layers=dec_layers,
                            batch_first=True, dropout=dropout if dec_layers>1 else 0.0)
        self.fc_out = nn.Linear(dec_hid_dim, output_dim)
    def forward(self, input, hidden, cell):
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc_hidden = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.fc_cell = nn.Linear(enc_hid_dim, dec_hid_dim)

    def forward(self, src, tgt):
        encoder_outputs, enc_hidden, enc_cell = self.encoder(src)
        n_slices, bsz, enc_hid_dim = enc_hidden.size()
        hidden_flat = enc_hidden.view(-1, enc_hid_dim)
        cell_flat = enc_cell.view(-1, enc_hid_dim)
        hidden_proj = torch.tanh(self.fc_hidden(hidden_flat)).view(n_slices, bsz, -1)
        cell_proj = torch.tanh(self.fc_cell(cell_flat)).view(n_slices, bsz, -1)

        dec_layers = self.decoder.lstm.num_layers
        if hidden_proj.size(0) != dec_layers:
            if hidden_proj.size(0) < dec_layers:
                reps = math.ceil(dec_layers / hidden_proj.size(0))
                hidden_proj = hidden_proj.repeat(reps, 1, 1)[:dec_layers]
                cell_proj = cell_proj.repeat(reps, 1, 1)[:dec_layers]
            else:
                hidden_proj = hidden_proj[:dec_layers]
                cell_proj = cell_proj[:dec_layers]

        output, hidden_dec, cell_dec = self.decoder(tgt, hidden_proj, cell_proj)
        return output

# --------- load tokenizers ----------
def safe_load_tokenizer(vocab_path, merges_path):
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        raise FileNotFoundError(f"Tokenizer files not found: {vocab_path}, {merges_path}")
    return ByteLevelBPETokenizer(vocab_path, merges_path)

tokenizer_roman = safe_load_tokenizer(TOKENIZER_ROMAN_VOCAB, TOKENIZER_ROMAN_MERGES)
tokenizer_urdu  = safe_load_tokenizer(TOKENIZER_URDU_VOCAB, TOKENIZER_URDU_MERGES)

pad_id = tokenizer_roman.token_to_id("<pad>") if tokenizer_roman.token_to_id("<pad>") is not None else 0
sos_id = tokenizer_roman.token_to_id("<sos>") if tokenizer_roman.token_to_id("<sos>") is not None else tokenizer_roman.token_to_id("<s>")
eos_id = tokenizer_roman.token_to_id("<eos>") if tokenizer_roman.token_to_id("<eos>") is not None else tokenizer_roman.token_to_id("</s>")
src_pad_id = tokenizer_urdu.token_to_id("<pad>") if tokenizer_urdu.token_to_id("<pad>") is not None else 0

INPUT_DIM = tokenizer_urdu.get_vocab_size()
OUTPUT_DIM = tokenizer_roman.get_vocab_size()

# --------- instantiate model and load weights ----------
enc = Encoder(INPUT_DIM, EMBEDDING_DIM, ENC_HID_DIM, ENC_LAYERS, 0.2, src_pad_id).to(device)
dec = Decoder(OUTPUT_DIM, EMBEDDING_DIM, DEC_HID_DIM, DEC_LAYERS, 0.2, pad_id).to(device)
model = Seq2Seq(enc, dec, ENC_HID_DIM, DEC_HID_DIM).to(device)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

# --------- helpers ----------
def decode_ids_to_text(tokenizer, ids, pad_id=pad_id, eos_id=eos_id):
    clean_ids = [int(i) for i in ids if int(i) != pad_id]
    try:
        txt = tokenizer.decode(clean_ids)
    except Exception:
        toks = []
        for iid in clean_ids:
            try:
                toks.append(tokenizer.id_to_token(iid))
            except Exception:
                toks.append(str(iid))
        txt = " ".join(toks)
    txt = txt.replace("<sos>", "").replace("<eos>", "").strip()
    return txt

def greedy_decode_single(model, src_tensor, max_len=50):
    # src_tensor: [1, src_len] on device
    with torch.no_grad():
        encoder_outputs, enc_hidden, enc_cell = model.encoder(src_tensor)
        n_slices, bsz, enc_hid_dim = enc_hidden.size()
        hidden_flat = enc_hidden.view(-1, enc_hid_dim)
        cell_flat = enc_cell.view(-1, enc_hid_dim)
        hidden_proj = torch.tanh(model.fc_hidden(hidden_flat)).view(n_slices, bsz, -1)
        cell_proj = torch.tanh(model.fc_cell(cell_flat)).view(n_slices, bsz, -1)

        dec_layers = model.decoder.lstm.num_layers
        if hidden_proj.size(0) != dec_layers:
            if hidden_proj.size(0) < dec_layers:
                reps = math.ceil(dec_layers / hidden_proj.size(0))
                hidden_proj = hidden_proj.repeat(reps, 1, 1)[:dec_layers]
                cell_proj = cell_proj.repeat(reps, 1, 1)[:dec_layers]
            else:
                hidden_proj = hidden_proj[:dec_layers]
                cell_proj = cell_proj[:dec_layers]

        input_tok = torch.tensor([[sos_id]], dtype=torch.long).to(device)
        hidden, cell = hidden_proj, cell_proj
        preds = []
        for _ in range(max_len):
            out, hidden, cell = model.decoder(input_tok, hidden, cell)   # out: [1,1,vocab]
            next_tok = out.argmax(2)  # [1,1]
            preds.append(next_tok.cpu().numpy()[0,0])
            input_tok = next_tok
        return preds

# --------- Streamlit UI ----------
st.title("Urdu → Roman Urdu Transliteration (Demo)")
st.write("Type an Urdu sentence below and click Translate.")

input_text = st.text_area("Urdu text", height=120)
if st.button("Translate"):
    if not input_text.strip():
        st.warning("Please enter some Urdu text.")
    else:
        # encode + pad
        enc = tokenizer_urdu.encode(input_text).ids
        if len(enc) > MAX_LEN_SRC:
            enc = enc[:MAX_LEN_SRC]
        else:
            enc = enc + [src_pad_id] * (MAX_LEN_SRC - len(enc))
        src_tensor = torch.tensor([enc], dtype=torch.long).to(device)
        pred_ids = greedy_decode_single(model, src_tensor, max_len=MAX_LEN_TGT)
        pred_text = decode_ids_to_text(tokenizer_roman, pred_ids, pad_id, eos_id)
        st.subheader("Predicted Roman Urdu")
        st.write(pred_text)

st.caption("Make sure `best_model.pt` and the `tokenizers/` folder are in the same directory as this app.py.")

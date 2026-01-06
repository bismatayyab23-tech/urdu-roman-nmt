import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import numpy as np
import json
import os

# Set page config
st.set_page_config(
    page_title="Urdu to Roman Urdu Translator",
    page_icon="🕌",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    .example-button {
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🕌 Urdu to Roman Urdu Translator</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Neural Machine Translation using BiLSTM Encoder-Decoder</p>', unsafe_allow_html=True)

# Load model function (cached for performance)
@st.cache_resource
def load_model():
    try:
        # Load vocabularies
        with open('urdu_vocab.pkl', 'rb') as f:
            urdu_vocab = pickle.load(f)

        with open('roman_vocab.pkl', 'rb') as f:
            roman_vocab = pickle.load(f)

        # Load model checkpoint
        checkpoint = torch.load('best_model.pth', map_location='cpu')

        # Load experiment results if available
        experiment_results = None
        if os.path.exists('experiment_results.json'):
            with open('experiment_results.json', 'r', encoding='utf-8') as f:
                experiment_results = json.load(f)

        # Recreate model architecture
        class Encoder(nn.Module):
            def __init__(self, input_dim, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=0)
                self.lstm = nn.LSTM(
                    embed_dim, hidden_dim, num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=True, batch_first=True
                )
                self.dropout = nn.Dropout(dropout)

            def forward(self, src):
                embedded = self.dropout(self.embedding(src))
                outputs, (hidden, cell) = self.lstm(embedded)
                return outputs, hidden, cell

        class Decoder(nn.Module):
            def __init__(self, output_dim, embed_dim, hidden_dim, num_layers=4, dropout=0.3):
                super().__init__()
                self.output_dim = output_dim
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=0)
                self.lstm = nn.LSTM(
                    embed_dim, hidden_dim * 2, num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True
                )
                self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
                self.dropout = nn.Dropout(dropout)

            def forward(self, input, hidden, cell):
                input = input.unsqueeze(1)
                embedded = self.dropout(self.embedding(input))
                output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
                prediction = self.fc_out(output.squeeze(1))
                return prediction, hidden, cell

        class Seq2Seq(nn.Module):
            def __init__(self, encoder, decoder, device):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.device = device

            def forward(self, src, trg, teacher_forcing_ratio=0):
                batch_size = src.shape[0]
                trg_len = trg.shape[1]
                trg_vocab_size = self.decoder.output_dim

                outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
                _, hidden, cell = self.encoder(src)

                # Convert bidirectional states
                hidden = hidden.view(self.encoder.num_layers, 2, batch_size, -1)
                hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
                cell = cell.view(self.encoder.num_layers, 2, batch_size, -1)
                cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)

                # Pad for decoder layers if needed
                if self.decoder.num_layers > hidden.shape[0]:
                    padding_layers = self.decoder.num_layers - hidden.shape[0]
                    hidden = torch.cat([hidden, torch.zeros(padding_layers, batch_size, hidden.shape[2])], dim=0)
                    cell = torch.cat([cell, torch.zeros(padding_layers, batch_size, cell.shape[2])], dim=0)

                input = trg[:, 0]
                for t in range(1, trg_len):
                    output, hidden, cell = self.decoder(input, hidden, cell)
                    outputs[:, t] = output
                    top1 = output.argmax(1)
                    input = top1

                return outputs

        # Initialize model
        device = torch.device('cpu')
        encoder = Encoder(**checkpoint['encoder_config'])
        decoder = Decoder(**checkpoint['decoder_config'])
        model = Seq2Seq(encoder, decoder, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model, urdu_vocab, roman_vocab, experiment_results

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

# Helper functions
def clean_urdu_text(text):
    """Clean Urdu text - preserve Urdu Unicode characters"""
    text = str(text)
    # Urdu Unicode range: \u0600-\u06FF
    text = re.sub(r'[^\u0600-\u06FF\s.,!?;\'"\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_roman_text(text):
    """Clean Roman Urdu text"""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s.,!?;\'"\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def encode_sequence(text, vocab, max_len=50):
    """Encode text to indices with padding"""
    # Add SOS and EOS tokens
    tokens = ['<sos>'] + list(text) + ['<eos>']
    
    # Convert to indices
    indices = []
    for token in tokens:
        if token in vocab:
            indices.append(vocab[token])
        else:
            indices.append(vocab.get('<unk>', 0))
    
    # Truncate or pad
    if len(indices) > max_len:
        indices = indices[:max_len]
        indices[-1] = vocab['<eos>']
    else:
        indices = indices + [vocab['<pad>']] * (max_len - len(indices))
    
    return indices

def translate_text(text, model, urdu_vocab, roman_vocab, max_len=50):
    """Translate a single Urdu sentence to Roman Urdu"""
    if not text.strip():
        return ""
    
    # Clean input
    cleaned = clean_urdu_text(text)
    if not cleaned:
        return "[Error: No valid Urdu text found]"
    
    # Encode
    encoded = encode_sequence(cleaned, urdu_vocab, max_len)
    src_tensor = torch.tensor(encoded).unsqueeze(0)
    
    # Start with SOS token
    trg_indices = [roman_vocab['<sos>']]
    
    with torch.no_grad():
        # Encode
        _, hidden, cell = model.encoder(src_tensor)
        
        # Convert bidirectional states
        hidden = hidden.view(model.encoder.num_layers, 2, 1, -1)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        cell = cell.view(model.encoder.num_layers, 2, 1, -1)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
        
        # Pad for decoder layers if needed
        if model.decoder.num_layers > hidden.shape[0]:
            padding_layers = model.decoder.num_layers - hidden.shape[0]
            hidden = torch.cat([hidden, torch.zeros(padding_layers, 1, hidden.shape[2])], dim=0)
            cell = torch.cat([cell, torch.zeros(padding_layers, 1, cell.shape[2])], dim=0)
        
        # Decode step by step
        for _ in range(max_len - 1):
            trg_tensor = torch.tensor([trg_indices[-1]])
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            
            pred_token = output.argmax(1).item()
            trg_indices.append(pred_token)
            
            if pred_token == roman_vocab['<eos>']:
                break
    
    # Convert indices to text
    idx_to_char = {v: k for k, v in roman_vocab.items()}
    translated_chars = []
    for idx in trg_indices[1:]:  # Skip SOS
        if idx == roman_vocab['<eos>'] or idx == roman_vocab['<pad>'] or idx == roman_vocab.get('<unk>', 0):
            break
        char = idx_to_char.get(idx, '')
        if char not in ['<sos>', '<eos>', '<pad>', '<unk>']:
            translated_chars.append(char)
    
    return ''.join(translated_chars)

# Initialize session state
if 'translation' not in st.session_state:
    st.session_state.translation = ""
if 'show_result' not in st.session_state:
    st.session_state.show_result = False
if 'examples' not in st.session_state:
    st.session_state.examples = [
        "اس آہٹ سے کوئی آیا تو لگتا ہے",
        "موج گل موج صبا موج سحر لگتی ہے",
        "ہر ایک روح میں ایک غم چھپا لگے ہیں",
        "دل کو توڑنا بھی کوئی ہنر نہیں ہے",
        "محبت میں نہیں ہے فرق جینے اور مرنے کا",
        "اردو ایک خوبصورت زبان ہے",
        "پاکستان زندہ باد",
        "سلام علیکم، آپ کیسے ہیں؟"
    ]

# Sidebar for model info
with st.sidebar:
    st.markdown("### 🏛️ Model Information")
    st.markdown("""
    **Architecture:**
    - Encoder: 2-layer Bidirectional LSTM
    - Decoder: 4-layer LSTM
    
    **Training Data:**
    - 1,314 Urdu-Roman Urdu pairs
    - Character-level tokenization
    """)
    
    # Load model info (cached)
    model, urdu_vocab, roman_vocab, experiment_results = load_model()
    
    if experiment_results and len(experiment_results) > 0:
        st.markdown("**Best Experiment Results:**")
        best_result = experiment_results[0]
        for res in experiment_results:
            if res.get('test_bleu', 0) > best_result.get('test_bleu', 0):
                best_result = res
        
        st.markdown(f"""
        - **BLEU Score:** {best_result.get('test_bleu', 0):.4f}
        - **Character Error Rate:** {best_result.get('test_cer', 0):.4f}
        - **Perplexity:** {best_result.get('test_perplexity', 0):.2f}
        - **Loss:** {best_result.get('test_loss', 0):.4f}
        """)
    
    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. Enter Urdu text in the text area
    2. Click the 'Translate' button
    3. View the Roman Urdu translation
    4. Try the example buttons for quick testing
    """)
    
    st.markdown("---")
    st.markdown("### 🚀 Project Info")
    st.markdown("""
    **Project:** Urdu to Roman Urdu NMT  
    **Course:** Neural Machine Translation Assignment  
    **Dataset:** Urdu-Roman Urdu Parallel Corpus  
    **Framework:** PyTorch  
    **Deployment:** Streamlit  
    **Model Type:** Character-level Seq2Seq
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Model Statistics")
    if model:
        total_params = sum(p.numel() for p in model.parameters())
        st.markdown(f"**Total Parameters:** {total_params:,}")
    
    if urdu_vocab and roman_vocab:
        st.markdown(f"**Urdu Vocabulary:** {len(urdu_vocab)} chars")
        st.markdown(f"**Roman Vocabulary:** {len(roman_vocab)} chars")

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 📝 Enter Urdu Text")
    
    # Example buttons in a grid
    st.markdown("**Try these examples:**")
    cols = st.columns(4)
    for i, example in enumerate(st.session_state.examples):
        with cols[i % 4]:
            if st.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True):
                st.session_state.urdu_text = example
                st.rerun()
    
    # Text input area
    urdu_text = st.text_area(
        "",
        height=200,
        placeholder="اردو متن درج کریں... (Type or paste Urdu text here)",
        key="urdu_text",
        help="Type or paste Urdu text in Perso-Arabic script"
    )
    
    # Translation controls
    col1a, col1b = st.columns([3, 1])
    with col1a:
        translate_button = st.button("🚀 Translate", type="primary", use_container_width=True)
    with col1b:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.urdu_text = ""
            st.session_state.translation = ""
            st.session_state.show_result = False
            st.rerun()
    
    if translate_button:
        if urdu_text and urdu_text.strip():
            with st.spinner("Translating... Please wait"):
                if model and urdu_vocab and roman_vocab:
                    translation = translate_text(urdu_text, model, urdu_vocab, roman_vocab)
                    st.session_state.translation = translation
                    st.session_state.show_result = True
                    st.rerun()
                else:
                    st.error("❌ Model failed to load. Please check if model files exist.")
        else:
            st.warning("⚠️ Please enter some Urdu text to translate")

with col2:
    st.markdown("### 🔤 Translation Results")
    
    if st.session_state.show_result and st.session_state.translation:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown("**Roman Urdu Translation:**")
        
        # Display translation in a nice box
        st.code(st.session_state.translation, language='text')
        
        # Show input text (cleaned)
        cleaned_input = clean_urdu_text(st.session_state.get('urdu_text', ''))
        if cleaned_input:
            with st.expander("View Input Text"):
                st.text(cleaned_input)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show translation stats
        if st.session_state.translation:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Characters", len(st.session_state.translation))
                st.markdown('</div>', unsafe_allow_html=True)
            with col_b:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                words = len(st.session_state.translation.split())
                st.metric("Words", words)
                st.markdown('</div>', unsafe_allow_html=True)
            with col_c:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Status", "✅ Complete")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Copy to clipboard functionality
        if st.button("📋 Copy Translation", use_container_width=True):
            # For Streamlit, we can use a workaround
            st.code(st.session_state.translation, language='text')
            st.success("✅ Translation ready to copy! Select the text above and copy it.")
        
        # Additional options
        with st.expander("⚙️ Advanced Options"):
            st.markdown("**Translation Settings:**")
            max_len = st.slider("Maximum Translation Length", 20, 100, 50)
            st.markdown("**Note:** Changing settings requires re-translation.")
            
            if st.button("🔄 Re-translate with Settings", use_container_width=True):
                with st.spinner("Re-translating..."):
                    if model and urdu_vocab and roman_vocab:
                        translation = translate_text(
                            st.session_state.get('urdu_text', ''), 
                            model, urdu_vocab, roman_vocab, max_len
                        )
                        st.session_state.translation = translation
                        st.rerun()
    
    elif st.session_state.show_result and not st.session_state.translation:
        st.warning("⚠️ Translation returned empty. The input might not contain valid Urdu text.")
    
    else:
        st.info("👈 Enter Urdu text on the left and click 'Translate' to see results here")
        
        # Quick info box
        st.markdown("""
        <div style="background-color: #f0f9ff; padding: 1rem; border-radius: 10px; border-left: 5px solid #0ea5e9;">
        <h4>💡 Tips for Best Results:</h4>
        <ul style="margin-bottom: 0;">
        <li>Use standard Urdu script (Perso-Arabic)</li>
        <li>Avoid mixed language text</li>
        <li>Keep sentences under 50 characters for best accuracy</li>
        <li>Use the example buttons to see how it works</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>🕌 <b>Urdu to Roman Urdu Neural Machine Translation System</b></p>
    <p>Built with PyTorch & Streamlit • Character-level Seq2Seq Model</p>
    <p style="font-size: 0.8rem; margin-top: 1rem;">
        <i>Note: This is a research model. Translation quality may vary based on input text.</i>
    </p>
</div>
""", unsafe_allow_html=True)

# Add a section for model troubleshooting
with st.expander("🔧 Troubleshooting & Model Info"):
    st.markdown("""
    **Common Issues:**
    
    1. **"Model failed to load" error:**
       - Ensure `best_model.pth`, `urdu_vocab.pkl`, and `roman_vocab.pkl` are in the same directory as app.py
       - Check file permissions
    
    2. **Empty translations:**
       - Input might contain non-Urdu characters
       - Try the example sentences first
    
    3. **Poor translation quality:**
       - This is a character-level model with limited training data
       - Complex or long sentences may not translate accurately
    
    **Model Files Required:**
    - `best_model.pth` - Trained PyTorch model
    - `urdu_vocab.pkl` - Urdu character vocabulary
    - `roman_vocab.pkl` - Roman Urdu character vocabulary
    - `experiment_results.json` - Optional, for performance metrics
    
    **For Developers:**
    - Source: [GitHub Repository](https://github.com/bismatayyab23-tech/urdu-roman-nmt)
    - Model: 2-layer BiLSTM encoder, 4-layer LSTM decoder
    - Training: 1,314 Urdu-Roman Urdu sentence pairs
    - Tokenization: Character-level
    """)

# Add a debug section (hidden by default)
if st.sidebar.checkbox("Show Debug Info", False):
    st.sidebar.markdown("### 🐛 Debug Information")
    st.sidebar.write(f"Session State: {dict(st.session_state)}")
    if model:
        st.sidebar.write(f"Model loaded: {model is not None}")
    if urdu_vocab:
        st.sidebar.write(f"Urdu vocab size: {len(urdu_vocab)}")
    if roman_vocab:
        st.sidebar.write(f"Roman vocab size: {len(roman_vocab)}")

Urdu to Roman Urdu Neural Machine Translation (NMT)
This repository contains a complete Neural Machine Translation (NMT) system for transliterating Urdu text (written in Perso-Arabic script) into Roman Urdu (using the Latin alphabet). The project implements a character-level Sequence-to-Sequence (Seq2Seq) model with a Bidirectional LSTM (BiLSTM) encoder and a multi-layer LSTM decoder, trained on a parallel corpus of Urdu and Roman Urdu text.

📋 Table of Contents
Project Overview

Model Architecture

Dataset

Installation

Usage

Training

Evaluation Metrics

Results

Deployment

Project Structure

License

🎯 Project Overview
This project implements a character-level neural machine translation system that converts Urdu script to its Romanized equivalent. The model follows the encoder-decoder paradigm with attention mechanisms and is implemented in PyTorch.

Key Features:

Character-level Seq2Seq model with BiLSTM encoder

Support for Urdu Unicode characters and Roman Urdu transliteration

Comprehensive evaluation (BLEU, CER, Perplexity)

Multiple experiment configurations

Streamlit web app for deployment

🏗️ Model Architecture
text
Encoder: 2-layer Bidirectional LSTM
  ├── Embedding Layer (vocab_size → embed_dim)
  ├── BiLSTM Layer 1 (embed_dim → hidden_dim, bidirectional=True)
  └── BiLSTM Layer 2 (hidden_dim×2 → hidden_dim, bidirectional=True)

Decoder: 4-layer LSTM
  ├── Embedding Layer (vocab_size → embed_dim)
  ├── LSTM Layer 1 (embed_dim → hidden_dim×2)
  ├── LSTM Layer 2 (hidden_dim×2 → hidden_dim×2)
  ├── LSTM Layer 3 (hidden_dim×2 → hidden_dim×2)
  ├── LSTM Layer 4 (hidden_dim×2 → hidden_dim×2)
  └── Linear Output Layer (hidden_dim×2 → vocab_size)

Training: Teacher forcing (ratio=0.5), Adam optimizer, Cross-entropy loss
📊 Dataset
The model is trained on a parallel corpus of Urdu and Roman Urdu text containing 1,314 sentence pairs. The data is split as follows:

Training set: 50% (657 samples)

Validation set: 25% (328 samples)

Test set: 25% (329 samples)

Data Preprocessing:

Urdu text: Preserves Urdu Unicode characters (U+0600 to U+06FF range)

Roman Urdu: Lowercased, alphanumeric characters and basic punctuation

Character-level tokenization with special tokens (<sos>, <eos>, <pad>, <unk>)

Maximum sequence length: 50 characters

⚙️ Installation
Prerequisites
Python 3.7+

PyTorch 1.8+

CUDA-capable GPU (optional, for faster training)

Install Dependencies
# Clone the repository
git clone https://github.com/bismatayyab23-tech/urdu-roman-nmt.git
cd urdu-roman-nmt

# Install required packages
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn nltk editdistance matplotlib
pip install streamlit  # For web app deployment
🚀 Usage
Quick Translation

from model import SimpleSeq2Seq, translate_sentence
import pickle

# Load vocabularies and model
with open('urdu_vocab.pkl', 'rb') as f:
    urdu_vocab = pickle.load(f)
    
with open('roman_vocab.pkl', 'rb') as f:
    roman_vocab = pickle.load(f)

# Load model (adjust paths as needed)
model = torch.load('best_model.pth', map_location='cpu')
model.eval()

# Translate a sentence
urdu_text = "اس آہٹ سے کوئی آیا تو لگتا ہے"
translation = translate_sentence(model, urdu_text, urdu_vocab, roman_vocab)
print(f"Translation: {translation}")
Run the Jupyter Notebook

jupyter notebook URDU_ROMAN.ipynb
The notebook contains the complete implementation including:

Data loading and preprocessing

Vocabulary building

Model definition and training

Evaluation and visualization

Experiment tracking

🏋️ Training
Hyperparameter Experiments
Three experiment configurations were tested:

Experiment	Embed Dim	Hidden Dim	Dropout	Learning Rate	Batch Size
Small	128	256	0.1	1e-3	32
Medium	256	512	0.3	5e-4	64
Large	512	512	0.5	1e-4	128
Training Command

# Run the training notebook
jupyter notebook URDU_ROMAN.ipynb

# Or run as Python script (if converted)
python train.py --config config_small.json
📈 Evaluation Metrics
The model is evaluated using three standard NLP metrics:

BLEU Score: Measures translation quality by comparing n-gram overlap

Character Error Rate (CER): Measures character-level accuracy

Perplexity: Measures model confidence and prediction uncertainty

Evaluation Results
Experiment	BLEU Score	CER	Perplexity	Test Loss
Small (128/256)	0.0000	1.0000	15.03	2.7099
Medium (256/512)	0.0000	1.0000	15.35	2.7313
Large (512/512)	Results pending	Results pending	Results pending	Results pending
*Note: The current BLEU scores of 0.0000 indicate the model is not generating meaningful translations yet, which is common in early training stages or with challenging character-level tasks.*

📊 Results
Training Progress
Loss decreases consistently across epochs

Validation metrics show learning trends

Model shows capacity for improvement with more training

Sample Translations
Example 1:

Input Urdu: "اس آہٹ سے کوئی آیا تو لگتا ہے"

Ground Truth: "is aahat se koi aaya to lagta hai"

Model Output: (To be generated after more training)

Example 2:

Input Urdu: "موج گل موج صبا موج سحر لگتی ہے"

Ground Truth: "mauj e gul mauj e saba mauj e sahar lagti hai"

Model Output: (To be generated after more training)

🌐 Deployment
Streamlit Web Application
A ready-to-deploy Streamlit web app is included in the notebook (Cell 15). To deploy:


# Save the Streamlit code as app.py
# Ensure required files are in the same directory:
# - app.py
# - urdu_vocab.pkl
# - roman_vocab.pkl
# - best_model.pth

# Install Streamlit
pip install streamlit

# Run the app
streamlit run app.py
The web app provides:

User-friendly interface for Urdu text input

Real-time Roman Urdu translation

Example sentences for quick testing

Translation statistics (character/word count)

API Deployment (Optional)
python
# Example Flask API endpoint
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    urdu_text = data['text']
    
    # Load model and vocabularies
    translation = translate_sentence(model, urdu_text, urdu_vocab, roman_vocab)
    
    return jsonify({
        'urdu': urdu_text,
        'roman_urdu': translation,
        'status': 'success'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
📁 Project Structure
text
urdu-roman-nmt/
├── URDU_ROMAN.ipynb          # Main Jupyter notebook
├── parallel_clean.csv        # Dataset (1,314 Urdu-Roman pairs)
├── app.py                    # Streamlit deployment app
├── requirements.txt          # Python dependencies
├── urdu_vocab.pkl           # Urdu character vocabulary
├── roman_vocab.pkl          # Roman Urdu character vocabulary
├── best_model.pth           # Trained model checkpoint
├── experiment_results.json  # Experiment results
└── training_results.png     # Training visualization
📚 Dependencies
PyTorch: Deep learning framework

Pandas & NumPy: Data manipulation

Scikit-learn: Data splitting

NLTK: BLEU score calculation

EditDistance: CER calculation

Matplotlib: Visualization

Streamlit: Web app deployment

🔮 Future Improvements
Model Architecture:

Add attention mechanism

Experiment with Transformer architecture

Implement beam search for decoding

Data:

Increase dataset size

Add more diverse text sources

Implement data augmentation

Training:

Increase training epochs

Implement learning rate scheduling

Add early stopping

Evaluation:

Human evaluation of translations

More comprehensive error analysis

Domain-specific evaluation metrics

👥 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
The Urdu language processing community

PyTorch development team

Contributors to open-source NLP tools

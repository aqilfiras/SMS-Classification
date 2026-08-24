# SMS Spam Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aqilfiras/SMS-Classification/blob/main/SMS_Spam_CLS.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A deep learning project that classifies SMS messages as **spam** or **ham** (legitimate) using sentence embeddings from a pretrained transformer model and a custom PyTorch MLP classifier.

## Overview

This project demonstrates how to leverage state-of-the-art sentence embeddings for binary text classification. Instead of using traditional NLP features (bag-of-words, TF-IDF), it encodes each SMS message into a dense 384-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer and then trains a multi-layer perceptron (MLP) to classify them.

### Pipeline

1. **Data Loading & Preprocessing** — Load the Kaggle SMS Spam Collection dataset, clean columns, and encode labels (`ham → 0`, `spam → 1`).
2. **Sentence Embedding** — Encode all SMS messages into 384-d vectors using [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
3. **Train/Validation/Test Split** — Stratified 70/10/20 split with `random_state=42` for reproducibility.
4. **Feature Scaling** — Apply `StandardScaler` to normalize the embedding features across all splits.
5. **Class Imbalance Handling** — Compute `pos_weight` from the training set to address the ham/spam imbalance via weighted `BCEWithLogitsLoss`.
6. **Model Training** — Train a 3-layer MLP (`384 → 128 → 32 → 1`) with ReLU activations, dropout (0.3), and Adam optimizer (lr=0.001, weight_decay=1e-4). Training uses **early stopping** (patience=10) based on validation loss, saving the best model checkpoint.
7. **Evaluation** — Generate predictions on the test set and report precision, recall, F1-score, confusion matrix, ROC curve (with AUC), and precision-recall curve (with AP).

## Results

The model achieves **~98% accuracy** on the test set with strong ROC and PR AUC scores:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Ham   | 0.99      | 0.98   | 0.99     | 966     |
| Spam  | 0.89      | 0.96   | 0.92     | 149     |
| **Weighted Avg** | **0.98** | **0.98** | **0.98** | **1115** |

| Metric   | Score  |
|----------|--------|
| ROC AUC  | 0.9919 |
| PR AUC   | 0.9800 |
| Accuracy | 0.9785 |

## Dataset

The project uses the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from Kaggle, which contains **5,572 SMS messages** labeled as:

- **Ham** (legitimate): 4,825 messages (86.6%)
- **Spam**: 747 messages (13.4%)

The dataset is included in the repository as `spam.csv`.

## Model Architecture

```
SpamClassifierMLP(
  (hidden1): Linear(in_features=384, out_features=128)
  (relu):    ReLU()
  (dropout): Dropout(p=0.3)
  (hidden2): Linear(in_features=128, out_features=32)
  (output):  Linear(in_features=32, out_features=1)
)
```

- **Input**: 384-dimensional sentence embeddings from `all-MiniLM-L6-v2`
- **Hidden layers**: 128 → 32 neurons, each followed by ReLU + Dropout (0.3)
- **Output**: Single logit, converted to probability via sigmoid at inference
- **Loss**: `BCEWithLogitsLoss` with `pos_weight` for class imbalance
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Early Stopping**: Patience of 10 epochs, monitoring validation loss
- **Checkpointing**: Best model saved as `best_model.pt`

## Inference

After training (or loading the saved `best_model.pt`), you can classify new SMS messages:

```python
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

# --- 1. Define the model class (must match training) ---
class SpamClassifierMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = torch.nn.Linear(384, 128)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.hidden2 = torch.nn.Linear(128, 32)
        self.output = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = self.dropout(self.relu(self.hidden1(x)))
        x = self.dropout(self.relu(self.hidden2(x)))
        return self.output(x)

# --- 2. Load the encoder and classifier ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

classifier = SpamClassifierMLP().to(device)
classifier.load_state_dict(torch.load("best_model.pt", map_location=device))
classifier.eval()

# --- 3. Classify a message ---
# Note: in production you should also load/fit the same StandardScaler used during training.
messages = ["Congratulations! You've won a free iPhone. Click here now!"]
embeddings = encoder.encode(messages, convert_to_tensor=True)

with torch.no_grad():
    logits = classifier(embeddings)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    labels = ["Spam" if p >= 0.5 else "Ham" for p in probs]

for msg, label, prob in zip(messages, labels, probs):
    print(f"[{label} ({prob:.2%})] {msg}")
# Output: [Spam (99.27%)] Congratulations! You've won a free iPhone. Click here now!
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended, but CPU works too)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Google Colab (Recommended)

Click the **Open in Colab** badge at the top to run the notebook directly in Google Colab with free GPU access.

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aqilfiras/SMS-Classification.git
   cd SMS-Classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**:
   ```bash
   jupyter notebook SMS_Spam_CLS.ipynb
   ```

## Project Structure

```
SMS-Classification/
├── SMS_Spam_CLS.ipynb   # Main notebook with full pipeline
├── spam.csv             # SMS Spam Collection dataset
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

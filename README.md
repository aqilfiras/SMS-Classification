# SMS Spam Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aqilfiras/SMS-Classification/blob/main/SMS_Spam_CLS.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A deep learning project that classifies SMS messages as **spam** or **ham** (legitimate) using sentence embeddings from a pretrained transformer model and a custom PyTorch neural network classifier.

## Overview

This project demonstrates how to leverage state-of-the-art sentence embeddings for binary text classification. Instead of using traditional NLP features (bag-of-words, TF-IDF), it encodes each SMS message into a dense 1024-dimensional vector using the `all-roberta-large-v1` sentence transformer and then trains a lightweight neural network to classify them.

### Pipeline

1. **Data Loading & Preprocessing** — Load the Kaggle SMS Spam Collection dataset, clean columns, and encode labels (`ham → 0`, `spam → 1`).
2. **Sentence Embedding** — Encode all SMS messages into 1024-d vectors using [`sentence-transformers/all-roberta-large-v1`](https://huggingface.co/sentence-transformers/all-roberta-large-v1).
3. **Train/Test Split** — 80/20 split with `random_state=42` for reproducibility.
4. **Feature Scaling** — Apply `StandardScaler` to normalize the embedding features.
5. **Model Training** — Train a 2-layer neural network (1024 → 16 → 1) with ReLU activation and `BCEWithLogitsLoss` for 100 epochs using the Adam optimizer.
6. **Evaluation** — Generate predictions on the test set and report precision, recall, F1-score, and a confusion matrix.

## Results

The model achieves **99% accuracy** on the test set:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Ham   | 1.00      | 0.99   | 1.00     | 965     |
| Spam  | 0.97      | 0.97   | 0.97     | 150     |
| **Weighted Avg** | **0.99** | **0.99** | **0.99** | **1115** |

## Dataset

The project uses the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from Kaggle, which contains **5,572 SMS messages** labeled as:

- **Ham** (legitimate): 4,825 messages (86.6%)
- **Spam**: 747 messages (13.4%)

The dataset is included in the repository as `spam.csv`.

## Model Architecture

```
LogisticRegression(
  (hidden): Linear(in_features=1024, out_features=16)
  (relu):   ReLU()
  (output): Linear(in_features=16, out_features=1)
  (sigmoid): Sigmoid()    # used at inference time
)
```

- **Input**: 1024-dimensional sentence embeddings
- **Hidden layer**: 16 neurons with ReLU activation
- **Output**: Single logit, converted to probability via sigmoid
- **Loss**: `BCEWithLogitsLoss`
- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 100

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

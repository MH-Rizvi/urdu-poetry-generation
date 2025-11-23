# Urdu Poetry Generation

## Project Overview
This project focuses on **comparative analysis of sequence-to-sequence models** (RNN, LSTM, Transformer) combined with different optimization algorithms (Adam, RMSprop, SGD) for **Urdu poetry text generation**. The goal is to generate coherent, grammatically correct, and contextually appropriate Urdu poetry while evaluating both **quantitative** and **qualitative** performance.

## Dataset
- **Source:** [ReySajju742/Urdu-Poetry-Dataset](https://huggingface.co/datasets/ReySajju742/Urdu-Poetry-Dataset)
- **Total poems:** 1,323
- **Content:** Poetry from classical poets such as Ghalib and Iqbal
- **Format:** Title and content pairs
- **Size:** 1.38 MB

## Project Methodology

### 1. Data Preprocessing
- Load dataset from Hugging Face
- Extract individual lines from poems
- Tokenization using Keras Tokenizer
- Vocabulary creation
- Sequence generation (n-grams)
- Padding sequences to uniform length
- Train-validation-test split (80-10-10)

### 2. Model Architectures
- Simple RNN
- LSTM (Long Short-Term Memory)
- Transformer

### 3. Optimization Algorithms
- Adam
- RMSprop
- SGD with Momentum

### 4. Training Configuration
- Epochs: 20–30 (with early stopping)
- Batch Size: 128
- Early Stopping: patience=5 on validation loss
- Additional hyperparameters will be tuned per experiment

### 5. Evaluation
- Quantitative: Loss, Perplexity, Accuracy, BLEU / ROUGE / METEOR (optional)
- Qualitative: Rhyme, Meter, Human evaluation

## Folder Structure
urdu-poetry-project/
│
├── data/ # Raw and processed dataset
├── notebooks/ # Jupyter notebooks for experiments
├── models/ # Saved models and checkpoints
├── results/ # Training metrics, generated poetry, plots
├── logs/ # TensorBoard or other logs
└── main.py # Main training and evaluation script


## Getting Started

### 1. Create Virtual Environment
```bash
python -m venv urdu-poetry-env
# Activate the environment:
# Linux/Mac
source urdu-poetry-env/bin/activate
# Windows
urdu-poetry-env\Scripts\activate
 ```

2. Install Dependencies
pip install tensorflow pandas numpy datasets scikit-learn matplotlib

3. Check GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

Notes

This project is designed to be fully reproducible.

GPU acceleration is recommended for training deep models.

Each model-optimizer combination will be logged and compared systematically.
# Urdu Poetry Generation

## Project Overview
This project explores **Urdu poetry text generation** using sequence models. We implement and compare **Simple RNN, LSTM, and Transformer models** with different optimization algorithms (**Adam, RMSprop, SGD**) to generate coherent, contextually relevant, and grammatically accurate Urdu poetry.  

The primary goal is to evaluate models both **quantitatively** (loss, perplexity, accuracy) and **qualitatively** (rhyme, meter, human evaluation).

---

## Dataset
- **Source:** [Hugging Face - ReySajju742/Urdu-Poetry-Dataset](https://huggingface.co/datasets/ReySajju742/Urdu-Poetry-Dataset)  
- **Total poems:** 1,323  
- **Content:** Classical Urdu poetry (Ghalib, Iqbal, others)  
- **Format:** Title and content pairs  
- **Size:** 1.38 MB  

**Preprocessing steps:**
1. Extract individual lines from poems  
2. Character-level tokenization using Keras Tokenizer  
3. Vocabulary creation  
4. Sequence generation using n-grams  
5. Padding sequences to uniform length  
6. Train-validation-test split: 80%-10%-10%  

---

## Project Methodology

### Model Architectures
- **Simple RNN** – Baseline model for sequential prediction  
- **LSTM (Long Short-Term Memory)** – Handles longer dependencies  
- **Transformer** – Attention-based sequence modeling  

### Optimization Algorithms
- **Adam**  
- **RMSprop**  
- **SGD with Momentum**  

### Training Configuration
- **Epochs:** 20–30 (with early stopping on validation loss)  
- **Batch Size:** 128  
- **Early Stopping:** Patience = 5  
- **Device:** GPU recommended for faster training  

---

## Evaluation

### Quantitative Metrics
- Training & Validation Loss  
- Test Perplexity  
- Accuracy (character prediction)  
- Optional: BLEU, ROUGE, METEOR (for advanced evaluation)  

### Qualitative Metrics
- Rhyme, meter, and flow of generated poetry  
- Human evaluation of coherence and style  

---

## Simple RNN Results

**Test Loss / Perplexity:**

| Optimizer | Test Loss | Perplexity |
|-----------|-----------|------------|
| Adam      | 1.6893    | 5.42       |
| RMSprop   | 1.7041    | 5.50       |
| SGD       | 1.7978    | 6.04       |

**Sample Generated Poetry (Seed: `"دل کی بات "`):**

- **Adam:**
دل کی بات چل لیا سی پھر باقی کی کاش میں کوئی یہ روح یا کے لیے اسی سن آمد کیا چیز میں پھر یہ تو تو میں نہ ہوا کر پھر بنا لہو دکھ بھی دعا دام میں پردۂ غفلہ و سوا دل سے مرحمندۂ رکھتا اس شیشہ سے بے حات بھلا گیا بھ


- **RMSprop:**
دل کی بات پہلے وہ تیز نہیں کیجے انساں جو پر ہے تیرے نہ نہیں اسے اے آنکھیں جو پردے کہ جھوٹ کے بال کا میں نے اب تو گرچہ و تیم آتک بھی اگر ان جاتی گا ترے گرش بن دے گی کہ کہیں اس کے مت دیا جتنے سے برباد راتے میں ہے


- **SGD:**
دل کی بات وہ سکائیں قیاط آتما تھا شرم ترا مانوں سے امشوف خراب کا غم میں نہ کوئی مجھو ہے ہم کو اس کبھی دیکھا ہے گا تھا جاتا ہے کر کچھ بات سے نے بھی تو سکتا ہوگا نہ ہے آگاہ کو حیرت داتے ہیں کہ مجھے آہ وہ وہ گھر ب


> Note: Adam and RMSprop generally generate smoother, more coherent sequences than SGD.  

---

## Folder Structure

urdu-poetry-project/
│
├── data/           # Raw and processed dataset
├── notebooks/      # Jupyter notebooks for experiments
├── models/         # Saved models and checkpoints
├── results/        # Training metrics, plots, generated poetry
├── logs/           # TensorBoard or other logs
├── scripts/        # Training and evaluation scripts
└── main.py         # Entry point for training & evaluation


---

## Getting Started

### 1. Setup Virtual Environment
```bash
python -m venv urdu-poetry-env
# Activate environment
# Linux/Mac
source urdu-poetry-env/bin/activate
# Windows
urdu-poetry-env\Scripts\activate

```

### 2. Install Dependencies
```bash
pip install tensorflow torch pandas numpy datasets scikit-learn matplotlib tqdm
```

### 3. Verify GPU
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Notes

All experiments are fully reproducible.

GPU acceleration is strongly recommended for training deep learning models.

Each model-optimizer combination is saved, logged, and compared systematically.

Results include both quantitative metrics and generated poetry samples.

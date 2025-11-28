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
- Training time comparison (minutes)  

### Qualitative Metrics
- Rhyme, meter, and flow of generated poetry  
- Human evaluation of coherence and style  

---

## Model Performance

### Perplexity Comparison (Test Set)

| Model | Optimizer | Test Loss | Perplexity | Training Time (mins) |
|-------|-----------|-----------|------------|--------------------|
| RNN | Adam | 1.6948 | 5.445 | 300 |
| RNN | RMSprop | 1.703 | 5.491 | 300 |
| RNN | SGD | 1.6898 | 5.418 | 300 |
| RNN 2 Layers Dropout 0.2 | Adam | 1.6798 | 5.36 | 1200 |
| LSTM | Adam | 1.5854 | 4.881 | 2400 |
| LSTM | RMSprop | 1.6406 | 5.158 | 1800 |
| LSTM | SGD | 1.5899 | 4.904 | 3420 |
| LSTM 3 Layers | Adam | 1.5563 | 4.742 | 4800 |
| Transformer Set1 | Adam | 1.7194 | 5.581 | 6000 |
| Transformer Set2 | Adam | 1.6902 | 5.420 | 10800 |

> Key findings:  
> - LSTM consistently outperformed RNN and Transformer in perplexity and accuracy.  
> - Transformers took the longest to train but achieved competitive perplexity on smaller architectures.  
> - SGD often resulted in slightly worse performance for both RNN and Transformer.  

---

### Visualizations
- **Perplexity Comparison Plot**  
- **Training Time Comparison Plot**  
- **Perplexity Heatmap by Model and Optimizer**

All plots are saved in the `visualizations/` folder as PNG files. These clearly show the trade-off between performance and training time across models and optimizers.

---

## Sample Generated Poetry

**Seed:** `"دل کی بات "`  

**RNN (Adam):**  
دل کی بات چل لیا سی پھر باقی کی کاش میں کوئی یہ روح یا کے لیے اسی سن آمد کیا چیز میں پھر یہ تو تو میں نہ ہوا کر پھر بنا لہو دکھ بھی دعا دام میں پردۂ غفلہ و سوا دل سے مرحمندۂ رکھتا اس شیشہ سے بے حات بھلا گیا بھ

**LSTM (Adam):**  
محبت نہیں ہوئی کہ آئی ہے مہرباں کیوں ہے کہ اس
کو ملنا نہ یہ تو مرے بعد منصفی سے ہم نے تو ان
تیز ہے انتظار نہیں ہوتی ان کی آنکھوں میں کچھ
کوئی کہانی تھی وہ بھی سمجھتے تھے پھر تو دیکھو
تو غالبؔ میں کسی سے ک

**Transformer (Adam):**  
دل کی بات وہ سب کچھ کہتا ہے، لیکن وقت کی رہنمائی میں ہم کہیں کھو جاتے ہیں، اور ہر لمحہ جو گزرتا ہے، نئے اشعار کی شکل اختیار کر لیتا ہے  

> Note: LSTM generated the most coherent and stylistically consistent sequences.  

---

## Folder Structure

urdu-poetry-project/  
│  
├── data/           # Raw and processed dataset  
├── models/         # Saved models and checkpoints  
├── results/        # Training metrics, plots, generated poetry  
├── visualizations/ # PNG plots for comparison   
├── scripts/        # Training and evaluation scripts  
└── requirements.txt   # necessary imports 

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
pip install tensorflow torch pandas numpy datasets scikit-learn matplotlib seaborn tqdm
```
### 3. Verify GPU
``` bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

```
Notes

All experiments are fully reproducible.

GPU acceleration is strongly recommended for LSTM and Transformer training.

Each model-optimizer combination is saved, logged, and compared systematically.

Both quantitative metrics and qualitative generated poetry are included in the results.

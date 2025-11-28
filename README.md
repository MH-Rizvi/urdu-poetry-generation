# Urdu Poetry Generation

This project explores **Urdu poetry text generation** using
state-of-the-art sequence models. The goal is to generate coherent,
stylistically rich, and meaningful Urdu poetry using RNN, LSTM, and
Transformer architectures, trained on a curated classical poetry
dataset.

The project includes **quantitative evaluation (loss, perplexity,
accuracy)** and **qualitative evaluation (flow, rhyme, human review)**
to compare model performance across architectures and optimizers.

## 📚 Dataset

**Source:** HuggingFace -- `ReySajju742/Urdu-Poetry-Dataset`\
**Total Poems:** 1,323\
**Content:** Classical Urdu poetry (Ghalib, Iqbal, Mir, etc.)\
**Size:** 1.38 MB

### Preprocessing Steps

-   Extract individual lines from poems\
-   Character-level tokenization using Keras Tokenizer\
-   Build vocabulary\
-   Create n-gram sequences for next-character prediction\
-   Pad sequences to uniform length\
-   Train/Val/Test split: **80% / 10% / 10%**

## 🧠 Models Implemented

### **1. Simple RNN**

-   Baseline sequential model\
-   Fast but struggles with long-term dependencies

### **2. LSTM (Long Short-Term Memory)**

-   Captures longer dependencies\
-   Best overall performance in this project

### **3. Transformer**

-   Self-attention architecture\
-   High training cost\
-   Competitive perplexity on tuned versions

## ⚙ Optimization Algorithms

-   **Adam**
-   **RMSprop**
-   **SGD with Momentum**

## 🏗 Training Configuration

-   **Epochs:** 20--30 (Early Stopping enabled)\
-   **Batch Size:** 128\
-   **Early Stopping Patience:** 5\
-   **Hardware:** GPU recommended (LSTM/Transformer especially)

## 📊 Evaluation Metrics

### Quantitative

-   Train & Validation Loss\
-   Test Perplexity\
-   Accuracy (character prediction)\
-   Training Time (minutes)

### Qualitative

-   Rhyme quality\
-   Meter consistency\
-   Stylistic coherence\
-   Human evaluation

## 📈 Model Performance Summary

### **Perplexity Comparison (Test Set)**

  Model                         Optimizer   Test Loss    Perplexity   Training Time (mins)
  ----------------------------- ----------- ------------ ------------ ----------------------
  RNN                           Adam        1.6948       5.445        300
  RNN                           RMSprop     1.703        5.491        300
  RNN                           SGD         1.6898       5.418        300
  RNN (2-Layer + Dropout 0.2)   Adam        1.6798       5.36         1200
  LSTM                          Adam        1.5854       4.881        2400
  LSTM                          RMSprop     1.6406       5.158        1800
  LSTM                          SGD         1.5899       4.904        3420
  LSTM (3-Layer)                Adam        **1.5563**   **4.742**    4800
  Transformer (Set1)            Adam        1.7194       5.581        6000
  Transformer (Set2)            Adam        1.6902       5.420        10800

## 🎨 Visualizations

All plots are available in the `visualizations/` folder:

-   **perplexity_comparison.png**
-   **training_time_comparison.png**
-   **perplexity_heatmap.png**

## ✍ Sample Generated Poetry

### Seed: **"دل کی بات"**

#### **RNN (Adam)**

دل کی بات چل لیا سی پھر باقی کی کاش میں کوئی یہ روح یا کے لیے اسی سن آمد
کیا چیز میں پھر یہ تو تو میں نہ ہوا کر پھر بنا لہو دکھ بھی دعا دام میں
پردۂ غفلہ و سوا دل سے مرحمندۂ رکھتا اس شیشہ سے بے حات بھلا گیا بھ

#### **LSTM (Adam)**
محبت نہیں ہوئی کہ آئی ہے مہرباں کیوں ہے کہ اس
کو ملنا نہ یہ تو مرے بعد منصفی سے ہم نے تو ان
تیز ہے انتظار نہیں ہوتی ان کی آنکھوں میں کچھ
کوئی کہانی تھی وہ بھی سمجھتے تھے پھر تو دیکھو
تو غالبؔ میں کسی سے ک
#### **Transformer (Adam)**

دل کی بات وہ سب کچھ کہتا ہے، لیکن وقت کی رہنمائی میں ہم کہیں کھو جاتے
ہیں،\
اور ہر لمحہ جو گزرتا ہے، نئے اشعار کی شکل اختیار کر لیتا ہے

## 📁 Folder Structure

    urdu-poetry-project/
    │
    ├── data/                # Raw and processed data
    ├── notebooks/           # Jupyter notebooks for experiments
    ├── models/              # Saved models & checkpoints
    ├── results/             # Metrics, generated poetry, logs
    ├── visualizations/      # PNG plots
    ├── logs/                # TensorBoard logs
    ├── scripts/             # Training & evaluation scripts
    └── main.py              # Entry point

## 🚀 Getting Started

### 1. Create Virtual Environment

    python -m venv urdu-poetry-env
    source urdu-poetry-env/bin/activate  # Linux/Mac
    urdu-poetry-env\Scripts\activate   # Windows

### 2. Install Dependencies

    pip install tensorflow torch pandas numpy datasets scikit-learn matplotlib seaborn tqdm

### 3. Verify GPU

    python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

## 📌 Future Work

-   Word-level tokenization for richer semantics\
-   Hyperparameter tuning with Optuna\
-   Fine-tune GPT-based architectures for Urdu\
-   Meter detection & automatic rhyme scoring

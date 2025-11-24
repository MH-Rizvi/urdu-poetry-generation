# 02_preprocess_data_char_keras.py

import pandas as pd
from datasets import load_dataset
import re
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

# ------------------------------
# Step 0: Ensure processed folder exists
# ------------------------------
os.makedirs("data/processed/", exist_ok=True)

# ------------------------------
# Step 1: Load dataset
# ------------------------------
dataset = load_dataset("ReySajju742/Urdu-Poetry-Dataset")
df = pd.DataFrame(dataset['train'])  # type: ignore # Hugging Face default split

# ------------------------------
# Step 2: Text cleaning
# ------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # keep only Urdu letters
    text = text.replace('ي', 'ی').replace('ك', 'ک')  # normalize letters
    return text

df['clean_content'] = df['content'].apply(clean_text)
df = df[df['clean_content'].str.strip() != ''].reset_index(drop=True)
print("Number of poems after cleaning:", len(df))
print("First cleaned poem:\n", df['clean_content'].iloc[0])

# ------------------------------
# Step 3: Character-level tokenization using Keras Tokenizer
# ------------------------------
tokenizer = Tokenizer(char_level=True, oov_token=None)  # char-level tokenizer
tokenizer.fit_on_texts(df['clean_content'].tolist())

# Save tokenizer
with open('data/processed/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

vocab_size = len(tokenizer.word_index) + 1  # +1 for padding
print("Character vocab size:", vocab_size)

# ------------------------------
# Step 4: Sequence preparation using sliding window
# ------------------------------
MAX_SEQ_LEN = 50  # characters per sequence
input_sequences = []

for line in df['clean_content']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    if len(token_list) < 2:
        continue
    for i in range(1, len(token_list)):
        start_idx = max(0, i-MAX_SEQ_LEN)
        n_gram_sequence = token_list[start_idx:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
input_sequences = pad_sequences(input_sequences, maxlen=MAX_SEQ_LEN, padding='pre')

# Split into X and y
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# Convert y to one-hot vectors
y = to_categorical(y, num_classes=vocab_size)

print("Number of sequences:", len(X))
print("Input shape:", X.shape, "Output shape:", y.shape)

# ------------------------------
# Step 5: Train/Val/Test split (80/10/10)
# ------------------------------
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1111, random_state=42
)  # 0.1111*0.9≈0.1

# Save sequences
np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/X_val.npy', X_val)
np.save('data/processed/y_val.npy', y_val)
np.save('data/processed/X_test.npy', X_test)
np.save('data/processed/y_test.npy', y_test)

print("Data splitting done: ")
print("Train:", X_train.shape[0], "Validation:", X_val.shape[0], "Test:", X_test.shape[0])

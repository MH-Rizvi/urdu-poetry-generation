# 02_preprocess_data.py

import pandas as pd
from datasets import load_dataset
import re
import os

# Step 0: Ensure processed folder exists
os.makedirs("../data/processed/", exist_ok=True)

# Step 1: Load dataset (same as Step 1)
dataset = load_dataset("ReySajju742/Urdu-Poetry-Dataset")
df = pd.DataFrame(dataset['train']) # type: ignore # Hugging Face datasets are split into 'train' by default

# Step 2: Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()                   # Remove leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text)     # Replace multiple spaces/newlines with single space
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Keep only Urdu letters and spaces
    # Normalize letters (example: convert Arabic Yeh 'ي' to Urdu Yeh 'ی')
    text = text.replace('ي', 'ی').replace('ك', 'ک')
    return text

# Step 3: Apply cleaning to content
df['clean_content'] = df['content'].apply(clean_text)

# Step 4: Filter out empty poems
df = df[df['clean_content'].str.strip() != ''].reset_index(drop=True)

# Step 5: Save cleaned dataset to CSV for future steps
df.to_csv("data/processed/cleaned_poetry.csv", index=False)

# Quick checks
print("Number of poems after cleaning:", len(df))
print("First cleaned poem:\n", df['clean_content'].iloc[0])

# ------------------------------
# Step 2.2: Tokenization
# ------------------------------
from tensorflow.keras.preprocessing.text import Tokenizer #type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore
import pickle
import numpy as np

# Initialize tokenizer
tokenizer = Tokenizer(oov_token='<OOV>')  # Use <OOV> for unknown words
tokenizer.fit_on_texts(df['clean_content'])  # Build vocabulary

# Save tokenizer for later use
with open('data/processed/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Vocabulary size:", len(tokenizer.word_index)+1)  # +1 for OOV token

# ------------------------------
# Step 2.3: Sequence Preparation
# ------------------------------
input_sequences = []

for line in df['clean_content']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    # Generate n-gram sequences
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]  # First i+1 tokens
        input_sequences.append(n_gram_sequence)

# Pad sequences to have same length
max_seq_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))

print("Number of sequences:", len(input_sequences))
print("Max sequence length:", max_seq_len)

# Split input sequences into X (inputs) and y (labels)
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# ------------------------------
# Step 2.4: Data Splitting
# ------------------------------
from sklearn.model_selection import train_test_split

# First split train+val vs test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Then split train vs validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1111, random_state=42
)
# 0.1111 x 0.9 ≈ 0.1 of total → 80/10/10 split

# Save sequences
np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/X_val.npy', X_val)
np.save('data/processed/y_val.npy', y_val)
np.save('data/processed/X_test.npy', X_test)
np.save('data/processed/y_test.npy', y_test)

print("Data splitting done: ")
print("Train:", X_train.shape[0], "Validation:", X_val.shape[0], "Test:", X_test.shape[0])
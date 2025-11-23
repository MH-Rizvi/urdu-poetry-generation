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

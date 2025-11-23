# 01_load_explore_dataset.py

# Step 1: Import required libraries
from datasets import load_dataset          # To fetch Urdu poetry dataset
import pandas as pd                       # For dataframe operations
import matplotlib.pyplot as plt           # For plots
from collections import Counter           # To count word frequencies
import re                                  # For text cleaning

# Step 2: Load the dataset from Hugging Face
dataset = load_dataset("ReySajju742/Urdu-Poetry-Dataset")

# Step 3: Convert to pandas DataFrame for easier handling
df = pd.DataFrame(dataset['train'])       # type: ignore # Hugging Face datasets are split into 'train' by default

# save raw dataset
df.to_csv("data/raw/raw_dataset.csv", index = False)
# Step 4: Basic exploration
print("Columns in dataset:", df.columns)
print("Number of poems:", len(df))
print("First 3 poems:\n", df.head(3))

# Step 5: Check for missing or null entries
print("Missing titles:", df['title'].isnull().sum())
print("Missing contents:", df['content'].isnull().sum())

# Step 6: Calculate poem lengths (number of words)
df['num_words'] = df['content'].apply(lambda x: len(str(x).split()))

# Step 7: Basic statistics
avg_words = df['num_words'].mean()
vocab = Counter()
for poem in df['content']:
    words = str(poem).split()
    vocab.update(words)

vocab_size = len(vocab)
most_common_words = vocab.most_common(20)

print(f"Average words per poem: {avg_words:.2f}")
print(f"Vocabulary size: {vocab_size}")
print("Top 20 most common words:", most_common_words)

# Step 8: Plot histogram of poem lengths
plt.figure(figsize=(10,5))
plt.hist(df['num_words'], bins=30, color='skyblue')
plt.title('Distribution of Poem Lengths (words per poem)')
plt.xlabel('Number of words')
plt.ylabel('Number of poems')
plt.grid(True)
plt.savefig("results/metrics/poem_length_distribution.png")
plt.show()
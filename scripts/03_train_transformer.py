# 03_train_transformer.py

import os
import pickle
import csv
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm
import random
from utils import compute_accuracy, evaluate_model, generate_text, save_loss_plot  # type: ignore

# ------------------------------
# Set random seeds for reproducibility
# ------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------
# Step 0: Settings
# ------------------------------
NUM_EPOCHS = 20
BATCH_SIZE = 128
PATIENCE = 5
NUM_CHARS_TO_GENERATE = 200
SEED_TEXTS = ["محبت", "دل", "شام", "یاد", "خوشی"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Using CPU. Training will be slower.")

MODEL_DIR = "models/"
RESULTS_DIR = "results/metrics/"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, "all_metrics.csv")

# ------------------------------
# Step 1: Load data + tokenizer
# ------------------------------
X_train = torch.from_numpy(np.load("data/processed/X_train.npy")).long()
y_train = torch.from_numpy(np.load("data/processed/y_train.npy")).long()
X_val = torch.from_numpy(np.load("data/processed/X_val.npy")).long()
y_val = torch.from_numpy(np.load("data/processed/y_val.npy")).long()
X_test = torch.from_numpy(np.load("data/processed/X_test.npy")).long()
y_test = torch.from_numpy(np.load("data/processed/y_test.npy")).long()

with open("data/processed/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
input_length = X_train.shape[1]

# ------------------------------
# Step 2: DataLoaders
# ------------------------------
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

# ------------------------------
# Step 3: Positional Encoding
# ------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :] # type: ignore

# ------------------------------
# Step 3b: Transformer Model
# ------------------------------
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, ff_dim=512, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
        x = self.transformer_encoder(x, mask=mask)
        return self.fc_out(x[:, -1, :])

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

# ------------------------------
# Step 4: Training + CSV logging
# ------------------------------
def train_model(model, optimizer_name, model_name, hyperparams):
    print(f"\nTraining {model_name} with {optimizer_name} optimizer...")
    model = model.to(DEVICE)

    # Optimizer selection
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    save_path = os.path.join(MODEL_DIR, f"{model_name}_{optimizer_name}")
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, total_correct = 0, 0
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")

        for xb, yb in loop:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_correct += compute_accuracy(outputs, yb) * xb.size(0)
            loop.set_postfix(loss=loss.item(), acc=f"{compute_accuracy(outputs, yb):.4f}")

        avg_train_loss = total_loss / len(train_loader.dataset)  # type: ignore
        avg_train_acc = total_correct / len(train_loader.dataset)  # type: ignore

        # Validation
        model.eval()
        val_loss_total, val_correct_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss_total += loss.item() * xb.size(0)
                val_correct_total += compute_accuracy(outputs, yb) * xb.size(0)

        avg_val_loss = val_loss_total / len(val_loader.dataset)  # type: ignore
        avg_val_acc = val_correct_total / len(val_loader.dataset)  # type: ignore

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} - "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered")
                break

    # Save history and loss plot
    np.save(os.path.join(save_path, "history.npy"), history)  # type: ignore
    save_loss_plot(history, optimizer_name, model_name, RESULTS_DIR)

    # Evaluate on test set
    test_loss, perplexity = evaluate_model(model, test_loader, DEVICE)
    print(f"Test Loss ({optimizer_name}): {test_loss:.4f}, Perplexity: {perplexity:.2f}")

    # Generate text for 3 temperatures and 5 seed words
    for temp in [0.7, 1.0, 1.3]:
        for seed in SEED_TEXTS:
            text = generate_text(
                model=model,
                tokenizer=tokenizer,
                seed_text=seed,
                input_length=input_length,
                device=DEVICE,
                num_chars=NUM_CHARS_TO_GENERATE,
                temperature=temp
            )
            filename = f"generated_{seed}_temp_{temp}.txt"
            with open(os.path.join(save_path, filename), "w", encoding="utf-8") as f:
                f.write(text)

            # Log metrics to CSV
            write_header = not os.path.exists(CSV_PATH)
            row = {
                "model": model_name,
                "optimizer": optimizer_name,
                "hyperparams": str(hyperparams),
                "seed_word": seed,
                "temperature": temp,
                "train_loss": avg_train_loss,
                "train_acc": avg_train_acc,
                "val_loss": avg_val_loss,
                "val_acc": avg_val_acc,
                "test_loss": test_loss,
                "perplexity": perplexity
            }
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

    # Clear GPU memory
    del model
    torch.cuda.empty_cache()

# ------------------------------
# Step 5: Run Transformer models
# ------------------------------
optimizers = ["adam", "rmsprop", "sgd"]

# Transformer: 2 hyperparameter settings
transformer_settings = [
    {"embed_dim":128, "num_heads":4, "ff_dim":512, "num_layers":2},
    {"embed_dim":128, "num_heads":8, "ff_dim":1024, "num_layers":2}
]

for idx, hp in enumerate(transformer_settings):
    for opt in optimizers:
        train_model(
            TransformerModel(
                vocab_size,
                embed_dim=hp["embed_dim"],
                num_heads=hp["num_heads"],
                ff_dim=hp["ff_dim"],
                num_layers=hp["num_layers"]
            ),
            opt,
            f"transformer_set{idx+1}",
            hp
        )

print("Transformer training complete. All metrics saved in:", CSV_PATH)

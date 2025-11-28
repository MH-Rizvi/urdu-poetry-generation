# 03_train_rnn.py

import os
import pickle
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm
import random
from utils import compute_accuracy, evaluate_model, generate_text, save_loss_plot  # type: ignore

# ------------------------------
# Random Seeds
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
# Settings
# ------------------------------
NUM_EPOCHS = 20
BATCH_SIZE = 128
PATIENCE = 5
NUM_CHARS_TO_GENERATE = 200
SEED_TEXTS = ["محبت", "دل", "شام", "یاد", "خوشی"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "models/RNN"
RESULTS_DIR = "results/metrics/RNN"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, "new_metrics.csv")

# ------------------------------
# Load Data
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

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

# ------------------------------
# RNN Model
# ------------------------------
class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=150, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# ------------------------------
# Training Function
# ------------------------------
def train_model(model, optimizer_name, model_name, hyperparams):
    print(f"\n==============================")
    print(f"Training model: {model_name} | Optimizer: {optimizer_name.upper()}")
    print(f"==============================\n")

    model = model.to(DEVICE)

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
        loop.set_description(f"[{optimizer_name.upper()}] Epoch {epoch+1}/{NUM_EPOCHS}")

        for xb, yb in loop:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            batch_acc = compute_accuracy(outputs, yb)
            total_correct += batch_acc * xb.size(0)

            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}")

        avg_train_loss = total_loss / len(train_loader.dataset)  # type: ignore
        avg_train_acc = total_correct / len(train_loader.dataset)  # type: ignore

        # ---- Validation ----
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

        # ---- Print Epoch Summary ----
        print(
            f"[{optimizer_name.upper()}] Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}"
        )

        # ---- Early Stopping ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[{optimizer_name.upper()}] Early stopping triggered.")
                break

    # ------------------------------
    # Save history & plot
    # ------------------------------
    np.save(os.path.join(save_path, "history.npy"), history)  # type: ignore
    save_loss_plot(history, optimizer_name, model_name, RESULTS_DIR)

    # ------------------------------
    # Test Evaluation
    # ------------------------------
    test_loss, perplexity = evaluate_model(model, test_loader, DEVICE)
    print(
        f"[{optimizer_name.upper()}] Test Loss: {test_loss:.4f} | Perplexity: {perplexity:.2f}"
    )

    # ------------------------------
    # CSV Logging
    # ------------------------------
    write_header = not os.path.exists(CSV_PATH)
    row = {
        "model": model_name,
        "optimizer": optimizer_name,
        "hyperparams": str(hyperparams),
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

    del model
    torch.cuda.empty_cache()

# ------------------------------
# Run
# ------------------------------
optimizers = ["adam", "rmsprop", "sgd"]

for opt in optimizers:
    train_model(
        model=SimpleRNNModel(vocab_size=vocab_size),
        optimizer_name=opt,
        model_name="rnn_Layers(2)_Dropout(0.2)",
        hyperparams={"embed_dim": 100, "hidden_dim": 150, "dropout": 0.2, "num_layers": 2}
    )

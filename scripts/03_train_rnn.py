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

MODEL_DIR = "models/"
RESULTS_DIR = "results/metrics/"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, "all_metrics.csv")

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
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=150):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
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
        loop = tqdm(train_loader)

        for xb, yb in loop:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_correct += compute_accuracy(outputs, yb) * xb.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset) # type: ignore
        avg_train_acc = total_correct / len(train_loader.dataset) # type: ignore

        model.eval()
        val_loss_total, val_correct_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss_total += loss.item() * xb.size(0)
                val_correct_total += compute_accuracy(outputs, yb) * xb.size(0)

        avg_val_loss = val_loss_total / len(val_loader.dataset) # type: ignore
        avg_val_acc = val_correct_total / len(val_loader.dataset) # type: ignore

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    np.save(os.path.join(save_path, "history.npy"), history) # type: ignore
    save_loss_plot(history, optimizer_name, model_name, RESULTS_DIR)

    test_loss, perplexity = evaluate_model(model, test_loader, DEVICE)

    for temp in [0.7, 1.0, 1.3]:
        for seed in SEED_TEXTS:
            text = generate_text(model, tokenizer, seed, input_length, DEVICE, NUM_CHARS_TO_GENERATE, temp)
            filename = f"generated_{seed}_temp_{temp}.txt"
            with open(os.path.join(save_path, filename), "w", encoding="utf-8") as f:
                f.write(text)

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
        model_name="rnn",
        hyperparams={"embed_dim": 100, "hidden_dim": 150}
    )

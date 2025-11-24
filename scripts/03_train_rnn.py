# 03_train_rnn.py

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

# ------------------------------
# Step 0: Settings
# ------------------------------
NUM_EPOCHS = 20
BATCH_SIZE = 128
PATIENCE = 5
NUM_CHARS_TO_GENERATE = 200
SEED_TEXT = "دل کی بات "
NUM_SAMPLES = 3
EMBED_DIM = 100
HIDDEN_DIM = 150
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check GPU
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Using CPU. Training will be slower.")

MODEL_DIR = "models/"
RESULTS_DIR = "results/metrics/"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------
# Step 1: Load preprocessed character data (from Keras Tokenizer)
# ------------------------------
X_train = torch.from_numpy(np.load("data/processed/X_train.npy")).long()
y_train = torch.from_numpy(np.load("data/processed/y_train.npy")).long()
X_val = torch.from_numpy(np.load("data/processed/X_val.npy")).long()
y_val = torch.from_numpy(np.load("data/processed/y_val.npy")).long()
X_test = torch.from_numpy(np.load("data/processed/X_test.npy")).long()
y_test = torch.from_numpy(np.load("data/processed/y_test.npy")).long()

with open("data/processed/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1  # +1 for padding
input_length = X_train.shape[1]

# ------------------------------
# Step 2: Create DataLoaders
# ------------------------------
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

# ------------------------------
# Step 3: Define RNN Model
# ------------------------------
class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SimpleRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim,batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # last time step
        return out

# ------------------------------
# Step 4: Accuracy helper
# ------------------------------
def compute_accuracy(preds, labels):
    predicted_ids = torch.argmax(preds, dim=1)
    return (predicted_ids == labels).sum().item() / labels.size(0)

# ------------------------------
# Step 5: Training function
# ------------------------------
def train_model(optimizer_name="adam"):
    print(f"\nTraining RNN with {optimizer_name} optimizer...")
    
    model = SimpleRNNModel(vocab_size, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), momentum=0.9)
    
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_correct = 0
        
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
        
        avg_train_loss = total_loss / len(train_loader.dataset) # type: ignore
        avg_train_acc = total_correct / len(train_loader.dataset) # type: ignore
        
        # Validation
        model.eval()
        val_loss_total = 0
        val_correct_total = 0
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
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} - "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(MODEL_DIR, f"rnn_{optimizer_name}")
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered")
                break
    
    # Save history
    np.save(os.path.join(save_path, "history.npy"), history) # type: ignore
    
    # Plot loss
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title(f"RNN Loss ({optimizer_name})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"rnn_{optimizer_name}_loss.png"))
    plt.close()
    
    return model

# ------------------------------
# Step 6: Evaluate
# ------------------------------
def evaluate_model(model):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(test_loader.dataset) # type: ignore
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

# ------------------------------
# Step 7: Character-level text generation using Keras Tokenizer
# ------------------------------
def generate_text(model, seed_text, temperature=0.8):
    """
    Generate text using the trained RNN model with temperature-based sampling.
    
    Args:
        model (nn.Module): Trained RNN model
        seed_text (str): Starting text for generation
        temperature (float): Controls randomness (higher = more creative)
    """
    model.eval()
    generated = list(seed_text)
    
    # Create a reverse mapping from index to character
    idx_to_char = {idx: char for char, idx in tokenizer.word_index.items()}
    
    for _ in range(NUM_CHARS_TO_GENERATE):
        token_list = tokenizer.texts_to_sequences([''.join(generated)])
        token_list = torch.tensor(token_list[0][-input_length:], dtype=torch.long).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred = model(token_list).squeeze()
        
        # Apply temperature scaling
        pred = pred / temperature
        probs = torch.softmax(pred, dim=-1)
        
        # Sample next character
        predicted_id = torch.multinomial(probs, num_samples=1).item()
        next_char = idx_to_char.get(predicted_id, '')
        generated.append(next_char)
    
    return ''.join(generated)

# ------------------------------
# Step 8: Run training for each optimizer
# ------------------------------
for opt_name in ["adam", "rmsprop", "sgd"]:
    model_path = os.path.join(MODEL_DIR, f"rnn_{opt_name}")
    os.makedirs(model_path, exist_ok=True)
    
    model = train_model(opt_name)
    
    test_loss, perplexity = evaluate_model(model)
    print(f"Test Loss ({opt_name}): {test_loss:.4f}, Perplexity: {perplexity:.2f}")
    
    for i in range(NUM_SAMPLES):
        text = generate_text(model, SEED_TEXT, temperature=0.8)
        with open(os.path.join(model_path, f"generated_sample_{i+1}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

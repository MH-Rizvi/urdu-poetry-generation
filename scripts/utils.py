# utils.py

import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Accuracy helper
# ------------------------------
def compute_accuracy(preds, labels):
    predicted_ids = torch.argmax(preds, dim=1)
    return (predicted_ids == labels).sum().item() / labels.size(0)

# ------------------------------
# Evaluation helper
# ------------------------------
def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(test_loader.dataset)
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity

# ------------------------------
# Plotting helper
# ------------------------------
def save_loss_plot(history, optimizer_name, model_name, results_dir):
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title(f"{model_name} Loss ({optimizer_name})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f"{model_name}_{optimizer_name}_loss.png"))
    plt.close()

# ------------------------------
# Text generation (reusable)
# ------------------------------
def generate_text(
    model,
    tokenizer,
    seed_text,
    input_length,
    device,
    num_chars=200,
    temperature=0.8
):
    """
    Character-level text generation with temperature scaling.
    """

    model.eval()
    generated = list(seed_text)

    # Reverse mapping: index -> char
    idx_to_char = {idx: char for char, idx in tokenizer.word_index.items()}

    for _ in range(num_chars):
        token_list = tokenizer.texts_to_sequences([''.join(generated)])
        token_list = torch.tensor(
            token_list[0][-input_length:], dtype=torch.long
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(token_list).squeeze()

        # Temperature scaling
        preds = preds / temperature
        probs = torch.softmax(preds, dim=-1)

        # Sample next char
        predicted_id = torch.multinomial(probs, num_samples=1).item()
        next_char = idx_to_char.get(predicted_id, '')

        generated.append(next_char)

    return ''.join(generated)

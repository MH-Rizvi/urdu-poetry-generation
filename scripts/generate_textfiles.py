import os
import torch
import pickle
import numpy as np
import torch.nn as nn

# ------------------------------
# Paths
# ------------------------------
MODEL_PATH = r"models/RNN/rnn_Layers(2)_Dropout(0.2)_sgd/model.pt"
SAVE_DIR = r"models/RNN/rnn_Layers(2)_Dropout(0.2)_sgd/generated_texts"
TOKENIZER_PATH = r"data/processed/tokenizer.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load tokenizer
# ------------------------------
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1

# ------------------------------
# Load X_train to get input_length
# ------------------------------
X_train = np.load("data/processed/X_train.npy")
input_length = X_train.shape[1]

# ------------------------------
# RNN Model class definition
# ------------------------------
class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=150, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            embed_dim, hidden_dim, batch_first=True,
            num_layers=num_layers, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# ------------------------------
# Model hyperparameters (set manually)
# ------------------------------
num_layers = 2   # change as needed
dropout = 0.2    # change as needed

# ------------------------------
# Instantiate model and load weights
# ------------------------------
model = SimpleRNNModel(
    vocab_size=vocab_size,
    embed_dim=100,
    hidden_dim=150,
    num_layers=num_layers,
    dropout=dropout
)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# ------------------------------
# Text generation function
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
    generated = list(seed_text)
    idx_to_char = {idx: char for char, idx in tokenizer.word_index.items()}

    for _ in range(num_chars):
        token_list = tokenizer.texts_to_sequences([''.join(generated)])
        token_list = torch.tensor(
            token_list[0][-input_length:], dtype=torch.long
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(token_list).squeeze()

        preds = preds / temperature
        probs = torch.softmax(preds, dim=-1)
        predicted_id = torch.multinomial(probs, num_samples=1).item()
        next_char = idx_to_char.get(predicted_id, "")
        generated.append(next_char)

    return ''.join(generated)

# ------------------------------
# Generation settings
# ------------------------------
NUM_CHARS_TO_GENERATE = 200
SEED_TEXTS = ["محبت", "دل", "شام", "یاد", "خوشی"]
TEMPERATURES = [0.7, 1.0, 1.3]

# ------------------------------
# Generate text files
# ------------------------------
os.makedirs(SAVE_DIR, exist_ok=True)

for temp in TEMPERATURES:
    for seed in SEED_TEXTS:
        output_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            seed_text=seed,
            input_length=input_length,
            device=DEVICE,
            num_chars=NUM_CHARS_TO_GENERATE,
            temperature=temp
        )

        filename = f"{seed}_temp_{temp}.txt"
        file_path = os.path.join(SAVE_DIR, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(output_text)

        print(f"Saved → {file_path}")

# ------------------------------
# Print model summary
# ------------------------------
print("Model Architecture:")
print(model)
print(f"Num layers: {num_layers}, Dropout: {dropout}")

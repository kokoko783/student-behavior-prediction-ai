print("PROGRAM BASLADI")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset import StudentDataset
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel  # İleride ekleyeceğin model

# --- CONFIG ---
model_choice = "LSTM"  # "LSTM" veya "Transformer"
dataset_path = "data/student_behavior.csv"
batch_size = 2
epochs = 20
lr = 0.001

# --- Dataset ---
dataset = StudentDataset(dataset_path)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Model Selection ---
if model_choice == "LSTM":
    model = LSTMModel(input_size=3, hidden_size=64, num_layers=1, num_classes=2)
elif model_choice == "Transformer":
    model = TransformerModel(input_size=3, num_classes=2)
else:
    raise ValueError(f"Unknown model_choice: {model_choice}")

# --- Loss & Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# --- Training ---
for epoch in range(epochs):
    for X, y in loader:
        X = X.unsqueeze(1)  # batch_size, seq_len=1, input_size
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch:", epoch, "Loss:", loss.item())

# --- Save Model ---
torch.save(model.state_dict(), f"{model_choice}_model.pth")
print(f"{model_choice} model saved!")
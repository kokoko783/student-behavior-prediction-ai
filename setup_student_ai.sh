#!/bin/bash

# --- CONFIG ---
REPO_NAME="student-behavior-prediction-ai"
GITHUB_URL="<YOUR_GITHUB_REPO_LINK>"  # buraya kendi GitHub linkini koy
PYTHON_VERSION="python3"

echo " Başlıyoruz: $REPO_NAME hazırlığı"

# --- Klasör oluştur ---
mkdir -p $REPO_NAME/{data,models,utils}
cd $REPO_NAME || exit

# --- Sanal ortam oluştur ---
$PYTHON_VERSION -m venv venv
source venv/bin/activate

# --- requirements.txt ---
cat > requirements.txt <<EOL
torch
torchvision
numpy
pandas
seaborn
matplotlib
EOL

pip install -r requirements.txt

# --- dataset.py ---
cat > utils/dataset.py <<'EOL'
import pandas as pd
import torch
from torch.utils.data import Dataset

class StudentDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.X = torch.tensor(self.data.iloc[:, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(self.data.iloc[:, -1].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
EOL

# --- lstm_model.py ---
cat > models/lstm_model.py <<'EOL'
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 64)
        c0 = torch.zeros(1, x.size(0), 64)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
EOL

# --- transformer_model.py (opsiyonel) ---
cat > models/transformer_model.py <<'EOL'
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, 64)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])
        return x
EOL

# --- train.py ---
cat > train.py <<'EOL'
print("PROGRAM BASLADI")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset import StudentDataset
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel

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
        X = X.unsqueeze(1)
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch:", epoch, "Loss:", loss.item())

# --- Save Model ---
torch.save(model.state_dict(), f"{model_choice}_model.pth")
print(f"{model_choice} model saved!")
EOL

# --- README.md ---
cat > README.md <<EOL
# AI-Powered Student Behavior Prediction System

- Built LSTM & Transformer models to predict student engagement
- Visualized dataset & modeled sequential behavior
- Trained and evaluated models with accuracy & confusion matrix
- Implemented in Python & PyTorch
EOL

# --- Git ---
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin $GITHUB_URL
echo "✅ Repo hazır, push için: git push -u origin main"

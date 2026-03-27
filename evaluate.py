import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils.dataset import StudentDataset
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel

# --- Dataset ---
dataset = StudentDataset("data/student_behavior.csv")
loader = DataLoader(dataset, batch_size=2, shuffle=False)

# --- Model ---
model_choice = "lstm"  # "lstm" veya "transformer"
if model_choice == "lstm":
    model = LSTMModel(3, 64, 1, 2)
else:
    model = TransformerModel(3, 2)

model.load_state_dict(torch.load(f"{model_choice}_model.pth"))
model.eval()

# --- Evaluation ---
y_true, y_pred = [], []

for X, y in loader:
    X = X.unsqueeze(1)
    with torch.no_grad():
        outputs = model(X)
    predictions = torch.argmax(outputs, dim=1)
    y_true.extend(y.tolist())
    y_pred.extend(predictions.tolist())

acc = accuracy_score(y_true, y_pred)
print("Model Accuracy:", acc)

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()
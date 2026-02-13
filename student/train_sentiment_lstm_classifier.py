# train_sentiment_lstm_classifier.py

print("\n========== Task 2: LSTM Sentiment Classifier ==========")

import os
import random
import numpy as np
import torch

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import nltk
from nltk.tokenize import word_tokenize

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from gensim.models import KeyedVectors


# -----------------------
#  Reproducibility
# -----------------------
SEED = 816
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -----------------------
#  Load dataset + split
# -----------------------
dataset = load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
texts = dataset["train"]["sentence"]
labels = dataset["train"]["label"]  # 0/1/2

X_trainval, X_test, y_trainval, y_test = train_test_split(
    texts, labels,
    test_size=0.15,
    random_state=SEED,
    stratify=labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.15,
    random_state=SEED,
    stratify=y_trainval
)


# -----------------------
# Tokenization 
# -----------------------
nltk.download("punkt", quiet=True)

def tokenize_text(text: str):
    return word_tokenize(text.lower(), preserve_line=True)


# -----------------------
# Load FastText vectors
# -----------------------

fasttext = KeyedVectors.load("fasttext-wiki-news-subwords-300.model", mmap="r")
EMB_DIM = 300
SEQ_LEN = 32



# -----------------------
# Sentence -> (32, 300) matrix
# -----------------------
def sentence_to_matrix(sentence: str, wv: KeyedVectors, seq_len: int = 32, emb_dim: int = 300) -> np.ndarray:
    tokens = tokenize_text(sentence)

    # Truncate
    tokens = tokens[:seq_len]

    mat = np.zeros((seq_len, emb_dim), dtype=np.float32)

    # Fill rows with word vectors (OOV -> zeros)
    for i, tok in enumerate(tokens):
        if tok in wv:
            mat[i] = wv[tok]

    # If tokens shorter than seq_len, remaining rows stay zero (padding)
    return mat


def build_sequence_tensor(sent_list):
    # returns shape: (N, 32, 300)
    mats = [sentence_to_matrix(s, fasttext, SEQ_LEN, EMB_DIM) for s in sent_list]
    return np.stack(mats, axis=0).astype(np.float32)



X_train_seq = build_sequence_tensor(X_train)
X_val_seq   = build_sequence_tensor(X_val)
X_test_seq  = build_sequence_tensor(X_test)

y_train_np = np.array(y_train, dtype=np.int64)
y_val_np   = np.array(y_val, dtype=np.int64)
y_test_np  = np.array(y_test, dtype=np.int64)

print("Train tensor:", X_train_seq.shape)
print("Val tensor:  ", X_val_seq.shape)
print("Test tensor: ", X_test_seq.shape)


# -----------------------
# DataLoaders
# -----------------------
X_train_t = torch.from_numpy(X_train_seq)  # float32
X_val_t   = torch.from_numpy(X_val_seq)
X_test_t  = torch.from_numpy(X_test_seq)

y_train_t = torch.from_numpy(y_train_np)
y_val_t   = torch.from_numpy(y_val_np)
y_test_t  = torch.from_numpy(y_test_np)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t, y_val_t)
test_ds  = TensorDataset(X_test_t, y_test_t)

BATCH_SIZE = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# -----------------------
# Class weights
# -----------------------
classes = np.array([0, 1, 2], dtype=np.int64)
class_w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_np)
class_w = torch.tensor(class_w, dtype=torch.float32)
print("Class weights:", class_w.tolist())


# -----------------------
# LSTM model
# -----------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=128, num_layers=1, bidirectional=True, dropout=0.3, num_classes=3):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Note: PyTorch only applies dropout inside LSTM if num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,        # input: (B, T, D)
            bidirectional=bidirectional,
            dropout=lstm_dropout
        )

        out_dim = hidden_dim * self.num_directions
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, num_classes)
        )

    def forward(self, x):
        # x: (B, 32, 300)
        _, (h_n, _) = self.lstm(x)

        # h_n: (num_layers * num_directions, B, hidden_dim)
        if self.bidirectional:
            # last layer forward and backward are the last two entries
            h_fwd = h_n[-2]  # (B, H)
            h_bwd = h_n[-1]  # (B, H)
            h = torch.cat([h_fwd, h_bwd], dim=1)  # (B, 2H)
        else:
            h = h_n[-1]  # (B, H)

        return self.head(h)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(input_dim=300, hidden_dim=128, num_layers=1, bidirectional=True, dropout=0.3, num_classes=3).to(device)

criterion = nn.CrossEntropyLoss(weight=class_w.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)


# -----------------------
# Eval helper
# -----------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    total_n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(yb.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss = total_loss / total_n
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


# -----------------------
# Train loop (>= 30 epochs) + save best
# -----------------------
os.makedirs("outputs", exist_ok=True)

EPOCHS = 30
best_val_f1 = -1.0
best_path = "outputs/best_lstm.pt"

history = {
    "train_loss": [], "train_acc": [], "train_f1": [],
    "val_loss": [], "val_acc": [], "val_f1": []
}

print("\n========== Training ==========")
for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    train_loss, train_acc, train_f1 = evaluate(model, train_loader, criterion, device)
    val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["train_f1"].append(train_f1)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    scheduler.step(val_f1)

    print(f"Epoch {epoch:02d} | "
          f"train loss {train_loss:.4f} acc {train_acc:.4f} f1 {train_f1:.4f} | "
          f"val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_f1": val_f1
        }, best_path)


# -----------------------
# Plot curves
# -----------------------
def plot_curve(train_vals, val_vals, title, ylabel, outpath):
    epochs = np.arange(1, len(train_vals) + 1)
    plt.figure()
    plt.plot(epochs, train_vals, label="train")
    plt.plot(epochs, val_vals, label="val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

plot_curve(history["train_loss"], history["val_loss"], "LSTM Loss vs Epochs", "Loss", "outputs/lstm_loss.png")
plot_curve(history["train_acc"],  history["val_acc"],  "LSTM Accuracy vs Epochs", "Accuracy", "outputs/lstm_acc.png")
plot_curve(history["train_f1"],   history["val_f1"],   "LSTM Macro F1 vs Epochs", "Macro F1", "outputs/lstm_f1.png")


# -----------------------
# Test eval + confusion matrix
# -----------------------
ckpt = torch.load(best_path, map_location=device)
model.load_state_dict(ckpt["model_state"])
print("\nLoaded best checkpoint from epoch:", ckpt["epoch"], "val_f1:", ckpt["val_f1"])

test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
print(f"TEST | loss {test_loss:.4f} acc {test_acc:.4f} macro_f1 {test_f1:.4f}")

@torch.no_grad()
def get_preds_labels(model, loader, device):
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(p)
        labels.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(labels)

y_pred, y_true = get_preds_labels(model, test_loader, device)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["neg", "neu", "pos"])
plt.figure()
disp.plot(values_format="d")
plt.title("LSTM Confusion Matrix (Test)")
plt.tight_layout()
plt.savefig("outputs/lstm_confusion_matrix.png")
plt.close()



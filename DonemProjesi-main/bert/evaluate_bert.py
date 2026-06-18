import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from transformers import BertTokenizer, BertForSequenceClassification

# ======================================================
# 1. AYARLAR
# ======================================================
MODEL_PATH = r"D:\Yazılım\DonemProje\text_sentiment_project\bert_gercekveri"
CSV_PATH = "original_dataset.csv"

id2label = {0: "olumsuz", 1: "notr", 2: "olumlu"}

# ======================================================
# 2. DATASET YÜKLE & TEMİZLE (EĞİTİM İLE AYNI)
# ======================================================
df = pd.read_csv(
    CSV_PATH,
    header=None,
    names=["text", "label"],
    encoding="utf-8",
    on_bad_lines="skip"
)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    return " ".join(text.split())

df["text"] = df["text"].apply(clean_text)

# LABEL TEMİZLEME (AYNI MANTIK)
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(str).str.lower().str.strip()

df["label"] = df["label"].replace({
    "pozitiz": "2",
    "positive": "2",
    "olumlu": "2",
    "nört": "1",
    "nötr": "1",
    "neutral": "1",
    "negative": "0",
    "negatif": "0",
    "olumsuz": "0",
    "nan": np.nan
})

df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# ======================================================
# 3. SADECE TEST SETİ AYIR
# ======================================================
_, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=df["label"]
)

test_df = test_df.reset_index(drop=True)

# ======================================================
# 4. MODEL & TOKENIZER YÜKLE
# ======================================================
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ======================================================
# 5. TAHMİN
# ======================================================
y_true = test_df["label"].tolist()
y_pred = []

for text in test_df["text"]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=-1).item()
    y_pred.append(pred)

# ======================================================
# 6. CONFUSION MATRIX
# ======================================================
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[id2label[i] for i in range(3)]
)

disp.plot(cmap="Blues", values_format="d")
plt.title("BERT Confusion Matrix")
plt.show()

# ======================================================
# 7. CLASSIFICATION REPORT
# ======================================================
print("\n📄 Classification Report:\n")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=[id2label[i] for i in range(3)]
    )
)

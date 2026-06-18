import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)


# 1. AYARLAR

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
CSV_PATH = "original_dataset.csv"
OUTPUT_DIR = "bert_gercekveri"

label2id = {0: 0, 1: 1, 2: 2}
id2label = {0: "olumsuz", 1: "notr", 2: "olumlu"}

# 2. DATASET YÜKLEME

df = pd.read_csv(
    CSV_PATH,
    header=None,
    names=["text", "label"],
    encoding="utf-8",
    on_bad_lines="skip"
)


# 3. TEXT TEMİZLEME

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text

df["text"] = df["text"].apply(clean_text)


# 4. LABEL TEMİZLEME (SAYISAL AĞIRLIKLI DATASET)


# NaN etiketleri at
df = df.dropna(subset=["label"])

# String'e çevir (karışık tipler için)
df["label"] = df["label"].astype(str).str.lower().str.strip()

# Bozuk stringleri düzelt
df["label"] = df["label"].replace({
    "pozitiz": "2",
    "positive": "2",
    "olumlu": "2",

    "nört": "1",
    "nötr": "1",
    "neutral": "1",

    "negatif": "0",
    "negative": "0",
    "olumsuz": "0",

    "nan": np.nan
})

# Sayıya çevir
df["label"] = pd.to_numeric(df["label"], errors="coerce")

# Hatalı satırları temizle
df = df.dropna(subset=["label", "text"])
df = df[df["text"].str.len() > 5]

df["label"] = df["label"].astype(int)


# 5. KONTROLLER

print("\n📊 Sınıf Dağılımı:")
print(df["label"].value_counts())

print("\n🧪 NaN Kontrol:")
print(df.isna().sum())


# 6. TRAIN / TEST SPLIT

train_df, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=df["label"]
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)


# 7. TOKENIZATION

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds = train_ds.remove_columns(["text"])
test_ds = test_ds.remove_columns(["text"])

train_ds.set_format("torch")
test_ds.set_format("torch")


# 8. METRİKLER

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }


# 9. MODEL

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    label2id={0: 0, 1: 1, 2: 2},
    id2label=id2label
)


# 10. TRAINING AYARLARI (RTX 4050 OPTİMİZE)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# 11. EĞİTİM

trainer.train()


# 12. KAYDET

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Model başarıyla eğitildi ve kaydedildi → {OUTPUT_DIR}")

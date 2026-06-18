import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_DIR = "bert_output"

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

print("\n🎤 BERT Türkçe Duygu Analizi Demo")
print("Çıkmak için Enter\n")

while True:
    text = input("📝 Cümle gir: ").strip()
    if not text:
        break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    confidence = probs[0][pred].item()
    label = model.config.id2label[pred]

    print(f"➡️ Duygu: {label.upper()}  (Güven: %{confidence*100:.1f})\n")

"""
src/text_analyzer.py

BERT tabanlı Türkçe duygu analizi.
Etiketler: olumsuz (0) | notr (1) | olumlu (2)
"""

import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Proje kökünden bert_gercekveri klasörü
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(_PROJECT_ROOT, "models", "bert_gercekveri")

LABEL_MAP = {0: "olumsuz", 1: "notr", 2: "olumlu"}

# Kalibrasyon eşikleri: güven bu değerin altındaysa notr döner
THRESHOLDS = {0: 0.45, 2: 0.30}


class TextAnalyzer:

    def __init__(self, hf_token=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📂 BERT model: {MODEL_PATH}  |  💻 {self.device}")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model klasörü bulunamadı: {MODEL_PATH}")

        self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        self.model     = BertForSequenceClassification.from_pretrained(
            MODEL_PATH, local_files_only=True
        ).to(self.device)
        self.model.eval()
        print("✅ BERT yüklendi.")

    def analyze(self, text: str) -> dict:
        """
        Döndürür: {"label": "olumsuz"|"notr"|"olumlu", "score": float}
        """
        if not text or len(text.strip()) < 2:
            return {"label": "notr", "score": 1.0}

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt",
                truncation=True, padding=True, max_length=128
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs      = torch.softmax(logits, dim=-1)
            pred       = torch.argmax(logits, dim=-1).item()
            confidence = probs[0][pred].item()

            # Kalibrasyon: düşük güvende notr'a çek
            label = LABEL_MAP[pred]
            if pred in THRESHOLDS and confidence < THRESHOLDS[pred]:
                label = "notr"

            return {"label": label, "score": round(confidence, 3)}

        except Exception as e:
            print(f"⚠️ TextAnalyzer hatası: {e}")
            return {"label": "notr", "score": 0.0}


if __name__ == "__main__":
    analyzer = TextAnalyzer()
    tests = [
        "İyi günler, size nasıl yardımcı olabilirim?",
        "İnternetim dünden beri çok yavaş, gerçekten mağdur oldum.",
        "Sorunumu hemen çözdüğünüz için çok teşekkürler.",
        "Artık sabrım kalmadı, iptal etmek istiyorum.",
    ]
    for t in tests:
        print(f"{analyzer.analyze(t)}  ←  {t}")

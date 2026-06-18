import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification

# 1. Yolun başına 'r' ekleyerek Windows yol hatalarını engelliyoruz
MODEL_PATH = r"D:\Yazılım\DonemProje\text_sentiment_project\bert_gercekveri"

class SentimentPredictor:
    def __init__(self):
        print("🚀 Model yükleniyor...")
        # Yerelden yükleme yaptığımızı belirtmek için local_files_only ekleyebilirsin
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        self.model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model.eval()
        
        # 2. YENİ ETİKET HARİTASI (Eğitimdeki sırayla aynı)
        self.labels = {0: "NEGATİF", 1: "NÖTR", 2: "POZİTİF"}

    def predict(self, text):
        if not text.strip():
            return "BOŞ", 0.0
            
        # Metni hazırla
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Olasılıkları hesapla
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        score, index = torch.max(probs, dim=1)
        
        # 3. İndeksi etiket haritasından çek
        label = self.labels.get(index.item(), "BİLİNMİYOR")
        return label, score.item()

if __name__ == "__main__":
    # Klasör kontrolü
    if not os.path.exists(MODEL_PATH):
        print(f"❌ HATA: Model klasörü bulunamadı: {MODEL_PATH}")
    else:
        predictor = SentimentPredictor()
        print("✅ Teste hazır! (Çıkış için 'q')")
        
        while True:
            sentence = input("\nTest cümlesi girin: ")
            if sentence.lower() == 'q': break
            
            label, confidence = predictor.predict(sentence)
            print(f"🔍 Sonuç: {label} (Güven: %{confidence*100:.2f})")
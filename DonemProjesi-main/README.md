# 🎙️ AI Çağrı Merkezi Analiz Platformu

Tekirdağ Namık Kemal Üniversitesi — Bilgisayar Mühendisliği Bitirme Projesi

## 📌 Proje Hakkında

Bu platform, çağrı merkezi ses kayıtlarını yapay zeka ile analiz eden çok modlu (multimodal) bir sistemdir. Ses kaydındaki konuşmayı metne döker, konuşmacıları ayırır ve hem metin hem ses tonu üzerinden duygu analizi yaparak çağrı kalitesini değerlendirir.

## 🚀 Özellikler

- **Metin Duygu Analizi** — Türkçe metinlerde BERT tabanlı duygu sınıflandırması (Pozitif / Negatif / Nötr)
- **Ses Tonu Analizi** — SVM tabanlı ses özelliği çıkarımı ve duygu tahmini
- **Multimodal Fusion** — Metin ve ses analizini birleştirerek kinaye tespiti dahil nihai duygu kararı
- **Otomatik Konuşma Tanıma (STT)** — Whisper ile Türkçe ses kaydından metin üretimi
- **Konuşmacı Ayrıştırma** — Pyannote ile temsilci/müşteri ayrımı
- **Streamlit Arayüzü** — Kullanımı kolay web tabanlı analiz platformu

## 🛠️ Kurulum

### Gereksinimler

- Python 3.10+
- CUDA destekli GPU (önerilen)


### Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### Çalıştır

```bash
streamlit run app.py
```

## 📁 Proje Yapısı

```
DonemProjesi/
├── app.py                  # Ana Streamlit uygulaması
├── recorder_page.py        # Canlı kayıt modülü
├── src/
│   ├── call_analyzer.py    # Çağrı analiz motoru
│   ├── audio_analyzer.py   # Ses özellik çıkarımı ve SVM tahmini
│   ├── stt_engine.py       # Whisper STT entegrasyonu
│   ├── text_analyzer.py    # BERT metin analizi
│   └── fusion_engine.py    # Multimodal fusion ve kinaye tespiti
├── models/
│   ├── bert_gercekveri/    # Fine-tuned BERT modeli
│   └── audio_model/        # Eğitilmiş SVM modeli
├── bert/                   # BERT eğitim ve değerlendirme scriptleri
├── scripts/                # Yardımcı scriptler
└── requirements.txt
```

## 🧠 Kullanılan Teknolojiler

| Katman | Teknoloji |
|---|---|
| Dil Modeli | BERT (bert-base-turkish-cased) |
| Ses Sınıflandırma | SVM + MFCC / Spectral özellikler |
| Konuşma Tanıma | OpenAI Whisper |
| Konuşmacı Ayrıştırma | Pyannote Audio |
| Arayüz | Streamlit |
| Ses İşleme | Librosa, SoundFile, SoundDevice |

## 👥 Geliştiriciler

- Uğur Arslan
- Fuat Kuşkonmaz
- Melih Can

---

Tekirdağ Namık Kemal Üniversitesi — Bilgisayar Mühendisliği, 2025

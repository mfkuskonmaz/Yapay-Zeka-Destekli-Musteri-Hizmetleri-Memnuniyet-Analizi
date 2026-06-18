from src.text_analyzer import TextAnalyzer
from src.audio_analyzer import ProfessionalAudioAnalyzer
from src.fusion_engine import MultimodalFusion
from src.stt_engine import SpeechToTextEngine # YENİ

# Modelleri Başlat
stt_engine = SpeechToTextEngine()
text_engine = TextAnalyzer(model_path="bert_output")
audio_engine = ProfessionalAudioAnalyzer()
fusion_engine = MultimodalFusion()

def process_audio_call(audio_file_path):
    print(f"\n--- Multimodal Analiz Başlıyor ---")
    
    # ADIM 1: Sesi Metne Çevir (Speech-to-Text)
    print("Ses metne dönüştürülüyor...")
    detected_text = stt_engine.transcribe(audio_file_path)
    print(f"📝 Tespit Edilen Metin: \"{detected_text}\"")
    
    # ADIM 2: Metin Üzerinden Duygu Analizi (BERT)
    t_res = text_engine.analyze(detected_text)
    
    # ADIM 3: Ses Tonu Analizi (Wav2Vec2)
    a_res = audio_engine.analyze(audio_file_path)
    
    # Analiz sonrası modelin tüm etiketlerini gör
    print(f"📊 Modelin Tüm Ses Analizi: {a_res}")
    
    # ADIM 4: Karar (Fusion)
    final = fusion_engine.merge(t_res, a_res)
    
    # SONUÇLARI YAZDIR
    print("-" * 30)
    print(f"BERT Tahmini: {t_res['label'].upper()}")
    print(f"Ses Öfke Skoru: %{a_res.get('ang', 0)*100:.1f}")
    print(f"NİHAİ DURUM: {final['final_sentiment'].upper()}")
    
    if final['sarcasm_detected']:
        print("⚠️ DİKKAT: Kinaye tespit edildi! Müşteri kelimeleriyle değil, ses tonuyla sinyal veriyor.")

if __name__ == "__main__":
    audio_path = "data/samples/WhatsApp Ptt 2025-12-17 at 01.09.06 (online-audio-converter.com).wav"
    process_audio_call(audio_path)
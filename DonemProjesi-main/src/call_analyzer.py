"""
src/call_analyzer.py

Tam çağrı analizi:
  1. Ses yükleme
  2. Konuşmacı ayırma (pyannote diarization)
  3. Metne döküm (Whisper)
  4. Her segment için: TextAnalyzer + AudioAnalyzer + MultimodalFusion
  5. Rapor döndürme
"""

import torch
import librosa
import numpy as np


# Kelime havuzları — rol belirleme için
AGENT_KEYWORDS    = ["merhaba", "hoş", "gel", "buyur", "yardımcı",
                     "günler", "destek", "kaydı", "beklediğiniz"]
CUSTOMER_KEYWORDS = ["internet", "çekmiyor", "bozuk", "şikayet", "fatura",
                     "kesildi", "iptal", "bıktım", "alo", "ya"]


class CallAnalyticEngine:

    def __init__(self, hf_token, text_engine, audio_engine, stt_engine, fusion_engine):
        self.text_engine   = text_engine
        self.audio_engine  = audio_engine
        self.stt_engine    = stt_engine
        self.fusion_engine = fusion_engine
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🚀 Modeller {self.device} üzerinde hazırlanıyor...")
        from pyannote.audio import Pipeline
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=hf_token
        ).to(self.device)

    # ─────────────────────────────────────────────────────────────────────────
    def analyze_full_audio(self, audio_path: str) -> list[dict]:
        print("⌛ Analiz devam ediyor, lütfen bekleyin...")

        # 1. Ses yükle
        y, sr = librosa.load(audio_path, sr=16000)
        audio_tensor = {"waveform": torch.from_numpy(y).unsqueeze(0),
                        "sample_rate": sr}

        # 2. Diarization
        diarization_result = self.diarization_pipeline(
            audio_tensor, num_speakers=2
        )
        tracks = self._extract_tracks(diarization_result)

        # 3. STT
        stt_result = self.stt_engine.model.transcribe(audio_path, language="tr")
        segments   = stt_result["segments"]

        # 4. Her segment → analiz
        speaker_map      = {}
        first_speaker_id = None
        call_report      = []

        for seg in segments:
            start = seg["start"]
            end   = seg["end"]
            text  = seg["text"].strip()
            text_lower = text.lower()

            # Konuşmacı tespit
            speaker_id = self._find_speaker(start, end, tracks)

            # Rol atama
            if speaker_id != "Bilinmiyor" and speaker_id not in speaker_map:
                if not speaker_map:
                    first_speaker_id = speaker_id
                    agent_score    = sum(1 for kw in AGENT_KEYWORDS    if kw in text_lower)
                    customer_score = sum(1 for kw in CUSTOMER_KEYWORDS if kw in text_lower)
                    if customer_score > agent_score:
                        speaker_map[speaker_id] = "MÜŞTERİ"
                    else:
                        speaker_map[speaker_id] = "TEMSİLCİ"
                else:
                    first_role = speaker_map[first_speaker_id]
                    speaker_map[speaker_id] = (
                        "MÜŞTERİ" if first_role == "TEMSİLCİ" else "TEMSİLCİ"
                    )

            role = speaker_map.get(speaker_id, "MÜŞTERİ")

            print(f"[DEBUG] {int(start//60):02d}:{int(start%60):02d} "
                  f"| {speaker_id} | {role} | {text[:40]}...")

            # 5. Ses segmenti kes
            audio_chunk = y[int(start * sr): int(end * sr)]

            # 6. Analiz — sadece MÜŞTERİ segmentlerinde tam fusion yap
            sentiment = "-"  # TEMSİLCİ için duygu analizi yapılmaz

            if role == "MÜŞTERİ":
                # Metin analizi
                text_res  = self.text_engine.analyze(text)

                # Ses analizi (yeni SVM tabanlı)
                audio_res = self.audio_engine.analyze_tone(audio_chunk, sr) \
                    if len(audio_chunk) > sr * 0.3 \
                    else self.audio_engine._neutral_result("kısa segment")

                # Fusion
                fusion = self.fusion_engine.merge(text_res, audio_res)

                sentiment = fusion["final_sentiment"].upper()

                if fusion["sarcasm_detected"]:
                    sentiment += " (⚠️ Kinaye)"

                # Debug
                print(f"         BERT={text_res['label']} | "
                      f"Audio={audio_res['label']} ({audio_res['confidence']:.0%}) | "
                      f"Fusion={fusion['final_sentiment']}")

            call_report.append({
                "time":      f"{int(start//60):02d}:{int(start%60):02d}",
                "role":      role,
                "text":      text,
                "sentiment": sentiment,
            })

        return call_report

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_tracks(diarization_result) -> list:
        try:
            for attr in ("speaker_diarization", "exclusive_speaker_diarization"):
                if hasattr(diarization_result, attr):
                    return list(getattr(diarization_result, attr)
                                .itertracks(yield_label=True))
            return list(diarization_result.itertracks(yield_label=True))
        except Exception as e:
            print(f"⚠️ Diarization okuma hatası: {e}")
            return []

    @staticmethod
    def _find_speaker(start: float, end: float, tracks: list) -> str:
        # En fazla çakışan konuşmacıyı bul
        best_speaker, best_overlap = "Bilinmiyor", 0.0
        for turn, _, speaker in tracks:
            overlap = min(end, turn.end) - max(start, turn.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        # Fallback: orta nokta
        if best_speaker == "Bilinmiyor":
            mid = (start + end) / 2
            for turn, _, speaker in tracks:
                if turn.start <= mid <= turn.end:
                    return speaker

        return best_speaker

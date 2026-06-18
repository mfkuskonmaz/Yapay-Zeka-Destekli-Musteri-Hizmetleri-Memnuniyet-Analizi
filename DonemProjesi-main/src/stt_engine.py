"""
src/stt_engine.py

Whisper tabanlı Türkçe konuşma-metin dönüştürücü.
"""

import torch
import static_ffmpeg
static_ffmpeg.add_paths()
import whisper


class SpeechToTextEngine:

    def __init__(self, model_size: str = "base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎤 Whisper ({model_size}) {self.device.upper()} üzerinde yükleniyor...")
        self.model = whisper.load_model("medium", device=self.device)
        print("✅ Whisper hazır.")

    def transcribe(self, audio_path: str) -> str:
        result = self.model.transcribe(audio_path, language="tr")
        return result["text"].strip()

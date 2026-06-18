"""
src/audio_analyzer.py

Eğitilmiş SVM modeliyle ses dosyasından duygu tahmini yapar.
Önce train_audio_classifier.py çalıştırılmış olmalı.
"""

import numpy as np
import librosa
import joblib
from pathlib import Path

MODEL_PATH      = Path("models/audio_model/audio_svm.pkl")
LABEL_INFO_PATH = Path("models/audio_model/label_info.pkl")

SR     = 16000
N_MFCC = 13


def extract_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean   = np.mean(mfcc, axis=1)
    mfcc_std    = np.std(mfcc,  axis=1)

    cent        = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff     = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr         = librosa.feature.zero_crossing_rate(y)[0]
    rms         = librosa.feature.rms(y=y)[0]

    chroma      = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    contrast      = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std  = np.std(contrast,  axis=1)

    f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
    f0_clean  = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
    f0_feats  = np.array([
        np.mean(f0_clean) if len(f0_clean) else 0.0,
        np.std(f0_clean)  if len(f0_clean) else 0.0,
        np.max(f0_clean)  if len(f0_clean) else 0.0,
    ])

    return np.concatenate([
        mfcc_mean, mfcc_std,
        [np.mean(cent),    np.std(cent)],
        [np.mean(rolloff), np.std(rolloff)],
        [np.mean(zcr),     np.std(zcr)],
        [np.mean(rms),     np.std(rms)],
        chroma_mean,
        contrast_mean, contrast_std,
        f0_feats,
    ])


class ProfessionalAudioAnalyzer:
    def __init__(self, hf_token=None):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Audio model bulunamadı: {MODEL_PATH}\n"
                "Önce 'python train_audio_classifier.py' çalıştır."
            )
        self.pipeline    = joblib.load(MODEL_PATH)
        label_info       = joblib.load(LABEL_INFO_PATH)
        self.id_to_label = label_info["id_to_label"]  # {0:"olumsuz", 1:"notr", 2:"olumlu"}
        print("🎙️ Audio Analyzer (SVM) hazır.")

    def analyze(self, audio_path: str) -> dict:
        """
        Döndürür:
        {
            "label":      "olumlu" | "notr" | "olumsuz",
            "label_id":   2 | 1 | 0,
            "confidence": 0.87,
            "probs":      {"olumsuz": 0.05, "notr": 0.08, "olumlu": 0.87}
        }
        """
        try:
            y, sr = librosa.load(audio_path, sr=SR)

            if len(y) < sr * 0.3:
                return self._neutral_result("Ses çok kısa")

            features = extract_features(y, sr).reshape(1, -1)
            label_id  = int(self.pipeline.predict(features)[0])
            proba     = self.pipeline.predict_proba(features)[0]

            label_names = [self.id_to_label[i] for i in range(len(proba))]
            probs_dict  = {name: round(float(p), 4)
                           for name, p in zip(label_names, proba)}

            return {
                "label":      self.id_to_label[label_id],
                "label_id":   label_id,
                "confidence": round(float(np.max(proba)), 4),
                "probs":      probs_dict,
            }

        except Exception as e:
            print(f"⚠️ Audio analiz hatası: {e}")
            return self._neutral_result(str(e))

    # Segment bazlı analiz (call_analyzer için)
    def analyze_tone(self, audio_segment: np.ndarray, sr: int = SR) -> dict:
        try:
            if len(audio_segment) < sr * 0.3:
                return self._neutral_result("Segment çok kısa")

            features = extract_features(audio_segment, sr).reshape(1, -1)
            label_id  = int(self.pipeline.predict(features)[0])
            proba     = self.pipeline.predict_proba(features)[0]

            label_names = [self.id_to_label[i] for i in range(len(proba))]
            probs_dict  = {name: round(float(p), 4)
                           for name, p in zip(label_names, proba)}

            return {
                "label":      self.id_to_label[label_id],
                "label_id":   label_id,
                "confidence": round(float(np.max(proba)), 4),
                "probs":      probs_dict,
            }
        except Exception as e:
            return self._neutral_result(str(e))

    @staticmethod
    def _neutral_result(reason: str = "") -> dict:
        return {
            "label":      "notr",
            "label_id":   1,
            "confidence": 0.0,
            "probs":      {"olumsuz": 0.0, "notr": 1.0, "olumlu": 0.0},
            "tone":       "notr",
            "reason":     reason,
        }

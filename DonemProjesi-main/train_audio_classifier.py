"""
Ses tabanlı duygu sınıflandırıcı eğitimi.
240 wav dosyasından MFCC + spektral özellikler çıkarır,
SVM ile eğitir ve modeli kaydeder.

Kullanım: python train_audio_classifier.py
"""

import numpy as np
import librosa
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# ── AYARLAR ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/samples")
OUTPUT_DIR = Path("audio_model")
SR         = 16000
N_MFCC     = 13

LABEL_MAP = {
    "olumsuz": 0,
    "notr":    1,
    "olumlu":  2
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# ── FEATURE ÇIKARIMI ─────────────────────────────────────────────────────────
def extract_features(path: str):
    try:
        y, sr = librosa.load(path, sr=SR)

        if len(y) < sr * 0.3:
            return None

        # MFCC (13 × 2 = 26)
        mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean   = np.mean(mfcc, axis=1)
        mfcc_std    = np.std(mfcc,  axis=1)

        # Spektral Centroid (2)
        cent        = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

        # Spektral Rolloff (2)
        rolloff     = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

        # Zero Crossing Rate (2)
        zcr         = librosa.feature.zero_crossing_rate(y)[0]

        # RMS Energy (2)
        rms         = librosa.feature.rms(y=y)[0]

        # Chroma (12)
        chroma      = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Spectral Contrast (7 × 2 = 14)
        contrast      = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std  = np.std(contrast,  axis=1)

        # Pitch / F0 istatistikleri (3)
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
        f0_clean  = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
        f0_feats  = np.array([
            np.mean(f0_clean) if len(f0_clean) else 0.0,
            np.std(f0_clean)  if len(f0_clean) else 0.0,
            np.max(f0_clean)  if len(f0_clean) else 0.0,
        ])

        return np.concatenate([
            mfcc_mean,                               # 13
            mfcc_std,                                # 13
            [np.mean(cent),    np.std(cent)],        #  2
            [np.mean(rolloff), np.std(rolloff)],     #  2
            [np.mean(zcr),     np.std(zcr)],         #  2
            [np.mean(rms),     np.std(rms)],         #  2
            chroma_mean,                             # 12
            contrast_mean,                           #  7
            contrast_std,                            #  7
            f0_feats,                                #  3
        ])  # Toplam: 63 feature

    except Exception as e:
        print(f"  ⚠️  {Path(path).name}: {e}")
        return None


# ── VERİ YÜKLEME ─────────────────────────────────────────────────────────────
def load_dataset():
    X, y, skipped = [], [], 0

    for label_name, label_id in LABEL_MAP.items():
        folder = DATA_DIR / label_name
        if not folder.exists():
            print(f"❌ Klasör yok: {folder}")
            continue

        files = sorted(folder.glob("*.wav"))
        print(f"\n📁 {label_name.upper()} ({len(files)} dosya)")

        for f in files:
            feat = extract_features(str(f))
            if feat is not None:
                X.append(feat)
                y.append(label_id)
            else:
                skipped += 1

    print(f"\n⏭️  Atlanan dosya: {skipped}")
    return np.array(X), np.array(y)


# ── ANA AKIŞ ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  SES DUYGU SINIFLANDIRICI — EĞİTİM")
    print("=" * 55)

    # 1. Veri yükle
    X, y = load_dataset()
    print(f"\n✅ Toplam örnek: {len(X)}")
    for lid, lname in ID_TO_LABEL.items():
        print(f"   {lname}: {np.sum(y == lid)}")

    if len(X) == 0:
        raise RuntimeError("Hiç ses yüklenemedi. data/samples/ klasörünü kontrol et.")

    # 2. GridSearch ile en iyi parametreleri bul
    print("\n🔍 GridSearch ile en iyi parametreler aranıyor...")
    param_grid = {
        "svm__C":     [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "svm__kernel": ["rbf", "linear"]
    }
    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(probability=True, random_state=42))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(base_pipeline, param_grid, cv=cv,
                        scoring="f1_weighted", n_jobs=-1, verbose=1)
    grid.fit(X, y)
    print(f"   En iyi parametreler : {grid.best_params_}")
    print(f"   En iyi CV F1        : {grid.best_score_:.3f}")
    pipeline = grid.best_estimator_

    # 5. Eğitim seti raporu
    y_pred = pipeline.predict(X)
    print("\n📊 Eğitim Seti Raporu:")
    print(classification_report(y, y_pred,
                                 target_names=["olumsuz", "notr", "olumlu"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    # 6. Kaydet
    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, OUTPUT_DIR / "audio_svm.pkl")
    joblib.dump({"label_map": LABEL_MAP, "id_to_label": ID_TO_LABEL},
                OUTPUT_DIR / "label_info.pkl")

    print(f"\n✅ Model kaydedildi  → {OUTPUT_DIR / 'audio_svm.pkl'}")
    print(f"✅ Label bilgisi     → {OUTPUT_DIR / 'label_info.pkl'}")
    print("\n🎉 Eğitim tamamlandı!")

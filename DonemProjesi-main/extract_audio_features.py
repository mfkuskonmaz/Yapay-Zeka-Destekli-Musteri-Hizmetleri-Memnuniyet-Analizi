import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# SES DOSYALARINDAN ÖZELLİK ÇIKARMA (Audio Features)
# =====================================================

class AudioFeatureExtractor:
    def __init__(self, sr=16000):
        self.sr = sr
    
    def extract_features(self, audio_path):
        """
        Ses dosyasından özellikler çıkart:
        - MFCC (Mel-Frequency Cepstral Coefficients)
        - Spektral özellikler (Centroid, Rolloff, Contrast)
        - Energy, Zero Crossing Rate
        """
        try:
            # Ses yükle
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # MFCC (40 coefficient)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Spektral Centroid
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spec_cent_mean = np.mean(spec_cent)
            spec_cent_std = np.std(spec_cent)
            
            # Spektral Rolloff
            spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spec_rolloff_mean = np.mean(spec_rolloff)
            spec_rolloff_std = np.std(spec_rolloff)
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = np.mean(zcr)
            zcr_std = np.std(zcr)
            
            # Energy (RMS)
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)
            
            # Chroma STFT
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Hepsi bir vektöre dönüştür
            features = np.concatenate([
                mfcc_mean,                    # 13
                mfcc_std,                     # 13
                [spec_cent_mean, spec_cent_std],  # 2
                [spec_rolloff_mean, spec_rolloff_std],  # 2
                [zcr_mean, zcr_std],          # 2
                [rms_mean, rms_std],          # 2
                chroma_mean                   # 12
            ])
            
            return features
        except Exception as e:
            print(f"❌ Hata {audio_path}: {e}")
            return None

def process_dataset():
    """
    data/samples/ klasörlerinden özellikleri çıkar ve CSV oluştur
    """
    extractor = AudioFeatureExtractor(sr=16000)
    data = []
    
    base_path = Path("data/samples")
    label_map = {
        "olumsuz": 0,
        "notr": 1,
        "olumlu": 2
    }
    
    print("🎵 Ses dosyalarından özellikler çıkarılıyor...\n")
    
    for label_name, label_id in label_map.items():
        folder_path = base_path / label_name
        
        if not folder_path.exists():
            print(f"⚠️  Klasör bulunamadı: {folder_path}")
            continue
        
        wav_files = list(folder_path.glob("*.wav"))
        print(f"📁 {label_name.upper()} ({len(wav_files)} dosya):")
        
        for i, wav_file in enumerate(wav_files, 1):
            features = extractor.extract_features(str(wav_file))
            
            if features is not None:
                # Feature vektörünü pipe (|) ile ayırmış string olarak kaydet
                feature_str = "|".join([f"{f:.6f}" for f in features])
                data.append({
                    "text": feature_str,  # Train_bert.py bunu metin olarak bekliyor
                    "label": label_id,
                    "filename": wav_file.name,
                    "source": "audio"
                })
                
                if i % 10 == 0:
                    print(f"  ✓ {i}/{len(wav_files)} işlendi")
        
        print(f"  ✅ {label_name.upper()} tamamlandı\n")
    
    # DataFrame oluştur
    df = pd.DataFrame(data)
    
    # CSV'ye kaydet
    output_csv = "bert/audio_features.csv"
    df.to_csv(output_csv, index=False, header=False, encoding="utf-8")
    
    print(f"\n📊 İstatistikler:")
    print(f"  Toplam: {len(df)} dosya")
    print(f"  Olumsuz: {len(df[df['label']==0])}")
    print(f"  Nötr: {len(df[df['label']==1])}")
    print(f"  Olumlu: {len(df[df['label']==2])}")
    print(f"\n✅ Kaydedildi: {output_csv}")
    
    return df

if __name__ == "__main__":
    df = process_dataset()

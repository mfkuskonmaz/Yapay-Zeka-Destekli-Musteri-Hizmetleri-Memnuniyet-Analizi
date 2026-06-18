"""
OGG dosyalarını WAV'a dönüştürür.
Her sınıf klasöründeki .ogg dosyalarını .wav yaparak aynı klasöre kaydeder.

Kullanım: python convert_ogg_to_wav.py
"""

import os
from pathlib import Path
import static_ffmpeg
static_ffmpeg.add_paths()
from pydub import AudioSegment

DATA_DIR = Path("data/samples")
LABELS   = ["olumsuz", "notr", "olumlu"]

def convert_folder(folder: Path):
    ogg_files = list(folder.glob("*.ogg"))
    if not ogg_files:
        print(f"  ⏭️  {folder.name}: OGG dosyası yok")
        return

    print(f"\n📁 {folder.name.upper()} ({len(ogg_files)} dosya)")
    converted, failed = 0, 0

    for ogg_path in ogg_files:
        wav_path = ogg_path.with_suffix(".wav")
        try:
            AudioSegment.from_ogg(str(ogg_path)).export(str(wav_path), format="wav")
            ogg_path.unlink()  # OGG'u sil
            converted += 1
            print(f"  ✅ {ogg_path.name} → {wav_path.name}")
        except Exception as e:
            failed += 1
            print(f"  ❌ {ogg_path.name}: {e}")

    print(f"  Tamamlandı: {converted} dönüştürüldü, {failed} hata")

if __name__ == "__main__":
    print("=" * 50)
    print("  OGG → WAV DÖNÜŞTÜRÜCÜ")
    print("=" * 50)

    for label in LABELS:
        folder = DATA_DIR / label
        if folder.exists():
            convert_folder(folder)
        else:
            print(f"❌ Klasör bulunamadı: {folder}")

    print("\n✅ Tüm dönüşümler tamamlandı!")

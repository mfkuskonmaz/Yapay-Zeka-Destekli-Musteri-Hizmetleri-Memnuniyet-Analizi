"""
stereo_recorder.py

Sol kanal  → Mikrofon (senin sesin / TEMSİLCİ)
Sağ kanal  → Voicemeeter Out B1 (WhatsApp sesi / MÜŞTERİ)

Kullanım:
    python stereo_recorder.py
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import datetime
import librosa
import time

SAMPLE_RATE = 16000
OUTPUT_FILE = f"cagri_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

MIC_ID     = 1   # Microphone Array (AMD Audio Device)
VIRTUAL_ID = 8   # Voicemeeter Out B1


def record_stereo():
    print(f"🎙️  Sol kanal  → Mikrofon    (ID: {MIC_ID})")
    print(f"🔊  Sağ kanal  → WhatsApp    (ID: {VIRTUAL_ID})")
    print(f"📁  Çıktı: {OUTPUT_FILE}")
    print("\n🔴 Kayıt başladı... Durdurmak için Enter'a bas.\n")

    mic_data     = []
    virtual_data = []

    mic_sr     = int(sd.query_devices(MIC_ID)['default_samplerate'])
    virtual_sr = int(sd.query_devices(VIRTUAL_ID)['default_samplerate'])

    def mic_callback(indata, frames, time, status):
        mic_data.append(indata[:, 0].copy())

    def virtual_callback(indata, frames, time, status):
        virtual_data.append(indata[:, 0].copy())

    with sd.InputStream(device=MIC_ID,     samplerate=mic_sr,
                        channels=1, callback=mic_callback), \
         sd.InputStream(device=VIRTUAL_ID, samplerate=virtual_sr,
                        channels=1, callback=virtual_callback):
        input()  # Enter'a basınca durur

    print("⏹️  Kayıt durduruldu, kaydediliyor...")

    left  = np.concatenate(mic_data)
    right = np.concatenate(virtual_data)

    if mic_sr != SAMPLE_RATE:
        left  = librosa.resample(left,  orig_sr=mic_sr,     target_sr=SAMPLE_RATE)
    if virtual_sr != SAMPLE_RATE:
        right = librosa.resample(right, orig_sr=virtual_sr, target_sr=SAMPLE_RATE)

    min_len = min(len(left), len(right))
    stereo  = np.stack([left[:min_len], right[:min_len]], axis=1)

    sf.write(OUTPUT_FILE, stereo, SAMPLE_RATE)
    print(f"\n✅ Kaydedildi: {OUTPUT_FILE}")
    print(f"   Sol kanal = Mikrofon  (TEMSİLCİ)")
    print(f"   Sağ kanal = WhatsApp  (MÜŞTERİ)")


if __name__ == "__main__":
    record_stereo()

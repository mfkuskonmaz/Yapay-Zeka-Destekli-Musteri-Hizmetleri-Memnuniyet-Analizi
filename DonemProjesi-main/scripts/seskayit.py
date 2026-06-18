import pyaudio
import wave

# Kayıt Ayarları
FORMAT = pyaudio.paInt16 # Ses derinliği
CHANNELS = 1             # Mono kayıt (2 yaparsanız Stereo olur)
RATE = 44100            # Örnekleme hızı (CD kalitesi)
CHUNK = 1024            # Veri blok boyutu
OUTPUT_FILENAME = "kayit.wav"

audio = pyaudio.PyAudio()

# Kaydı Başlatma
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Kayıt yapılıyor... (Durdurmak için Ctrl+C tuşlarına basın)")

frames = []

try:
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
except KeyboardInterrupt:
    print("\nKayıt durduruldu.")

# Akışı Kapatma
stream.stop_stream()
stream.close()
audio.terminate()

# WAV Dosyasına Yazma
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Dosya kaydedildi: {OUTPUT_FILENAME}")
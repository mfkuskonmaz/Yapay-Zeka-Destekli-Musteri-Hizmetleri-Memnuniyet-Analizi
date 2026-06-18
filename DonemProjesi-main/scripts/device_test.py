"""
device_test.py

Her giriş cihazını tek tek test eder, hangisinden ses geldiğini bulur.
Kullanım: python device_test.py
"""

import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
TEST_DURATION = 3  # her cihaz için 3 saniye


def test_device(device_id, device_name):
    try:
        recording = sd.rec(
            int(TEST_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            device=device_id,
            dtype='float32'
        )
        sd.wait()
        volume = np.abs(recording).mean()
        print(f"  [{device_id:3d}] {device_name[:50]:<50} | Ses: {volume:.6f} {'<<< SES VAR' if volume > 0.001 else ''}")
        return volume
    except Exception as e:
        print(f"  [{device_id:3d}] {device_name[:50]:<50} | HATA: {str(e)[:40]}")
        return 0


if __name__ == "__main__":
    devices = sd.query_devices()

    # Sadece Voicemeeter cihazlarını test et
    test_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    print("\nWhatsApp araması YAP ve konuş, test ediliyor...\n")

    for dev_id in test_ids:
        d = devices[dev_id]
        if d['max_input_channels'] > 0:
            test_device(dev_id, d['name'])

    print("\nDone. 'SES VAR' yazan cihaz doğru olan.")

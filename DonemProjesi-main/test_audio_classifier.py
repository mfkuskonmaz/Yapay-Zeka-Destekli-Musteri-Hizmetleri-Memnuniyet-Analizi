"""
Audio model test scripti.
Her sınıftan rastgele 5 ses alır, tahmin yapar, sonucu gösterir.

Kullanım: python test_audio_classifier.py
"""

import random
import numpy as np
from pathlib import Path
from src.audio_analyzer import ProfessionalAudioAnalyzer

DATA_DIR  = Path("data/samples")
LABEL_MAP = {"olumsuz": 0, "notr": 1, "olumlu": 2}

def test():
    analyzer = ProfessionalAudioAnalyzer()

    total, correct = 0, 0
    results = []

    for label_name, label_id in LABEL_MAP.items():
        folder = DATA_DIR / label_name
        files  = sorted(folder.glob("*.wav"))
        sample = random.sample(files, min(5, len(files)))

        print(f"\n{'='*50}")
        print(f"  {label_name.upper()} — {len(sample)} test")
        print(f"{'='*50}")

        for f in sample:
            res        = analyzer.analyze(str(f))
            predicted  = res["label"]
            confidence = res["confidence"]
            is_correct = (predicted == label_name)

            total   += 1
            correct += int(is_correct)

            status = "✅" if is_correct else "❌"
            print(f"  {status} {f.name[:35]:<35} → {predicted:<8} ({confidence:.0%})")

            if not is_correct:
                print(f"     Olasılıklar: {res['probs']}")

    print(f"\n{'='*50}")
    print(f"  SONUÇ: {correct}/{total} doğru  ({correct/total:.0%})")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    test()

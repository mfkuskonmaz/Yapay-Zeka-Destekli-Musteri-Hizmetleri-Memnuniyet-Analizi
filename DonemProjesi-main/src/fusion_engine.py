"""
src/fusion_engine.py

BERT (metin) + SVM (ses) sonuçlarını birleştirerek nihai duygu kararı verir.

Ağırlıklar:
  - Metin (BERT) : %60
  - Ses   (SVM)  : %40

Kinaye tespiti:
  Metin → olumsuz ama ses → olumlu ise kinaye/ironi işaretlenir.
"""


# Etiket → sayısal index eşlemesi (her iki model için ortak)
LABEL_TO_ID = {"olumsuz": 0, "negatif": 0,
               "notr":    1, "nötr":    1, "neutral": 1,
               "olumlu":  2, "pozitif": 2, "positive": 2}

ID_TO_LABEL = {0: "olumsuz", 1: "notr", 2: "olumlu"}

TEXT_WEIGHT  = 0.60
AUDIO_WEIGHT = 0.40


def _normalize_label(raw_label: str) -> int:
    """Herhangi bir etiket stringini 0/1/2 indexine çevirir."""
    return LABEL_TO_ID.get(str(raw_label).lower().strip(), 1)


def _build_prob_vector(result: dict, source: str) -> list[float]:
    """
    Modelin döndürdüğü sonuçtan [p_olumsuz, p_notr, p_olumlu] vektörü üretir.
    Olasılık yoksa one-hot tahmin kullanır.
    """
    # SVM path: result["probs"] mevcutsa kullan
    if "probs" in result:
        probs = result["probs"]
        return [
            probs.get("olumsuz", probs.get("negatif", 0.0)),
            probs.get("notr",    probs.get("nötr",    0.0)),
            probs.get("olumlu",  probs.get("pozitif", 0.0)),
        ]

    # BERT path: result["scores"] listesi varsa kullan
    if "scores" in result:
        scores = result["scores"]           # [{"label":..., "score":...}, ...]
        vec    = [0.0, 0.0, 0.0]
        for item in scores:
            idx = _normalize_label(item["label"])
            vec[idx] = item["score"]
        return vec

    # Sadece etiket varsa one-hot
    idx = _normalize_label(result.get("label", "notr"))
    vec = [0.0, 0.0, 0.0]
    vec[idx] = 1.0
    return vec


class MultimodalFusion:

    def merge(self, text_res: dict, audio_res: dict) -> dict:
        """
        Parametreler
        ------------
        text_res  : TextAnalyzer.analyze() çıktısı
                    Beklenen alanlar: "label", opsiyonel "scores"
        audio_res : ProfessionalAudioAnalyzer.analyze() çıktısı
                    Beklenen alanlar: "label", "probs", "confidence"

        Döndürür
        --------
        {
            "final_sentiment" : "olumlu" | "notr" | "olumsuz",
            "final_label_id"  : 0 | 1 | 2,
            "confidence"      : float,
            "sarcasm_detected": bool,
            "text_label"      : str,
            "audio_label"     : str,
            "fusion_probs"    : [p_olumsuz, p_notr, p_olumlu],
            "details"         : {"text": text_res, "audio": audio_res}
        }
        """
        text_vec  = _build_prob_vector(text_res,  "text")
        audio_vec = _build_prob_vector(audio_res, "audio")

        # Ağırlıklı ortalama
        fused = [TEXT_WEIGHT * t + AUDIO_WEIGHT * a
                 for t, a in zip(text_vec, audio_vec)]

        final_id   = int(fused.index(max(fused)))
        confidence = round(max(fused), 4)

        text_id  = _normalize_label(text_res.get("label",  "notr"))
        audio_id = _normalize_label(audio_res.get("label", "notr"))

        # Kinaye: metin olumsuz, ses olumlu — ve fark belirginse
        sarcasm = (
            text_id  == 0 and
            audio_id == 2 and
            audio_res.get("confidence", 0) > 0.55
        )

        if sarcasm:
            final_id = 0   # Kinayede metni önceliklendir

        return {
            "final_sentiment" : ID_TO_LABEL[final_id],
            "final_label_id"  : final_id,
            "confidence"      : confidence,
            "sarcasm_detected": sarcasm,
            "text_label"      : ID_TO_LABEL.get(text_id,  "notr"),
            "audio_label"     : ID_TO_LABEL.get(audio_id, "notr"),
            "fusion_probs"    : [round(p, 4) for p in fused],
            "details"         : {"text": text_res, "audio": audio_res},
        }

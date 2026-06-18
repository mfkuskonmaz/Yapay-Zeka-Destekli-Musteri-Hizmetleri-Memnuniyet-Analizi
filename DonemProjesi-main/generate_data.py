import pandas as pd
import random

# =====================================================
# ULTIMATE CALL CENTER DATASET GENERATOR (ACADEMIC FINAL)
# =====================================================

def create_call_center_dataset(size_per_class=8000):
    data = []

    # -------------------
    # KELİME HAVUZLARI
    # -------------------
    selamlar = [
        "İyi günler", "Merhaba", "İyi akşamlar", "Günaydın",
        "Merhabalar", "İyi çalışmalar", "Selamlar"
    ]

    isimler = [
        "Uğur", "Ahmet", "Mehmet", "Ayşe", "Fatma", "Zeynep",
        "Elif", "Can", "Mert", "Burcu", "Selin", "Ceren", "Emre"
    ]

    hitaplar = ["Hanım", "Bey", "", "efendim"]

    departmanlar = [
        "müşteri hizmetlerinden", "teknik destekten",
        "altyapı biriminden", "fatura servisinden"
    ]

    servisler = [
        "internetim", "bağlantım", "wifi", "modem",
        "fiber hat", "ev interneti"
    ]

    sorunlar = [
        "çalışmıyor", "sürekli kopuyor", "çok yavaş",
        "ışıklar yanmıyor", "hata veriyor"
    ]

    zamanlar = ["bir saattir", "iki gündür", "dünden beri", "sabahdan beri"]
    baglaclar = ["ama", "ancak", "fakat", "çünkü", "bu yüzden"]

    # ---- DUYGU HAVUZLARI ----
    olumsuz_duygular = [
        "gerçekten mağdur oldum", "artık sabrım kalmadı",
        "kimse ilgilenmedi", "böyle hizmet olmaz",
        "iptal etmeyi düşünüyorum"
    ]

    olumlu_duygular = [
        "çok memnun kaldım", "teşekkür ederim",
        "elinize sağlık", "harika oldu",
        "sorunum çözüldü", "beklediğimden hızlı çözüldü",
        "sağ olun", "çok teşekkürler"
    ]

    notr_aksiyonlar = [
        "sistemi kontrol ediyorum",
        "kaydınızı oluşturuyorum",
        "işlemi başlatıyorum",
        "adres bilgilerinizi teyit eder misiniz",
        "talebinizi üst birime iletiyorum"
    ]

    kurumsal_notr = [
        "KVKK kapsamında onayınızı almam gerekiyor.",
        "Şu an hattınızda aktif bir arıza görünmüyor.",
        "İşleminiz onay sürecindedir, lütfen hattan ayrılmayın.",
        "Sistemde kayıtlı adresinizi teyit edebilir miyim?"
    ]

    # -------------------
    # ASR HATA SİMÜLASYONU
    # -------------------
    def asr_noise(text):
        if random.random() < 0.20:
            replacements = {"ğ":"g","ş":"s","ç":"c","ö":"o","ü":"u","ı":"i"}
            for k, v in replacements.items():
                if random.random() < 0.4:
                    text = text.replace(k, v)
        if random.random() < 0.10:
            words = text.split()
            if len(words) > 4:
                words.pop(random.randint(0, len(words)-1))
                text = " ".join(words)
        return text

    # -------------------
    # VERİ ÜRETİMİ (BALANCED TRAINING)
    # -------------------
    for _ in range(size_per_class):
        # NÖTR (1)
        if random.random() > 0.5:
            n_text = f"{random.choice(selamlar)} {random.choice(isimler)} {random.choice(hitaplar)}, {random.choice(departmanlar)} arıyorum. {random.choice(notr_aksiyonlar)}."
        else:
            n_text = random.choice(kurumsal_notr)
        data.append({"text": asr_noise(n_text), "label": 1})

        # OLUMSUZ (0)
        neg_text = f"{random.choice(zamanlar)} {random.choice(servisler)} {random.choice(sorunlar)}. {random.choice(baglaclar)} {random.choice(olumsuz_duygular)}."
        data.append({"text": asr_noise(neg_text), "label": 0})

        # OLUMLU (2)
        pos_text = f"{random.choice(zamanlar)} sorun yaşıyordum. {random.choice(notr_aksiyonlar)}, {random.choice(baglaclar)} {random.choice(olumlu_duygular)}."
        data.append({"text": asr_noise(pos_text), "label": 2})

    df = pd.DataFrame(data).sample(frac=1).reset_index(drop=True)
    return df


# -------------------
# DATASET OLUŞTUR
# -------------------
if __name__ == "__main__":
    df = create_call_center_dataset(8000)
    df.to_csv("cagri_merkezi_ACADEMIC_FINAL.csv", index=False, encoding="utf-8")
    print("Dataset hazır:")
    print(df['label'].value_counts())

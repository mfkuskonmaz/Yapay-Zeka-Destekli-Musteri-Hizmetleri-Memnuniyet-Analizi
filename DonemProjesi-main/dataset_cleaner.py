import pandas as pd

LABEL_MAP = {"pozitif": "pozitif", "negatif": "negatif"}

def clean_dataset(input_path, output_path):
    df = pd.read_csv(input_path, encoding="utf-8", on_bad_lines="skip")
    
    # Sütunları netleştir
    df = df[["review", "label"]]
    
    # Normalizasyon
    df["review"] = df["review"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map(LABEL_MAP)
    
    # Geçersizleri temizle
    df = df.dropna(subset=["label"])
    
    print(f"✔ Temizleme tamamlandı. Kalan veri: {len(df)}")
    print(df["label"].value_counts())
    
    df.to_csv(output_path, index=False, encoding="utf-8")
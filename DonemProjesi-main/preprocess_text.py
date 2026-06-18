import re
import pandas as pd

def clean_text(text: str) -> str:
    """
    BERT için hafifletilmiş temizlik.
    """
    if pd.isna(text):
        return ""

    # Küçük harfe çevir
    text = text.lower()
    
    # URL ve kullanıcı adlarını temizle
    text = re.sub(r"http\S+|www\S+|@\w+", "", text)

    # Sadece harfleri ve boşlukları koru (BERT Türkçe karakterleri destekler)
    text = re.sub(r"[^a-zçğıöşü\s]", " ", text)

    # Fazla boşlukları temizle
    text = re.sub(r"\s+", " ", text).strip()

    return text

def preprocess_csv(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    if "review" not in df.columns:
        raise ValueError("CSV dosyasında 'review' kolonu bulunmalı")

    df["review"] = df["review"].apply(clean_text)
    
    # Boş kalan veya çok kısa satırları at
    df = df[df["review"].str.len() > 2]
    
    df.to_csv(output_path, index=False)
    print(f"✔ BERT uyumlu temizleme tamamlandı: {output_path}")
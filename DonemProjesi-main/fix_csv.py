import pandas as pd

def fix_my_csv(input_path, output_path):
    # Bozuk satırları atlamadan, ham metin olarak oku
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    fixed_data = []
    # Header'ı geç (ilk satır)
    for line in lines[1:]:
        line = line.strip()
        if not line: continue
        
        # Satırı sondaki virgülden ayır (etiketi al)
        parts = line.rsplit(',', 1) 
        
        if len(parts) == 2:
            review = parts[0].replace('"', '') # Varsa eski tırnakları temizle
            label = parts[1]
            # Metni tırnak içine alarak güvenli hale getir
            fixed_data.append([review, label])

    # Yeni, temiz CSV'yi oluştur
    df_fixed = pd.DataFrame(fixed_data, columns=["review", "label"])
    df_fixed.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✔ {len(df_fixed)} satır düzeltildi ve {output_path} olarak kaydedildi.")

if __name__ == "__main__":
    fix_my_csv("data/raw/text_reviews.csv", "data/raw/text_reviews_fixed.csv")
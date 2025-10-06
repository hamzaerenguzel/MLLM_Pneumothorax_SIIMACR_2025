from pathlib import Path
import base64
import google.generativeai as genai
import csv
import os

# === Ayarlar ===
ROOT = Path(r"C:\Users\Güzel\Desktop\pngs")   # klasör yolu
OUT_CSV = ROOT / "gemini_results.csv"         # çıktı dosyası
MODEL = "gemini-2.0-flash"                    # en güncel hızlı sürüm (dilersen "gemini-2.0-pro" da seçebilirsin)

# API key ortam değişkeninden oku
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def encode_image(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# tüm png dosyalarını bul
all_pngs = list(ROOT.rglob("*.png"))
print(f"{len(all_pngs)} dosya bulundu.")

results = []

# ✅ Güncellenmiş prompt
PROMPT = (
    "You are a careful radiology assistant. Examine this chest X-ray for pneumothorax.\n"
    "Return exactly ONE character: 1 if pneumothorax is clearly present or strongly suspected, "
    "and 0 if pneumothorax is absent.\n\n"
    "Answer only with 1 or 0."
)

for img_path in all_pngs:
    img_b64 = encode_image(img_path)

    try:
        response = genai.GenerativeModel(MODEL).generate_content(
            [PROMPT, {"mime_type": "image/png", "data": img_b64}]
        )
        answer = response.text.strip()
    except Exception as e:
        answer = f"Error: {e}"

    results.append((str(img_path), answer))
    print(f"{img_path} -> {answer}")

# CSV çıktısı
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "result"])
    writer.writerows(results)

print(f"\nTüm sonuçlar kaydedildi: {OUT_CSV}")
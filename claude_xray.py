from pathlib import Path
import base64
import os
import csv
from anthropic import Anthropic

# === Ayarlar ===
ROOT = Path(r"C:\Users\Güzel\Desktop\pngs")   # klasör yolu
OUT_CSV = ROOT / "claude4_results.csv"        # çıktı dosyası
MODEL = "claude-sonnet-4-20250514"            # Claude 4 Sonnet (Opus için: "claude-opus-4-20250514")

# API key ortam değişkeninden oku
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def encode_image(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# tüm png dosyalarını bul
all_pngs = list(ROOT.rglob("*.png"))
print(f"{len(all_pngs)} dosya bulundu.")

results = []

# ✅ Senin istediğin prompt
PROMPT = (
    "You are a radiology assistant. Carefully analyze this chest X-ray for pneumothorax.\n"
    "Return exactly ONE character: 1 if pneumothorax is present, 0 if absent.\n\n"
    "If uncertain, choose the answer that best minimizes diagnostic error.\n\n"
    "Answer only with 1 or 0."
)

for img_path in all_pngs:
    img_b64 = encode_image(img_path)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=5,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                    ],
                }
            ],
        )
        answer = response.content[0].text.strip()
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

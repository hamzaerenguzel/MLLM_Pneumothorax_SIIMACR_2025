# run_pnx.py 
# PA akciğer grafilerinde pnömotoraks: 1/0 (kararsızsa 1) 
# Klasör: C:\Users\Güzel\Desktop\pngs ve tüm alt klasörler 

from pathlib import Path 
import base64, csv, os, time 
from openai import OpenAI 
from tqdm import tqdm 

# === Ayarlar === 
ROOT = Path(r"C:\Users\Güzel\Desktop\pngs") 
OUT_CSV = ROOT / "pnx_results.csv" 
MODEL = "gpt-4o"  # erişiminize göre: "gpt-5", "gpt-4o" vb. 

PROMPT = ( 
    "You are a careful radiology assistant. Examine this chest X-ray for pneumothorax.\n"
    "Return exactly ONE character: 1 if pneumothorax is clearly present or strongly suspected, "
    "and 0 if pneumothorax is absent.\n\n"
    "Answer only with 1 or 0."
) 

# === Başlat === 
client = OpenAI() 

def classify_png(p: Path) -> str: 
    """Görseli gönder, yalnızca '1' veya '0' döndür. Hata/kararsızlıkta '1'.""" 
    with p.open("rb") as f: 
        b64 = base64.b64encode(f.read()).decode("utf-8") 

    resp = client.responses.create( 
        model=MODEL, 
        input=[{ 
            "role": "user", 
            "content": [ 
                {"type": "input_text", "text": PROMPT}, 
                {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}, 
            ], 
        }], 
        timeout=120.0, 
    ) 

    text = (resp.output_text or "").strip().lower() 

    if text in ("1", "0"): 
        return text 
    if text in ("yes", "var", "evet"): 
        return "1" 
    if text in ("no", "yok", "hayir", "hayır"): 
        return "0" 
    return "1"  # emniyet: kararsızlıkta 1 

def main(): 
    files = sorted(ROOT.rglob("*.png")) 
    if not files: 
        print(f"No PNG files found under: {ROOT}") 
        return 

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True) 
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f: 
        w = csv.writer(f) 
        w.writerow(["filename", "model", "pnx_1_or_0"]) 

        for p in tqdm(files, desc="Processing", unit="img"): 
            for attempt in range(4): 
                try: 
                    label = classify_png(p) 
                    w.writerow([p.name, MODEL, label]) 
                    break 
                except Exception as e: 
                    if attempt == 3: 
                        w.writerow([p.name, MODEL, "1"]) 
                    time.sleep(2 * (attempt + 1)) 

    print(f"\nDone. Results -> {OUT_CSV}") 

if __name__ == "__main__": 
    main()

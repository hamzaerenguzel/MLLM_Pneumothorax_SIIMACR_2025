# -*- coding: utf-8 -*-
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut

# ==========================
# KULLANICI AYARLARI
# ==========================
IN_ROOT  = Path(r"C:\Users\Güzel\Desktop\dicoms")   # DICOM kök klasörü
OUT_ROOT = Path(r"C:\Users\Güzel\Desktop\pngs")     # PNG çıktılarının gideceği klasör (yapı korunur)

USE_VOI = True              # VOI LUT/Window Center-Width uygula
P_LOW, P_HIGH = 0.5, 99.5   # Robust normalize yüzdelikleri
MAX_SIDE = None             # Uzun kenar sınırı (örn. 2048); None -> orijinal boyut korunur
OVERWRITE = True            # PNG varsa üzerine yaz

SAVE_RGB = False            # Model 3-kanal bekliyorsa True: griyi RGB'ye kopyalar (aynı piksel)
MAKE_MANIFEST = True        # SOPInstanceUID -> PNG yolu eşlemesi kaydı

# --- İsteğe bağlı geliştirmeler (varsayılan: kapalı) ---
USE_CLAHE   = False         # CLAHE uygula (skimage gerekiyorsa, yoksa atlar)
USE_UNSHARP = False         # Hafif unsharp mask uygula
CLAHE_CLIP  = 2.0
CLAHE_TILE  = (8, 8)
UNSHARP_RADIUS  = 1.0
UNSHARP_PERCENT = 150

# skimage varsa CLAHE için kullan
try:
    from skimage import exposure as _skex
    _HAS_SK = True
except Exception:
    _HAS_SK = False

# ==========================
# YARDIMCI FONKSİYONLAR
# ==========================
def is_dicom(path: Path) -> bool:
    """Basit DICOM kontrolü: .dcm uzantısı veya 128. bayttan 'DICM' imzası."""
    if path.suffix.lower() == ".dcm":
        return True
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False

def load_dicom_float(ds, use_voi=True):
    """
    DICOM pikselini float32'e getir:
    1) apply_modality_lut (slope/intercept)
    2) opsiyonel apply_voi_lut (window/VOI)
    MONOCHROME1 ise invert bilgisi döndür.
    Multi-frame gelirse ilk frame alınır.
    """
    arr = ds.pixel_array
    # Multi-frame ise ilk frame
    if isinstance(arr, np.ndarray) and arr.ndim >= 3:
        # (frames, H, W) veya (H, W, channels) olabilir; chest X-ray genellikle mono
        if arr.shape[0] <= 4 and arr.ndim == 3:     # muhtemel (frames, H, W)
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] in (3, 4):  # renkli ise luminance'a indir (nadirdir)
            # Yalın ortalama (basit ve hızlı)
            arr = arr[..., :3].mean(axis=-1)

    # 1) Modality LUT
    try:
        arr = apply_modality_lut(arr, ds)
    except Exception:
        pass

    # 2) VOI LUT / Window Center-Width
    if use_voi:
        try:
            arr = apply_voi_lut(arr, ds)
        except Exception:
            pass

    phot = str(ds.get("PhotometricInterpretation", "")).upper()
    invert = (phot == "MONOCHROME1")
    # Float32'e çevir
    arr = np.asarray(arr, dtype=np.float32)
    return arr, invert

def normalize_to_u8(arr: np.ndarray, invert=False, p_low=0.5, p_high=99.5) -> np.ndarray:
    """
    Robust yüzdeliklere göre [0..255] aralığına normalizasyon.
    Düzgün değilse min-max fallback. MONOCHROME1 için invert uygular.
    """
    # Güvenlik: NaN/Inf temizle
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=np.max(arr[np.isfinite(arr)]) if np.any(np.isfinite(arr)) else 0.0, neginf=np.min(arr[np.isfinite(arr)]) if np.any(np.isfinite(arr)) else 0.0)
        arr = arr.astype(np.float32)

    lo, hi = np.percentile(arr, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.uint8)

    arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr, 0.0, 1.0)
    if invert:
        arr = 1.0 - arr
    return (arr * 255.0 + 0.5).astype(np.uint8)

def resize_long_side(gray: np.ndarray, max_side=None) -> np.ndarray:
    """Uzun kenarı max_side olacak şekilde yeniden boyutlandır (downsample LANCZOS önerilir)."""
    if not max_side:
        return gray
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return gray
    scale = max_side / float(m)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = Image.fromarray(gray, mode="L")
    # LANCZOS downsample'da daha net sonuç verir
    img = img.resize((new_w, new_h), resample=Image.LANCZOS)
    return np.array(img)

def maybe_enhance(gray_u8: np.ndarray) -> np.ndarray:
    """Opsiyonel CLAHE ve/veya unsharp mask. Girdi/çıktı: uint8 [0..255]."""
    out = gray_u8
    if USE_CLAHE:
        if _HAS_SK:
            f = (out / 255.0)
            f = _skex.equalize_adapthist(f, clip_limit=CLAHE_CLIP, nbins=256)
            out = (f * 255.0 + 0.5).astype(np.uint8)
        else:
            print("[WARN] CLAHE aktif ama scikit-image bulunamadı; atlanıyor.")
    if USE_UNSHARP:
        im = Image.fromarray(out, mode="L").filter(
            ImageFilter.UnsharpMask(radius=UNSHARP_RADIUS, percent=UNSHARP_PERCENT, threshold=0)
        )
        out = np.array(im, dtype=np.uint8)
    return out

def convert_one(dcm_path: Path, out_png: Path,
                use_voi=True, p_low=0.5, p_high=99.5,
                max_side=None, overwrite=False, save_rgb=False):
    """Tek DICOM'u PNG'ye çevir ve kaydet."""
    if out_png.exists() and not overwrite:
        return "skip", None
    try:
        ds = pydicom.dcmread(str(dcm_path), force=True)
        arr_f32, invert = load_dicom_float(ds, use_voi=use_voi)
        u8 = normalize_to_u8(arr_f32, invert=invert, p_low=p_low, p_high=p_high)
        u8 = resize_long_side(u8, max_side=max_side)
        u8 = maybe_enhance(u8)

        out_png.parent.mkdir(parents=True, exist_ok=True)

        if save_rgb:
            rgb = np.stack([u8, u8, u8], axis=-1)
            Image.fromarray(rgb, mode="RGB").save(out_png, optimize=True)
        else:
            Image.fromarray(u8, mode="L").save(out_png, optimize=True)

        sop = getattr(ds, "SOPInstanceUID", None)
        return "ok", sop
    except Exception as e:
        # TransferSyntax bilgisini eklemek, sıkıştırma kaynaklı hatalarda yol gösterir
        try:
            ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
        except Exception:
            ts = None
        return f"err: {e} (TS={ts})", None

# ==========================
# ANA AKIŞ
# ==========================
def main():
    total, ok, skip, err = 0, 0, 0, 0
    manifest_rows = []

    for path in IN_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if not is_dicom(path):
            continue

        rel = path.relative_to(IN_ROOT)
        out_png = OUT_ROOT.joinpath(rel).with_suffix(".png")
        total += 1

        status, sop = convert_one(
            path, out_png,
            use_voi=USE_VOI,
            p_low=P_LOW, p_high=P_HIGH,
            max_side=MAX_SIDE,
            overwrite=OVERWRITE,
            save_rgb=SAVE_RGB
        )

        if status == "ok":
            ok += 1
            if MAKE_MANIFEST:
                manifest_rows.append({
                    "sop_instance_uid": sop,
                    "png_path": str(out_png),
                    "src_rel": str(rel)
                })
        elif status == "skip":
            skip += 1
        else:
            err += 1
            print(f"[ERR] {rel}: {status}")

    print(f"Done. total={total} ok={ok} skip={skip} err={err}")
    print(f"PNG root: {OUT_ROOT}")

    if MAKE_MANIFEST and manifest_rows:
        try:
            import pandas as pd
            OUT_ROOT.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(manifest_rows).to_csv(OUT_ROOT / "manifest.csv", index=False)
            print(f"Manifest yazıldı: {OUT_ROOT / 'manifest.csv'}")
        except Exception as e:
            print(f"[WARN] Manifest yazılamadı: {e}")

if __name__ == "__main__":
    main()

"""
Aplikasi web kompresi gambar dengan PCA -- FastAPI backend.

Endpoint:
  GET  /              -> halaman utama (upload + kontrol + hasil)
  POST /api/spectrum  -> terima gambar, kembalikan kurva nilai singular
                          (untuk grafik pemilihan rank di UI) + info ukuran
  POST /api/compress  -> terima gambar + parameter kompresi, kembalikan
                          gambar hasil (base64) + metrik (runtime, ukuran,
                          rasio kompresi, SSIM/PSNR)

Tidak ada penyimpanan file sementara di server: gambar hasil dikirim
langsung sebagai base64 dalam respons JSON, dan diunduh dari sisi browser
(Blob). Ini membuat aplikasi stateless dan sederhana untuk dijalankan
secara lokal.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from starlette.requests import Request

from image_metrics import build_metrics, encode_png
from pca_engine import (
    compute_psnr,
    compute_ssim,
    decompose_channel,
    decompose_image,
    find_optimal_k,
    luminance,
    reconstruct_image,
)

BASE_DIR = Path(__file__).resolve().parent

# Batas atas eksplorasi nilai k. Di atas titik ini, tambahan singular value
# pada foto natural praktis tidak terlihat lagi, jadi nilai ini cukup besar
# untuk kualitas tinggi sekaligus menjaga performa & ukuran respons.
MAX_K_CAP = 300

# Sisi terpanjang gambar yang diizinkan sebelum diperkecil otomatis. SVD
# pada gambar yang jauh lebih besar dari ini akan terasa lambat untuk demo
# interaktif, jadi gambar besar diperkecil (bukan ditolak) dengan kualitas
# resampling tinggi (LANCZOS) -- pengguna tetap diberi tahu lewat resize_info.
MAX_IMAGE_SIDE = 3000

QUALITY_PRESETS = {
    "high": 0.98,
    "balanced": 0.95,
    "aggressive": 0.90,
}

app = FastAPI(title="PCA Image Compressor")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def load_and_prepare_image(raw_bytes: bytes) -> tuple[np.ndarray, dict]:
    """Decode bytes upload -> array RGB uint8, memperkecil otomatis jika sisi
    terpanjangnya melebihi MAX_IMAGE_SIDE. Dipakai bersama oleh /api/spectrum
    dan /api/compress supaya kedua endpoint selalu konsisten."""
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes))
        pil_image.load()
    except Exception as exc:  # noqa: BLE001 - tampilkan pesan ramah ke user
        raise HTTPException(status_code=400, detail=f"Gagal membaca gambar: {exc}") from exc

    # Konversi ke RGB (menghapus alpha/palette) agar konsisten 3 kanal warna.
    pil_image = pil_image.convert("RGB")
    original_width, original_height = pil_image.size

    resized = False
    if max(original_width, original_height) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(original_width, original_height)
        new_size = (round(original_width * scale), round(original_height * scale))
        pil_image = pil_image.resize(new_size, Image.LANCZOS)
        resized = True

    image_array = np.asarray(pil_image, dtype=np.uint8)
    resize_info = {
        "resized": resized,
        "original_width": original_width,
        "original_height": original_height,
        "width": pil_image.width,
        "height": pil_image.height,
    }
    return image_array, resize_info


def read_uploaded_image(file: UploadFile, raw_bytes: bytes) -> None:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diunggah harus berupa gambar.")
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="File gambar kosong atau gagal dibaca.")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/spectrum")
async def spectrum(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    read_uploaded_image(file, raw_bytes)
    image_array, resize_info = load_and_prepare_image(raw_bytes)
    height, width = image_array.shape[:2]
    k_max = min(min(height, width), MAX_K_CAP)

    gray = luminance(image_array)
    decomp = decompose_channel(gray, k_cap=k_max)

    return {
        "singular_values": decomp.S.tolist(),
        "k_max": k_max,
        "width": width,
        "height": height,
        "resize_info": resize_info,
    }


@app.post("/api/compress")
async def compress(
    file: UploadFile = File(...),
    mode: Literal["manual", "auto"] = Form(...),
    k: Optional[int] = Form(None),
    quality: Optional[str] = Form(None),
):
    raw_bytes = await file.read()
    read_uploaded_image(file, raw_bytes)
    image_array, resize_info = load_and_prepare_image(raw_bytes)
    height, width = image_array.shape[:2]

    true_k_max = min(height, width)
    k_max = min(true_k_max, MAX_K_CAP)

    start = time.perf_counter()
    decomps = decompose_image(image_array, k_cap=k_max)

    if mode == "manual":
        if k is None:
            raise HTTPException(status_code=400, detail="Parameter k wajib diisi pada mode manual.")
        k_used = max(1, min(k, k_max))
        reconstructed = reconstruct_image(decomps, k_used)
    else:
        target_ssim = QUALITY_PRESETS.get(quality or "balanced")
        if target_ssim is None:
            raise HTTPException(status_code=400, detail="Preset kualitas tidak dikenali.")
        k_used = find_optimal_k(decomps, image_array, target_ssim, k_max)
        reconstructed = reconstruct_image(decomps, k_used)

    runtime_ms = (time.perf_counter() - start) * 1000

    ssim_score = compute_ssim(image_array, reconstructed)
    psnr_score = compute_psnr(image_array, reconstructed)
    compressed_png_bytes = encode_png(reconstructed)

    metrics = build_metrics(
        original_image=image_array,
        original_bytes=len(raw_bytes),
        compressed_png_bytes=compressed_png_bytes,
        k_used=k_used,
        k_max=k_max,
        runtime_ms=runtime_ms,
        ssim_score=ssim_score,
        psnr_score=psnr_score,
    )

    return {
        "image_base64": base64.b64encode(compressed_png_bytes).decode("ascii"),
        "resize_info": resize_info,
        "metrics": {
            "original_bytes": metrics.original_bytes,
            "compressed_bytes": metrics.compressed_bytes,
            "file_reduction_pct": round(metrics.file_reduction_pct, 2),
            "pca_reduction_pct": round(metrics.pca_reduction_pct, 2),
            "width": metrics.width,
            "height": metrics.height,
            "k_used": metrics.k_used,
            "k_max": metrics.k_max,
            "runtime_ms": round(metrics.runtime_ms, 2),
            "ssim": round(metrics.ssim, 4),
            "psnr": None if psnr_score == float("inf") else round(metrics.psnr, 2),
        },
    }

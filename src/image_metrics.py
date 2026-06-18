"""
Modul untuk menghitung metrik hasil kompresi: ukuran data, rasio kompresi,
dan kualitas (PSNR/SSIM sudah ada di pca_engine, di sini hanya yang terkait
ukuran & runtime).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class CompressionMetrics:
    original_bytes: int
    compressed_bytes: int
    file_reduction_pct: float
    pca_reduction_pct: float
    width: int
    height: int
    k_used: int
    k_max: int
    runtime_ms: float
    ssim: float
    psnr: float


def encode_png(image: np.ndarray) -> bytes:
    """Encode array gambar (uint8 RGB) menjadi bytes PNG (lossless).

    PNG dipakai (bukan JPEG) agar efek "kompresi" yang terlihat murni
    berasal dari rank-reduction PCA, tidak tercampur artefak kompresi
    lossy lain -- representasi paling jujur untuk demo akademik.
    """
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def pca_storage_values(height: int, width: int, k: int, channels: int = 3) -> int:
    """Jumlah nilai numerik yang perlu disimpan untuk merepresentasikan
    gambar hasil rank-k approximation: U_k (h*k) + S_k (k) + Vt_k (k*w),
    dikali jumlah kanal warna.
    """
    return channels * (height * k + k + k * width)


def build_metrics(
    original_image: np.ndarray,
    original_bytes: int,
    compressed_png_bytes: bytes,
    k_used: int,
    k_max: int,
    runtime_ms: float,
    ssim_score: float,
    psnr_score: float,
) -> CompressionMetrics:
    height, width = original_image.shape[:2]

    compressed_bytes = len(compressed_png_bytes)

    original_values = height * width * 3
    compressed_values = pca_storage_values(height, width, k_used)
    pca_reduction_pct = max(0.0, (1 - compressed_values / original_values) * 100)

    # Tidak di-clamp ke 0: untuk gambar dengan warna flat/sintetis, PNG asli
    # bisa sudah sangat efisien sehingga rekonstruksi PCA (gradasi halus)
    # justru menghasilkan file lebih besar. Nilai negatif di sini jujur
    # menunjukkan itu, alih-alih disembunyikan menjadi "0%".
    file_reduction_pct = (1 - compressed_bytes / original_bytes) * 100 if original_bytes else 0.0

    return CompressionMetrics(
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        file_reduction_pct=file_reduction_pct,
        pca_reduction_pct=pca_reduction_pct,
        width=width,
        height=height,
        k_used=k_used,
        k_max=k_max,
        runtime_ms=runtime_ms,
        ssim=ssim_score,
        psnr=psnr_score,
    )

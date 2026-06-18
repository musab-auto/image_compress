"""
Modul inti algoritma kompresi gambar dengan Principal Component Analysis (PCA).

Langkah PCA klasik (mean -> covariance -> eigenvalue/eigenvector -> reduksi)
diimplementasikan lewat Singular Value Decomposition (SVD), karena untuk data
X yang sudah di-mean-center:

    X' = U S V^T

eigenvector dari covariance matrix C = X'^T X' / (n-1) adalah kolom-kolom V,
dan eigenvalue-nya adalah S^2 / (n-1). Pendekatan ini equivalen secara
matematis dengan dekomposisi eigen pada matriks kovarians, tapi jauh lebih
stabil & efisien secara numerik -- inilah yang dipakai oleh hampir semua
library (numpy, scikit-learn, dll).

Setiap kanal warna (R, G, B) dikompresi secara independen dengan nilai k yang
sama, lalu digabung kembali. Ini menjaga agar warna asli gambar tetap
terjaga sesuai requirement tugas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.metrics import structural_similarity as ssim


@dataclass
class ChannelSVD:
    """Hasil SVD penuh (sampai k_cap) untuk satu kanal warna, dipakai ulang
    agar pencarian k tidak perlu menghitung SVD berkali-kali."""

    mean: np.ndarray
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray


def decompose_channel(channel: np.ndarray, k_cap: int) -> ChannelSVD:
    """Hitung SVD satu kanal (matriks 2D, nilai 0-255) dan simpan hanya
    k_cap komponen pertama. k_cap adalah batas atas eksplorasi nilai k
    (bukan hasil akhir kompresi) supaya performa tetap cepat untuk
    pencarian otomatis (Smart Auto-Compress).
    """
    centered = channel.astype(np.float64)
    mean = centered.mean(axis=0)
    centered = centered - mean

    # full_matrices=False -> hanya hitung komponen yang berguna (thin SVD)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    k_cap = min(k_cap, S.shape[0])
    return ChannelSVD(mean=mean, U=U[:, :k_cap], S=S[:k_cap], Vt=Vt[:k_cap, :])


def reconstruct_channel(decomp: ChannelSVD, k: int) -> np.ndarray:
    """Rekonstruksi kanal dari k komponen utama (singular value) pertama.

    Ini adalah langkah "menghasilkan dataset baru" pada algoritma PCA --
    versi rank-k approximation dari data asli.
    """
    k = max(1, min(k, decomp.S.shape[0]))
    U_k = decomp.U[:, :k]
    S_k = decomp.S[:k]
    Vt_k = decomp.Vt[:k, :]
    reconstructed = (U_k * S_k) @ Vt_k + decomp.mean
    return np.clip(reconstructed, 0, 255)


def decompose_image(image: np.ndarray, k_cap: int) -> list[ChannelSVD]:
    """Pisahkan gambar RGB menjadi 3 kanal dan hitung SVD masing-masing."""
    return [decompose_channel(image[:, :, c], k_cap) for c in range(3)]


def luminance(image: np.ndarray) -> np.ndarray:
    """Konversi RGB ke satu kanal grayscale (bobot luma ITU-R BT.601).

    Dipakai khusus untuk menghasilkan satu kurva spektrum nilai singular
    yang representatif untuk UI -- kompresi sesungguhnya tetap dihitung
    per-kanal R/G/B terpisah (lihat decompose_image) agar warna terjaga.
    """
    weights = np.array([0.299, 0.587, 0.114])
    return image.astype(np.float64) @ weights


def reconstruct_image(decomps: list[ChannelSVD], k: int) -> np.ndarray:
    """Gabungkan kembali 3 kanal hasil rekonstruksi rank-k menjadi gambar RGB."""
    channels = [reconstruct_channel(d, k) for d in decomps]
    return np.stack(channels, axis=-1).astype(np.uint8)


def compute_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Structural Similarity Index -- ukuran kemiripan struktural/persepsi
    antara gambar asli dan hasil kompresi (1.0 = identik)."""
    return float(
        ssim(original, reconstructed, channel_axis=2, data_range=255)
    )


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio dalam dB -- makin tinggi makin mirip."""
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10((255.0 ** 2) / mse))


def find_optimal_k(
    decomps: list[ChannelSVD],
    original: np.ndarray,
    target_ssim: float,
    k_max: int,
) -> int:
    """Smart Auto-Compress: cari nilai k TERKECIL yang SSIM hasil rekonstruksi
    masih >= target_ssim, lewat binary search.

    SSIM monoton naik (secara umum) terhadap k karena menambah komponen
    PCA hanya menambah informasi, sehingga binary search valid dan jauh
    lebih cepat daripada mencoba semua k satu per satu. Karena SVD sudah
    dihitung sekali di k_max (lihat decompose_image), tiap kandidat k di
    sini hanya butuh slicing + matrix multiply yang murah.
    """
    lo, hi = 1, k_max
    best_k = k_max

    # Jika bahkan k_max tidak mencapai target, gunakan k_max (kualitas terbaik
    # yang bisa dicapai dalam batas eksplorasi).
    best_reconstruction = reconstruct_image(decomps, k_max)
    if compute_ssim(original, best_reconstruction) < target_ssim:
        return k_max

    while lo < hi:
        mid = (lo + hi) // 2
        candidate = reconstruct_image(decomps, mid)
        score = compute_ssim(original, candidate)
        if score >= target_ssim:
            best_k = mid
            hi = mid
        else:
            lo = mid + 1

    return best_k

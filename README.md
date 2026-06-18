# Spektrum — Kompresi Gambar dengan PCA

Aplikasi web untuk mengompresi gambar menggunakan **Principal Component
Analysis (PCA)**, dibuat untuk Tugas Project ke-2 mata kuliah **Aljabar
Linear Kelas C** (Pengampu: Drs. Bambang Harjito, M.App.Sc., Ph.D.),
Program Studi Informatika, FATISDA UNS.

Dokumen ini ditulis agar bisa dipakai langsung sebagai bahan laporan
(Bab 2 — Teori, Bab 3 — Implementasi), sekaligus sebagai dokumentasi teknis
proyek.

---

## Daftar Isi

1. [Cara Menjalankan](#cara-menjalankan)
2. [Struktur Folder](#struktur-folder)
3. [Landasan Teori (Bab 2)](#landasan-teori-bab-2)
4. [Algoritma & Implementasi (Bab 3)](#algoritma--implementasi-bab-3)
5. [Fitur Aplikasi](#fitur-aplikasi)
6. [API](#api)
7. [Metrik yang Ditampilkan](#metrik-yang-ditampilkan)
8. [Batasan & Asumsi](#batasan--asumsi)
9. [Verifikasi Kesesuaian dengan Materi Kuliah](#verifikasi-kesesuaian-dengan-materi-kuliah)
10. [Referensi](#referensi)

---

## Cara Menjalankan

```bash
cd src
pip install -r requirements.txt
uvicorn main:app --reload
```

Lalu buka `http://127.0.0.1:8000` di browser. Tidak ada database atau
penyimpanan file sementara di server — aplikasi sepenuhnya stateless,
gambar hasil dikirim sebagai base64 dalam respons JSON dan diunduh langsung
dari browser.

## Struktur Folder

Sesuai ketentuan tugas (poin g): folder `src/`, `test/`, `doc/`.

```
aljabar-linear/
├── src/
│   ├── main.py             # FastAPI app: routing & endpoint HTTP
│   ├── pca_engine.py       # Algoritma inti PCA/SVD (well-commented)
│   ├── image_metrics.py    # Hitung ukuran data, rasio kompresi, encode PNG
│   ├── requirements.txt
│   ├── static/
│   │   ├── style.css       # Desain UI ("kertas grafik laboratorium")
│   │   └── script.js       # Logika frontend: upload, grafik spektrum, hasil
│   └── templates/
│       └── index.html      # Halaman utama (satu halaman, tiga panel)
├── test/                   # Gambar uji (gradient, geometris, tekstur, dll.)
└── doc/
    └── README.md           # Dokumen ini
```

---

## Landasan Teori (Bab 2)

### 1. PCA secara umum

*Principal Component Analysis* (PCA) — ditemukan Karl Pearson (1901), juga
dikenal sebagai Transformasi Karhunen–Loève / *Singular Value Decomposition*
pada matriks — adalah teknik **transformasi linear orthogonal** yang
mereduksi data berdimensi tinggi menjadi dimensi lebih kecil sambil
mempertahankan karakteristik (variansi) data tersebut. Transformasi ini
menghilangkan korelasi antar variabel asal dengan memproyeksikannya ke
variabel baru yang saling tidak berkorelasi, disebut **principal
component** — diurutkan berdasarkan seberapa besar variansi data yang
dijelaskan masing-masing komponen.

Pada kasus citra digital, satu gambar berwarna adalah tiga matriks angka
(kanal Merah, Hijau, Biru) berukuran `tinggi × lebar`. PCA memanfaatkan
fakta bahwa baris/kolom piksel yang berdekatan sangat berkorelasi (area
yang halus/berulang) — sehingga sebagian besar "informasi" gambar
sebenarnya hidup di segelintir komponen dengan variansi terbesar, dan
komponen sisanya bisa dibuang dengan kehilangan visual minimal.

### 2. Konsep matematika yang melandasi

**Perkalian matriks & vektor.** Operasi inti PCA seluruhnya berupa
perkalian matriks: memproyeksikan data ke basis baru (`Vᵗ·X'`) dan
merekonstruksinya kembali (`U·S·Vᵗ`) adalah perkalian matriks berurutan.
Asosiativitas dan dimensi perkalian matriks (`(h×k)·(k×k)·(k×w)`) inilah
yang membuat representasi rank-k lebih hemat daripada matriks penuh
`h×w` aslinya (lihat [Reduksi Data](#3-reduksi-data-dan-rank-k-approximation)).

**Nilai eigen & vektor eigen.** Untuk matriks bujur sangkar simetris `C`
(matriks kovarians), vektor `v` disebut **eigenvector** dengan
**eigenvalue** `λ` jika:

```
C·v = λ·v        atau ekuivalen   (C − λI)·v = 0
```

Vektor eigen menunjukkan *arah* di mana data tersebar (variansinya
maksimal di sepanjang arah tersebut), dan nilai eigen menunjukkan *seberapa
besar* penyebaran (variansi) data pada arah itu. Karena matriks kovarians
selalu simetris dan *positive semi-definite*, seluruh eigenvalue-nya real
dan ≥ 0, dan eigenvector-eigenvectornya saling orthogonal — sifat inilah
yang membuat PCA menghasilkan sumbu-sumbu baru yang tidak berkorelasi satu
sama lain.

**Singular Value Decomposition (SVD).** Untuk sembarang matriks `X'`
(berukuran `n×m`, tidak harus bujur sangkar ataupun simetris), SVD
menyatakan:

```
X' = U · S · Vᵗ
```

dengan `U` (`n×n`, orthogonal), `S` (matriks diagonal berisi *singular
value* ≥ 0, terurut menurun), dan `V` (`m×m`, orthogonal). Hubungan SVD
dengan eigen-dekomposisi matriks kovarians adalah:

```
C = X'ᵗX' / (n−1)
  = V · (S² / (n−1)) · Vᵗ
```

Karena bentuk ini sudah merupakan dekomposisi eigen dari `C` (matriks
orthogonal `V` di kiri-kanan, matriks diagonal di tengah), maka:

- kolom-kolom `V` **adalah** eigenvector dari `C` — yaitu *principal
  component* yang dicari;
- `λₘ = Sₘ² / (n−1)` **adalah** eigenvalue yang berkorespondensi.

Singular value besar ⇒ eigenvalue besar ⇒ komponen tersebut menjelaskan
variansi data yang besar ⇒ layak dipertahankan saat kompresi.

---

## Algoritma & Implementasi (Bab 3)

### Pemetaan langkah PCA pada soal tugas → kode

Soal tugas mendefinisikan algoritma PCA dalam 5 langkah (mean → matriks
kovarians → eigenvalue/eigenvector → urutkan descending → hasilkan dataset
baru). Tabel berikut memetakan setiap langkah ke implementasi nyata di
`pca_engine.py`:

| # | Langkah pada soal | Persamaan soal | Implementasi |
|---|---|---|---|
| 1 | Hitung mean `X̄` tiap dimensi | `X̄ = (1/n)Σ Xᵢ` | `mean = centered.mean(axis=0)` di `decompose_channel` |
| 2 | Hitung matriks kovarians `Cₓ` | `Cₓ = (1/(n−1))Σ(Xᵢ−X̄)(Xᵢ−X̄)ᵗ` | **Tidak dihitung eksplisit** — digantikan SVD langsung pada data terpusat (`X' = X − mean`), karena `Cₓ` secara matematis sama dengan `X'ᵗX'/(n−1)` (lihat bagian SVD di atas) |
| 3 | Hitung eigenvalue `λₘ` & eigenvector `vₘ`: `Cₓvₘ = λₘvₘ` | persamaan (3) | `U, S, Vt = np.linalg.svd(centered, full_matrices=False)` — baris `Vt` = eigenvector `vₘ` (sebagai `Vᵗ`), dan `λₘ = Sₘ²/(n−1)` |
| 4 | Urutkan eigenvalue descending | — | **Otomatis** — `np.linalg.svd` selalu mengembalikan singular value (≡ akar eigenvalue) dalam urutan menurun |
| 5 | Hasilkan dataset baru | — | `reconstruct_channel`: ambil `k` komponen pertama lalu rekonstruksi `(U_k·S_k)@Vt_k + mean` |

**Mengapa SVD, bukan menghitung `Cₓ` secara eksplisit?** Materi kuliah
sendiri menyatakan PCA "dalam perhitungannya melibatkan nilai eigen dari
matriks kovarians *(singular value decomposition)*" — slide PCA (slide 10)
bahkan menulis langkah ini sebagai *"Most data mining packages do this for
you"*, mengakui bahwa praktiknya orang tidak menghitung dekomposisi eigen
manual. Menghitung `Cₓ` (`m×m`) secara eksplisit lalu mencari
eigenvector-nya jauh lebih mahal & kurang stabil secara numerik untuk
gambar (m bisa ribuan kolom) dibanding SVD langsung pada `X'`, yang
**matematinya identik** seperti dibuktikan di atas. Ini juga konsisten
dengan poin 9–10 spesifikasi tugas yang membolehkan pemakaian
fungsi-fungsi siap pakai.

### Alur kompresi penuh (`pca_engine.py` + `main.py`)

Kompresi dilakukan **per kanal warna (R, G, B) secara independen**, sesuai
poin 4 spesifikasi tugas ("kompresi image tetap mempertahankan warna dari
image asli") — bila ketiga kanal digabung jadi satu PCA, korelasi
antar-warna akan tercampur dan bisa menggeser hue gambar.

```
Untuk setiap kanal c ∈ {R, G, B} (matriks h×w):
  1. mean = rata-rata tiap kolom kanal
  2. X'   = kanal - mean                      (mean-centering)
  3. U,S,Vᵗ = SVD(X')                          (≡ eigen-dekomposisi Cₓ, lihat tabel)
  4. ambil k kolom/baris pertama: U_k, S_k, Vᵗ_k
  5. rekonstruksi: kanal_approx = (U_k · S_k) · Vᵗ_k + mean
  6. clip ke rentang [0, 255]

Gabungkan kembali (np.stack) ketiga kanal → gambar RGB hasil kompresi (rank-k approximation).
```

Fungsi-fungsi terkait (`src/pca_engine.py`):

- `decompose_channel(channel, k_cap)` — langkah 1–3, hanya menyimpan
  `k_cap` komponen pertama (batas atas eksplorasi `k`, bukan hasil akhir)
  supaya pencarian `k` otomatis tidak perlu mengulang SVD.
- `reconstruct_channel(decomp, k)` — langkah 4–6, slicing murah dari hasil
  SVD yang sudah dihitung sekali.
- `decompose_image` / `reconstruct_image` — versi 3-kanal dari dua fungsi
  di atas.
- `luminance(image)` — konversi RGB → satu kanal grayscale (bobot luma
  ITU-R BT.601: `0.299R + 0.587G + 0.114B`), dipakai **khusus untuk
  menggambar satu kurva spektrum nilai singular yang representatif di
  UI** (lihat [Fitur](#fitur-aplikasi)). Kompresi sesungguhnya tetap
  dihitung per-kanal R/G/B terpisah, fungsi ini tidak pernah dipakai untuk
  hasil kompresi akhir.
- `compute_ssim` / `compute_psnr` — metrik kualitas (lihat
  [Metrik](#metrik-yang-ditampilkan)).
- `find_optimal_k` — algoritma *Smart Auto-Compress* (lihat di bawah).

Persentase variansi yang dijelaskan komponen ke-`j`, sesuai rumus materi
kuliah (`Vⱼ = 100·λⱼ / Σλₓ`), dihitung di sisi **frontend** (lihat
`script.js`, `prepareSpectrum`/`energyAt`) sebagai **energi kumulatif**:
karena `λ ∝ S²`, energi kumulatif sampai komponen `k` adalah
`Σᵢ₌₁ᵏ Sᵢ² / Σᵢ Sᵢ²` × 100% — inilah angka "% energi dipertahankan" yang
ditampilkan di UI saat menggeser titik potong `k`.

---

## Fitur Aplikasi

### Mode kompresi

1. **Manual** — pengguna menarik garis potong langsung pada **grafik
   spektrum nilai singular** (lihat di bawah) untuk memilih `k`. Ini
   menggantikan slider biasa: posisi garis terhubung langsung secara visual
   dengan magnitudo nilai singular yang dipertahankan vs dibuang, dan
   pembacaan "% energi dipertahankan" diperbarui live.
2. **Smart Auto-Compress** (fitur kreatif, sesuai poin 7 spesifikasi
   tugas) — pengguna memilih target kualitas (Tinggi/Seimbang/Maksimal →
   ambang SSIM 0.98/0.95/0.90). Sistem menghitung SVD satu kali sampai
   `k_max`, lalu melakukan **binary search** pada `k` (memanfaatkan sifat
   SSIM yang naik monoton terhadap `k`, karena menambah komponen PCA hanya
   menambah informasi) untuk menemukan nilai `k` **terkecil** yang masih
   memenuhi ambang kualitas — inilah titik kompresi paling ringan tanpa
   penurunan kualitas visual yang signifikan, sesuai permintaan agar
   sistem bisa **menemukan otomatis** titik kompresi paling ringan.

### Grafik spektrum nilai singular (elemen signature UI)

Saat gambar diunggah, frontend memanggil `POST /api/spectrum`, yang
menghitung SVD dari kanal *luminance* gambar dan mengembalikan seluruh
kurva nilai singularnya. Grafik ini secara langsung memparalelkan
*scree plot* (grafik variansi per komponen) yang diajarkan pada materi
kuliah — bedanya di sini interaktif: pengguna men-drag garis potong dan
melihat langsung berapa banyak "energi" (variansi) gambar yang masih
dipertahankan pada nilai `k` tersebut, sebelum bahkan menekan tombol
kompresi.

### Auto-resize gambar berukuran besar

Sisi terpanjang gambar dibatasi `3000px` demi menjaga SVD tetap cepat untuk
demo interaktif. Gambar yang lebih besar **tidak ditolak**, melainkan
diperkecil otomatis (`PIL.Image.LANCZOS`, resampling kualitas tinggi) ke
batas tersebut, dan pengguna diberi tahu secara transparan lewat notifikasi
`resize_info` di UI (ukuran asli vs ukuran setelah diperkecil). Ini berbeda
dari versi awal aplikasi yang menolak gambar oversize dengan pesan error —
diubah agar pengalaman pengguna lebih mulus tanpa mengorbankan kejujuran
(pengguna tetap tahu bahwa resize terjadi).

### Validasi & penanganan kesalahan

- File yang diunggah harus bertipe gambar (`content-type` divalidasi) dan
  tidak boleh kosong.
- Gambar dengan kanal alpha (transparansi) dikonversi ke RGB (alpha
  dibuang) agar konsisten 3 kanal warna untuk PCA.
- Output selalu disimpan sebagai **PNG (lossless)** sehingga efek
  "kompresi" yang terlihat murni berasal dari reduksi rank PCA, bukan
  tercampur artefak kompresi lossy lain seperti JPEG — representasi paling
  jujur untuk demo akademik.

---

## API

| Endpoint | Metode | Input | Output |
|---|---|---|---|
| `/` | GET | — | Halaman utama (HTML) |
| `/api/spectrum` | POST | `file` (gambar) | `singular_values` (array), `k_max`, `width`, `height`, `resize_info` |
| `/api/compress` | POST | `file`, `mode` (`manual`/`auto`), `k` *(mode manual)* atau `quality` *(mode auto)* | `image_base64`, `resize_info`, `metrics` (lihat di bawah) |

`/api/spectrum` dan `/api/compress` berbagi fungsi `load_and_prepare_image`
(decode + auto-resize) dan `read_uploaded_image` (validasi) di `main.py`
agar kedua endpoint selalu konsisten dalam memperlakukan gambar input.

---

## Metrik yang Ditampilkan

Sesuai poin 2 spesifikasi tugas ("runtime algoritma, dan persentase hasil
kompresi gambar / perubahan jumlah pixel gambar"):

- **Runtime algoritma** (ms) — diukur dengan `time.perf_counter()` membungkus
  proses SVD + rekonstruksi (dan, pada mode auto, pencarian `k`).
- **Ukuran file** asli vs hasil (bytes) dan **persentase reduksi ukuran
  file** aktual.
- **Reduksi data PCA** — ini adalah metrik "perubahan jumlah pixel/data"
  yang diminta soal: membandingkan jumlah nilai yang perlu disimpan untuk
  representasi rank-k (`k·(h + 1 + w)` per kanal, dari matriks `U_k`
  (`h×k`), `S_k` (`k`), `Vᵗ_k` (`k×w`)) terhadap jumlah piksel asli
  (`h·w` per kanal). Dihitung di `image_metrics.pca_storage_values`.
- **SSIM** (*Structural Similarity Index*) dan **PSNR** (*Peak
  Signal-to-Noise Ratio*) — metrik kuantitatif objektif untuk membuktikan
  kualitas visual tetap terjaga meski data dipangkas signifikan; sangat
  relevan untuk analisis eksperimen di Bab 4 laporan (mis. menunjukkan
  kurva SSIM vs `k` untuk berbagai gambar uji).
- **Catatan kejujuran metrik**: untuk gambar dengan warna sangat flat atau
  sintetis (blok warna solid, gradient sangat halus), PNG asli bisa sudah
  sangat efisien (PNG memanfaatkan kompresi entropi pada area homogen),
  sedangkan rekonstruksi PCA rank-rendah cenderung menghasilkan gradasi
  halus yang justru *lebih* sulit dikompresi PNG. Pada kasus ini,
  **persentase reduksi ukuran file bisa negatif** (file bertambah besar).
  Nilai ini **tidak disembunyikan/diclamp ke 0** — UI menampilkannya apa
  adanya dengan warna peringatan (`warn`), karena metrik yang jujur lebih
  penting daripada angka yang selalu terlihat bagus. Reduksi *data PCA*
  (jumlah nilai numerik) tetap valid dan positif pada kasus ini — yang
  berkurang adalah ukuran representasi sebelum encoding PNG, bukan
  ukuran file PNG itu sendiri.

---

## Batasan & Asumsi

- Sisi terpanjang gambar dibatasi `3000px`; gambar yang lebih besar
  **diperkecil otomatis** (lihat [Auto-resize](#auto-resize-gambar-berukuran-besar)),
  bukan ditolak.
- Eksplorasi nilai `k` dibatasi maksimum `300` (`MAX_K_CAP`) demi performa
  interaktif — di atas titik ini, tambahan singular value pada foto natural
  praktis tidak lagi terlihat secara visual.
- Gambar dengan kanal alpha dikonversi ke RGB (alpha dibuang).
- Tidak ada penyimpanan file di server (stateless); seluruh hasil dikirim
  sebagai base64 dan diunduh dari sisi browser.

---

## Verifikasi Kesesuaian dengan Materi Kuliah

Implementasi ini **tetap merupakan PCA sesuai materi yang diajarkan**.
Ringkasan korespondensinya:

- **Mean & mean-centering** (slide "Steps of PCA" / soal langkah 1,
  persamaan 1) → `decompose_channel`: `mean = centered.mean(axis=0)`.
- **Matriks kovarians & eigen-dekomposisi** (soal langkah 2–3, persamaan
  2–3: `Cₓvₘ = λₘvₘ`) → tidak dihitung eksplisit, digantikan SVD yang
  matematinya identik (`V` ≡ eigenvector, `S²/(n−1)` ≡ eigenvalue) —
  pendekatan ini **secara eksplisit diizinkan materi kuliah sendiri**
  (slide PCA: *"Most data mining packages do this for you"*) dan oleh poin
  9–10 spesifikasi tugas (boleh pakai fungsi siap pakai).
- **Pengurutan eigenvalue descending & pemilihan principal component**
  (soal langkah 4) → otomatis dari sifat `np.linalg.svd` yang selalu
  mengurutkan singular value menurun; pemilihan `k` komponen
  teratas dilakukan di `reconstruct_channel`/`reconstruct_image`.
  Persentase variansi per komponen (`Vⱼ = 100λⱼ/Σλₓ` pada materi kuliah)
  diimplementasikan di frontend sebagai "% energi dipertahankan".
- **Menghasilkan dataset baru** (soal langkah 5) → rekonstruksi rank-k:
  `(U_k · S_k) · Vᵗ_k + mean`, identik dengan rumus rekonstruksi pada
  materi kuliah (`RetrievedRowData = (RowFeatureVectorᵗ × FinalData) +
  OriginalMean`).

Jadi tidak ada langkah algoritma yang "dilewati" — yang berubah hanya cara
*menghitung* eigenvector/eigenvalue (lewat SVD, bukan dekomposisi eigen
manual pada matriks kovarians eksplisit), dan ini adalah pendekatan numerik
standar yang diakui sendiri oleh materi kuliah dan diizinkan oleh
spesifikasi tugas.

---

## Referensi

- Materi kuliah Aljabar Linear Kelas C, FATISDA UNS: `pca (1).ppt.pdf`
  (slide "Steps of PCA"), `kelas C Project Kompresi image dengan PCA.pdf`
  (spesifikasi tugas).
- Pearson, K. (1901). *On Lines and Planes of Closest Fit to Systems of
  Points in Space*.
- Jolliffe, I.T. *Principal Component Analysis*. Springer.
- Dokumentasi NumPy `numpy.linalg.svd`.
- Dokumentasi scikit-image `skimage.metrics.structural_similarity`.
- Wang, Z. et al. (2004). *Image Quality Assessment: From Error
  Visibility to Structural Similarity* (rujukan SSIM).

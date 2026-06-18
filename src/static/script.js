// ---------------------------------------------------------------------
// Frontend logic: upload, ambil spektrum nilai singular dari backend,
// render grafik spektrum interaktif (kontrol pemilihan rank k), panggil
// API kompresi, render before/after + metrik, dan unduh hasil.
// ---------------------------------------------------------------------

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const dropzoneEmpty = document.getElementById("dropzoneEmpty");
const previewImage = document.getElementById("previewImage");
const uploadMeta = document.getElementById("uploadMeta");
const resizeNotice = document.getElementById("resizeNotice");

const rankPanel = document.getElementById("rankPanel");
const manualTabBtn = document.getElementById("manualTabBtn");
const autoTabBtn = document.getElementById("autoTabBtn");
const manualHint = document.getElementById("manualHint");
const autoPanel = document.getElementById("autoPanel");

const spectrumChart = document.getElementById("spectrumChart");
const spectrumCanvas = document.getElementById("spectrumCanvas");
const energyValue = document.getElementById("energyValue");
const kReadout = document.getElementById("kReadout");
const kMaxReadout = document.getElementById("kMaxReadout");

const compressBtn = document.getElementById("compressBtn");
const compressBtnLabel = document.getElementById("compressBtnLabel");
const compressSpinner = document.getElementById("compressSpinner");
const errorText = document.getElementById("errorText");

const resultsPanel = document.getElementById("resultsPanel");
const beforeImage = document.getElementById("beforeImage");
const afterImage = document.getElementById("afterImage");
const metricsGrid = document.getElementById("metricsGrid");
const downloadBtn = document.getElementById("downloadBtn");

let currentFile = null;
let currentMode = "manual";
let currentQuality = "balanced";
let currentK = 1;
let spectrumState = null; // { values, kMax, total, prefix }
let isDragging = false;
let lastResultBase64 = null;
let lastFileBaseName = "image";

// --- Spektrum: persiapan data & matematika --------------------------------

function prepareSpectrum(values) {
  const squared = values.map((v) => v * v);
  const total = squared.reduce((a, b) => a + b, 0) || 1;
  const prefix = [];
  let running = 0;
  for (let i = 0; i < squared.length; i++) {
    running += squared[i];
    prefix.push(running);
  }
  return { values, kMax: values.length, total, prefix };
}

function energyAt(k) {
  if (!spectrumState) return 0;
  const idx = Math.max(1, Math.min(spectrumState.kMax, k));
  return (spectrumState.prefix[idx - 1] / spectrumState.total) * 100;
}

function defaultK(spectrum) {
  for (let k = 1; k <= spectrum.kMax; k++) {
    if ((spectrum.prefix[k - 1] / spectrum.total) * 100 >= 90) return k;
  }
  return Math.max(1, Math.round(spectrum.kMax * 0.15));
}

// --- Canvas helpers --------------------------------------------------------

function setupCanvasDPR(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, rect.width * dpr);
  canvas.height = Math.max(1, rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function drawHeroChart() {
  const canvas = document.getElementById("heroCanvas");
  if (!canvas) return;
  const { ctx, width, height } = setupCanvasDPR(canvas);
  ctx.clearRect(0, 0, width, height);
  const n = 60;
  const barWidth = width / n;
  for (let i = 0; i < n; i++) {
    const v = Math.exp(-i / 11);
    const barHeight = Math.max(1.5, v * (height - 2));
    ctx.fillStyle = i < 9 ? "#0e6b5c" : "#cdddd2";
    ctx.fillRect(i * barWidth, height - barHeight, Math.max(1, barWidth - 1.5), barHeight);
  }
}

function drawSpectrumChart() {
  if (!spectrumState) return;
  const { ctx, width, height } = setupCanvasDPR(spectrumCanvas);
  ctx.clearRect(0, 0, width, height);

  const kMax = spectrumState.kMax;
  const barCount = Math.min(kMax, 110);
  const barWidth = width / barCount;
  const maxVal = spectrumState.values[0] || 1;

  for (let i = 0; i < barCount; i++) {
    const idx = Math.min(kMax - 1, Math.floor((i * kMax) / barCount));
    const v = spectrumState.values[idx];
    const norm = Math.log(1 + v) / Math.log(1 + maxVal);
    const barHeight = Math.max(2, norm * (height - 14));
    const x = i * barWidth;
    const y = height - barHeight;
    ctx.fillStyle = idx < currentK ? "#0e6b5c" : "#cdddd2";
    ctx.fillRect(x, y, Math.max(1, barWidth - 1), barHeight);
  }

  const cutX = (currentK / kMax) * width;
  ctx.strokeStyle = "#16241f";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cutX, 0);
  ctx.lineTo(cutX, height);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(cutX, 8, 4, 0, Math.PI * 2);
  ctx.fillStyle = "#16241f";
  ctx.fill();
}

function updateReadouts() {
  if (!spectrumState) return;
  energyValue.textContent = energyAt(currentK).toFixed(1);
  kReadout.textContent = currentK;
  kMaxReadout.textContent = spectrumState.kMax;
}

// --- Drag interaksi grafik spektrum (kontrol k pada mode manual) ----------

function kFromClientX(clientX) {
  const rect = spectrumCanvas.getBoundingClientRect();
  const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
  return Math.max(1, Math.min(spectrumState.kMax, Math.round(ratio * spectrumState.kMax)));
}

function setKFromEvent(e) {
  currentK = kFromClientX(e.clientX);
  drawSpectrumChart();
  updateReadouts();
}

spectrumCanvas.addEventListener("pointerdown", (e) => {
  if (currentMode !== "manual" || !spectrumState) return;
  isDragging = true;
  spectrumCanvas.setPointerCapture(e.pointerId);
  spectrumChart.classList.add("dragging");
  setKFromEvent(e);
});

spectrumCanvas.addEventListener("pointermove", (e) => {
  if (!isDragging) return;
  setKFromEvent(e);
});

["pointerup", "pointercancel"].forEach((evt) =>
  spectrumCanvas.addEventListener(evt, () => {
    isDragging = false;
    spectrumChart.classList.remove("dragging");
  })
);

// --- Upload & dropzone -------------------------------------------------

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);

["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});

async function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("File yang dipilih bukan gambar.");
    return;
  }
  hideError();
  currentFile = file;
  lastFileBaseName = file.name.replace(/\.[^.]+$/, "") || "image";

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    previewImage.hidden = false;
    dropzoneEmpty.hidden = true;
  };
  reader.readAsDataURL(file);

  resultsPanel.hidden = true;
  rankPanel.hidden = true;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/spectrum", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Gagal membaca gambar.");

    spectrumState = prepareSpectrum(data.singular_values);
    currentK = defaultK(spectrumState);

    uploadMeta.hidden = false;
    uploadMeta.textContent = `${file.name} · ${data.width}×${data.height}px · ${formatBytes(file.size)}`;

    if (data.resize_info && data.resize_info.resized) {
      resizeNotice.hidden = false;
      resizeNotice.textContent =
        `Diperkecil otomatis dari ${data.resize_info.original_width}×${data.resize_info.original_height}px ` +
        `ke ${data.resize_info.width}×${data.resize_info.height}px agar tetap responsif.`;
    } else {
      resizeNotice.hidden = true;
    }

    rankPanel.hidden = false;
    drawSpectrumChart();
    updateReadouts();
  } catch (err) {
    showError(err.message);
  }
}

// --- Mode toggle ---------------------------------------------------------

manualTabBtn.addEventListener("click", () => switchMode("manual"));
autoTabBtn.addEventListener("click", () => switchMode("auto"));

function switchMode(mode) {
  currentMode = mode;
  manualTabBtn.classList.toggle("active", mode === "manual");
  autoTabBtn.classList.toggle("active", mode === "auto");
  manualHint.hidden = mode !== "manual";
  autoPanel.hidden = mode !== "auto";
  spectrumChart.classList.toggle("readonly", mode === "auto");
}

document.querySelectorAll(".quality-tick").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".quality-tick").forEach((b) => b.classList.remove("selected"));
    btn.classList.add("selected");
    currentQuality = btn.dataset.quality;
  });
});

// --- Compress action -------------------------------------------------

compressBtn.addEventListener("click", async () => {
  if (!currentFile) {
    showError("Unggah gambar terlebih dahulu.");
    return;
  }
  hideError();
  setLoading(true);

  const formData = new FormData();
  formData.append("file", currentFile);
  formData.append("mode", currentMode);
  if (currentMode === "manual") {
    formData.append("k", currentK);
  } else {
    formData.append("quality", currentQuality);
  }

  try {
    const res = await fetch("/api/compress", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Terjadi kesalahan saat memproses gambar.");
    }
    renderResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  compressBtn.disabled = isLoading;
  compressSpinner.hidden = !isLoading;
  compressBtnLabel.textContent = isLoading ? "Memproses…" : "Kompres Gambar";
}

function renderResult(data) {
  lastResultBase64 = data.image_base64;
  beforeImage.src = previewImage.src;
  afterImage.src = `data:image/png;base64,${data.image_base64}`;

  currentK = data.metrics.k_used;
  drawSpectrumChart();
  updateReadouts();

  const m = data.metrics;
  const stats = [
    { label: "Runtime algoritma", value: `${m.runtime_ms} ms` },
    { label: "Ukuran asli", value: formatBytes(m.original_bytes) },
    { label: "Ukuran hasil", value: formatBytes(m.compressed_bytes) },
    {
      label: "Reduksi ukuran file",
      value: `${m.file_reduction_pct > 0 ? "-" : "+"}${Math.abs(m.file_reduction_pct)}%`,
      highlight: m.file_reduction_pct > 0,
      warn: m.file_reduction_pct <= 0,
    },
    { label: "Reduksi data PCA", value: `${m.pca_reduction_pct}%`, highlight: true },
    { label: "SSIM (kemiripan)", value: m.ssim },
    { label: "PSNR", value: m.psnr !== null ? `${m.psnr} dB` : "∞ dB" },
    { label: "Singular value (k)", value: `${m.k_used} / ${m.k_max}` },
  ];

  metricsGrid.innerHTML = stats
    .map(
      (s) => `
      <div class="metric-row">
        <span class="metric-label">${s.label}</span>
        <span class="metric-value ${s.highlight ? "highlight" : ""} ${s.warn ? "warn" : ""}">${s.value}</span>
      </div>`
    )
    .join("");

  resultsPanel.hidden = false;
  resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

downloadBtn.addEventListener("click", () => {
  if (!lastResultBase64) return;
  const link = document.createElement("a");
  link.href = `data:image/png;base64,${lastResultBase64}`;
  link.download = `${lastFileBaseName}_compressed.png`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
});

// --- Helpers -------------------------------------------------------------

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function showError(message) {
  errorText.textContent = message;
  errorText.hidden = false;
}

function hideError() {
  errorText.hidden = true;
}

// --- Init ------------------------------------------------------------------

drawHeroChart();

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    drawHeroChart();
    if (spectrumState) drawSpectrumChart();
  }, 100);
});

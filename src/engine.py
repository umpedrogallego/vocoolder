"""
Vocoolder - Motor de audio leve
===============================

Sem torch/torchaudio/librosa/sklearn no caminho em tempo real.

- Indexacao do corpus em ``amostras/``: soundfile (MP3/WAV/FLAC/OGG),
  deteccao de onsets por fluxo espectral em numpy, graos multi-escala
  (60/120/180/350 ms, hop 70 ms), selecao por classe x escala.
- Features: 13 MFCCs (numpy/scipy) + centroid + ZCR; descritores de
  flatness, HF 4-8 kHz, f0, vozeamento e ruidosidade.
- Cache versionado ``corpus_cache/corpus_data.pkl`` com assinatura dos
  arquivos (reindexa sozinho quando as amostras mudam).
- Busca k-NN por similaridade cosine em numpy, com resposta fonetica
  (percentil de ruidosidade) e contraste timbrico por registro.
- Streaming com sounddevice: InputStream/OutputStream separados,
  callbacks leves, thread de processamento, filas nao bloqueantes.
- Pitch: desligado, desvio microtonal, casar com o f0 da voz ou intervalo
  fixo; 24-EDO/JI; varispeed e WSOLA para preservar duracao; fade Hann 6 ms.
- Gate RMS com calibracao automatica do ruido ambiente.
- Biblioteca completa em cache x conjunto ativo (10-1800 graos) filtrado por
  fonte e estratificado por classe x escala; audio inativo e liberado da RAM.
- Gravacao de sessao pos-limiter em WAV + log de eventos JSONL, assincrona.

Tudo em ASCII nos prints (console Windows cp1252).
"""

from __future__ import annotations

import math
import os
import pickle
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.fft import dct, rfft
from scipy.signal import resample_poly

from .microtonal_tables import (
    HARMONIC_SERIES_DEVIATIONS_CENTS,
    JI_INTERVALS_CENTS,
    quantize_absolute_cents,
    quantize_to_microtone,
)
from .session import SessionRecorder

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

AMOSTRAS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "amostras")
CORPUS_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "corpus_cache")
CORPUS_CACHE_FILE = os.path.join(CORPUS_CACHE_DIR, "corpus_data.pkl")
DEBUG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "corpus_debug")
CACHE_VERSION = 5

SAMPLE_RATE = 44100
GRAIN_MIN_DURATION = 0.060
GRAIN_MAX_DURATION = 0.350
GRAIN_TARGET_DURATION = 0.180
GRAIN_HOP_DURATION = 0.070
GRAIN_SCALES = (0.060, 0.120, 0.180, 0.350)
MAX_GRAINS_PER_FILE = 400
MAX_TOTAL_GRAINS = 1800
# Biblioteca (cache em disco) x conjunto ativo (RAM/uso live).
MIN_ACTIVE_GRAINS = 10
MAX_ACTIVE_GRAINS = 1800
DEFAULT_ACTIVE_GRAINS = MAX_ACTIVE_GRAINS
MIN_GRAIN_RMS = 0.004
GRAIN_PEAK = 0.7

# Cotas do palheta: transiente/curto, tonal/sustentado, ruidoso/fricativo
CLASS_QUOTAS = {
    "transient": 0.38,
    "tonal": 0.24,
    "noisy": 0.38,
}
# Distribuicao de escalas dentro de cada classe (60/120/180/350 ms)
CLASS_SCALE_WEIGHTS = {
    "transient": (0.35, 0.30, 0.30, 0.05),
    "tonal": (0.02, 0.05, 0.18, 0.75),
    "noisy": (0.25, 0.20, 0.35, 0.20),
}
SELECTION_BIN_SECONDS = 5.0
SELECTION_PER_BIN = 4
FPS_NOVELTY_WEIGHT = 0.55
FPS_QUALITY_WEIGHT = 0.30
FPS_BIN_WEIGHT = 0.15
RECENT_GRAIN_AVOID = 12

N_FFT = 1024
FEATURE_HOP = 256
N_MELS = 26
N_MFCC = 13
FMIN = 60.0
FMAX = 8000.0

FEATURE_WINDOW = 0.200
EVENT_INTERVAL = 0.170
CASCADE_STEP = 0.040
FADE_MS = 6.0

GRAIN_GAIN = 0.85
DEFAULT_MIX = 0.30
DEFAULT_PHONETIC_RESPONSE = 0.60
DEFAULT_MUTATION_RATE = 0.20

PITCH_MODE_OFF = "off"
PITCH_MODE_MICROTONAL = "microtonal"
PITCH_MODE_MATCH = "match"
PITCH_MODE_INTERVAL = "interval"
PITCH_MODES = (
    PITCH_MODE_OFF,
    PITCH_MODE_MICROTONAL,
    PITCH_MODE_MATCH,
    PITCH_MODE_INTERVAL,
)
DEFAULT_PITCH_MODE = PITCH_MODE_MICROTONAL

# Intervalos fixos permitidos (sem setima maior)
PITCH_INTERVALS = {
    "unison": 0.0,
    "fourth": 500.0,
    "tritone": 600.0,
    "fifth": 700.0,
    "minor_seventh": 1000.0,
    "octave": 1200.0,
}
DEFAULT_PITCH_INTERVAL = "fifth"
DEFAULT_PRESERVE_DURATION = 0.0
PITCH_F0_CONFIDENCE = 0.30
PITCH_MATCH_LIMIT_CENTS = 1200.0
PITCH_HYSTERESIS_CENTS = 20.0
PITCH_F0_HISTORY = 3
SOFT_LIMIT_CEILING = 0.891  # -1 dBFS
SOFT_LIMIT_KNEE = 0.70

IN_BLOCK = 512
OUT_BLOCK = 512
IN_QUEUE_MAX = 8
OUT_QUEUE_MAX = 16

DEFAULT_RICOCHET_DENSITY = 3
MIN_RICOCHET_DENSITY = 1
MAX_RICOCHET_DENSITY = 6
DEFAULT_RMS_THRESHOLD = 0.005
MIN_RMS_THRESHOLD = 0.0008
AUTO_GATE_FLOOR = 0.006
GATE_ATTACK_BLOCKS = 2
GATE_HOLD_BLOCKS = 3
MAX_RMS_THRESHOLD = 0.08
AUTO_GATE_DURATION = 0.8


# ---------------------------------------------------------------------------
# Features (numpy/scipy)
# ---------------------------------------------------------------------------

_MEL_CACHE: Dict[Tuple[int, int, int, float, float], np.ndarray] = {}


def _mel_filterbank(
    sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    key = (int(sr), int(n_fft), int(n_mels), float(fmin), float(fmax))
    cached = _MEL_CACHE.get(key)
    if cached is not None:
        return cached

    def hz_to_mel(f: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    fft_freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fb = np.zeros((n_mels, fft_freqs.size), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        rising = (fft_freqs - left) / max(center - left, 1e-9)
        falling = (right - fft_freqs) / max(right - center, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(rising, falling))
    _MEL_CACHE[key] = fb
    return fb


def analyze_grain(
    audio: np.ndarray, sr: int, with_f0: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Features (15 dims) + descritores para selecao fonetica.

    Descritores: zcr, centroid, flatness, hf_ratio, rms, peak, duration,
    f0, voicing (confianca) e noisy (0..1).
    """
    raw = np.asarray(audio, dtype=np.float32).reshape(-1)
    duration = float(raw.size) / float(sr) if sr else 0.0
    if raw.size == 0:
        feats = np.zeros(N_MFCC + 2, dtype=np.float32)
        desc = {
            "zcr": 0.0, "centroid": 0.0, "flatness": 0.0, "hf_ratio": 0.0,
            "rms": 0.0, "peak": 0.0, "duration": duration,
            "f0": None, "voicing": 0.0, "noisy": 0.0,
        }
        return feats, desc

    peak = float(np.max(np.abs(raw)))
    rms = float(np.sqrt(np.mean(raw.astype(np.float32) ** 2)))

    y = raw.copy()
    if y.size < N_FFT:
        y = np.pad(y, (0, N_FFT - y.size))
    y = y - float(np.mean(y))

    n_frames = 1 + (y.size - N_FFT) // FEATURE_HOP
    if n_frames < 1:
        n_frames = 1
    frames = np.lib.stride_tricks.sliding_window_view(y, N_FFT)[::FEATURE_HOP]
    frames = frames[:n_frames].astype(np.float32, copy=False)

    window = np.hanning(N_FFT).astype(np.float32)
    spectrum = np.abs(rfft(frames * window, axis=1)) ** 2 / N_FFT
    spectrum = np.maximum(spectrum, 1e-12)
    fmax = min(FMAX, sr / 2.0)
    fb = _mel_filterbank(sr, N_FFT, N_MELS, FMIN, fmax)
    mel = spectrum @ fb.T
    logmel = np.log(mel + 1e-10)
    mfcc = dct(logmel, type=2, axis=1, norm="ortho")[:, :N_MFCC]

    freqs = np.linspace(0.0, sr / 2.0, spectrum.shape[1])
    power_sum = np.sum(spectrum, axis=1) + 1e-12
    centroid = np.sum(spectrum * freqs, axis=1) / power_sum

    # Flatness espectral (0 = tonal, ~1 = ruido)
    gmean = np.exp(np.mean(np.log(spectrum), axis=1))
    amean = np.mean(spectrum, axis=1) + 1e-12
    flatness = float(np.mean(np.clip(gmean / amean, 0.0, 1.0)))

    hf_mask = freqs >= 4000.0
    if np.any(hf_mask):
        hf_ratio = float(np.mean(np.sum(spectrum[:, hf_mask], axis=1) / power_sum))
    else:
        hf_ratio = 0.0

    zcr = float(np.mean(np.abs(np.diff(np.signbit(y)).astype(np.float32))))

    if with_f0:
        f0, voicing = estimate_f0_details(y, sr)
    else:
        f0, voicing = None, 0.0

    zcr_n = float(np.clip(zcr / 0.35, 0.0, 1.0))
    flat_n = float(np.clip(flatness / 0.60, 0.0, 1.0))
    hf_n = float(np.clip(hf_ratio / 0.50, 0.0, 1.0))
    noisy = float(np.clip(
        0.35 * zcr_n + 0.35 * flat_n + 0.20 * hf_n
        + 0.10 * (1.0 - float(np.clip(voicing, 0.0, 1.0))),
        0.0, 1.0,
    ))

    feats = np.concatenate(
        [mfcc.mean(axis=0), [float(np.mean(centroid)), zcr]]
    ).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    desc = {
        "zcr": zcr,
        "centroid": float(np.mean(centroid)),
        "flatness": flatness,
        "hf_ratio": hf_ratio,
        "rms": rms,
        "peak": peak,
        "duration": duration,
        "f0": f0,
        "voicing": float(np.clip(voicing, 0.0, 1.0)),
        "noisy": noisy,
    }
    return feats, desc


def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """13 MFCCs + centroid + ZCR (medias temporais), 15 dimensoes."""
    feats, _ = analyze_grain(audio, sr, with_f0=False)
    return feats


# Alias de compatibilidade
extract_acoustic_features = extract_features


def estimate_f0_details(
    audio: np.ndarray, sr: int
) -> Tuple[Optional[float], float]:
    """Autocorrelacao via FFT; retorna (f0, confianca 0..1)."""
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if y.size < 512:
        return None, 0.0
    y = y - float(np.mean(y))
    if float(np.max(np.abs(y))) < 1e-5:
        return None, 0.0

    factor = max(1, int(sr // 12000))
    if factor > 1:
        y = resample_poly(y, 1, factor).astype(np.float32)
        sr2 = sr / factor
    else:
        sr2 = sr

    lag_min = max(2, int(sr2 / 1000.0))
    lag_max = min(len(y) - 2, int(sr2 / 70.0))
    if lag_max <= lag_min + 2:
        return None, 0.0

    nfft = 1
    while nfft < 2 * len(y):
        nfft <<= 1
    spec = np.fft.rfft(y, nfft)
    corr = np.fft.irfft(spec * np.conj(spec), nfft)[: len(y)]
    if corr[0] <= 1e-9:
        return None, 0.0
    corr = corr / corr[0]

    segment = corr[lag_min:lag_max]
    peak_idx = int(np.argmax(segment))
    peak_val = float(segment[peak_idx])
    if peak_val < 0.30:
        return None, float(max(0.0, peak_val))
    f0 = sr2 / float(lag_min + peak_idx)
    if not (60.0 <= f0 <= 1200.0):
        return None, float(np.clip(peak_val, 0.0, 1.0))
    return float(f0), float(np.clip(peak_val, 0.0, 1.0))


def estimate_f0(audio: np.ndarray, sr: int) -> Optional[float]:
    """Autocorrelacao simples; retorna f0 em Hz ou None se nao for vozeado."""
    f0, _ = estimate_f0_details(audio, sr)
    return f0


# ---------------------------------------------------------------------------
# Leitura e fatiamento
# ---------------------------------------------------------------------------


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Le WAV/MP3/FLAC/OGG com soundfile e devolve mono float32 em ``sr``."""
    data, file_sr = sf.read(path, dtype="float32", always_2d=True)
    if data.size == 0:
        raise ValueError(f"Arquivo vazio: {path}")
    y = data.mean(axis=1).astype(np.float32)
    if int(file_sr) != int(sr):
        g = math.gcd(int(file_sr), int(sr))
        y = resample_poly(y, int(sr) // g, int(file_sr) // g).astype(np.float32)
    return y


def detect_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    """Deteccao de onsets por fluxo de energia (numpy)."""
    y = np.asarray(y, dtype=np.float32)
    hop = 256
    frame = 1024
    if y.size < frame:
        return np.zeros(0, dtype=np.int64)

    n_frames = 1 + (y.size - frame) // hop
    frames = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop][:n_frames]
    energy = np.sum(frames.astype(np.float32) ** 2, axis=1) + 1e-10
    flux = np.maximum(0.0, np.diff(np.log(energy)))
    if flux.size == 0:
        return np.zeros(0, dtype=np.int64)

    threshold = float(np.mean(flux) + 1.0 * np.std(flux))
    min_gap = max(1, int(0.050 * sr / hop))
    onsets: List[int] = []
    last = -min_gap
    for i in range(1, len(flux) - 1):
        if flux[i] > threshold and flux[i] >= flux[i - 1] and flux[i] > flux[i + 1]:
            if i - last >= min_gap:
                onsets.append(i * hop)
                last = i
    return np.asarray(onsets, dtype=np.int64)


def normalize_grain(grain: np.ndarray, peak: float = GRAIN_PEAK) -> Optional[np.ndarray]:
    """Remove DC e normaliza o pico; devolve float32."""
    g = np.asarray(grain, dtype=np.float32).reshape(-1).copy()
    if g.size < 256:
        return None
    g -= float(np.mean(g))
    p = float(np.max(np.abs(g)))
    if p < 1e-5:
        return None
    return (g / p * peak).astype(np.float32)


def grain_candidates(y: np.ndarray, sr: int) -> List[Dict[str, Any]]:
    """Gera candidatos multi-escala (60/120/180/350 ms) com filtro de silencio.

    Retorna metadados leves; o audio so e recortado depois da selecao.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    n = y.size
    min_len = int(GRAIN_MIN_DURATION * sr)
    if n < min_len:
        return []

    # Integral do quadrado para RMS O(1) por janela
    sq = np.square(y.astype(np.float64))
    cum = np.concatenate(([0.0], np.cumsum(sq)))

    def rms_at(start: int, length: int) -> float:
        end = min(start + length, n)
        length = end - start
        if length <= 0:
            return 0.0
        return float(np.sqrt(max((cum[end] - cum[start]) / length, 0.0)))

    out: List[Dict[str, Any]] = []
    seen = set()

    def add(start: int, duration: float, onset: bool = False) -> None:
        length = int(duration * sr)
        if start < 0 or start + length > n or length < min_len:
            return
        key = (int(start), int(length))
        if key in seen:
            return
        if rms_at(start, length) < MIN_GRAIN_RMS:
            return
        seen.add(key)
        out.append(
            {
                "start": int(start),
                "length": int(length),
                "scale": float(duration),
                "onset": bool(onset),
            }
        )

    onsets = detect_onsets(y, sr)

    # 1) Alinhados a onsets: curtos e medios (transientes/consoantes)
    for onset in onsets:
        for duration in (0.060, 0.120, 0.180):
            add(int(onset), duration, onset=True)

    # 2) Segmentos entre onsets (podem ser vogais sustentadas)
    for a, b in zip(onsets[:-1], onsets[1:]):
        a, b = int(a), int(b)
        dur = (b - a) / sr
        if GRAIN_MIN_DURATION <= dur <= GRAIN_MAX_DURATION:
            add(a, dur, onset=True)
        elif dur > GRAIN_MAX_DURATION:
            start = a
            while start + int(GRAIN_TARGET_DURATION * sr) <= b:
                add(start, GRAIN_TARGET_DURATION)
                start += int(GRAIN_TARGET_DURATION * sr)

    # 3) Grade regular: 180 ms sempre; 60/120 ms a cada 140 ms; 350 ms a cada 210 ms
    hop = int(GRAIN_HOP_DURATION * sr)
    idx = 0
    for start in range(0, n - min_len + 1, hop):
        if rms_at(start, min_len) < MIN_GRAIN_RMS:
            idx += 1
            continue
        add(start, 0.180)
        if idx % 2 == 0:
            add(start, 0.060)
            add(start, 0.120)
        if idx % 3 == 0:
            add(start, 0.350)
        idx += 1

    return out


def _norm01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def _fps_select(
    pool: np.ndarray,
    k: int,
    feats_unit: np.ndarray,
    quality: np.ndarray,
    bins: np.ndarray,
    file_arr: np.ndarray,
    file_counts: Dict[str, int],
    file_cap: int,
) -> List[int]:
    """Seleciona k indices maximizando novidade + qualidade + cobertura temporal."""
    if k <= 0 or pool.size == 0:
        return []

    pool = np.asarray(pool, dtype=np.int64)
    q = np.asarray(quality, dtype=np.float32)
    q_norm = _norm01(q[pool])
    pool_bins = bins[pool]
    pool_files = file_arr[pool]

    selected: List[int] = []
    selected_mask = np.zeros(pool.size, dtype=bool)
    min_dist = np.ones(pool.size, dtype=np.float32)

    # Semente: melhor de cada arquivo no pool (garante cobertura)
    for fname in np.unique(pool_files):
        file_mask = pool_files == fname
        candidates_local = np.where(file_mask)[0]
        if candidates_local.size == 0:
            continue
        best_local = int(candidates_local[np.argmax(q_norm[candidates_local])])
        global_idx = int(pool[best_local])
        if file_counts.get(str(fname), 0) >= file_cap:
            continue
        selected.append(global_idx)
        selected_mask[best_local] = True
        file_counts[str(fname)] = file_counts.get(str(fname), 0) + 1
        min_dist = np.minimum(
            min_dist, 1.0 - feats_unit[pool] @ feats_unit[global_idx]
        )
        if len(selected) >= k:
            return selected

    covered_bins = set(int(b) for b in pool_bins[selected_mask])

    while len(selected) < k:
        file_full = np.array(
            [file_counts.get(str(f), 0) >= file_cap for f in pool_files],
            dtype=bool,
        )
        bin_bonus = np.array(
            [0.0 if int(b) in covered_bins else 1.0 for b in pool_bins],
            dtype=np.float32,
        )
        score = (
            FPS_NOVELTY_WEIGHT * min_dist
            + FPS_QUALITY_WEIGHT * q_norm
            + FPS_BIN_WEIGHT * bin_bonus
        )
        score[selected_mask] = -np.inf
        score[file_full] = -np.inf
        best_local = int(np.argmax(score))
        if not np.isfinite(score[best_local]):
            break

        global_idx = int(pool[best_local])
        selected.append(global_idx)
        selected_mask[best_local] = True
        fname = str(pool_files[best_local])
        file_counts[fname] = file_counts.get(fname, 0) + 1
        covered_bins.add(int(pool_bins[best_local]))
        min_dist = np.minimum(
            min_dist, 1.0 - feats_unit[pool] @ feats_unit[global_idx]
        )

    return selected


def _allocate_counts(total: int, weights) -> List[int]:
    """Reparte um total inteiro proporcionalmente, com metodo do maior resto."""
    if total <= 0:
        return [0 for _ in weights]
    raw = [float(total) * float(w) for w in weights]
    base = [int(math.floor(x)) for x in raw]
    remainder = int(total - sum(base))
    order = sorted(
        range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True
    )
    for i in order[:remainder]:
        base[i] += 1
    return base


def select_candidates(
    candidates: List[Dict[str, Any]],
    sr: int,
    budget: int = MAX_TOTAL_GRAINS,
    per_file_cap: int = MAX_GRAINS_PER_FILE,
    quotas: Dict[str, float] = CLASS_QUOTAS,
    progress=print,
) -> Tuple[List[int], Dict[str, Any]]:
    """Seleciona o palheta final por cotas de classe x escala, diversidade e tempo."""
    n = len(candidates)
    if n <= budget:
        return list(range(n)), {"note": "todos os candidatos couberam no orcamento"}

    feats = np.vstack([c["features"] for c in candidates]).astype(np.float32)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std[std < 1e-6] = 1.0
    feats_unit = (feats - mean) / std
    norms = np.linalg.norm(feats_unit, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    feats_unit = (feats_unit / norms).astype(np.float32)

    scales = np.array([c["scale"] for c in candidates], dtype=np.float32)
    onset = np.array([1.0 if c["onset"] else 0.0 for c in candidates], dtype=np.float32)
    noisy = np.array([c["noisy"] for c in candidates], dtype=np.float32)
    voicing = np.array([c["voicing"] for c in candidates], dtype=np.float32)

    short_factor = np.clip(
        (GRAIN_MAX_DURATION - scales) / (GRAIN_MAX_DURATION - GRAIN_MIN_DURATION),
        0.0, 1.0,
    )
    long_factor = np.clip(
        (scales - GRAIN_MIN_DURATION) / (GRAIN_MAX_DURATION - GRAIN_MIN_DURATION),
        0.0, 1.0,
    )
    transient_raw = (
        (1.0 - 0.5 * noisy) * (0.5 + 0.5 * (1.0 - voicing)) * short_factor
        + 0.30 * onset
    )
    tonal_raw = (1.0 - noisy) * (0.35 + 0.65 * voicing) * long_factor
    noisy_raw = noisy

    class_scores = np.stack(
        [_norm01(transient_raw), _norm01(tonal_raw), _norm01(noisy_raw)], axis=1
    )
    class_names = ["transient", "tonal", "noisy"]
    class_idx = np.argmax(class_scores, axis=1)

    # Bucket de escala: cada candidato pertence a 60/120/180/350 ms
    scale_points = np.array(GRAIN_SCALES, dtype=np.float32)
    scale_ids = np.argmin(
        np.abs(scales[:, None] - scale_points[None, :]), axis=1
    ).astype(np.int64)

    bins = np.array(
        [int(c["start"] // max(1, int(SELECTION_BIN_SECONDS * sr))) for c in candidates],
        dtype=np.int64,
    )
    file_arr = np.array([c["file"] for c in candidates], dtype=object)

    # Cotas de classe. Ordem: tonal primeiro (mais escasso em 350 ms),
    # depois ruidoso, e transiente por ultimo.
    class_order = ["tonal", "noisy", "transient"]
    targets: Dict[str, int] = {}
    remaining = int(budget)
    for i, name in enumerate(class_order):
        if i == len(class_order) - 1:
            target = remaining
        else:
            target = int(round(budget * float(quotas.get(name, 0.0))))
        target = max(0, min(target, remaining))
        targets[name] = target
        remaining -= target

    def _shortlist(pool: np.ndarray, quality: np.ndarray, k: int) -> np.ndarray:
        shortlist = set()
        pool_bins = bins[pool]
        unique_bins = np.unique(pool_bins)
        per_bin = max(
            2, int(np.ceil(k / max(1, unique_bins.size)) * 2)
        )
        per_bin = min(16, per_bin)
        for b in unique_bins:
            idx = pool[pool_bins == b]
            take = min(per_bin, idx.size)
            top = idx[np.argsort(-quality[idx])[:take]]
            shortlist.update(int(i) for i in top)
        top_global = pool[
            np.argsort(-quality[pool])[: min(pool.size, max(4, k * 3))]
        ]
        shortlist.update(int(i) for i in top_global)
        for fname in np.unique(file_arr[pool]):
            idx = pool[file_arr[pool] == fname]
            shortlist.add(int(idx[np.argmax(quality[idx])]))
        arr = np.array(sorted(shortlist), dtype=np.int64)
        return pool if arr.size < k else arr

    selected: List[int] = []
    selected_mask = np.zeros(n, dtype=bool)
    assigned_class = np.full(n, -1, dtype=np.int8)
    file_counts: Dict[str, int] = {}

    def _take(picked: List[int], ci: int) -> int:
        added = 0
        for idx in picked:
            idx = int(idx)
            if selected_mask[idx]:
                continue
            selected.append(idx)
            selected_mask[idx] = True
            assigned_class[idx] = ci
            added += 1
        return added

    for ci, cname in enumerate(class_names):
        target = targets.get(cname, 0)
        if target <= 0:
            continue
        per_scale = _allocate_counts(target, CLASS_SCALE_WEIGHTS[cname])
        picked_class = 0
        scale_report: List[str] = []

        for si, scale_k in enumerate(per_scale):
            if scale_k <= 0:
                continue
            pool = np.where((scale_ids == si) & (~selected_mask))[0]
            if pool.size == 0:
                scale_report.append(f"{GRAIN_SCALES[si] * 1000:.0f}ms:0/{scale_k}")
                continue
            k = min(int(scale_k), int(pool.size))
            short_pool = _shortlist(pool, class_scores[:, ci], k)
            picked = _fps_select(
                short_pool,
                k,
                feats_unit,
                class_scores[:, ci],
                bins,
                file_arr,
                file_counts,
                per_file_cap,
            )
            added = _take(picked, ci)
            picked_class += added
            scale_report.append(
                f"{GRAIN_SCALES[si] * 1000:.0f}ms:{added}/{scale_k}"
            )

        # Se a classe ficou abaixo da cota (pool pequeno ou limite por arquivo),
        # completa com os melhores restantes da mesma classe, em qualquer escala.
        if picked_class < target:
            need = target - picked_class
            pool = np.where(~selected_mask)[0]
            if pool.size:
                short_pool = _shortlist(pool, class_scores[:, ci], need)
                picked = _fps_select(
                    short_pool,
                    need,
                    feats_unit,
                    class_scores[:, ci],
                    bins,
                    file_arr,
                    file_counts,
                    per_file_cap,
                )
                picked_class += _take(picked, ci)

        progress(
            "  classe %s: alvo %d, selecionados %d | %s"
            % (cname, target, picked_class, " ".join(scale_report))
        )

    # Preenche eventuais faltas com o restante, priorizando qualidade e diversidade
    if len(selected) < budget:
        leftovers = np.where(~selected_mask)[0]
        if leftovers.size:
            max_quality = class_scores.max(axis=1)
            need = budget - len(selected)
            picked = _fps_select(
                leftovers,
                need,
                feats_unit,
                max_quality,
                bins,
                file_arr,
                file_counts,
                per_file_cap,
            )
            for idx in picked:
                idx = int(idx)
                if selected_mask[idx]:
                    continue
                selected.append(idx)
                selected_mask[idx] = True
                assigned_class[idx] = int(class_idx[idx])

    selected_arr = np.asarray(selected, dtype=np.int64)
    class_counts = {
        cname: int(np.sum(assigned_class[selected_arr] == ci))
        for ci, cname in enumerate(class_names)
    }
    scale_counts: Dict[str, int] = {}
    for idx in selected:
        key = f"{GRAIN_SCALES[int(scale_ids[int(idx)])] * 1000:.0f}ms"
        scale_counts[key] = scale_counts.get(key, 0) + 1

    info = {
        "candidates": n,
        "selected": len(selected),
        "targets": targets,
        "class_counts": class_counts,
        "scale_counts": scale_counts,
        "selected_class": {
            int(i): int(assigned_class[int(i)]) for i in selected
        },
    }
    return selected, info


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def select_active_indices(
    grains: List[Dict[str, Any]],
    indices: List[int],
    budget: int,
) -> List[int]:
    """Seleciona um subconjunto ativo estratificado por classe e escala.

    A biblioteca continua completa em disco; o conjunto ativo limita apenas
    os graos carregados para selecao/playback. A escolha e deterministica e
    preserva as cotas de classe e escala do corpus.
    """
    unique = sorted({int(i) for i in indices})
    budget = int(max(0, budget))
    if budget <= 0:
        return []
    if len(unique) <= budget:
        return unique

    by_class: Dict[str, List[int]] = {}
    for idx in unique:
        cls = str(grains[idx].get("class", "transient"))
        by_class.setdefault(cls, []).append(idx)

    class_order = [c for c in ("transient", "tonal", "noisy") if c in by_class]
    class_order.extend(sorted(c for c in by_class if c not in class_order))

    # 1) Cotas por classe.
    targets: Dict[str, int] = {c: 0 for c in class_order}
    remaining = budget
    for cls in class_order:
        weight = CLASS_QUOTAS.get(cls, 1.0 / max(1, len(class_order)))
        target = min(
            len(by_class[cls]),
            max(0, remaining),
            int(round(budget * weight)),
        )
        targets[cls] = target
        remaining -= target
    while remaining > 0:
        progressed = False
        for cls in class_order:
            if remaining <= 0:
                break
            if targets[cls] < len(by_class[cls]):
                targets[cls] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    picked: List[int] = []
    for cls in class_order:
        target = targets[cls]
        if target <= 0:
            continue
        by_scale: Dict[int, List[int]] = {}
        for idx in by_class[cls]:
            raw_scale = float(grains[idx].get("scale", 0.0))
            nominal = min(
                GRAIN_SCALES, key=lambda value: abs(float(value) - raw_scale)
            )
            scale_key = int(round(nominal * 1000.0))
            by_scale.setdefault(scale_key, []).append(idx)
        for group in by_scale.values():
            group.sort(
                key=lambda i: (
                    str(grains[i].get("file", "")),
                    float(grains[i].get("start_s", 0.0)),
                )
            )
        scale_keys = sorted(by_scale)
        scale_targets: Dict[int, int] = {key: 0 for key in scale_keys}
        left = target
        weights = CLASS_SCALE_WEIGHTS.get(cls)
        if weights and len(scale_keys) == len(GRAIN_SCALES):
            for key, weight in zip(scale_keys, weights):
                take = min(
                    len(by_scale[key]),
                    max(0, left),
                    int(round(target * weight)),
                )
                scale_targets[key] = take
                left -= take
        while left > 0:
            progressed = False
            for key in scale_keys:
                if left <= 0:
                    break
                if scale_targets[key] < len(by_scale[key]):
                    scale_targets[key] += 1
                    left -= 1
                    progressed = True
            if not progressed:
                break
        for key in scale_keys:
            take = scale_targets[key]
            group = by_scale[key]
            if take <= 0:
                continue
            if take >= len(group):
                picked.extend(group)
                continue
            positions = np.linspace(0, len(group) - 1, take)
            picked.extend(group[int(round(float(pos)))] for pos in positions)

    # Garante o budget mesmo se alguma classe tiver capacidade insuficiente.
    if len(picked) < budget:
        used = set(picked)
        for idx in unique:
            if len(picked) >= budget:
                break
            if idx not in used:
                picked.append(idx)
                used.add(idx)
    return sorted(set(picked[:budget]))


@dataclass
class Corpus:
    grains: List[Dict[str, Any]]
    features: np.ndarray               # raw (n, 15) float32
    features_norm: np.ndarray          # escalado + L2 (n, 15) float32
    mean: np.ndarray
    std: np.ndarray
    sr: int
    signature: List[List[Any]]
    file_stats: Dict[str, Any] = field(default_factory=dict)
    selection_info: Dict[str, Any] = field(default_factory=dict)
    version: int = CACHE_VERSION

    # -- resumo ---------------------------------------------------------
    def summary(self) -> str:
        return f"{len(self.grains)} graos @ {self.sr} Hz"

    def subset(self, indices: List[int]) -> "Corpus":
        """Cria a visao ativa sem copiar o audio dos graos."""
        chosen = sorted({int(i) for i in indices})
        if not chosen:
            raise ValueError("Conjunto ativo vazio.")
        grains = [self.grains[i] for i in chosen]
        features = self.features[chosen]
        features_norm = self.features_norm[chosen]
        class_counts: Dict[str, int] = {}
        for grain in grains:
            cls = str(grain.get("class", "transient"))
            class_counts[cls] = class_counts.get(cls, 0) + 1
        info = dict(self.selection_info)
        info["selected"] = int(len(grains))
        info["class_counts"] = class_counts
        info["active"] = int(len(grains))
        info["library_total"] = int(len(self.grains))
        return Corpus(
            grains=grains,
            features=features,
            features_norm=features_norm,
            mean=self.mean,
            std=self.std,
            sr=self.sr,
            signature=self.signature,
            file_stats=self.file_stats,
            selection_info=info,
        )

    def cache_size_mb(self) -> float:
        try:
            return os.path.getsize(CORPUS_CACHE_FILE) / 1e6
        except OSError:
            return 0.0

    # -- features -------------------------------------------------------
    @staticmethod
    def _normalize(
        features: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        scaled = (features - mean) / std
        norms = np.linalg.norm(scaled, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        return (scaled / norms).astype(np.float32)

    def query_norm(self, features: np.ndarray) -> np.ndarray:
        q = (np.asarray(features, dtype=np.float32) - self.mean) / self.std
        n = float(np.linalg.norm(q))
        if n < 1e-9:
            return np.zeros_like(q, dtype=np.float32)
        return (q / n).astype(np.float32)

    # -- assinatura -----------------------------------------------------
    @staticmethod
    def _signature(amostras_dir: str) -> List[List[Any]]:
        sig: List[List[Any]] = []
        if not os.path.isdir(amostras_dir):
            return sig
        for name in sorted(os.listdir(amostras_dir)):
            path = os.path.join(amostras_dir, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name)[1].lower() not in (
                ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"
            ):
                continue
            st = os.stat(path)
            sig.append([name, int(st.st_size), int(st.st_mtime)])
        return sig

    # -- build ----------------------------------------------------------
    @classmethod
    def build(
        cls,
        amostras_dir: str = AMOSTRAS_DIR,
        sr: int = SAMPLE_RATE,
        budget: int = MAX_TOTAL_GRAINS,
        per_file_cap: int = MAX_GRAINS_PER_FILE,
        progress=print,
    ) -> "Corpus":
        if not os.path.isdir(amostras_dir):
            raise FileNotFoundError(f"Pasta de amostras nao encontrada: {amostras_dir}")

        budget = int(max(1, budget))
        per_file_cap = int(max(1, per_file_cap))
        signature = cls._signature(amostras_dir)
        if not signature:
            raise FileNotFoundError(f"Nenhum audio encontrado em {amostras_dir}")

        progress(f"Indexando {len(signature)} arquivo(s) de audio...")
        candidates: List[Dict[str, Any]] = []
        file_stats: Dict[str, Any] = {}

        for name, _size, _mtime in signature:
            path = os.path.join(amostras_dir, name)
            try:
                y = load_audio(path, sr)
                duration = len(y) / sr

                # Proporcao de tempo nao-silencio (frames de 100 ms)
                frame_len = max(1, int(0.100 * sr))
                n_frames = len(y) // frame_len
                if n_frames > 0:
                    frames = y[: n_frames * frame_len].reshape(-1, frame_len)
                    frame_rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))
                    non_silent = float(np.mean(frame_rms >= MIN_GRAIN_RMS))
                else:
                    non_silent = 0.0

                cands = grain_candidates(y, sr)
                kept = 0
                for cand in cands:
                    start = int(cand["start"])
                    length = int(cand["length"])
                    audio = y[start:start + length]
                    feats, desc = analyze_grain(audio, sr, with_f0=True)
                    if desc["rms"] < MIN_GRAIN_RMS:
                        continue
                    candidates.append(
                        {
                            "start": start,
                            "length": length,
                            "scale": float(cand["scale"]),
                            "onset": bool(cand["onset"]),
                            "features": feats,
                            "zcr": desc["zcr"],
                            "centroid": desc["centroid"],
                            "flatness": desc["flatness"],
                            "hf_ratio": desc["hf_ratio"],
                            "rms": desc["rms"],
                            "f0": desc["f0"],
                            "voicing": desc["voicing"],
                            "noisy": desc["noisy"],
                            "file": name,
                        }
                    )
                    kept += 1

                file_stats[name] = {
                    "duration": float(duration),
                    "non_silent": float(non_silent),
                    "candidates": int(kept),
                    "selected": 0,
                }
                progress(
                    "  %s: %.1f s, %.0f%% nao-silencio, %d candidatos"
                    % (name, duration, non_silent * 100.0, kept)
                )
            except Exception as exc:
                progress(f"  ERRO em {name}: {exc}")

        if not candidates:
            raise ValueError("Nenhum grao util extraido do corpus.")

        progress(
            f"Selecionando ate {budget} graos de {len(candidates)} candidatos "
            f"(cotas {CLASS_QUOTAS})..."
        )
        selected, info = select_candidates(
            candidates, sr, budget=budget, per_file_cap=per_file_cap,
            progress=progress,
        )
        class_names = ["transient", "tonal", "noisy"]
        selected_class = info.get("selected_class", {})

        # Recorta o audio apenas dos selecionados e recalcula features no
        # audio normalizado (consistencia entre indexacao e consulta).
        by_file: Dict[str, List[int]] = {}
        for idx in selected:
            by_file.setdefault(candidates[idx]["file"], []).append(idx)

        grains: List[Dict[str, Any]] = []
        for name, idxs in by_file.items():
            path = os.path.join(amostras_dir, name)
            y = load_audio(path, sr)
            kept = 0
            for idx in idxs:
                cand = candidates[idx]
                raw = y[cand["start"]:cand["start"] + cand["length"]]
                grain_audio = normalize_grain(raw)
                if grain_audio is None:
                    continue
                feats, desc = analyze_grain(grain_audio, sr, with_f0=True)
                if desc["rms"] < MIN_GRAIN_RMS:
                    continue
                if idx in selected_class:
                    class_name = class_names[int(selected_class[idx])]
                else:
                    class_name = (
                        "noisy" if cand["noisy"] >= 0.55
                        else "tonal" if cand["voicing"] >= 0.40
                        else "transient"
                    )
                grains.append(
                    {
                        "audio": grain_audio.astype(np.float16),
                        "features": feats,
                        "zcr": desc["zcr"],
                        "centroid": desc["centroid"],
                        "flatness": desc["flatness"],
                        "hf_ratio": desc["hf_ratio"],
                        "rms": desc["rms"],
                        "f0": desc["f0"],
                        "voicing": desc["voicing"],
                        "noisy": desc["noisy"],
                        "duration": desc["duration"],
                        "scale": cand["scale"],
                        "start": cand["start"],
                        "start_s": cand["start"] / sr,
                        "onset": cand["onset"],
                        "class": class_name,
                        "file": name,
                    }
                )
                kept += 1
            file_stats[name]["selected"] = int(kept)
            progress(f"  {name}: {kept} graos selecionados")

        if not grains:
            raise ValueError("Nenhum grao valido apos a selecao.")

        grains.sort(key=lambda g: (g["file"], g["start"]))
        features = np.vstack([g["features"] for g in grains]).astype(np.float32)
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std < 1e-6] = 1.0
        features_norm = cls._normalize(features, mean, std)

        actual_classes: Dict[str, int] = {}
        for g in grains:
            actual_classes[g["class"]] = actual_classes.get(g["class"], 0) + 1

        selection_info = {
            "candidates": int(info.get("candidates", len(candidates))),
            "selected": int(len(grains)),
            "targets": info.get("targets", {}),
            "class_counts": actual_classes,
            "budget": int(budget),
            "per_file_cap": int(per_file_cap),
            "hop": float(GRAIN_HOP_DURATION),
            "scales": [float(s) for s in GRAIN_SCALES],
            "quotas": dict(CLASS_QUOTAS),
        }

        progress(
            f"Corpus pronto: {len(grains)} graos | classes {actual_classes}."
        )
        return cls(
            grains=grains,
            features=features,
            features_norm=features_norm,
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            sr=sr,
            signature=signature,
            file_stats=file_stats,
            selection_info=selection_info,
        )

    # -- persistencia ---------------------------------------------------
    def save(self, path: str = CORPUS_CACHE_FILE, progress=print) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "version": CACHE_VERSION,
            "grains": self.grains,
            "features": self.features,
            "features_norm": self.features_norm,
            "mean": self.mean,
            "std": self.std,
            "sr": self.sr,
            "signature": self.signature,
            "file_stats": self.file_stats,
            "selection_info": self.selection_info,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        progress(f"Cache salvo: {path} ({self.cache_size_mb():.1f} MB)")

    @classmethod
    def load(cls, path: str = CORPUS_CACHE_FILE) -> "Corpus":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        if int(data.get("version", -1)) != CACHE_VERSION:
            raise ValueError("versao de cache incompativel")
        for key in ("grains", "features", "features_norm", "mean", "std", "signature"):
            if key not in data:
                raise ValueError(f"cache sem campo '{key}'")
        return cls(
            grains=data["grains"],
            features=data["features"],
            features_norm=data["features_norm"],
            mean=data["mean"],
            std=data["std"],
            sr=int(data["sr"]),
            signature=data["signature"],
            file_stats=data.get("file_stats", {}),
            selection_info=data.get("selection_info", {}),
        )

    @classmethod
    def load_or_build(
        cls,
        amostras_dir: str = AMOSTRAS_DIR,
        path: str = CORPUS_CACHE_FILE,
        force: bool = False,
        progress=print,
        budget: int = MAX_TOTAL_GRAINS,
        per_file_cap: int = MAX_GRAINS_PER_FILE,
    ) -> "Corpus":
        current_sig = cls._signature(amostras_dir)
        if not force and os.path.exists(path):
            try:
                corpus = cls.load(path)
                if corpus.signature == current_sig:
                    cached_budget = int(
                        corpus.selection_info.get("budget", len(corpus.grains))
                    )
                    if int(budget) > cached_budget:
                        progress(
                            "Aviso: o cache atual tem orcamento %d, menor que "
                            "o solicitado (%d). Use --force-reindex para "
                            "reconstruir a biblioteca."
                            % (cached_budget, int(budget))
                        )
                    progress(f"Cache carregado: {corpus.summary()} "
                             f"({corpus.cache_size_mb():.1f} MB)")
                    return corpus
                progress("Amostras mudaram desde o ultimo cache; reindexando...")
            except Exception as exc:
                progress(f"Cache invalido ({exc}); reindexando...")
        corpus = cls.build(
            amostras_dir,
            progress=progress,
            budget=int(budget),
            per_file_cap=int(per_file_cap),
        )
        try:
            corpus.save(path, progress=progress)
        except Exception as exc:
            progress(f"Aviso: nao foi possivel salvar o cache: {exc}")
        return corpus

    # -- playback -------------------------------------------------------
    def prepare_playback(self, out_sr: int, progress=print) -> None:
        """Resamplica os grãos para a taxa de saída (float32) uma unica vez."""
        out_sr = int(out_sr)
        for grain in self.grains:
            if grain.get("playback_sr") == out_sr:
                continue
            audio = np.asarray(grain["audio"])
            if int(self.sr) != out_sr:
                g = math.gcd(int(self.sr), out_sr)
                audio = resample_poly(
                    audio.astype(np.float32), out_sr // g, int(self.sr) // g
                ).astype(np.float16)
            # Mantem float16 no playback: o corpus ja usa essa precisao e
            # isso evita duplicar o audio em float32 na RAM.
            grain["audio_playback"] = audio
            grain["playback_sr"] = out_sr
        progress(f"Playback preparado a {out_sr} Hz ({len(self.grains)} graos).")


# ---------------------------------------------------------------------------
# Pitch shift microtonal (varispeed)
# ---------------------------------------------------------------------------


def soft_limit(
    audio: np.ndarray,
    ceiling: float = SOFT_LIMIT_CEILING,
    knee: float = SOFT_LIMIT_KNEE,
) -> np.ndarray:
    """Limiter suave: transparente abaixo do knee, ceiling rígido acima."""
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x
    ax = np.abs(x)
    out = ax.copy()
    over = ax > knee
    if np.any(over):
        span = max(1e-6, ceiling - knee)
        out[over] = knee + span * np.tanh((ax[over] - knee) / span)
    return (np.sign(x) * out).astype(np.float32)


def apply_hann_fade(
    grain: np.ndarray, fade_ms: float = FADE_MS, sr: int = SAMPLE_RATE
) -> np.ndarray:
    grain = np.asarray(grain, dtype=np.float32).copy()
    fade = min(int(fade_ms * sr / 1000.0), max(0, len(grain) // 2))
    if fade <= 1:
        return grain
    window = np.hanning(2 * fade).astype(np.float32)
    grain[:fade] *= window[:fade]
    grain[-fade:] *= window[fade:]
    return grain


def quantized_shift_cents(cents_shift: float, tuning: str = "24edo") -> int:
    """Quantiza o desvio microtonal e devolve o valor em cents."""
    if tuning == "ji":
        target = min(
            HARMONIC_SERIES_DEVIATIONS_CENTS.values(),
            key=lambda c: abs(c - cents_shift),
        )
        return int(round(float(target) / 10.0) * 10)
    return int(round(float(quantize_to_microtone(cents_shift, mode="24edo"))))


def wsola_time_stretch(
    audio: np.ndarray,
    target_length: int,
    sample_rate: int,
) -> np.ndarray:
    """Time-stretch WSOLA em numpy, sem mudar o pitch.

    Usado para restaurar total ou parcialmente a duracao depois do
    varispeed. Nao preserva formantes de forma explicita: apenas
    remove o encurtamento/alongamento temporal do grão.
    """
    x = np.asarray(audio, dtype=np.float32)
    n = int(x.size)
    target_length = int(round(float(target_length)))
    if target_length <= 0 or n < 128:
        return x.copy()
    if abs(target_length - n) <= 1:
        return x.copy()

    if n < 4096:
        frame = 256
    elif n < 8192:
        frame = 512
    else:
        frame = 1024
    frame = int(min(frame, max(64, n // 2)))
    if frame < 64:
        return x.copy()
    hop = max(16, frame // 4)
    overlap = max(16, frame - hop)
    search = max(8, frame // 8)

    analysis_hop = hop * float(n) / float(max(1, target_length))
    window = np.hanning(frame).astype(np.float32)
    out = np.zeros(target_length + frame, dtype=np.float32)
    norm = np.zeros(target_length + frame, dtype=np.float32)
    max_start = max(0, n - frame)

    pos = 0.0
    out_pos = 0
    prev_start: Optional[int] = None
    frames = 0
    max_frames = int(math.ceil(target_length / float(hop))) + 2

    while out_pos < target_length and frames < max_frames:
        nominal = int(round(pos))
        nominal = max(0, min(max_start, nominal))
        start = nominal

        if prev_start is not None:
            expected_start = prev_start + hop
            expected = x[expected_start:expected_start + overlap]
            if expected.size >= max(16, overlap // 2):
                target = expected
                t_norm = float(np.sqrt(np.dot(target, target)))
                lo = max(0, nominal - search)
                hi = min(max_start, nominal + search)
                if t_norm > 1e-9 and hi >= lo:
                    starts = lo + 4 * np.arange(
                        (hi - lo) // 4 + 1, dtype=np.int64
                    )
                    windows = np.lib.stride_tricks.sliding_window_view(
                        x, target.size
                    )
                    segments = windows[starts]
                    norms = np.sqrt(
                        np.einsum("ij,ij->i", segments, segments)
                    )
                    valid = norms > 1e-9
                    if np.any(valid):
                        scores = np.full(
                            starts.size, -np.inf, dtype=np.float32
                        )
                        scores[valid] = (
                            segments[valid] @ target
                        ) / (norms[valid] * t_norm)
                        best = int(np.argmax(scores))
                        if scores[best] > -1e29:
                            start = int(starts[best])

        start = max(0, min(max_start, start))
        segment = x[start:start + frame]
        if segment.size < frame:
            segment = np.pad(segment, (0, frame - segment.size))
        out[out_pos:out_pos + frame] += segment * window
        norm[out_pos:out_pos + frame] += window
        prev_start = start
        pos += analysis_hop
        out_pos += hop
        frames += 1

    out = out[:target_length]
    norm = norm[:target_length]
    if not np.any(norm):
        return x.copy()
    out = out / np.maximum(norm, 0.10)
    return out.astype(np.float32)


def apply_microtonal_pitch_shift(
    audio: np.ndarray,
    sample_rate: int,
    cents_shift: float,
    tuning: str = "24edo",
    preserve_duration: float = 0.0,
    absolute: bool = False,
    fade_ms: float = FADE_MS,
) -> np.ndarray:
    """Varispeed microtonal, com WSOLA opcional para preservar duracao.

    ``absolute=False`` quantiza um desvio microtonal (24-EDO/JI).
    ``absolute=True`` quantiza um intervalo absoluto (quarta, quinta, etc.).
    ``preserve_duration`` 0 = varispeed puro; 1 = WSOLA restaura o
    comprimento original; valores intermediarios restauram parcialmente.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if absolute:
        cents = int(round(quantize_absolute_cents(cents_shift, tuning)))
    else:
        cents = quantized_shift_cents(cents_shift, tuning)
    if cents == 0 or audio.size < 32:
        return apply_hann_fade(audio, fade_ms, sample_rate)

    factor = 2.0 ** (cents / 1200.0)  # >1 = mais agudo
    ratio = Fraction(1.0 / factor).limit_denominator(512)
    up = max(1, int(ratio.numerator))
    down = max(1, int(ratio.denominator))
    shifted = resample_poly(audio, up, down).astype(np.float32)
    if shifted.size < 32:
        return apply_hann_fade(audio, fade_ms, sample_rate)

    preserve = float(max(0.0, min(1.0, preserve_duration)))
    if preserve > 0.001:
        target = int(round(shifted.size * (1.0 - preserve) + audio.size * preserve))
        if abs(target - shifted.size) > 2:
            try:
                candidate = wsola_time_stretch(shifted, target, sample_rate)
                if candidate.size >= 32:
                    shifted = candidate
            except Exception:
                # Fallback seguro: mantem o varispeed se o WSOLA falhar.
                pass

    return apply_hann_fade(shifted, fade_ms, sample_rate)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class VocoolderEngine:
    """Motor reativo leve (numpy/scipy/sounddevice)."""

    def __init__(
        self,
        ricochet_density: int = DEFAULT_RICOCHET_DENSITY,
        mutation_rate: float = DEFAULT_MUTATION_RATE,
        rms_threshold: float = DEFAULT_RMS_THRESHOLD,
        sample_rate: int = SAMPLE_RATE,
        buffer_duration: float = 1.25,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        tuning: str = "24edo",
        mix: float = DEFAULT_MIX,
        pitch_mode: str = DEFAULT_PITCH_MODE,
        pitch_interval: str = DEFAULT_PITCH_INTERVAL,
        preserve_duration: float = DEFAULT_PRESERVE_DURATION,
        grain_gain: float = GRAIN_GAIN,
        phonetic_response: float = DEFAULT_PHONETIC_RESPONSE,
        auto_calibrate: bool = True,
        active_grains: int = DEFAULT_ACTIVE_GRAINS,
        sources: Optional[List[str]] = None,
        library_grains: int = MAX_TOTAL_GRAINS,
        per_file_cap: int = MAX_GRAINS_PER_FILE,
        progress=print,
    ) -> None:
        self.ricochet_density = int(
            max(MIN_RICOCHET_DENSITY, min(MAX_RICOCHET_DENSITY, ricochet_density))
        )
        self.mutation_rate = float(max(0.0, min(1.0, mutation_rate)))
        self.phonetic_response = float(max(0.0, min(1.0, phonetic_response)))
        self.rms_threshold = float(
            max(MIN_RMS_THRESHOLD, min(MAX_RMS_THRESHOLD, rms_threshold))
        )
        self.analysis_sr = int(sample_rate)
        self.buffer_duration = float(buffer_duration)
        self.tuning = tuning if tuning in ("24edo", "ji") else "24edo"
        self.mix = float(max(0.0, min(1.0, mix)))
        self._mix_smooth = self.mix
        self.pitch_mode = (
            pitch_mode if pitch_mode in PITCH_MODES else DEFAULT_PITCH_MODE
        )
        self.pitch_interval = (
            pitch_interval
            if pitch_interval in PITCH_INTERVALS
            else DEFAULT_PITCH_INTERVAL
        )
        self.preserve_duration = float(
            max(0.0, min(1.0, preserve_duration))
        )
        self._f0_history: List[float] = []
        self._last_voice_f0: Optional[float] = None
        self.grain_gain = float(grain_gain)
        self.auto_calibrate = bool(auto_calibrate)
        self.progress = progress

        # Biblioteca completa (cache) x conjunto ativo (selecao/playback).
        self.library_grains = int(max(1, library_grains))
        self.per_file_cap = int(max(1, per_file_cap))
        self.active_grains = int(
            max(MIN_ACTIVE_GRAINS, min(MAX_ACTIVE_GRAINS, active_grains))
        )
        self.sources = [
            os.path.basename(str(s).strip())
            for s in (sources or [])
            if str(s).strip()
        ]
        self._library_total = 0
        self._active_sources: List[str] = []
        self._loaded_config: Optional[Tuple[int, int, int, Tuple[str, ...]]] = None

        self.input_device = input_device
        self.output_device = output_device
        self.input_samplerate: Optional[int] = None
        self.output_samplerate: Optional[int] = None
        self.in_channels = 1
        self.out_channels = 2

        self.corpus: Optional[Corpus] = None
        self.is_running = False
        self.last_error = ""

        self.input_rms = 0.0
        self.input_peak = 0.0
        self.output_peak = 0.0

        self._in_cb_count = 0
        self._out_cb_count = 0
        self._out_nonzero = 0
        self._render_count = 0
        self._gate_blocks = 0
        self._gate_attack = 0
        self._gate_hold = 0
        self._events = 0
        self._event_time_warn = 0
        self._event_time_max = 0.0
        self._calibrating = False

        self._stream_in: Optional[sd.InputStream] = None
        self._stream_out: Optional[sd.OutputStream] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._in_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=IN_QUEUE_MAX)
        self._out_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=OUT_QUEUE_MAX)
        self._render_sink: Optional[List[np.ndarray]] = None

        self._active_grains: List[Dict[str, Any]] = []
        self._shift_cache: Dict[Tuple[int, int, int, str, bool], np.ndarray] = {}
        self._samples_to_event = 0.0
        self._recent = np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32)
        self._recent_grains: List[int] = []
        self._last_selection: List[Dict[str, Any]] = []
        self._selection_history: List[Dict[str, Any]] = []
        self._last_query: Dict[str, Any] = {}

        self.session_recorder: Optional[SessionRecorder] = None
        self.last_recording: Optional[Dict[str, Any]] = None
        self._session_samples = 0

    # -- corpus ---------------------------------------------------------
    def load_corpus(self, force_reindex: bool = False) -> bool:
        try:
            library = Corpus.load_or_build(
                AMOSTRAS_DIR,
                CORPUS_CACHE_FILE,
                force=force_reindex,
                progress=self.progress,
                budget=self.library_grains,
                per_file_cap=self.per_file_cap,
            )
            self._library_total = len(library.grains)
            self.corpus = self._build_active_view(library)
            self._loaded_config = (
                self.library_grains,
                self.per_file_cap,
                self.active_grains,
                tuple(self.sources),
            )
            if self.output_samplerate:
                self.corpus.prepare_playback(self.output_samplerate, progress=self.progress)
            self.progress(
                "Corpus pronto: %s | biblioteca: %d | ativos: %d | fontes: %s"
                % (
                    self.corpus.summary(),
                    self._library_total,
                    len(self.corpus.grains),
                    ", ".join(self._active_sources) if self._active_sources else "todas",
                )
            )
            return True
        except Exception as exc:
            self.last_error = f"Falha ao carregar corpus: {exc}"
            self.progress(self.last_error)
            return False

    def _build_active_view(self, library: Corpus) -> Corpus:
        """Filtra fontes e limita o conjunto ativo, liberando o resto da RAM."""
        all_indices = list(range(len(library.grains)))
        available = sorted(
            {str(g.get("file", "")) for g in library.grains if g.get("file")}
        )
        requested = [s for s in self.sources if s in available]
        missing = [s for s in self.sources if s not in available]
        if missing:
            self.progress(
                "Aviso: fonte(s) nao encontrada(s) na biblioteca: %s"
                % ", ".join(missing)
            )
        if self.sources and not requested:
            self.progress("Aviso: nenhuma fonte solicitada existe; usando todas.")
            requested = available
        candidates = all_indices
        if self.sources and requested:
            wanted = set(requested)
            candidates = [
                i for i in all_indices if str(library.grains[i].get("file", "")) in wanted
            ]
        if not candidates:
            candidates = all_indices
        target = min(int(self.active_grains), len(candidates))
        chosen = select_active_indices(library.grains, candidates, target)
        if not chosen:
            chosen = candidates
        self._active_sources = sorted(
            {str(library.grains[i].get("file", "")) for i in chosen}
        )

        # A biblioteca completa permanece no cache em disco. Em memoria,
        # libera o audio dos graos que nao pertencem ao conjunto ativo.
        keep = set(chosen)
        for idx, grain in enumerate(library.grains):
            if idx in keep:
                continue
            grain["audio"] = np.empty(0, dtype=np.float16)
            grain.pop("audio_playback", None)
            grain.pop("playback_sr", None)
        if len(chosen) == len(library.grains):
            return library
        return library.subset(chosen)

    # -- dispositivos ---------------------------------------------------
    def _resolve_devices(self) -> None:
        defaults = sd.default.device
        if self.input_device is None or int(self.input_device) < 0:
            self.input_device = defaults[0]
        if self.output_device is None or int(self.output_device) < 0:
            self.output_device = defaults[1]
        if (
            self.input_device is None
            or int(self.input_device) < 0
            or self.output_device is None
            or int(self.output_device) < 0
        ):
            raise RuntimeError("Windows nao informou dispositivos padrao de audio.")
        self.progress(
            "Dispositivos: entrada #%d (%s) | saida #%d (%s)"
            % (
                self.input_device,
                sd.query_devices(self.input_device)["name"],
                self.output_device,
                sd.query_devices(self.output_device)["name"],
            )
        )

    @staticmethod
    def _negotiate(device: int, kind: str) -> Tuple[int, int]:
        info = sd.query_devices(device)
        if kind == "input":
            max_ch = int(info["max_input_channels"])
            ch_candidates = [c for c in (1, 2) if c <= max_ch] or [max_ch]
        else:
            max_ch = int(info["max_output_channels"])
            ch_candidates = [c for c in (2, 1) if c <= max_ch] or [max_ch]
        if max_ch <= 0:
            raise RuntimeError(f"Dispositivo {device} sem canais de {kind}.")

        base_sr = int(round(float(info["default_samplerate"])))
        sr_candidates: List[int] = []
        for sr in (base_sr, 48000, 44100, 32000):
            if sr not in sr_candidates:
                sr_candidates.append(sr)

        last_err: Optional[Exception] = None
        for sr in sr_candidates:
            for ch in ch_candidates:
                try:
                    if kind == "input":
                        sd.check_input_settings(
                            device=device, samplerate=sr, channels=ch, dtype="float32"
                        )
                    else:
                        sd.check_output_settings(
                            device=device, samplerate=sr, channels=ch, dtype="float32"
                        )
                    return sr, ch
                except Exception as exc:
                    last_err = exc
        raise RuntimeError(
            f"Nenhuma configuracao valida para {kind} no dispositivo {device}: {last_err}"
        )

    # -- parametros -----------------------------------------------------
    def set_parameters(
        self,
        ricochet_density: Optional[int] = None,
        mutation_rate: Optional[float] = None,
        rms_threshold: Optional[float] = None,
        phonetic_response: Optional[float] = None,
        mix: Optional[float] = None,
        tuning: Optional[str] = None,
        pitch_mode: Optional[str] = None,
        pitch_interval: Optional[str] = None,
        preserve_duration: Optional[float] = None,
    ) -> None:
        if ricochet_density is not None:
            self.ricochet_density = int(
                max(MIN_RICOCHET_DENSITY, min(MAX_RICOCHET_DENSITY, ricochet_density))
            )
        if mutation_rate is not None:
            self.mutation_rate = float(max(0.0, min(1.0, mutation_rate)))
        if rms_threshold is not None:
            self.rms_threshold = float(
                max(MIN_RMS_THRESHOLD, min(MAX_RMS_THRESHOLD, rms_threshold))
            )
        if phonetic_response is not None:
            self.phonetic_response = float(
                max(0.0, min(1.0, phonetic_response))
            )
        if mix is not None:
            self.mix = float(max(0.0, min(1.0, mix)))
        if tuning in ("24edo", "ji"):
            self.tuning = tuning
        if pitch_mode is not None and pitch_mode in PITCH_MODES:
            self.pitch_mode = pitch_mode
        if pitch_interval is not None and pitch_interval in PITCH_INTERVALS:
            self.pitch_interval = pitch_interval
        if preserve_duration is not None:
            self.preserve_duration = float(
                max(0.0, min(1.0, preserve_duration))
            )

    def status_text(self) -> str:
        if not self.is_running:
            return "Parado"
        db = 20.0 * math.log10(max(self.input_rms, 1e-9))
        return f"Ativo | Entrada {db:.1f} dBFS | Saida {self.output_peak:.2f}"

    # -- start / stop ---------------------------------------------------
    def start(self) -> bool:
        if self.is_running:
            return True
        self.last_error = ""

        try:
            self._resolve_devices()
            assert self.input_device is not None and self.output_device is not None
            self.input_samplerate, self.in_channels = self._negotiate(self.input_device, "input")
            self.output_samplerate, self.out_channels = self._negotiate(self.output_device, "output")
        except Exception as exc:
            self.last_error = f"Dispositivo de audio: {exc}"
            self.progress(self.last_error)
            return False

        config = (
            self.library_grains,
            self.per_file_cap,
            self.active_grains,
            tuple(self.sources),
        )
        if self.corpus is None or self._loaded_config != config:
            if not self.load_corpus():
                return False
        else:
            self.corpus.prepare_playback(self.output_samplerate, progress=self.progress)

        self._reset_runtime_state()

        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="VocoolderWorker"
        )
        self._worker_thread.start()

        try:
            self._stream_in = sd.InputStream(
                device=self.input_device,
                samplerate=self.input_samplerate,
                channels=self.in_channels,
                blocksize=IN_BLOCK,
                dtype="float32",
                callback=self._input_callback,
            )
            self._stream_out = sd.OutputStream(
                device=self.output_device,
                samplerate=self.output_samplerate,
                channels=self.out_channels,
                blocksize=OUT_BLOCK,
                dtype="float32",
                callback=self._output_callback,
            )
            self._stream_in.start()
            self._stream_out.start()
        except Exception as exc:
            self.last_error = f"Falha ao abrir streams: {exc}"
            self.progress(self.last_error)
            self._cleanup_streams()
            self._stop_event.set()
            return False

        self.is_running = True
        self.progress("Vocoolder ativo.")

        if self.auto_calibrate:
            self._calibrating = True
            self.progress("Calibrando gate: fique em silencio por ~1 segundo...")
            try:
                threshold = self.calibrate_gate(AUTO_GATE_DURATION)
                self.progress(f"Gate calibrado em {threshold:.4f} RMS.")
            except Exception as exc:
                self.progress(f"Aviso: calibracao do gate falhou: {exc}")
            finally:
                self._calibrating = False
                self._samples_to_event = 0.0

        return True

    def _reset_runtime_state(self) -> None:
        self._stop_event.clear()
        self._in_queue = queue.Queue(maxsize=IN_QUEUE_MAX)
        self._out_queue = queue.Queue(maxsize=OUT_QUEUE_MAX)
        self._active_grains = []
        self._shift_cache.clear()
        self._recent = np.zeros(int(0.3 * int(self.output_samplerate or SAMPLE_RATE)),
                                dtype=np.float32)
        self._samples_to_event = 0.0
        self._session_samples = 0
        self._in_cb_count = 0
        self._out_cb_count = 0
        self._out_nonzero = 0
        self._render_count = 0
        self._gate_blocks = 0
        self._gate_attack = 0
        self._gate_hold = 0
        self._events = 0
        self._event_time_warn = 0
        self._event_time_max = 0.0
        self._calibrating = False
        self._mix_smooth = self.mix
        self._f0_history = []
        self._last_voice_f0 = None
        self._recent_grains = []
        self._last_selection = []
        self._selection_history = []
        self._last_query = {}

    def stop(self) -> None:
        self.is_running = False
        self._stop_event.set()
        self._cleanup_streams()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None
        self.stop_recording()
        self.progress("Vocoolder parado.")

    # -- gravacao de sessao --------------------------------------------
    def start_recording(self) -> Dict[str, Any]:
        if not self.is_running:
            raise RuntimeError("Inicie o Vocoolder antes de gravar a sessao.")
        if self.session_recorder is not None and self.session_recorder.is_recording:
            return {
                "audio_path": self.session_recorder.audio_path,
                "event_path": self.session_recorder.event_path,
            }
        sr = int(self.output_samplerate or SAMPLE_RATE)
        channels = max(1, int(self.out_channels or 2))
        self.session_recorder = SessionRecorder(
            sample_rate=sr,
            channels=channels,
        )
        self.progress(
            "Gravando sessao: %s (log %s)"
            % (self.session_recorder.audio_path, self.session_recorder.event_path)
        )
        return {
            "audio_path": self.session_recorder.audio_path,
            "event_path": self.session_recorder.event_path,
        }

    def stop_recording(self) -> Optional[Dict[str, Any]]:
        recorder = self.session_recorder
        if recorder is None:
            return None
        info = recorder.stop()
        self.session_recorder = None
        self.last_recording = info
        if info.get("error"):
            self.progress("Aviso na gravacao: %s" % info["error"])
        else:
            self.progress(
                "Sessao gravada: %.1f s | %s"
                % (info.get("duration_seconds", 0.0), info.get("audio_path", ""))
            )
        if info.get("dropped_audio_blocks"):
            self.progress(
                "Aviso: %d bloco(s) de audio descartados pela fila do gravador."
                % int(info["dropped_audio_blocks"])
            )
        return info

    def _cleanup_streams(self) -> None:
        for stream in (self._stream_in, self._stream_out):
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
        self._stream_in = None
        self._stream_out = None

    def calibrate_gate(self, duration: float = AUTO_GATE_DURATION) -> float:
        """Mede o ruido ambiente e ajusta o gate (mediana * 3.5, piso 0.006)."""
        if not self.is_running and self._stream_in is None:
            raise RuntimeError("Streams de entrada nao estao ativos.")
        # Deixa o stream estabilizar e ignora amostras iniciais zeradas
        time.sleep(0.2)
        samples: List[float] = []
        t_end = time.time() + max(0.2, duration)
        while time.time() < t_end:
            samples.append(float(self.input_rms))
            time.sleep(0.05)
        samples = [s for s in samples if s > 1e-6]
        if not samples:
            raise RuntimeError("Sem amostras de entrada.")
        noise = float(np.percentile(samples, 50))
        threshold = max(noise * 3.5, AUTO_GATE_FLOOR, MIN_RMS_THRESHOLD)
        threshold = min(threshold, MAX_RMS_THRESHOLD)
        self.rms_threshold = float(threshold)
        self.progress(
            "Ruido medido: %.5f RMS -> gate %.5f RMS (%.1f dBFS)"
            % (noise, self.rms_threshold,
               20.0 * math.log10(max(self.rms_threshold, 1e-9)))
        )
        return self.rms_threshold

    # -- callbacks ------------------------------------------------------
    def _input_callback(self, indata, frames, time_info, status) -> None:
        try:
            self._in_cb_count += 1
            data = np.asarray(indata, dtype=np.float32)
            if data.ndim == 2:
                mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0].copy()
            else:
                mono = data.copy()
            mono = mono.astype(np.float32, copy=False)
            self.input_rms = float(np.sqrt(np.mean(mono ** 2)))
            self.input_peak = float(np.max(np.abs(mono))) if mono.size else 0.0
            try:
                self._in_queue.put_nowait(mono)
            except queue.Full:
                try:
                    self._in_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._in_queue.put_nowait(mono)
                except queue.Full:
                    pass
        except Exception as exc:
            self.last_error = f"input callback: {exc}"

    def _output_callback(self, outdata, frames, time_info, status) -> None:
        try:
            self._out_cb_count += 1
            block = self._out_queue.get_nowait()
        except queue.Empty:
            outdata.fill(0)
            return

        n = min(frames, len(block))
        if outdata.ndim == 2:
            outdata[:n, 0] = block[:n]
            for ch in range(1, outdata.shape[1]):
                outdata[:n, ch] = block[:n]
            outdata[n:, :] = 0.0
        else:
            outdata[:n] = block[:n]
            outdata[n:] = 0.0
        self.output_peak = float(np.max(np.abs(outdata))) if n else 0.0
        if self.output_peak > 1e-6:
            self._out_nonzero += 1

    # -- worker ---------------------------------------------------------
    def _push_output(self, block: np.ndarray, max_depth: int) -> None:
        if self._render_sink is not None:
            self._render_sink.append(block)
            return
        # Gravacao pos-limiter: apenas enfileira; nunca escreve no callback.
        recorder = self.session_recorder
        if recorder is not None:
            recorder.push(block)
        if self._out_queue.qsize() >= max_depth:
            return
        try:
            self._out_queue.put_nowait(block)
        except queue.Full:
            pass

    def _worker_loop(self) -> None:
        in_sr = int(self.input_samplerate or SAMPLE_RATE)
        out_sr = int(self.output_samplerate or SAMPLE_RATE)
        pending = np.zeros(0, dtype=np.float32)
        g = math.gcd(in_sr, out_sr)
        up, down = out_sr // g, in_sr // g

        while not self._stop_event.is_set():
            try:
                chunk = self._in_queue.get(timeout=0.15)
            except queue.Empty:
                self._push_output(np.zeros(OUT_BLOCK, dtype=np.float32), max_depth=3)
                continue
            try:
                if in_sr != out_sr:
                    chunk = resample_poly(chunk, up, down).astype(np.float32)
                pending = np.concatenate([pending, chunk])
                while pending.size >= OUT_BLOCK and not self._stop_event.is_set():
                    block = pending[:OUT_BLOCK]
                    pending = pending[OUT_BLOCK:]
                    self._render_block(block)
            except Exception:
                self.last_error = "worker: " + traceback.format_exc(limit=1).strip()
                self._push_output(np.zeros(OUT_BLOCK, dtype=np.float32), max_depth=3)

    def _render_block(self, block: np.ndarray) -> None:
        n = len(block)
        sr = int(self.output_samplerate or SAMPLE_RATE)
        self._render_count += 1

        self._recent = np.roll(self._recent, -n)
        self._recent[-n:] = block

        rms = float(np.sqrt(np.mean(block ** 2)))
        above = (not self._calibrating) and rms >= self.rms_threshold
        if above:
            self._gate_attack += 1
        else:
            self._gate_attack = 0
        if self._gate_attack >= GATE_ATTACK_BLOCKS:
            self._gate_hold = GATE_HOLD_BLOCKS
        if self._gate_hold > 0:
            gate_open = True
            self._gate_hold -= 1
        else:
            gate_open = False
        if gate_open:
            self._gate_blocks += 1

        self._mix_smooth += (self.mix - self._mix_smooth) * 0.20
        mix_now = float(max(0.0, min(1.0, self._mix_smooth)))
        angle = mix_now * (math.pi / 2.0)
        dry_mix = math.sin(angle)
        grain_mix = math.cos(angle)
        out = np.zeros(n, dtype=np.float32)
        if gate_open:
            out += dry_mix * block

        still_active: List[Dict[str, Any]] = []
        for grain in self._active_grains:
            audio = grain["audio"]
            delay = grain["delay"]
            pos = grain["pos"]
            start = 0
            if delay > 0:
                if delay >= n:
                    grain["delay"] = delay - n
                    still_active.append(grain)
                    continue
                start = delay
                grain["delay"] = 0

            count = min(n - start, len(audio) - pos)
            if count > 0:
                out[start:start + count] += grain_mix * grain["gain"] * audio[pos:pos + count]
                grain["pos"] = pos + count
            if grain["pos"] < len(audio):
                still_active.append(grain)
        self._active_grains = still_active

        self._samples_to_event -= n
        if gate_open and self._samples_to_event <= 0:
            self._samples_to_event = EVENT_INTERVAL * sr
            t0 = time.perf_counter()
            try:
                self._trigger_event(rms)
            except Exception:
                self.last_error = "evento: " + traceback.format_exc(limit=1).strip()
            elapsed = (time.perf_counter() - t0) * 1000.0
            if elapsed > self._event_time_max:
                self._event_time_max = float(elapsed)
            if elapsed > 80.0:
                self._event_time_warn += 1
                if self._event_time_warn <= 3:
                    self.progress(f"Aviso: evento granular levou {elapsed:.0f} ms.")

        out = soft_limit(out)
        self._push_output(out, max_depth=12 if gate_open else 3)
        self._session_samples += n

    def _trigger_event(self, rms: float) -> None:
        if self.corpus is None or not self.corpus.grains:
            return
        if self.mix >= 0.999:
            return
        self._events += 1
        sr = int(self.output_samplerate or SAMPLE_RATE)
        analysis_sr = int(self.corpus.sr)

        window = self._recent[-int(FEATURE_WINDOW * sr):]
        if window.size < 512:
            return
        query = window
        if sr != analysis_sr:
            g = math.gcd(sr, analysis_sr)
            query = resample_poly(window, analysis_sr // g, sr // g).astype(np.float32)
            query = query[:int(FEATURE_WINDOW * analysis_sr)]

        feats, query_desc = analyze_grain(query, analysis_sr, with_f0=True)
        self._ensure_selection_cache()
        f0 = query_desc.get("f0")
        voice_f0 = self._track_voice_f0(query_desc)
        self._last_query = {
            "f0": f0,
            "f0_smooth": voice_f0,
            "voicing": query_desc.get("voicing", 0.0),
            "noisy": query_desc.get("noisy", 0.0),
            "centroid": query_desc.get("centroid", 0.0),
            "rms": rms,
        }

        density = int(
            max(MIN_RICOCHET_DENSITY, min(MAX_RICOCHET_DENSITY, self.ricochet_density))
        )
        pitch_mode = self.pitch_mode
        if pitch_mode == PITCH_MODE_MICROTONAL:
            base_cents = self._microtonal_shift_cents(voice_f0, rms)
        elif pitch_mode == PITCH_MODE_INTERVAL:
            base_cents = self._interval_shift_cents()
        else:
            base_cents = 0.0
        absolute_shift = pitch_mode in (PITCH_MODE_MATCH, PITCH_MODE_INTERVAL)
        preserve_q = int(round(self.preserve_duration * 20.0))
        preserve = preserve_q / 20.0
        gain = self.grain_gain / math.sqrt(density)
        event_indices: List[int] = []
        event_log: List[Dict[str, Any]] = []

        for d in range(density):
            idx = self._select_grain(
                feats, query_desc, self.mutation_rate,
                self.phonetic_response, exclude=set(event_indices),
            )
            event_indices.append(idx)
            grain = self.corpus.grains[int(idx)]
            if pitch_mode == PITCH_MODE_MATCH:
                cents = self._grain_match_cents(grain, voice_f0, query_desc)
            elif pitch_mode == PITCH_MODE_INTERVAL:
                cents = int(round(quantize_absolute_cents(base_cents, self.tuning)))
            elif pitch_mode == PITCH_MODE_OFF:
                cents = 0
            else:
                cents = quantized_shift_cents(
                    base_cents + (d % 3 - 1) * 50, self.tuning
                )
            key = (
                int(idx),
                int(cents),
                int(preserve_q),
                self.tuning,
                bool(absolute_shift),
            )
            shifted = self._shift_cache.get(key)
            if shifted is None:
                base = grain.get("audio_playback")
                if base is None:
                    base = grain["audio"].astype(np.float32)
                shifted = apply_microtonal_pitch_shift(
                    base,
                    sr,
                    float(cents),
                    self.tuning,
                    preserve_duration=preserve,
                    absolute=absolute_shift,
                )
                if len(self._shift_cache) > 96:
                    self._shift_cache.clear()
                self._shift_cache[key] = shifted
            self._active_grains.append(
                {
                    "audio": shifted,
                    "pos": 0,
                    "delay": int(d * CASCADE_STEP * sr),
                    "gain": float(gain),
                }
            )
            event_log.append(
                {
                    "idx": int(idx),
                    "grain_id": int(idx),
                    "file": grain.get("file", ""),
                    "class": grain.get("class", ""),
                    "scale": float(grain.get("scale", 0.0)),
                    "start_s": float(grain.get("start_s", 0.0)),
                    "onset": bool(grain.get("onset", False)),
                    "duration_s": float(grain.get("duration", 0.0)),
                    "noisy": float(grain.get("noisy", 0.0)),
                    "query_noisy": float(query_desc.get("noisy", 0.0)),
                    "f0": grain.get("f0"),
                    "query_f0": query_desc.get("f0"),
                    "f0_smooth": voice_f0,
                    "pitch_mode": pitch_mode,
                    "preserve": preserve,
                    "cents": int(cents),
                }
            )

        self._last_selection = event_log
        self._selection_history.extend(event_log)
        if len(self._selection_history) > 1000:
            self._selection_history = self._selection_history[-1000:]
        self._recent_grains.extend(event_indices)
        if len(self._recent_grains) > RECENT_GRAIN_AVOID:
            self._recent_grains = self._recent_grains[-RECENT_GRAIN_AVOID:]

        recorder = self.session_recorder
        if recorder is not None:
            try:
                recorder.log_event(
                    {
                        "type": "event",
                        "event_index": int(self._events),
                        "session_time_s": round(
                            self._session_samples / float(sr), 4
                        ),
                        "sample_position": int(self._session_samples),
                        "output_sr": int(sr),
                        "input_rms": float(rms),
                        "query": {
                            "f0": query_desc.get("f0"),
                            "f0_smooth": voice_f0,
                            "voicing": float(query_desc.get("voicing", 0.0)),
                            "noisy": float(query_desc.get("noisy", 0.0)),
                            "centroid": float(query_desc.get("centroid", 0.0)),
                        },
                        "parameters": {
                            "density": int(density),
                            "contrast": float(self.mutation_rate),
                            "phonetic": float(self.phonetic_response),
                            "mix": float(self.mix),
                            "tuning": self.tuning,
                            "pitch_mode": pitch_mode,
                            "pitch_interval": self.pitch_interval,
                            "preserve": float(preserve),
                            "active_grains": int(len(self.corpus.grains)),
                            "library_grains": int(self._library_total),
                            "sources": list(self._active_sources),
                        },
                        "grains": event_log,
                    }
                )
            except (TypeError, ValueError) as exc:
                self.last_error = "log de sessao: %s" % exc

    def _track_voice_f0(self, query_desc: Dict[str, Any]) -> Optional[float]:
        """Mediana de 3 eventos vozeados + histerese de 20 cents."""
        raw = query_desc.get("f0")
        confidence = float(query_desc.get("voicing", 0.0))
        if (
            raw is None
            or float(raw) <= 40.0
            or confidence <= PITCH_F0_CONFIDENCE
        ):
            return None
        value = float(raw)
        self._f0_history.append(value)
        if len(self._f0_history) > PITCH_F0_HISTORY:
            self._f0_history = self._f0_history[-PITCH_F0_HISTORY:]
        median = float(np.median(self._f0_history))
        if self._last_voice_f0 is not None and self._last_voice_f0 > 0:
            delta = abs(1200.0 * math.log2(median / self._last_voice_f0))
            if delta < PITCH_HYSTERESIS_CENTS:
                median = self._last_voice_f0
        self._last_voice_f0 = median
        return median

    def _interval_shift_cents(self) -> float:
        cents = float(PITCH_INTERVALS[self.pitch_interval])
        if self.tuning == "ji":
            return float(JI_INTERVALS_CENTS.get(self.pitch_interval, cents))
        return cents

    def _grain_match_cents(
        self,
        grain: Dict[str, Any],
        voice_f0: Optional[float],
        query_desc: Dict[str, Any],
    ) -> int:
        if voice_f0 is None or voice_f0 <= 40.0:
            return 0
        if float(query_desc.get("voicing", 0.0)) <= PITCH_F0_CONFIDENCE:
            return 0
        grain_f0 = grain.get("f0")
        if grain_f0 is None:
            return 0
        try:
            grain_f0 = float(grain_f0)
        except (TypeError, ValueError):
            return 0
        if grain_f0 <= 40.0:
            return 0
        raw = 1200.0 * math.log2(float(voice_f0) / grain_f0)
        raw = float(
            np.clip(raw, -PITCH_MATCH_LIMIT_CENTS, PITCH_MATCH_LIMIT_CENTS)
        )
        return int(round(quantize_absolute_cents(raw, self.tuning)))

    def _microtonal_shift_cents(self, f0: Optional[float], rms: float) -> float:
        pitch_cents = 0.0
        if f0 is not None and f0 > 40:
            midi = 69.0 + 12.0 * math.log2(f0 / 440.0)
            pitch_cents = (midi - round(midi)) * 100.0
        threshold = max(self.rms_threshold, 1e-6)
        energy_cents = float(np.clip((rms / threshold) * 20.0 - 20.0, -50.0, 50.0))
        raw = 0.6 * pitch_cents + 0.4 * energy_cents
        return raw

    def _ensure_selection_cache(self) -> None:
        if self.corpus is None:
            return
        if getattr(self, "_sel_corpus_id", None) == id(self.corpus):
            return
        grains = self.corpus.grains
        self._grain_noisy = np.array(
            [float(g.get("noisy", 0.0)) for g in grains], dtype=np.float32
        )
        self._grain_f0 = np.array(
            [float(g.get("f0") or 0.0) for g in grains], dtype=np.float32
        )
        self._grain_voicing = np.array(
            [float(g.get("voicing", 0.0)) for g in grains], dtype=np.float32
        )
        order = np.argsort(self._grain_noisy, kind="stable")
        pct = np.empty(self._grain_noisy.size, dtype=np.float32)
        if self._grain_noisy.size > 1:
            pct[order] = np.linspace(0.0, 1.0, self._grain_noisy.size, dtype=np.float32)
        else:
            pct[:] = 0.5
        self._grain_noisy_sorted = self._grain_noisy[order]
        self._grain_noisy_pct = pct
        self._sel_corpus_id = id(self.corpus)

    def _select_grain(
        self,
        features: np.ndarray,
        query_desc: Dict[str, Any],
        contrast: float,
        phonetic_response: float,
        exclude: Optional[set] = None,
    ) -> int:
        assert self.corpus is not None
        corpus = self.corpus
        self._ensure_selection_cache()
        exclude = exclude or set()

        q = corpus.query_norm(features)
        sims = corpus.features_norm @ q
        lo = float(np.min(sims))
        hi = float(np.max(sims))
        sim_norm = (sims - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(sims)

        contrast = float(np.clip(contrast, 0.0, 1.0))
        phonetic = float(np.clip(phonetic_response, 0.0, 1.0))
        target_sim = 1.0 - contrast
        base_raw = -np.abs(sim_norm - target_sim)
        bmin = float(np.min(base_raw))
        bmax = float(np.max(base_raw))
        if bmax - bmin > 1e-9:
            base_norm = (base_raw - bmin) / (bmax - bmin)
        else:
            base_norm = np.ones_like(base_raw)

        if phonetic > 0.0:
            q_noisy = float(query_desc.get("noisy", 0.0))
            n_grains = int(self._grain_noisy.size)
            if n_grains > 1:
                rank = float(
                    np.searchsorted(self._grain_noisy_sorted, q_noisy, side="right")
                )
                q_pct = float(np.clip(rank / float(n_grains), 0.0, 1.0))
            else:
                q_pct = 0.5
            sigma_pct = 0.05 + 0.45 * (1.0 - phonetic)
            phon = np.exp(
                -0.5 * ((self._grain_noisy_pct - q_pct) / sigma_pct) ** 2
            )
            score = (1.0 - 0.5 * phonetic) * base_norm + phonetic * phon
        else:
            score = base_norm

        # Contraste dentro do mesmo registro: penaliza fora de +-12 semitons
        q_f0 = query_desc.get("f0")
        if q_f0 is not None and q_f0 > 40.0:
            valid = self._grain_f0 > 40.0
            semis = np.zeros_like(self._grain_f0)
            semis[valid] = np.abs(12.0 * np.log2(self._grain_f0[valid] / float(q_f0)))
            register_penalty = np.clip((semis - 12.0) / 12.0, 0.0, 1.0)
            score = score - register_penalty * 0.4 * (0.5 + 0.5 * contrast)

        # Nao trocar de registro (vozeado <-> nao vozeado) ao buscar contraste
        q_voicing = float(query_desc.get("voicing", 0.0))
        q_voiced = (
            (q_f0 is not None and q_f0 > 40.0) or q_voicing >= 0.35
        )
        grain_voiced = (self._grain_f0 > 40.0) | (self._grain_voicing >= 0.35)
        mismatch = (~grain_voiced) if q_voiced else grain_voiced
        score = score - mismatch.astype(np.float32) * (0.12 + 0.25 * contrast)

        # Diversidade dentro da cascata: os graos do mesmo evento devem diferir
        if exclude:
            chosen_idx = np.array(sorted(exclude), dtype=np.int64)
            chosen = corpus.features_norm[chosen_idx]
            sim_to_chosen = np.max(chosen @ corpus.features_norm.T, axis=0)
            score = score + (0.15 + 0.55 * contrast) * (1.0 - sim_to_chosen)

        for idx in self._recent_grains:
            if 0 <= idx < len(score):
                score[idx] -= 0.6
        for idx in exclude:
            if 0 <= idx < len(score):
                score[idx] = -np.inf

        if not np.any(np.isfinite(score)):
            for idx in exclude:
                if 0 <= idx < len(score):
                    score[idx] = 0.0
        return int(np.argmax(score))

    # -- render offline / selftest -------------------------------------
    def render_array(
        self,
        audio: np.ndarray,
        sr: int,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Processa um array como se fosse entrada ao vivo e devolve a saida."""
        self.output_samplerate = int(sr)
        self.input_samplerate = int(sr)
        if self.corpus is None:
            if not self.load_corpus():
                raise RuntimeError(self.last_error or "corpus indisponivel")
        self.corpus.prepare_playback(int(sr), progress=self.progress)
        self._reset_runtime_state()
        self._render_sink = []

        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        block_rms = []
        for start in range(0, y.size, OUT_BLOCK):
            block = y[start:start + OUT_BLOCK]
            if block.size < OUT_BLOCK:
                block = np.pad(block, (0, OUT_BLOCK - block.size))
            block_rms.append(float(np.sqrt(np.mean(block ** 2))))

        if threshold is None:
            noise = float(np.percentile(block_rms, 20)) if block_rms else 0.0
            threshold = max(noise * 3.5, MIN_RMS_THRESHOLD)
        self.rms_threshold = float(np.clip(threshold, MIN_RMS_THRESHOLD, MAX_RMS_THRESHOLD))

        for start in range(0, y.size, OUT_BLOCK):
            block = y[start:start + OUT_BLOCK]
            if block.size < OUT_BLOCK:
                block = np.pad(block, (0, OUT_BLOCK - block.size))
            self._render_block(block)

        out = (
            np.concatenate(self._render_sink)
            if self._render_sink
            else np.zeros(0, dtype=np.float32)
        )
        self._render_sink = None
        return out.astype(np.float32)

    # -- status ---------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        return {
            "running": self.is_running,
            "input_device": self.input_device,
            "output_device": self.output_device,
            "input_samplerate": self.input_samplerate,
            "output_samplerate": self.output_samplerate,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "corpus": self.corpus.summary() if self.corpus else None,
            "library_grains": self._library_total,
            "active_grains": (
                len(self.corpus.grains) if self.corpus is not None else 0
            ),
            "requested_active_grains": self.active_grains,
            "active_sources": list(self._active_sources),
            "recording": (
                {
                    "audio_path": self.session_recorder.audio_path,
                    "event_path": self.session_recorder.event_path,
                    "duration_seconds": self.session_recorder.duration_seconds,
                    "dropped_audio_blocks": self.session_recorder.dropped_audio_blocks,
                    "error": self.session_recorder.error,
                }
                if self.session_recorder is not None
                else None
            ),
            "last_recording": self.last_recording,
            "rms_threshold": self.rms_threshold,
            "density": self.ricochet_density,
            "mutation": self.mutation_rate,
            "phonetic_response": self.phonetic_response,
            "mix": self.mix,
            "tuning": self.tuning,
            "pitch_mode": self.pitch_mode,
            "pitch_interval": self.pitch_interval,
            "preserve_duration": self.preserve_duration,
            "last_query": self._last_query,
            "last_selection": self._last_selection,
            "last_error": self.last_error,
            "counters": {
                "in_callbacks": self._in_cb_count,
                "out_callbacks": self._out_cb_count,
                "out_nonzero": self._out_nonzero,
                "blocks_rendered": self._render_count,
                "gate_open_blocks": self._gate_blocks,
                "events": self._events,
                "event_slow": self._event_time_warn,
                "event_time_max_ms": self._event_time_max,
            },
        }


# ---------------------------------------------------------------------------
# Ferramentas de validacao
# ---------------------------------------------------------------------------


def audit_corpus(corpus: Corpus, sample_files: int = 5, progress=print) -> Dict[str, Any]:
    """Audita graos, classes, duracoes, niveis e features do corpus."""
    durations = np.array([len(g["audio"]) / corpus.sr for g in corpus.grains])
    peaks = np.array(
        [float(np.max(np.abs(g["audio"].astype(np.float32)))) for g in corpus.grains]
    )
    rms_vals = np.array(
        [float(np.sqrt(np.mean(g["audio"].astype(np.float32) ** 2))) for g in corpus.grains]
    )

    by_file: Dict[str, int] = {}
    by_class: Dict[str, int] = {}
    by_scale: Dict[float, int] = {}
    for g in corpus.grains:
        by_file[g["file"]] = by_file.get(g["file"], 0) + 1
        cls = g.get("class", "?")
        by_class[cls] = by_class.get(cls, 0) + 1
        scale = float(g.get("scale", g.get("duration", 0.0)))
        key = round(scale, 3)
        by_scale[key] = by_scale.get(key, 0) + 1

    progress("=== Auditoria do Corpus ===")
    progress(f"Graos: {len(corpus.grains)} | taxa de analise: {corpus.sr} Hz")
    progress(f"Cache: {corpus.cache_size_mb():.1f} MB")
    if corpus.selection_info:
        info = corpus.selection_info
        progress(
            "Candidatos: %d | orcamento: %s | limite/arquivo: %s | hop: %.0f ms"
            % (
                int(info.get("candidates", 0)),
                info.get("budget", "?"),
                info.get("per_file_cap", "?"),
                float(info.get("hop", 0.0)) * 1000.0,
            )
        )

    progress("Arquivos (duracao / nao-silencio / candidatos / selecionados):")
    for name, count in sorted(by_file.items()):
        stats = corpus.file_stats.get(name, {})
        progress(
            "  %s: %.1f s / %.0f%% / %s / %d"
            % (
                name,
                float(stats.get("duration", 0.0)),
                float(stats.get("non_silent", 0.0)) * 100.0,
                stats.get("candidates", "?"),
                count,
            )
        )

    progress(
        "Classes: %s | cotas alvo: %s"
        % (
            {k: by_class.get(k, 0) for k in ("transient", "tonal", "noisy")},
            {k: round(v * 100.0, 0) for k, v in CLASS_QUOTAS.items()},
        )
    )
    progress(f"Escalas: {dict(sorted(by_scale.items()))}")

    for cls in ("transient", "tonal", "noisy"):
        vals = [
            float(g.get("noisy", 0.0)) for g in corpus.grains
            if g.get("class") == cls
        ]
        if vals:
            progress(
                "  noisy[%s]: min %.2f | mediana %.2f | max %.2f"
                % (cls, min(vals), float(np.median(vals)), max(vals))
            )

    voiced = [g for g in corpus.grains if (g.get("f0") or 0.0) > 40.0]
    progress(
        "Vozeados: %d/%d (%.0f%%)"
        % (len(voiced), len(corpus.grains),
           100.0 * len(voiced) / max(1, len(corpus.grains)))
    )
    progress(
        "Duracao (s): min %.3f | mediana %.3f | max %.3f"
        % (durations.min(), float(np.median(durations)), durations.max())
    )
    progress(
        "Pico: min %.3f | mediana %.3f | max %.3f"
        % (peaks.min(), float(np.median(peaks)), peaks.max())
    )
    progress(
        "RMS: min %.4f | mediana %.4f | max %.4f"
        % (rms_vals.min(), float(np.median(rms_vals)), rms_vals.max())
    )
    zcr = np.array([g["zcr"] for g in corpus.grains])
    progress(
        "ZCR: min %.3f | mediana %.3f | max %.3f"
        % (zcr.min(), float(np.median(zcr)), zcr.max())
    )
    progress(
        "Features: %d dims | mean[0]=%.2f std[0]=%.2f"
        % (corpus.features.shape[1], float(corpus.mean[0]), float(corpus.std[0]))
    )

    problems: List[str] = []
    warnings: List[str] = []
    if durations.min() < GRAIN_MIN_DURATION - 0.005:
        problems.append("grao mais curto que 60 ms")
    if durations.max() > GRAIN_MAX_DURATION + 0.005:
        problems.append("grao mais longo que 350 ms")
    if peaks.min() < 0.5:
        problems.append("grao com pico muito baixo")
    if not np.all(np.isfinite(corpus.features)):
        problems.append("features com NaN/inf")
    if len(corpus.grains) < 20:
        problems.append("corpus muito pequeno")

    budget = int(corpus.selection_info.get("budget", MAX_TOTAL_GRAINS))
    if len(corpus.grains) < int(0.85 * budget):
        warnings.append(
            f"corpus com {len(corpus.grains)} graos (alvo {budget})"
        )
    for cls, quota in CLASS_QUOTAS.items():
        actual = by_class.get(cls, 0) / max(1, len(corpus.grains))
        if abs(actual - quota) > 0.08:
            warnings.append(
                f"classe {cls}: {actual * 100:.0f}% (alvo {quota * 100:.0f}%)"
            )

    if problems:
        progress("PROBLEMAS: " + "; ".join(problems))
    if warnings:
        progress("AVISOS: " + "; ".join(warnings))
    if not problems:
        progress("Auditoria OK: nenhum problema estrutural encontrado.")

    return {
        "ok": not problems,
        "problems": problems,
        "warnings": warnings,
        "grains": len(corpus.grains),
        "by_file": by_file,
        "by_class": by_class,
        "by_scale": {str(k): v for k, v in sorted(by_scale.items())},
        "duration_min": float(durations.min()),
        "duration_median": float(np.median(durations)),
        "duration_max": float(durations.max()),
        "cache_mb": corpus.cache_size_mb(),
    }


def dump_grains(
    corpus: Corpus, count: int = 12, out_dir: str = DEBUG_DIR, progress=print
) -> List[str]:
    """Exporta grãos representativos e um mosaico para escuta."""
    os.makedirs(out_dir, exist_ok=True)
    count = max(1, min(int(count), len(corpus.grains)))
    idxs = np.linspace(0, len(corpus.grains) - 1, count).astype(int)

    paths: List[str] = []
    montage_parts: List[np.ndarray] = []
    gap = np.zeros(int(0.12 * corpus.sr), dtype=np.float32)

    for n, idx in enumerate(idxs):
        audio = corpus.grains[int(idx)]["audio"].astype(np.float32)
        path = os.path.join(out_dir, f"grain_{n:02d}_zcr{corpus.grains[int(idx)]['zcr']:.2f}.wav")
        sf.write(path, audio, corpus.sr, subtype="PCM_16")
        paths.append(path)
        montage_parts.append(audio)
        montage_parts.append(gap)

    montage_path = os.path.join(out_dir, "montage.wav")
    sf.write(montage_path, np.concatenate(montage_parts), corpus.sr, subtype="PCM_16")
    paths.append(montage_path)

    progress(f"Exportados {count} graos + mosaico em {out_dir}")
    return paths


def run_selftest(progress=print) -> Dict[str, Any]:
    """Testa corpus + selecao fonetica + cascata + pitch + render offline."""
    progress("=== Selftest (offline, sem dispositivos) ===")
    corpus = Corpus.load_or_build(progress=progress)
    audit = audit_corpus(corpus, progress=progress)

    engine = VocoolderEngine(
        phonetic_response=DEFAULT_PHONETIC_RESPONSE,
        mutation_rate=DEFAULT_MUTATION_RATE,
        progress=progress,
        auto_calibrate=False,
    )
    engine.corpus = corpus
    engine.output_samplerate = corpus.sr
    engine.input_samplerate = corpus.sr

    sr = corpus.sr
    t = np.arange(int(2.2 * sr)) / sr
    # "voz" sintetica: silencio, vogal modulada, consoante ruidosa
    envelope = np.zeros_like(t)
    envelope[int(0.3 * sr):int(1.2 * sr)] = np.hanning(int(0.9 * sr)) * 0.25
    envelope[int(1.3 * sr):int(1.7 * sr)] = np.hanning(int(0.4 * sr)) * 0.20
    voice = envelope * np.sin(2 * np.pi * 220.0 * t)
    noise = np.random.default_rng(42).normal(0.0, 1.0, t.size).astype(np.float32)
    noise_env = np.zeros_like(t)
    noise_env[int(1.3 * sr):int(1.7 * sr)] = np.hanning(int(0.4 * sr)) * 0.12
    signal = (voice + noise * noise_env).astype(np.float32)

    out = engine.render_array(signal, sr)
    rms = float(np.sqrt(np.mean(out ** 2))) if out.size else 0.0
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    info = engine.describe()
    counters = info["counters"]
    history = list(engine._selection_history)

    progress(
        "Saida: %d amostras | RMS %.4f | pico %.3f"
        % (out.size, rms, peak)
    )
    progress(
        "Eventos: %d | blocos com gate aberto: %d | graos ativos ao final: %d"
        % (counters["events"], counters["gate_open_blocks"], len(engine._active_grains))
    )
    progress(
        "Shifts em cache: %d | erros: %s"
        % (len(engine._shift_cache), info["last_error"] or "nenhum")
    )

    # Cascata: cada evento deve escolher indices distintos
    density = int(engine.ricochet_density)
    groups = [history[i:i + density] for i in range(0, len(history), density)]
    unique_groups = sum(
        1 for group in groups
        if len({entry["idx"] for entry in group}) == len(group)
    )
    unique_ok = bool(groups) and unique_groups == len(groups)
    progress(
        "Cascata: %d/%d eventos com %d graos distintos."
        % (unique_groups, len(groups), density)
    )

    # Teste fonetico: ruido deve puxar graos ruidosos; tom, graos tonais
    def render_probe(probe: np.ndarray, phonetic: float) -> float:
        probe_engine = VocoolderEngine(
            ricochet_density=1,
            mutation_rate=0.0,
            phonetic_response=phonetic,
            progress=lambda *_args, **_kwargs: None,
            auto_calibrate=False,
        )
        probe_engine.corpus = corpus
        probe_engine.render_array(probe, sr)
        # O primeiro evento coincide com o onset (janela de 200 ms ainda
        # mistura silencio); avaliamos o regime permanente.
        values = [
            float(e["noisy"]) for e in probe_engine._selection_history[1:]
        ]
        return float(np.mean(values)) if values else 0.0

    probe_len = int(1.0 * sr)
    probe_env = np.zeros(probe_len, dtype=np.float32)
    probe_env[int(0.2 * sr):] = np.hanning(probe_len - int(0.2 * sr)) * 0.15
    rng = np.random.default_rng(7)
    noise_probe = (rng.normal(0, 1, probe_len).astype(np.float32) * probe_env)
    tone_probe = (
        np.sin(2 * np.pi * 180.0 * np.arange(probe_len) / sr).astype(np.float32)
        * probe_env
    )

    noisy_off = render_probe(noise_probe, 0.0)
    noisy_on = render_probe(noise_probe, 1.0)
    tonal_off = render_probe(tone_probe, 0.0)
    tonal_on = render_probe(tone_probe, 1.0)
    phonetic_ok = (
        (noisy_on >= noisy_off - 0.005)
        and (noisy_on >= 0.60)
        and (tonal_on < tonal_off - 0.04)
    )
    progress(
        "Fonetica (noisy medio): ruido 0%%=%.2f 100%%=%.2f | "
        "tom 0%%=%.2f 100%%=%.2f"
        % (noisy_off, noisy_on, tonal_off, tonal_on)
    )

    # Mix voz/graos: 100% = so voz (sem eventos); 0% = so graos
    solo_voice_engine = VocoolderEngine(
        ricochet_density=1,
        mutation_rate=0.0,
        mix=1.0,
        progress=lambda *_args, **_kwargs: None,
        auto_calibrate=False,
    )
    solo_voice_engine.corpus = corpus
    solo_voice_out = solo_voice_engine.render_array(signal, sr)
    solo_voice_rms = (
        float(np.sqrt(np.mean(solo_voice_out ** 2))) if solo_voice_out.size else 0.0
    )
    solo_voice_info = solo_voice_engine.describe()
    voice_only_ok = (
        solo_voice_info["counters"]["events"] == 0
        and len(solo_voice_engine._active_grains) == 0
        and solo_voice_rms > 1e-3
    )

    solo_grains_engine = VocoolderEngine(
        ricochet_density=1,
        mutation_rate=0.0,
        mix=0.0,
        progress=lambda *_args, **_kwargs: None,
        auto_calibrate=False,
    )
    solo_grains_engine.corpus = corpus
    solo_grains_out = solo_grains_engine.render_array(signal, sr)
    solo_grains_rms = (
        float(np.sqrt(np.mean(solo_grains_out ** 2))) if solo_grains_out.size else 0.0
    )
    solo_grains_info = solo_grains_engine.describe()
    grains_only_ok = (
        solo_grains_info["counters"]["events"] > 0
        and solo_grains_rms > 1e-4
    )
    mix_ok = voice_only_ok and grains_only_ok
    progress(
        "Mix: so voz RMS %.4f/eventos %d | so graos RMS %.4f/eventos %d"
        % (
            solo_voice_rms,
            solo_voice_info["counters"]["events"],
            solo_grains_rms,
            solo_grains_info["counters"]["events"],
        )
    )

    # Rastreamento de f0: mediana de 3 eventos + histerese de 20 cents
    track_engine = VocoolderEngine(
        progress=lambda *_args, **_kwargs: None,
        auto_calibrate=False,
    )
    tracked = [
        track_engine._track_voice_f0({"f0": f0, "voicing": 0.9})
        for f0 in (220.0, 221.0, 220.5)
    ]
    unvoiced_ok = (
        track_engine._track_voice_f0({"f0": 330.0, "voicing": 0.2}) is None
    )
    match_up = track_engine._grain_match_cents(
        {"f0": 50.0}, 1000.0, {"voicing": 0.9}
    )
    match_down = track_engine._grain_match_cents(
        {"f0": 1000.0}, 50.0, {"voicing": 0.9}
    )
    match_exact = track_engine._grain_match_cents(
        {"f0": 220.0}, 220.0, {"voicing": 0.9}
    )
    match_octave = track_engine._grain_match_cents(
        {"f0": 110.0}, 220.0, {"voicing": 0.9}
    )
    match_limit_ok = (
        match_up == 1200
        and match_down == -1200
        and match_exact == 0
        and match_octave == 1200
    )
    tracking_ok = (
        tracked[-1] is not None
        and abs(float(tracked[-1]) - 220.0) < 1e-6
        and unvoiced_ok
        and match_limit_ok
    )

    interval_table_ok = (
        "major_seventh" not in PITCH_INTERVALS
        and all(
            abs(float(PITCH_INTERVALS[name]) - expected) < 1e-9
            for name, expected in (
                ("unison", 0.0),
                ("fourth", 500.0),
                ("tritone", 600.0),
                ("fifth", 700.0),
                ("minor_seventh", 1000.0),
                ("octave", 1200.0),
            )
        )
    )
    for name in PITCH_INTERVALS:
        track_engine.pitch_interval = name
        track_engine.tuning = "24edo"
        interval_table_ok = interval_table_ok and abs(
            track_engine._interval_shift_cents() - PITCH_INTERVALS[name]
        ) < 1e-9
        track_engine.tuning = "ji"
        interval_table_ok = interval_table_ok and abs(
            track_engine._interval_shift_cents() - JI_INTERVALS_CENTS[name]
        ) < 1e-9
    track_engine.tuning = "24edo"

    def render_pitch_mode(
        mode: str, interval: str = "fifth", tuning: str = "24edo"
    ) -> List[int]:
        pitch_engine = VocoolderEngine(
            ricochet_density=1,
            mutation_rate=0.0,
            phonetic_response=0.6,
            pitch_mode=mode,
            pitch_interval=interval,
            tuning=tuning,
            progress=lambda *_args, **_kwargs: None,
            auto_calibrate=False,
        )
        pitch_engine.corpus = corpus
        pitch_engine.render_array(tone_probe, sr)
        return [int(e["cents"]) for e in pitch_engine._selection_history]

    off_cents = render_pitch_mode(PITCH_MODE_OFF)
    fifth_cents = render_pitch_mode(PITCH_MODE_INTERVAL, "fifth")
    fifth_ji_cents = render_pitch_mode(PITCH_MODE_INTERVAL, "fifth", "ji")
    match_cents = render_pitch_mode(PITCH_MODE_MATCH)
    off_ok = bool(off_cents) and all(c == 0 for c in off_cents)
    fifth_ok = bool(fifth_cents) and all(c == 700 for c in fifth_cents)
    fifth_ji_ok = bool(fifth_ji_cents) and all(c == 702 for c in fifth_ji_cents)
    match_ok = (
        bool(match_cents)
        and all(abs(c) <= 1200 and c % 50 == 0 for c in match_cents)
        and any(c != 0 for c in match_cents)
    )
    pitch_ok = (
        tracking_ok
        and interval_table_ok
        and off_ok
        and fifth_ok
        and fifth_ji_ok
        and match_ok
    )
    progress(
        "Pitch: tracking=%s intervalos=%s off=%s quinta=%s quinta_ji=%s match=%s | %s"
        % (
            tracking_ok,
            interval_table_ok,
            off_ok,
            fifth_ok,
            fifth_ji_ok,
            match_ok,
            match_cents[:8],
        )
    )

    # Teste de pitch shift isolado + WSOLA
    base = max(corpus.grains, key=lambda g: len(g["audio"]))["audio"].astype(
        np.float32
    )
    lengths = {}
    for cents in (0, 50, -50):
        shifted = apply_microtonal_pitch_shift(base, sr, cents)
        lengths[cents] = len(shifted)

    t0 = time.perf_counter()
    wsola_full = apply_microtonal_pitch_shift(
        base, sr, 700.0, "24edo", preserve_duration=1.0, absolute=True
    )
    wsola_ms = (time.perf_counter() - t0) * 1000.0
    wsola_half = apply_microtonal_pitch_shift(
        base, sr, 700.0, "24edo", preserve_duration=0.5, absolute=True
    )
    varispeed = apply_microtonal_pitch_shift(
        base, sr, 700.0, "24edo", preserve_duration=0.0, absolute=True
    )
    preserve_ok = (
        abs(len(wsola_full) - len(base)) <= 1
        and len(varispeed) != len(base)
        and min(len(varispeed), len(base))
        <= len(wsola_half)
        <= max(len(varispeed), len(base))
    )
    wsola_time_ok = wsola_ms < 80.0
    progress(
        "Pitch shift (amostras): 0=%d | +50=%d | -50=%d"
        % (lengths[0], lengths[50], lengths[-50])
    )
    progress(
        "WSOLA: base=%d varispeed=%d 50%%=%d 100%%=%d | %.1f ms"
        % (
            len(base),
            len(varispeed),
            len(wsola_half),
            len(wsola_full),
            wsola_ms,
        )
    )

    ok = (
        audit["ok"]
        and rms > 1e-4
        and peak > 0.01
        and counters["events"] > 0
        and counters["gate_open_blocks"] > 0
        and unique_ok
        and phonetic_ok
        and mix_ok
        and pitch_ok
        and preserve_ok
        and wsola_time_ok
        and counters["event_slow"] == 0
        and lengths[50] != lengths[0]
        and lengths[-50] != lengths[0]
    )
    progress(
        "Watchdog: eventos acima de 80 ms = %d | max %.1f ms"
        % (counters["event_slow"], counters["event_time_max_ms"])
    )
    progress("SELFTEST: " + ("PASS" if ok else "FAIL"))
    return {
        "ok": ok,
        "rms": rms,
        "peak": peak,
        "events": counters["events"],
        "gate_blocks": counters["gate_open_blocks"],
        "event_slow": counters["event_slow"],
        "event_time_max_ms": counters["event_time_max_ms"],
        "cascade_unique": unique_ok,
        "phonetic": {
            "noise_off": noisy_off,
            "noise_on": noisy_on,
            "tone_off": tonal_off,
            "tone_on": tonal_on,
        },
        "mix": {
            "voice_only_ok": voice_only_ok,
            "voice_only_rms": solo_voice_rms,
            "voice_only_events": solo_voice_info["counters"]["events"],
            "grains_only_ok": grains_only_ok,
            "grains_only_rms": solo_grains_rms,
            "grains_only_events": solo_grains_info["counters"]["events"],
        },
        "pitch": {
            "tracking_ok": tracking_ok,
            "unvoiced_ok": unvoiced_ok,
            "match_limit_ok": match_limit_ok,
            "interval_table_ok": interval_table_ok,
            "off_ok": off_ok,
            "fifth_ok": fifth_ok,
            "fifth_ji_ok": fifth_ji_ok,
            "match_ok": match_ok,
            "match_cents": match_cents,
        },
        "wsola": {
            "preserve_ok": preserve_ok,
            "time_ok": wsola_time_ok,
            "time_ms": wsola_ms,
            "base_samples": len(base),
            "varispeed_samples": len(varispeed),
            "half_samples": len(wsola_half),
            "full_samples": len(wsola_full),
        },
        "audit": audit,
    }


def render_file(
    input_path: str,
    output_path: str,
    density: int = DEFAULT_RICOCHET_DENSITY,
    mutation: float = DEFAULT_MUTATION_RATE,
    phonetic_response: float = DEFAULT_PHONETIC_RESPONSE,
    mix: float = DEFAULT_MIX,
    tuning: str = "24edo",
    pitch_mode: str = DEFAULT_PITCH_MODE,
    pitch_interval: str = DEFAULT_PITCH_INTERVAL,
    preserve_duration: float = DEFAULT_PRESERVE_DURATION,
    active_grains: int = DEFAULT_ACTIVE_GRAINS,
    sources: Optional[List[str]] = None,
    library_grains: int = MAX_TOTAL_GRAINS,
    per_file_cap: int = MAX_GRAINS_PER_FILE,
    progress=print,
) -> Dict[str, Any]:
    """Renderiza um arquivo de audio com o pipeline granular/microtonal."""
    progress(f"Renderizando {input_path} -> {output_path}")
    corpus = Corpus.load_or_build(
        progress=progress,
        budget=int(library_grains),
        per_file_cap=int(per_file_cap),
    )
    engine = VocoolderEngine(
        ricochet_density=density,
        mutation_rate=mutation,
        phonetic_response=phonetic_response,
        mix=mix,
        tuning=tuning,
        pitch_mode=pitch_mode,
        pitch_interval=pitch_interval,
        preserve_duration=preserve_duration,
        active_grains=active_grains,
        sources=sources,
        library_grains=library_grains,
        per_file_cap=per_file_cap,
        progress=progress,
        auto_calibrate=False,
    )
    engine._library_total = len(corpus.grains)
    engine.corpus = engine._build_active_view(corpus)
    sr = corpus.sr
    y = load_audio(input_path, sr)
    progress(f"Entrada: {y.size / sr:.2f} s @ {sr} Hz")
    out = engine.render_array(y, sr)
    sf.write(output_path, out, sr, subtype="PCM_16")
    rms = float(np.sqrt(np.mean(out ** 2))) if out.size else 0.0
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    info = engine.describe()
    selected = info.get("last_selection") or []
    progress(
        "Saida: %.2f s | RMS %.4f | pico %.3f | eventos %d"
        % (out.size / sr, rms, peak, info["counters"]["events"])
    )
    if selected:
        progress(
            "Ultimo evento: " + ", ".join(
                "%s[%s] noisy=%.2f cents=%+d"
                % (e["file"], e["class"], e["noisy"], e["cents"])
                for e in selected
            )
        )
    progress(f"Arquivo salvo: {output_path}")
    return {
        "ok": out.size > 0 and rms > 1e-4,
        "output": output_path,
        "rms": rms,
        "peak": peak,
        "events": info["counters"]["events"],
        "event_slow": info["counters"]["event_slow"],
        "event_time_max_ms": info["counters"]["event_time_max_ms"],
        "pitch_mode": info["pitch_mode"],
        "pitch_interval": info["pitch_interval"],
        "preserve_duration": info["preserve_duration"],
        "tuning": info["tuning"],
        "mix": info["mix"],
    }


# ---------------------------------------------------------------------------
# API de conveniencia
# ---------------------------------------------------------------------------


def start_engine(
    ricochet_density: int = DEFAULT_RICOCHET_DENSITY,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    rms_threshold: float = DEFAULT_RMS_THRESHOLD,
    sample_rate: int = SAMPLE_RATE,
    buffer_duration: float = 1.25,
    input_device: Optional[int] = None,
    output_device: Optional[int] = None,
    tuning: str = "24edo",
    phonetic_response: float = DEFAULT_PHONETIC_RESPONSE,
    mix: float = DEFAULT_MIX,
    pitch_mode: str = DEFAULT_PITCH_MODE,
    pitch_interval: str = DEFAULT_PITCH_INTERVAL,
    preserve_duration: float = DEFAULT_PRESERVE_DURATION,
    auto_calibrate: bool = True,
    active_grains: int = DEFAULT_ACTIVE_GRAINS,
    sources: Optional[List[str]] = None,
    library_grains: int = MAX_TOTAL_GRAINS,
    per_file_cap: int = MAX_GRAINS_PER_FILE,
    progress=print,
) -> VocoolderEngine:
    engine = VocoolderEngine(
        ricochet_density=ricochet_density,
        mutation_rate=mutation_rate,
        rms_threshold=rms_threshold,
        sample_rate=sample_rate,
        buffer_duration=buffer_duration,
        input_device=input_device,
        output_device=output_device,
        tuning=tuning,
        phonetic_response=phonetic_response,
        mix=mix,
        pitch_mode=pitch_mode,
        pitch_interval=pitch_interval,
        preserve_duration=preserve_duration,
        auto_calibrate=auto_calibrate,
        active_grains=active_grains,
        sources=sources,
        library_grains=library_grains,
        per_file_cap=per_file_cap,
        progress=progress,
    )
    if not engine.start():
        raise RuntimeError(engine.last_error or "Falha ao iniciar o engine.")
    return engine


def stop_engine(engine: Optional[VocoolderEngine]) -> None:
    if engine is not None:
        engine.stop()

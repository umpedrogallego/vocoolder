"""
Vocoolder - Audio Device Diagnostics
=====================================

Utilities to inspect audio devices, record a short microphone sample
and play a test tone. Por padrao respeita os dispositivos definidos no
Windows (inclusive saidas HDMI/monitor); a heuristica de recomendacao e
apenas informativa.

No GUI dependencies. Mensagens em ASCII para compatibilidade com o
console do Windows (cp1252).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db(value: float) -> float:
    """Convert linear amplitude to dBFS (clamped)."""
    return 20.0 * np.log10(max(float(value), 1e-9))


def list_devices() -> List[Dict[str, Any]]:
    """Return a list of audio devices with useful metadata."""
    devices: List[Dict[str, Any]] = []
    hostapis = sd.query_hostapis()
    for idx, dev in enumerate(sd.query_devices()):
        api = hostapis[dev["hostapi"]]["name"] if dev["hostapi"] < len(hostapis) else "?"
        devices.append(
            {
                "index": idx,
                "name": dev["name"],
                "hostapi": api,
                "hostapi_index": dev["hostapi"],
                "max_input_channels": int(dev["max_input_channels"]),
                "max_output_channels": int(dev["max_output_channels"]),
                "default_samplerate": float(dev["default_samplerate"]),
            }
        )
    return devices


def _score_output(dev: Dict[str, Any]) -> float:
    """Heuristic score: higher = more likely to be audible on speakers."""
    if dev["max_output_channels"] <= 0:
        return -999.0
    name = dev["name"].lower()
    api = dev["hostapi"].lower()
    score = 0.0

    if "wasapi" in api:
        score += 40
    elif "directsound" in api:
        score += 20
    elif "mme" in api:
        score += 10

    if any(k in name for k in ("alto-falante", "alto falante", "speakers", "fone", "headphone")):
        score += 30
    if "realtek" in name:
        score += 20
    if any(k in name for k in ("hdmi", "nvidia", "lg hdr", "monitor", "projector")):
        score -= 80
    if "voicemeeter" in name:
        score -= 20
    if "hands-free" in name or "bluetooth" in name:
        score -= 10
    return score


def _score_input(dev: Dict[str, Any]) -> float:
    """Heuristic score: higher = more likely to be a usable microphone."""
    if dev["max_input_channels"] <= 0:
        return -999.0
    name = dev["name"].lower()
    api = dev["hostapi"].lower()
    score = 0.0

    if "wasapi" in api:
        score += 40
    elif "directsound" in api:
        score += 20
    elif "mme" in api:
        score += 10

    if "microfone" in name or "microphone" in name:
        score += 30
    if "realtek" in name:
        score += 20
    if "mixagem" in name or "stereo mix" in name or "loopback" in name:
        score -= 40
    if "voicemeeter" in name or "voicemod" in name:
        score -= 20
    return score


def choose_devices(prefer_same_hostapi: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Pick the best guess for input and output devices."""
    devices = list_devices()
    inputs = [d for d in devices if d["max_input_channels"] > 0]
    outputs = [d for d in devices if d["max_output_channels"] > 0]
    if not inputs or not outputs:
        raise RuntimeError("Nenhum dispositivo de entrada/saida de audio encontrado.")

    best_in = max(inputs, key=_score_input)
    best_out = max(outputs, key=_score_output)

    if prefer_same_hostapi:
        same_api = [
            d
            for d in outputs
            if d["hostapi_index"] == best_in["hostapi_index"] and _score_output(d) > 0
        ]
        if same_api:
            best_out = max(same_api, key=_score_output)

    return best_in, best_out


def _negotiate_input(device: Optional[int], duration: float = 1.5) -> Tuple[int, int, int]:
    """Find a working (samplerate, channels, device) for input."""
    if device is None:
        device = sd.default.device[0]
    info = sd.query_devices(device)
    max_ch = int(info["max_input_channels"])
    ch_candidates = [c for c in (1, 2) if c <= max_ch] or [max_ch]
    base_sr = int(round(float(info["default_samplerate"])))
    sr_candidates: List[int] = []
    for sr in (base_sr, 48000, 44100, 32000):
        if sr not in sr_candidates:
            sr_candidates.append(sr)

    last_err: Optional[Exception] = None
    for sr in sr_candidates:
        for ch in ch_candidates:
            try:
                sd.check_input_settings(
                    device=device, samplerate=sr, channels=ch, dtype="float32"
                )
                return sr, ch, int(device)
            except Exception as exc:  # pragma: no cover - depends on hardware
                last_err = exc
    raise RuntimeError(f"Entrada invalida no dispositivo {device}: {last_err}")


def _negotiate_output(device: Optional[int]) -> Tuple[int, int, int]:
    """Find a working (samplerate, channels, device) for output."""
    if device is None:
        device = sd.default.device[1]
    info = sd.query_devices(device)
    max_ch = int(info["max_output_channels"])
    ch_candidates = [c for c in (2, 1) if c <= max_ch] or [max_ch]
    base_sr = int(round(float(info["default_samplerate"])))
    sr_candidates: List[int] = []
    for sr in (base_sr, 48000, 44100, 32000):
        if sr not in sr_candidates:
            sr_candidates.append(sr)

    last_err: Optional[Exception] = None
    for sr in sr_candidates:
        for ch in ch_candidates:
            try:
                sd.check_output_settings(
                    device=device, samplerate=sr, channels=ch, dtype="float32"
                )
                return sr, ch, int(device)
            except Exception as exc:  # pragma: no cover - depends on hardware
                last_err = exc
    raise RuntimeError(f"Saida invalida no dispositivo {device}: {last_err}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def play_test_tone(
    device: Optional[int] = None,
    samplerate: Optional[int] = None,
    duration: float = 0.9,
    progress=None,
) -> Dict[str, Any]:
    """Play a short two-tone test signal on the chosen output device."""
    sr, channels, device = _negotiate_output(device)
    if samplerate is not None:
        sr = int(samplerate)

    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    # Two-note sequence (A4 then E5) with small fade to avoid clicks
    half = len(t) // 2
    signal = np.zeros(len(t), dtype=np.float32)
    signal[:half] = 0.35 * np.sin(2 * np.pi * 440.0 * t[:half])
    signal[half:] = 0.35 * np.sin(2 * np.pi * 659.25 * t[half:])
    fade = max(1, int(0.01 * sr))
    signal[:fade] *= np.linspace(0.0, 1.0, fade)
    signal[-fade:] *= np.linspace(1.0, 0.0, fade)
    if channels > 1:
        signal = np.repeat(signal[:, None], channels, axis=1)

    if progress:
        progress(f"Reproduzindo tom de teste (A4/E5) em: {sd.query_devices(device)['name']}")

    try:
        sd.play(signal, samplerate=sr, device=device, blocking=True)
    except Exception as exc:  # pragma: no cover - depends on hardware
        raise RuntimeError(f"Falha ao reproduzir audio: {exc}") from exc

    return {
        "ok": True,
        "device": device,
        "name": sd.query_devices(device)["name"],
        "samplerate": sr,
        "channels": channels,
        "duration": duration,
    }


def record_test(
    device: Optional[int] = None,
    duration: float = 1.5,
    progress=None,
) -> Dict[str, Any]:
    """Record a short sample from the chosen input device and measure levels."""
    sr, channels, device = _negotiate_input(device)

    if progress:
        progress(f"Gravando {duration:.1f}s do microfone: {sd.query_devices(device)['name']}")
        progress("FALE ALGO AGORA...")

    # Pre-roll para a pessoa reagir ao aviso antes da gravacao comecar
    time.sleep(1.0)
    if progress:
        progress("Gravando...")

    try:
        recording = sd.rec(
            int(sr * duration),
            samplerate=sr,
            channels=channels,
            device=device,
            dtype="float32",
        )
        sd.wait()
    except Exception as exc:  # pragma: no cover - depends on hardware
        raise RuntimeError(f"Falha ao gravar audio: {exc}") from exc

    audio = np.asarray(recording, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    clipping = float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 0.0

    if peak < 0.001:
        level = "SILENCIOSO"
        hint = "Nenhum sinal captado. Verifique o microfone selecionado e o volume no Windows."
    elif rms < 0.003:
        level = "MUITO BAIXO"
        hint = (
            "Sinal muito fraco. Fale mais perto do microfone e aumente o volume "
            "em Windows > Configuracoes > Sistema > Som > Entrada > Microfone."
        )
    elif clipping > 0.01:
        level = "SATURADO"
        hint = "Sinal saturado. Afaste-se do microfone ou reduza o ganho de entrada."
    else:
        level = "OK"
        hint = "Entrada de audio funcionando."

    return {
        # SILENCIOSO/SATURADO indicam problema; sinal fraco ainda conta
        # como entrada detectada (o usuario pode estar longe do microfone).
        "ok": level in ("OK", "MUITO BAIXO"),
        "detected": peak >= 0.001,
        "level": level,
        "hint": hint,
        "device": device,
        "name": sd.query_devices(device)["name"],
        "samplerate": sr,
        "channels": channels,
        "duration": duration,
        "rms": rms,
        "peak": peak,
        "rms_db": _db(rms),
        "peak_db": _db(peak),
        "clipping": clipping,
    }


def meter(
    device: Optional[int] = None,
    seconds: float = 10.0,
    progress=None,
) -> Dict[str, Any]:
    """Mostra um medidor de nivel do microfone em tempo real (ASCII)."""
    sr, channels, device = _negotiate_input(device)
    name = sd.query_devices(device)["name"]
    if progress:
        progress(f"Medidor de entrada: #{device} {name} ({sr} Hz, {channels} ch)")
        progress("Fale normalmente. Ctrl+C para sair antes do fim.")

    block = 1024
    width = 28
    t0 = time.time()
    last_print = 0.0
    max_rms = 0.0
    max_peak = 0.0

    with sd.InputStream(
        device=device,
        samplerate=sr,
        channels=channels,
        dtype="float32",
        blocksize=block,
    ) as stream:
        while time.time() - t0 < seconds:
            data, _ = stream.read(block)
            audio = np.asarray(data, dtype=np.float32)
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1) if audio.shape[1] > 1 else audio[:, 0]
            rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            max_rms = max(max_rms, rms)
            max_peak = max(max_peak, peak)

            now = time.time() - t0
            if now - last_print >= 0.15:
                last_print = now
                db = _db(rms)
                frac = max(0.0, min(1.0, (db + 60.0) / 60.0))
                filled = int(round(frac * width))
                bar = "#" * filled + "-" * (width - filled)
                print(
                    "\r%6.2fs  %7.1f dBFS  [%s]  pico %.1f"
                    % (now, db, bar, _db(peak)),
                    end="",
                    flush=True,
                )

    print()
    return {
        "ok": max_peak >= 0.01,
        "device": device,
        "name": name,
        "samplerate": sr,
        "channels": channels,
        "max_rms_db": _db(max_rms),
        "max_peak_db": _db(max_peak),
    }


def record_to_file(
    path: str,
    seconds: float = 5.0,
    device: Optional[int] = None,
    progress=None,
) -> Dict[str, Any]:
    """Grava ``seconds`` do microfone em WAV (PCM 16) para testes offline."""
    import soundfile as sf

    sr, channels, device = _negotiate_input(device)
    name = sd.query_devices(device)["name"]
    total_frames = int(sr * seconds)
    block = 1024

    if progress:
        progress(f"Gravando {seconds:.1f}s de #{device} {name} ({sr} Hz, {channels} ch)")
        progress("FALE AGORA...")

    time.sleep(0.8)
    if progress:
        progress("Gravando...")

    chunks: List[np.ndarray] = []
    recorded = 0
    next_report = time.time() + 0.5

    with sd.InputStream(
        device=device,
        samplerate=sr,
        channels=channels,
        dtype="float32",
        blocksize=block,
    ) as stream:
        while recorded < total_frames:
            data, _ = stream.read(min(block, total_frames - recorded))
            chunks.append(np.asarray(data, dtype=np.float32).copy())
            recorded += len(data)
            if progress and time.time() >= next_report:
                mono = chunks[-1].mean(axis=1) if chunks[-1].ndim == 2 else chunks[-1]
                rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
                progress(f"  {recorded / sr:.1f}s ... {_db(rms):.0f} dBFS")
                next_report = time.time() + 0.5

    audio = np.concatenate(chunks, axis=0)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")

    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0

    return {
        "ok": peak >= 0.001,
        "path": os.path.abspath(path),
        "duration": len(audio) / sr,
        "samplerate": sr,
        "channels": channels,
        "rms_db": _db(rms),
        "peak_db": _db(peak),
        "device": device,
        "name": name,
    }


def run_diagnostics(
    input_device: Optional[int] = None,
    output_device: Optional[int] = None,
    duration_in: float = 1.5,
    duration_out: float = 0.9,
    progress=None,
) -> Dict[str, Any]:
    """Run a complete input/output audio diagnostic.

    Returns a dict with a human-readable ``text`` report plus structured data.
    """
    lines: List[str] = []

    def log(msg: str) -> None:
        lines.append(msg)
        if progress:
            progress(msg)

    log("=== Diagnostico de Audio - Vocoolder ===")

    # Resolve devices: por padrao usa o que o Windows definiu (respeita HDMI).
    try:
        defaults = sd.default.device
        if input_device is None:
            input_device = defaults[0]
        if output_device is None:
            output_device = defaults[1]
        if input_device is None or output_device is None:
            raise RuntimeError("Windows nao informou dispositivos padrao de entrada/saida.")

        log(
            "Dispositivos padrao do Windows: entrada #%d (%s) | saida #%d (%s)"
            % (
                input_device,
                sd.query_devices(input_device)["name"],
                output_device,
                sd.query_devices(output_device)["name"],
            )
        )
    except Exception as exc:
        log(f"ERRO ao resolver dispositivos: {exc}")
        return {"ok": False, "text": "\n".join(lines), "input": None, "output": None}

    # Sugestao apenas informativa (nao altera a escolha do usuario)
    try:
        rec_in, rec_out = choose_devices()
        log(
            "Sugestao (nao aplicada): entrada #%d (%s) | saida #%d (%s)"
            % (
                rec_in["index"],
                rec_in["name"],
                rec_out["index"],
                rec_out["name"],
            )
        )
    except Exception:
        pass

    # Input test
    try:
        input_report = record_test(input_device, duration_in, progress=log)
        log(
            "Entrada: %s | RMS %.1f dBFS | pico %.1f dBFS | %s"
            % (input_report["level"], input_report["rms_db"],
               input_report["peak_db"], input_report["hint"])
        )
    except Exception as exc:
        input_report = {"ok": False, "hint": f"Falha na entrada: {exc}"}
        log(f"ERRO na entrada: {exc}")

    # Output test
    try:
        output_report = play_test_tone(output_device, progress=log)
        log(
            "Saida: tom reproduzido em #%d (%s, %d Hz, %d canais)"
            % (output_report["device"], output_report["name"],
               output_report["samplerate"], output_report["channels"])
        )
        log("Voce deve ter ouvido DOIS tons (A4 e E5). "
            "Se nao ouviu, confira o volume/dispositivo de saida do Windows.")
    except Exception as exc:
        output_report = {"ok": False, "hint": f"Falha na saida: {exc}"}
        log(f"ERRO na saida: {exc}")

    input_ok = bool(input_report.get("ok"))
    output_ok = bool(output_report.get("ok"))
    overall = input_ok and output_ok
    log("=== Resultado: %s ===" % ("TUDO OK" if overall else "ATENCAO NECESSARIA"))

    return {
        "ok": overall,
        "text": "\n".join(lines),
        "input": input_report,
        "output": output_report,
    }

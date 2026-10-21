"""
Vocoolder - Main Entry Point
============================

Uso:
    python main.py                          # GUI
    python main.py --diagnose               # testa microfone + saida (leve)
    python main.py --list-devices           # lista dispositivos
    python main.py --meter --seconds 15     # medidor de entrada
    python main.py --record 5               # grava voice_test.wav
    python main.py --audit-corpus           # audita extracao dos graos
    python main.py --dump-grains 12         # exporta graos para escuta
    python main.py --selftest               # testa pipeline offline
    python main.py --render voz.wav -o saida.wav
    python main.py --render voz.wav -o saida.wav --pitch-mode interval --interval fifth --preserve 1
    python main.py --render voz.wav -o saida.wav --active-grains 100 --sources "01 Goticulas.mp3"
    python main.py --force-reindex --library-grains 2400 --per-file-cap 600 --audit-corpus
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Dependencias minimas por modo (evita importar torch/librosa/sklearn).
DIAGNOSE_DEPS = ("numpy", "sounddevice")
RECORD_DEPS = ("numpy", "sounddevice", "soundfile")
AUDIO_DEPS = ("numpy", "scipy", "soundfile", "sounddevice")
GUI_DEPS = ("customtkinter", "sounddevice", "soundfile", "numpy", "scipy")


def _check_dependencies(modules) -> bool:
    missing = []
    for mod in modules:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print("Dependencias ausentes:")
        for name in missing:
            print(f"  - {name}")
        return False
    return True


# ---------------------------------------------------------------------------
# Modos leves (sem engine)
# ---------------------------------------------------------------------------


def _run_list_devices() -> int:
    if not _check_dependencies(DIAGNOSE_DEPS):
        return 1

    import sounddevice as sd

    from src.audio_diag import list_devices

    defaults = sd.default.device
    dev_in, dev_out = defaults[0], defaults[1]

    print("=" * 72)
    print("Dispositivos de audio")
    print("=" * 72)
    print(f"Padrao do Windows: entrada #{dev_in} | saida #{dev_out}")
    print("-" * 72)
    for dev in list_devices():
        caps = []
        if dev["max_input_channels"] > 0:
            caps.append(f"IN {dev['max_input_channels']}")
        if dev["max_output_channels"] > 0:
            caps.append(f"OUT {dev['max_output_channels']}")
        marks = []
        if dev["index"] == dev_in:
            marks.append("PADRAO-IN")
        if dev["index"] == dev_out:
            marks.append("PADRAO-OUT")
        print(
            "#%-3d %-28s %-18s %s %s"
            % (
                dev["index"],
                dev["name"][:28],
                dev["hostapi"].replace("Windows ", "")[:18],
                " ".join(caps),
                " ".join(marks),
            )
        )
    print("-" * 72)
    print("Use: python main.py --diagnose --input-device N --output-device N")
    return 0


def _run_meter(input_device=None, seconds: float = 10.0) -> int:
    print("=" * 60)
    print("Vocoolder - Medidor de Entrada (modo leve)")
    print("=" * 60)
    if not _check_dependencies(DIAGNOSE_DEPS):
        return 1

    from src.audio_diag import meter

    try:
        report = meter(input_device, seconds, progress=print)
    except KeyboardInterrupt:
        print("\nMedidor interrompido.")
        return 0
    except Exception as exc:
        print(f"ERRO no medidor: {exc}")
        return 2

    print(
        "Pico maximo: %.1f dBFS | RMS maximo: %.1f dBFS"
        % (report["max_peak_db"], report["max_rms_db"])
    )
    if report["max_peak_db"] < -40.0:
        print(
            "Sinal muito baixo. Aumente o volume do microfone em "
            "Windows > Configuracoes > Sistema > Som > Entrada."
        )
    return 0


def _run_diagnose(input_device=None, output_device=None) -> int:
    print("=" * 60)
    print("Vocoolder - Diagnostico de Audio (modo leve)")
    print("=" * 60)
    print("Sem engine, sem torch/librosa/sklearn. Apenas numpy + sounddevice.")
    print()

    if not _check_dependencies(DIAGNOSE_DEPS):
        print("\nInstale as dependencias acima antes de continuar.")
        return 1

    from src.audio_diag import run_diagnostics

    print("-" * 60)
    print("1) Gravacao do microfone (fale algo quando solicitado)")
    print("2) Reproducao de dois tons na saida escolhida")
    print("-" * 60)

    report = run_diagnostics(
        input_device=input_device,
        output_device=output_device,
        duration_in=2.0,
        progress=print,
    )

    print("-" * 60)
    if report.get("ok"):
        print("Diagnostico concluido: entrada e saida de audio OK.")
    else:
        print("Diagnostico concluido com avisos. Veja as mensagens acima.")
        inp = report.get("input") or {}
        if inp and inp.get("level") != "OK":
            print(f"  Entrada: {inp.get('level')} - {inp.get('hint')}")
    print("-" * 60)
    return 0 if report.get("ok") else 2


def _run_record(seconds: float, output_path: str, input_device=None) -> int:
    print("=" * 60)
    print("Vocoolder - Gravacao de voz para teste")
    print("=" * 60)
    if not _check_dependencies(RECORD_DEPS):
        return 1

    from src.audio_diag import record_to_file

    try:
        report = record_to_file(
            output_path, seconds=seconds, device=input_device, progress=print
        )
    except Exception as exc:
        print(f"ERRO ao gravar: {exc}")
        return 2

    print(
        "Gravado: %s | %.1f s | RMS %.1f dBFS | pico %.1f dBFS"
        % (
            report["path"],
            report["duration"],
            report["rms_db"],
            report["peak_db"],
        )
    )
    return 0


# ---------------------------------------------------------------------------
# Modos de audio/corpus (engine leve)
# ---------------------------------------------------------------------------


def _load_corpus(
    force: bool = False,
    library_grains: int = 1800,
    per_file_cap: int = 400,
):
    from src.engine import Corpus

    return Corpus.load_or_build(
        force=force,
        budget=int(library_grains),
        per_file_cap=int(per_file_cap),
    )


def _run_audit(
    force: bool = False,
    library_grains: int = 1800,
    per_file_cap: int = 400,
) -> int:
    print("=" * 60)
    print("Vocoolder - Auditoria do Corpus")
    print("=" * 60)
    if not _check_dependencies(AUDIO_DEPS):
        return 1

    from src.engine import audit_corpus

    try:
        corpus = _load_corpus(force, library_grains, per_file_cap)
        report = audit_corpus(corpus)
    except Exception as exc:
        print(f"ERRO: {exc}")
        return 2
    return 0 if report["ok"] else 2


def _run_dump_grains(
    count: int,
    force: bool = False,
    library_grains: int = 1800,
    per_file_cap: int = 400,
) -> int:
    print("=" * 60)
    print("Vocoolder - Exportar Graos")
    print("=" * 60)
    if not _check_dependencies(AUDIO_DEPS):
        return 1

    from src.engine import dump_grains

    try:
        corpus = _load_corpus(force, library_grains, per_file_cap)
        paths = dump_grains(corpus, count)
    except Exception as exc:
        print(f"ERRO: {exc}")
        return 2
    for path in paths:
        print(f"  {path}")
    return 0


def _run_selftest(
    force: bool = False,
    library_grains: int = 1800,
    per_file_cap: int = 400,
) -> int:
    print("=" * 60)
    print("Vocoolder - Selftest (offline, sem dispositivos)")
    print("=" * 60)
    if not _check_dependencies(AUDIO_DEPS):
        return 1

    from src.engine import run_selftest

    try:
        if force:
            _load_corpus(force=True, library_grains=library_grains,
                         per_file_cap=per_file_cap)
        report = run_selftest()
    except Exception as exc:
        print(f"ERRO: {exc}")
        return 2
    return 0 if report["ok"] else 2


def _run_render(
    input_path: str,
    output_path: str,
    density: int = 3,
    mutation: float = 0.2,
    phonetic_response: float = 0.6,
    mix: float = 0.3,
    tuning: str = "24edo",
    pitch_mode: str = "microtonal",
    pitch_interval: str = "fifth",
    preserve_duration: float = 0.0,
    force: bool = False,
    active_grains: int = 1800,
    sources: Optional[list] = None,
    library_grains: int = 1800,
    per_file_cap: int = 400,
) -> int:
    print("=" * 60)
    print("Vocoolder - Render offline")
    print("=" * 60)
    print(
        "Config: pitch=%s/%s tuning=%s preserve=%.2f mix=%.2f"
        % (pitch_mode, pitch_interval, tuning, preserve_duration, mix)
    )
    if not _check_dependencies(AUDIO_DEPS):
        return 1
    if not os.path.isfile(input_path):
        print(f"ERRO: arquivo nao encontrado: {input_path}")
        return 2

    from src.engine import render_file

    try:
        if force:
            _load_corpus(
                force=True,
                library_grains=library_grains,
                per_file_cap=per_file_cap,
            )
        report = render_file(
            input_path,
            output_path,
            density=density,
            mutation=mutation,
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
        )
    except Exception as exc:
        print(f"ERRO: {exc}")
        return 2
    if report.get("event_slow"):
        print("Aviso: %d evento(s) acima de 80 ms." % report["event_slow"])
    print("Evento mais lento: %.1f ms" % report.get("event_time_max_ms", 0.0))
    return 0 if report["ok"] else 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Vocoolder")
    parser.add_argument("--diagnose", action="store_true",
                        help="testa microfone e saida de audio (modo leve)")
    parser.add_argument("--list-devices", action="store_true",
                        help="lista os dispositivos de audio e sai")
    parser.add_argument("--input-device", type=int, default=None,
                        help="indice do dispositivo de entrada")
    parser.add_argument("--output-device", type=int, default=None,
                        help="indice do dispositivo de saida")
    parser.add_argument("--meter", action="store_true",
                        help="medidor de nivel do microfone em tempo real")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="duracao do --meter em segundos (padrao: 10)")
    parser.add_argument("--record", type=float, default=None, metavar="SEGUNDOS",
                        help="grava a voz para um WAV de teste")
    parser.add_argument("--record-output", default=os.path.join(PROJECT_ROOT, "voice_test.wav"),
                        help="arquivo de saida do --record")
    parser.add_argument("--audit-corpus", action="store_true",
                        help="audita a extracao dos graos")
    parser.add_argument("--dump-grains", type=int, default=None, metavar="N",
                        help="exporta N graos para corpus_debug/")
    parser.add_argument("--selftest", action="store_true",
                        help="testa corpus + pitch + render sem dispositivos")
    parser.add_argument("--render", default=None, metavar="ARQUIVO",
                        help="renderiza um arquivo de audio pelo pipeline")
    parser.add_argument("--output", "-o", default=None, metavar="SAIDA",
                        help="arquivo de saida do --render")
    parser.add_argument("--density", type=int, default=3,
                        help="densidade do ricochete no --render (1-6)")
    parser.add_argument("--mutation", "--contrast", dest="mutation", type=float,
                        default=0.2,
                        help="contraste timbrico no --render (0-1)")
    parser.add_argument("--phonetic", type=float, default=0.6,
                        help="resposta fonetica no --render (0-1)")
    parser.add_argument("--mix", type=float, default=0.3,
                        help="mix voz/graos no --render (0=so graos, 1=so voz)")
    parser.add_argument("--tuning", choices=("24edo", "ji"), default="24edo",
                        help="afinacao microtonal no --render")
    parser.add_argument("--pitch-mode",
                        choices=("off", "microtonal", "match", "interval"),
                        default="microtonal",
                        help="modo de pitch no --render")
    parser.add_argument("--interval",
                        choices=("unison", "fourth", "tritone", "fifth",
                                 "minor_seventh", "octave"),
                        default="fifth",
                        help="intervalo fixo quando --pitch-mode interval")
    parser.add_argument("--preserve", type=float, default=0.0,
                        help="preservar duracao no --render (0=varispeed, 1=WSOLA)")
    parser.add_argument("--active-grains", type=int, default=1800,
                        help="graos do conjunto ativo live (10-1800)")
    parser.add_argument("--sources", nargs="+", default=None,
                        help="nomes dos arquivos de amostra usados no --render")
    parser.add_argument("--library-grains", type=int, default=1800,
                        help="orcamento da biblioteca ao reindexar")
    parser.add_argument("--per-file-cap", type=int, default=400,
                        help="maximo de graos por arquivo ao reindexar")
    parser.add_argument("--force-reindex", action="store_true",
                        help="forca a reindexacao do corpus")
    args = parser.parse_args()

    # Modos leves primeiro
    if args.list_devices:
        return _run_list_devices()
    if args.meter:
        return _run_meter(args.input_device, args.seconds)
    if args.diagnose:
        return _run_diagnose(args.input_device, args.output_device)
    if args.record is not None:
        return _run_record(args.record, args.record_output, args.input_device)
    if args.audit_corpus:
        return _run_audit(
            args.force_reindex, args.library_grains, args.per_file_cap
        )
    if args.dump_grains is not None:
        return _run_dump_grains(
            args.dump_grains,
            args.force_reindex,
            args.library_grains,
            args.per_file_cap,
        )
    if args.selftest:
        return _run_selftest(
            args.force_reindex, args.library_grains, args.per_file_cap
        )
    if args.render:
        output_path = args.output
        if output_path is None:
            stem, _ = os.path.splitext(os.path.basename(args.render))
            output_path = os.path.join(PROJECT_ROOT, f"{stem}_vocoolder.wav")
        return _run_render(
            args.render,
            output_path,
            args.density,
            args.mutation,
            args.phonetic,
            args.mix,
            args.tuning,
            args.pitch_mode,
            args.interval,
            args.preserve,
            args.force_reindex,
            args.active_grains,
            args.sources,
            args.library_grains,
            args.per_file_cap,
        )

    # GUI
    print("=" * 60)
    print("Vocoolder - Vocal Performance Reactive System")
    print("=" * 60)

    if not _check_dependencies(GUI_DEPS):
        print("\nInstale as dependencias acima antes de continuar.")
        return 1

    from src.microtonal_tables import cents_to_semitones, quantize_to_microtone

    print("  Microtonal tables: OK")
    print(f"    cents_to_semitones(100) = {cents_to_semitones(100)}")
    print(f"    quantize_to_microtone(45, '24edo') = {quantize_to_microtone(45, '24edo')}")

    amostras = os.path.join(PROJECT_ROOT, "amostras")
    if os.path.isdir(amostras):
        samples = [
            f for f in os.listdir(amostras)
            if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))
        ]
        print(f"  Amostras: {len(samples)} arquivo(s) em {amostras}")
    else:
        print(f"  AVISO: pasta de amostras ausente: {amostras}")

    print("-" * 60)
    print("Abrindo interface grafica...")
    print("-" * 60)
    from src.gui import main as gui_main

    gui_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

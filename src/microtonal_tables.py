"""
Vocoolder - Microtonal Tables and Utilities
============================================

Pure computation modules for microtonal pitch handling, formant constants,
and vocal synthesis support. No real-time audio or GUI dependencies.

This module provides:
  - 24-EDO (quarter-tone) tuning tables
  - Just Intonation ratios with harmonic series deviations
  - Cents-to-semitones conversion for torchaudio.PitchShift
  - Microtone quantization for microphone-derived pitch variations
  - Absolute JI intervals for fixed/matched pitch shifts
  - Vocal formant frequency zones
"""

from __future__ import annotations

import math
from typing import Final, Dict, List, Tuple

# ---------------------------------------------------------------------------
# 24-EDO (Quarter-tone) Constants
# ---------------------------------------------------------------------------

# Step size in cents: 1200 cents / 24 = 50 cents per step
TWENTY_FOUR_EDO_CENTS_PER_STEP: Final[float] = 50.0

# Step size in semitones: 12 / 24 = 0.5 semitones per step
TWENTY_FOUR_EDO_STEPS_PER_SEMITONE: Final[float] = 2.0

# Mapping: step index -> cents (relative to A440 / C0)
_TWENTY_FOUR_EDO_CENTS: Final[List[float]] = [
    i * TWENTY_FOUR_EDO_CENTS_PER_STEP for i in range(24)
]

# Mapping: step index -> semitones (relative to C0)
_TWENTY_FOUR_EDO_SEMITONES: Final[List[float]] = [
    i / TWENTY_FOUR_EDO_STEPS_PER_SEMITONE for i in range(24)
]

# Convenience: dict mapping step name to (cents, semitones)
TWENTY_FOUR_EDO_TABLE: Final[Dict[str, Tuple[float, float]]] = {
    f"{i}": (_TWENTY_FOUR_EDO_CENTS[i], _TWENTY_FOUR_EDO_SEMITONES[i])
    for i in range(24)
}

# ---------------------------------------------------------------------------
# Just Intonation & Harmonic Series Resonance
# ---------------------------------------------------------------------------

# JI intervals as (frequency_ratio, deviation_from_12tET_in_cents).
# A deviation is measured against the nearest 12-TET degree.
JUST_INTONATION_TABLE: Final[List[Tuple[float, float]]] = [
    (1.0, 0.0),             # Unison
    (16.0 / 15.0, 11.73),   # Minor second (111.73 vs 100)
    (9.0 / 8.0, 3.91),      # Major second (203.91 vs 200)
    (6.0 / 5.0, 15.64),     # Minor third (315.64 vs 300)
    (5.0 / 4.0, -13.69),    # Major third (386.31 vs 400)
    (4.0 / 3.0, -1.96),     # Perfect fourth (498.04 vs 500)
    (3.0 / 2.0, 1.96),      # Perfect fifth (701.96 vs 700)
    (7.0 / 4.0, -31.17),    # Harmonic seventh (968.83 vs 1000)
    (15.0 / 8.0, -11.73),   # Major seventh (1088.27 vs 1100)
    (2.0, 0.0),             # Octave
]

# Common JI deviations from 12-TET. Positive = sharper than 12-TET.
JI_DEVIATIONS_CENTS: Final[Dict[str, float]] = {
    "unison": 0.0,
    "minor_second_just": 11.73,
    "major_second_pythagorean": 3.91,
    "minor_third_just": 15.64,
    "major_third_just": -13.69,
    "fourth_perfect": -1.96,
    "fifth_perfect": 1.96,
    "harmonic_seventh": -31.17,
    "minor_sixth_just": 13.69,
    "major_sixth_just": -15.64,
    "major_seventh_just": -11.73,
}

# Absolute Just Intonation intervals in cents (not deviations from 12-TET).
# Used for fixed intervals and for matching the voice to a JI grid.
JI_INTERVALS_CENTS: Final[Dict[str, float]] = {
    "unison": 0.0,
    "minor_second": 111.73,    # 16/15
    "major_second": 203.91,    # 9/8
    "minor_third": 315.64,     # 6/5
    "major_third": 386.31,     # 5/4
    "fourth": 498.04,          # 4/3
    "tritone": 590.22,         # 45/32
    "fifth": 701.96,           # 3/2
    "minor_sixth": 813.69,     # 8/5
    "major_sixth": 884.36,     # 5/3
    "minor_seventh": 968.83,   # 7/4
    "major_seventh": 1088.27,  # 15/8
    "octave": 1200.0,          # 2/1
}

# Desvios microtonais JI usados pelo motor, em cents.
# A tabela inclui os harmônicos principais (5, 7, 11, 13, 17) e algumas
# aproximações auxiliares; a quantização final arredonda para passos de 10¢.
HARMONIC_SERIES_DEVIATIONS_CENTS: Final[Dict[int, float]] = {
    1: 0.0,
    2: 0.0,
    3: -2.0,
    4: 0.0,
    5: -13.7,     # 5:4, terça maior
    6: -3.0,
    7: -31.2,     # 7:4, sétima harmônica
    8: 0.0,
    9: -4.0,
    10: -14.0,
    11: -49.0,    # 11º harmônico
    12: -2.0,
    13: -41.1,    # 13º harmônico
    14: -31.0,
    15: -3.2,
    16: 0.0,
    17: +5.0,     # 17º harmônico
    18: -3.0,
    19: +8.0,
    20: -14.0,
}

# ---------------------------------------------------------------------------
# Micro-vibrato and Vocal Tension Micro-steps
# ---------------------------------------------------------------------------

# Micro-vibrato rate ranges (Hz) and extent (cents)
VIBRATO_EXTENT_CENTS: Final[float] = 5.0  # typical vibrato extent
VIBRATO_RATE_RANGE: Final[Tuple[float, float]] = (5.0, 8.0)  # Hz

# Vocal tension micro-steps in cents (10-25 cent range)
MICRO_TENSION_STEPS_CENTS: Final[List[float]] = [
    10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0
]

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def cents_to_semitones(cents: float) -> float:
    """
    Convert a pitch deviation in cents to semitones (float).

    This is designed to feed directly into torchaudio.transforms.PitchShift,
    which expects the shift amount in semitones (can be fractional).

    Parameters
    ----------
    cents : float
        Pitch deviation in cents (positive = sharp, negative = flat).

    Returns
    -------
    float
        Equivalent pitch shift in semitones (float).
    """
    return cents / 100.0


def quantize_to_microtone(
    pitch_shift_cents: float,
    mode: str = "24edo",
) -> float:
    """
    Quantize a continuous pitch variation (e.g., from microphone input) to the
    nearest microtone step in the specified tuning mode.

    Parameters
    ----------
    pitch_shift_cents : float
        Pitch deviation in cents from the continuous source (mic, sensor, etc.).
    mode : str, optional
        Tuning mode for quantization. Currently supports:
        - ``'24edo'`` : Quarter-tone system (50-cent steps)
        - Default: ``'24edo'``

    Returns
    -------
    float
        Quantized pitch shift in cents (nearest microtone step).
    """
    if mode == "24edo":
        step_size_cents = TWENTY_FOUR_EDO_CENTS_PER_STEP  # 50 cents
        # Round to nearest 50-cent step
        nearest_step = round(pitch_shift_cents / step_size_cents)
        quantized_cents = nearest_step * step_size_cents
        return quantized_cents
    else:
        # Fallback: just round to nearest cent
        return round(pitch_shift_cents)


def quantize_absolute_cents(
    pitch_shift_cents: float,
    mode: str = "24edo",
) -> float:
    """Quantiza um deslocamento absoluto em cents.

    Diferente de ``quantize_to_microtone`` (que arredonda desvios para
    degraus microtonais), aqui o valor pode ser um intervalo grande, como
    uma quinta ou uma oitava. Em JI, encaixa no intervalo justo mais proximo.
    """
    value = float(pitch_shift_cents)
    if mode == "24edo":
        return float(round(value / TWENTY_FOUR_EDO_CENTS_PER_STEP) * TWENTY_FOUR_EDO_CENTS_PER_STEP)
    sign = -1.0 if value < 0.0 else 1.0
    target = abs(value)
    nearest = min(JI_INTERVALS_CENTS.values(), key=lambda c: abs(c - target))
    return float(sign * nearest)


# ---------------------------------------------------------------------------
# Vocal Formant Constants (Frequency Zones)
# ---------------------------------------------------------------------------

# Fundamental frequency range (F0) for normal speech and singing
F0_MIN_HZ: Final[float] = 80.0   # Lower bound for male speech / chest voice
F0_MAX_HZ: Final[float] = 300.0  # Upper bound for female/falsetto

# First formants (F1 - resonância laringal / abertura vocal)
# F1 corresponds to vowel height (open/close)
F1_GRAVES_HZ: Final[float] = 250.0  # Low F1 = "a" / "o" vowels (dark/resonant)
F1_AGUDAS_HZ: Final[float] = 750.0  # High F1 = "i" vowels (bright/tensed)

# Second formants (F2 - posição da língua / morfologia vocal)
# F2 corresponds to vowel frontness/backness
F2_FRONT_HZ: Final[float] = 1200.0  # Front vowels ("i", "e")
F2_BACK_HZ: Final[float] = 800.0   # Back vowels ("u", "o")

# Third formants (F3 - mordedura tímbrica / cantora)
# F3 is crucial for singer's formant and phonetic discrimination
F3_SINGER_FORMANT_HZ: Final[float] = 2800.0  # Singer's formant cluster
F3_MORDEDURA_HZ: Final[float] = 3200.0      # Timbral bite / clarity

# Fricatives and sibilance zones
# High-ZCR (Zero Crossing Rate) regions for consonant classification
FRICATIVE_ZONE_MIN_HZ: Final[float] = 2000.0   # Start of fricative region
FRICATIVE_ZONE_MAX_HZ: Final[float] = 8000.0   # End of fricative region
SIBILANCE_THRESHOLD_HZ: Final[float] = 4000.0  # Typical sibilance peak (s, sh, f)
SIBILANCE_UPPER_HZ: Final[float] = 8000.0      # Extended sibilance range

# Vocal zone boundaries summary
VOCAL_ZONES: Final[Dict[str, Tuple[float, float]]] = {
    "F0": (F0_MIN_HZ, F0_MAX_HZ),
    "F1": (F1_GRAVES_HZ, F1_AGUDAS_HZ),
    "F2": (F2_BACK_HZ, F2_FRONT_HZ),
    "F3": (F3_MORDEDURA_HZ, F3_SINGER_FORMANT_HZ),
    "fricatives": (FRICATIVE_ZONE_MIN_HZ, FRICATIVE_ZONE_MAX_HZ),
    "sibilance": (SIBILANCE_THRESHOLD_HZ, SIBILANCE_UPPER_HZ),
}

# ---------------------------------------------------------------------------
# 24-EDO Note Reference (C0-relative, octave 0)
# ---------------------------------------------------------------------------

# All 24-EDO steps from C0, in cents and semitones
# Useful for constructing microtonal scales or mapping MIDI-like note numbers
TWENTY_FOUR_EDO_NOTES: Final[Dict[int, Tuple[float, float]]] = {
    octave * 24 + step: (
        step * TWENTY_FOUR_EDO_CENTS_PER_STEP + octave * 1200.0,
        step / TWENTY_FOUR_EDO_STEPS_PER_SEMITONE + octave * 12.0,
    )
    for octave in range(5)  # C0 through C4 (5 octaves, 24 steps each)
    for step in range(24)
}

# ---------------------------------------------------------------------------
# Module-level constants for quick access
# ---------------------------------------------------------------------------

# Quick lookup: just intonation ratio -> cents deviation from 12-TET
JI_DEVIATIONS: Final[Dict[str, float]] = JI_DEVIATIONS_CENTS

# Harmonic series deviations
HARMONICS: Final[Dict[int, float]] = HARMONIC_SERIES_DEVIATIONS_CENTS

# 24-EDO step size
EDO24_STEP_CENTS: Final[float] = TWENTY_FOUR_EDO_CENTS_PER_STEP

# Vocal formant constants
F0_MIN: Final[float] = F0_MIN_HZ
F0_MAX: Final[float] = F0_MAX_HZ
F1_GRAVES: Final[float] = F1_GRAVES_HZ
F1_AGUDAS: Final[float] = F1_AGUDAS_HZ
F2_FRONT: Final[float] = F2_FRONT_HZ
F2_BACK: Final[float] = F2_BACK_HZ
F3_MORDEDURA: Final[float] = F3_MORDEDURA_HZ
F3_SINGER: Final[float] = F3_SINGER_FORMANT_HZ
SIBILANCE: Final[float] = SIBILANCE_THRESHOLD_HZ

# Micro-tension steps
MICRO_TENSION: Final[List[float]] = MICRO_TENSION_STEPS_CENTS

# ---------------------------------------------------------------------------
# End of microtonal_tables.py
# ---------------------------------------------------------------------------
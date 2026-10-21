"""Persistência versionada de presets do Vocoolder.

Formato:
- JSON na pasta ``presets/``.
- Fonte de áudio sempre relativa a ``amostras/`` (apenas o nome do arquivo).
- Dispositivos identificados por nome/hostapi, nunca por índice.
- Seções opcionais explícitas em ``sections``.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

PRESET_VERSION = 1
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESET_DIR = os.path.join(PROJECT_ROOT, "presets")

SECTION_KEYS = ("library", "voice", "grains", "pitch", "output")
DEFAULT_SECTIONS: Dict[str, bool] = {key: True for key in SECTION_KEYS}

_NAME_RE = re.compile(r"[^A-Za-z0-9 _.-]+")


def _clean_name(name: str) -> str:
    cleaned = _NAME_RE.sub("_", str(name or "").strip())
    cleaned = cleaned.strip(" ._-")
    if not cleaned:
        raise ValueError("Nome do preset vazio.")
    return cleaned[:80]


def preset_path(name: str) -> str:
    return os.path.join(PRESET_DIR, _clean_name(name) + ".json")


def ensure_preset_dir() -> None:
    os.makedirs(PRESET_DIR, exist_ok=True)


def list_presets() -> List[str]:
    ensure_preset_dir()
    names: List[str] = []
    for entry in sorted(os.listdir(PRESET_DIR)):
        if not entry.lower().endswith(".json"):
            continue
        path = os.path.join(PRESET_DIR, entry)
        if not os.path.isfile(path):
            continue
        names.append(os.path.splitext(entry)[0])
    return names


def _normalize_sections(value: Any) -> Dict[str, bool]:
    if not isinstance(value, dict):
        return dict(DEFAULT_SECTIONS)
    return {
        key: bool(value.get(key, DEFAULT_SECTIONS[key]))
        for key in SECTION_KEYS
    }


def validate_payload(payload: Any) -> List[str]:
    """Retorna uma lista de erros; vazia significa payload válido."""
    errors: List[str] = []
    if not isinstance(payload, dict):
        return ["O preset precisa ser um objeto JSON."]
    version = payload.get("version")
    if not isinstance(version, int) or version < 1:
        errors.append("Campo 'version' ausente ou inválido.")
    elif version > PRESET_VERSION:
        errors.append(
            "Preset de versão %s é mais novo que o suportado (%s)."
            % (version, PRESET_VERSION)
        )
    sections = payload.get("sections")
    if sections is not None and not isinstance(sections, dict):
        errors.append("Campo 'sections' deve ser um objeto.")
    for key in SECTION_KEYS:
        value = payload.get(key)
        if value is not None and not isinstance(value, dict):
            errors.append("Seção '%s' deve ser um objeto." % key)
    library = payload.get("library")
    if isinstance(library, dict):
        sources = library.get("sources")
        if sources is not None and not isinstance(sources, list):
            errors.append("library.sources deve ser uma lista.")
        active = library.get("active_grains")
        if active is not None and not isinstance(active, int):
            errors.append("library.active_grains deve ser inteiro.")
    return errors


def save_preset(name: str, payload: Dict[str, Any]) -> str:
    """Grava o preset de forma atômica e devolve o caminho final."""
    ensure_preset_dir()
    clean = _clean_name(name)
    payload = dict(payload)
    payload["version"] = PRESET_VERSION
    payload["name"] = clean
    payload["saved_at"] = datetime.now(timezone.utc).isoformat()
    payload["sections"] = _normalize_sections(payload.get("sections"))
    errors = validate_payload(payload)
    if errors:
        raise ValueError("; ".join(errors))
    path = preset_path(clean)
    fd, tmp_path = tempfile.mkstemp(
        prefix=clean + ".", suffix=".tmp", dir=PRESET_DIR
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(
                payload, fh, ensure_ascii=False, indent=2, allow_nan=False
            )
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return path


def load_preset(name: str) -> Dict[str, Any]:
    path = preset_path(name)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    errors = validate_payload(payload)
    if errors:
        raise ValueError("; ".join(errors))
    payload["sections"] = _normalize_sections(payload.get("sections"))
    return payload


def delete_preset(name: str) -> bool:
    path = preset_path(name)
    if not os.path.isfile(path):
        return False
    os.unlink(path)
    return True


def relative_sources(names: Optional[List[str]]) -> List[str]:
    """Normaliza fontes para nomes relativos, sem caminhos absolutos."""
    result: List[str] = []
    for item in names or []:
        value = os.path.basename(str(item).strip())
        if value and value not in result:
            result.append(value)
    return result

"""
Vocoolder - Vocal Performance Reactive GUI
===========================================

Interface customtkinter (dark mode) com:

- Toggle Iniciar/Parar (motor em thread separada)
- Seleção de dispositivos de entrada/saída
- Botão "Testar Áudio" (grava o microfone e reproduz um tom de teste)
- Abas Voz / Grãos / Pitch / Saída / Sessão
- Sliders: Mix, Densidade, Contraste, Resposta Fonética, Preservar Duração,
  Gate RMS e Grãos ativos
- Biblioteca em dois níveis, filtro de fontes, presets JSON e gravação de sessão
- Pitch: modo, intervalo fixo e afinação 24-EDO/JI
- Medidor de nível em tempo real e log de diagnóstico
"""

from __future__ import annotations

import os
import threading
from tkinter import messagebox
from typing import Any, Dict, List, Optional

import customtkinter
import sounddevice as sd

from src.audio_diag import (
    choose_devices,
    list_devices,
    run_diagnostics,
)
from src.engine import (
    AMOSTRAS_DIR,
    DEFAULT_ACTIVE_GRAINS,
    DEFAULT_MIX,
    DEFAULT_MUTATION_RATE,
    DEFAULT_PHONETIC_RESPONSE,
    DEFAULT_PITCH_INTERVAL,
    DEFAULT_PITCH_MODE,
    DEFAULT_PRESERVE_DURATION,
    DEFAULT_RICOCHET_DENSITY,
    DEFAULT_RMS_THRESHOLD,
    MAX_ACTIVE_GRAINS,
    MAX_RICOCHET_DENSITY,
    MAX_RMS_THRESHOLD,
    MIN_ACTIVE_GRAINS,
    MIN_RICOCHET_DENSITY,
    MIN_RMS_THRESHOLD,
    VocoolderEngine,
    start_engine,
    stop_engine,
)
from src.presets import (
    SECTION_KEYS,
    delete_preset,
    load_preset,
    list_presets,
    relative_sources,
    save_preset,
)

customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")

INPUT, OUTPUT = "input", "output"

PITCH_MODE_OPTIONS = {
    "Desligado": "off",
    "Desvio microtonal": "microtonal",
    "Casar com a voz": "match",
    "Intervalo fixo": "interval",
}
PITCH_MODE_LABELS = {v: k for k, v in PITCH_MODE_OPTIONS.items()}

PITCH_INTERVAL_OPTIONS = {
    "Uníssono": "unison",
    "Quarta justa": "fourth",
    "Trítone": "tritone",
    "Quinta justa": "fifth",
    "Sétima menor": "minor_seventh",
    "Oitava": "octave",
}
PITCH_INTERVAL_LABELS = {v: k for k, v in PITCH_INTERVAL_OPTIONS.items()}

TUNING_OPTIONS = {"24-EDO": "24edo", "JI": "ji"}
TUNING_LABELS = {v: k for k, v in TUNING_OPTIONS.items()}


def _device_label(dev: dict) -> str:
    api = dev["hostapi"].replace("Windows ", "")
    return f"#{dev['index']} {dev['name']} [{api}]"


class VocoolderGUI(customtkinter.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Vocoolder - Vocal Performance Reactive")
        self.geometry("830x780+40+10")
        self.minsize(760, 760)
        self.update_idletasks()

        self.engine: Optional[VocoolderEngine] = None
        self._busy = False
        self._testing = False
        self._record_stopping = False
        self._in_map: Dict[str, int] = {}
        self._out_map: Dict[str, int] = {}
        self._devices: List[Dict[str, Any]] = []
        self._selected_sources: List[str] = []

        self.status_var = customtkinter.StringVar(value="Parado")
        self.meter_var = customtkinter.StringVar(value="Entrada: -- dBFS")
        self.devices_var = customtkinter.StringVar(value="Dispositivos nao carregados")

        self.density_var = customtkinter.IntVar(value=DEFAULT_RICOCHET_DENSITY)
        self.mutation_var = customtkinter.DoubleVar(value=DEFAULT_MUTATION_RATE * 100.0)
        self.phonetic_var = customtkinter.DoubleVar(value=DEFAULT_PHONETIC_RESPONSE * 100.0)
        self.mix_var = customtkinter.DoubleVar(value=DEFAULT_MIX * 100.0)
        self.pitch_mode_var = customtkinter.StringVar(
            value=PITCH_MODE_LABELS[DEFAULT_PITCH_MODE]
        )
        self.pitch_interval_var = customtkinter.StringVar(
            value=PITCH_INTERVAL_LABELS[DEFAULT_PITCH_INTERVAL]
        )
        self.tuning_var = customtkinter.StringVar(value=TUNING_LABELS["24edo"])
        self.preserve_var = customtkinter.DoubleVar(
            value=DEFAULT_PRESERVE_DURATION * 100.0
        )
        self.rms_var = customtkinter.DoubleVar(value=DEFAULT_RMS_THRESHOLD)
        self.active_grains_var = customtkinter.IntVar(
            value=DEFAULT_ACTIVE_GRAINS
        )
        self.sources_var = customtkinter.StringVar(value="Todas as fontes")
        self.preset_name_var = customtkinter.StringVar(value="")
        self.preset_choice_var = customtkinter.StringVar(value="(nenhum)")
        self.record_status_var = customtkinter.StringVar(value="Não gravando")
        self.section_vars = {
            key: customtkinter.BooleanVar(value=True) for key in SECTION_KEYS
        }

        self._build_ui()
        self._populate_devices()
        self._update_loop()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)

        # Status
        status_frame = customtkinter.CTkFrame(self, corner_radius=10)
        status_frame.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = customtkinter.CTkLabel(
            status_frame, textvariable=self.status_var,
            font=("Segoe UI", 18, "bold"),
        )
        self.status_label.grid(row=0, column=0, padx=12, pady=(6, 0), sticky="w")

        self.meter_label = customtkinter.CTkLabel(
            status_frame, textvariable=self.meter_var, font=("Segoe UI", 11)
        )
        self.meter_label.grid(row=1, column=0, padx=12, sticky="w")

        self.meter = customtkinter.CTkProgressBar(status_frame, height=10)
        self.meter.grid(row=2, column=0, padx=12, pady=(2, 0), sticky="ew")
        self.meter.set(0.0)

        self.devices_label = customtkinter.CTkLabel(
            status_frame, textvariable=self.devices_var,
            font=("Segoe UI", 10), text_color="gray70",
        )
        self.devices_label.grid(row=3, column=0, padx=12, pady=(0, 6), sticky="w")

        # Devices
        dev_frame = customtkinter.CTkFrame(self, corner_radius=10)
        dev_frame.grid(row=1, column=0, padx=16, pady=8, sticky="ew")
        dev_frame.grid_columnconfigure((0, 1), weight=1)

        customtkinter.CTkLabel(
            dev_frame, text="Microfone (entrada)", font=("Segoe UI", 11, "bold")
        ).grid(row=0, column=0, padx=10, pady=(6, 0), sticky="w")
        self.input_menu = customtkinter.CTkOptionMenu(
            dev_frame, values=["carregando..."], width=280,
        )
        self.input_menu.grid(row=1, column=0, padx=10, pady=(0, 6), sticky="ew")

        customtkinter.CTkLabel(
            dev_frame, text="Saída (caixas/fones)", font=("Segoe UI", 11, "bold")
        ).grid(row=0, column=1, padx=10, pady=(6, 0), sticky="w")
        self.output_menu = customtkinter.CTkOptionMenu(
            dev_frame, values=["carregando..."], width=280,
        )
        self.output_menu.grid(row=1, column=1, padx=10, pady=(0, 6), sticky="ew")

        button_row = customtkinter.CTkFrame(dev_frame, fg_color="transparent")
        button_row.grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 6), sticky="ew")
        button_row.grid_columnconfigure((0, 1), weight=1)

        self.refresh_btn = customtkinter.CTkButton(
            button_row, text="Recarregar dispositivos",
            command=self._populate_devices, height=28,
        )
        self.refresh_btn.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        self.test_btn = customtkinter.CTkButton(
            button_row, text="Testar Áudio", command=self._on_test_audio, height=28,
        )
        self.test_btn.grid(row=0, column=1, padx=(5, 0), sticky="ew")

        # Controls: seções Voz / Grãos / Pitch / Saída
        tabs = customtkinter.CTkTabview(self, height=160, corner_radius=10)
        tabs.grid(row=2, column=0, padx=16, pady=8, sticky="ew")
        self.tabs = tabs
        tab_voice = tabs.add("Voz")
        tab_grains = tabs.add("Grãos")
        tab_pitch = tabs.add("Pitch")
        tab_output = tabs.add("Saída")
        tab_session = tabs.add("Sessão")
        for tab in (tab_voice, tab_grains, tab_pitch, tab_output, tab_session):
            tab.grid_columnconfigure(0, weight=1)

        # Voz
        voice_row = customtkinter.CTkFrame(tab_voice, fg_color="transparent")
        voice_row.grid(row=0, column=0, padx=12, pady=(10, 2), sticky="ew")
        voice_row.grid_columnconfigure(1, weight=1)
        self.mix_label = customtkinter.CTkLabel(
            voice_row, text="Mix Voz/Grãos: 30% (0=grãos, 100=voz)",
            width=220, anchor="w", font=("Segoe UI", 11),
        )
        self.mix_label.grid(row=0, column=0, sticky="w")
        self.mix_slider = customtkinter.CTkSlider(
            voice_row, from_=0, to=100, command=self._on_mix,
        )
        self.mix_slider.grid(row=0, column=1, padx=(8, 0), sticky="ew")
        self.mix_slider.set(DEFAULT_MIX * 100.0)
        customtkinter.CTkLabel(
            tab_voice,
            text="0% = só grãos  |  100% = só voz seca. Sem ducking.",
            anchor="w", font=("Segoe UI", 11), text_color="gray70",
        ).grid(row=1, column=0, padx=12, pady=(0, 8), sticky="w")

        # Grãos
        density_row = customtkinter.CTkFrame(tab_grains, fg_color="transparent")
        density_row.grid(row=0, column=0, padx=12, pady=(8, 2), sticky="ew")
        density_row.grid_columnconfigure(1, weight=1)
        self.density_label = customtkinter.CTkLabel(
            density_row, text=f"Densidade do Ricochete: {self.density_var.get()}",
            width=220, anchor="w", font=("Segoe UI", 11),
        )
        self.density_label.grid(row=0, column=0, sticky="w")
        self.density_slider = customtkinter.CTkSlider(
            density_row, from_=MIN_RICOCHET_DENSITY, to=MAX_RICOCHET_DENSITY,
            command=self._on_density,
        )
        self.density_slider.grid(row=0, column=1, padx=(8, 0), sticky="ew")
        self.density_slider.set(DEFAULT_RICOCHET_DENSITY)

        mutation_row = customtkinter.CTkFrame(tab_grains, fg_color="transparent")
        mutation_row.grid(row=1, column=0, padx=12, pady=2, sticky="ew")
        mutation_row.grid_columnconfigure(1, weight=1)
        self.mutation_label = customtkinter.CTkLabel(
            mutation_row, text="Contraste Tímbrico: 20%",
            width=220, anchor="w", font=("Segoe UI", 11),
        )
        self.mutation_label.grid(row=0, column=0, sticky="w")
        self.mutation_slider = customtkinter.CTkSlider(
            mutation_row, from_=0, to=100, command=self._on_mutation,
        )
        self.mutation_slider.grid(row=0, column=1, padx=(8, 0), sticky="ew")
        self.mutation_slider.set(DEFAULT_MUTATION_RATE * 100.0)

        phonetic_row = customtkinter.CTkFrame(tab_grains, fg_color="transparent")
        phonetic_row.grid(row=2, column=0, padx=12, pady=(2, 8), sticky="ew")
        phonetic_row.grid_columnconfigure(1, weight=1)
        self.phonetic_label = customtkinter.CTkLabel(
            phonetic_row, text="Resposta Fonética: 60%",
            width=220, anchor="w", font=("Segoe UI", 11),
        )
        self.phonetic_label.grid(row=0, column=0, sticky="w")
        self.phonetic_slider = customtkinter.CTkSlider(
            phonetic_row, from_=0, to=100, command=self._on_phonetic,
        )
        self.phonetic_slider.grid(row=0, column=1, padx=(8, 0), sticky="ew")
        self.phonetic_slider.set(DEFAULT_PHONETIC_RESPONSE * 100.0)

        # Pitch
        tab_pitch.grid_columnconfigure((0, 1, 2), weight=1)
        for col, text in enumerate(("Modo", "Intervalo", "Afinação")):
            customtkinter.CTkLabel(
                tab_pitch, text=text, anchor="w",
                font=("Segoe UI", 10, "bold"),
            ).grid(row=0, column=col, padx=(12, 4), pady=(6, 0), sticky="w")

        self.pitch_mode_menu = customtkinter.CTkOptionMenu(
            tab_pitch,
            values=list(PITCH_MODE_OPTIONS.keys()),
            variable=self.pitch_mode_var,
            command=self._on_pitch_mode,
        )
        self.pitch_mode_menu.grid(
            row=1, column=0, padx=(12, 4), pady=(0, 4), sticky="ew"
        )

        self.interval_menu = customtkinter.CTkOptionMenu(
            tab_pitch,
            values=list(PITCH_INTERVAL_OPTIONS.keys()),
            variable=self.pitch_interval_var,
            command=self._on_interval,
        )
        self.interval_menu.grid(
            row=1, column=1, padx=(4, 4), pady=(0, 4), sticky="ew"
        )

        self.tuning_menu = customtkinter.CTkOptionMenu(
            tab_pitch,
            values=list(TUNING_OPTIONS.keys()),
            variable=self.tuning_var,
            command=self._on_tuning,
        )
        self.tuning_menu.grid(
            row=1, column=2, padx=(4, 12), pady=(0, 4), sticky="ew"
        )

        preserve_row = customtkinter.CTkFrame(tab_pitch, fg_color="transparent")
        preserve_row.grid(
            row=2, column=0, columnspan=3, padx=12, pady=(6, 8), sticky="ew"
        )
        preserve_row.grid_columnconfigure(1, weight=1)
        self.preserve_label = customtkinter.CTkLabel(
            preserve_row,
            text="Preservar Duração: 0% (0=varispeed, 100=WSOLA)",
            width=300, anchor="w", font=("Segoe UI", 11),
        )
        self.preserve_label.grid(row=0, column=0, sticky="w")
        self.preserve_slider = customtkinter.CTkSlider(
            preserve_row, from_=0, to=100, command=self._on_preserve,
        )
        self.preserve_slider.grid(row=0, column=1, padx=(8, 0), sticky="ew")
        self.preserve_slider.set(DEFAULT_PRESERVE_DURATION * 100.0)

        # Saída
        rms_row = customtkinter.CTkFrame(tab_output, fg_color="transparent")
        rms_row.grid(row=0, column=0, padx=12, pady=(8, 2), sticky="ew")
        rms_row.grid_columnconfigure(1, weight=1)
        self.rms_label = customtkinter.CTkLabel(
            rms_row, text=f"Gate de Silêncio (RMS): {self.rms_var.get():.4f}",
            width=220, anchor="w", font=("Segoe UI", 11),
        )
        self.rms_label.grid(row=0, column=0, sticky="w")
        self.rms_slider = customtkinter.CTkSlider(
            rms_row, from_=MIN_RMS_THRESHOLD, to=MAX_RMS_THRESHOLD,
            command=self._on_rms,
        )
        self.rms_slider.grid(row=0, column=1, padx=(8, 0), sticky="ew")
        self.rms_slider.set(DEFAULT_RMS_THRESHOLD)

        self.calibrate_btn = customtkinter.CTkButton(
            tab_output,
            text="Calibrar Gate (ficar em silêncio)",
            command=self._on_calibrate_gate,
            height=30,
        )
        self.calibrate_btn.grid(row=1, column=0, padx=12, pady=(2, 4), sticky="ew")
        customtkinter.CTkLabel(
            tab_output,
            text="Limiter suave em −1 dBFS (0,891). Saída padrão do Windows.",
            anchor="w", font=("Segoe UI", 10), text_color="gray70",
        ).grid(row=2, column=0, padx=12, pady=(0, 8), sticky="w")

        # Sessão: presets, biblioteca ativa, fontes e gravação
        preset_row = customtkinter.CTkFrame(tab_session, fg_color="transparent")
        preset_row.grid(row=0, column=0, padx=12, pady=(6, 2), sticky="ew")
        preset_row.grid_columnconfigure(1, weight=1)
        customtkinter.CTkLabel(
            preset_row, text="Preset", width=54, anchor="w",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.preset_menu = customtkinter.CTkOptionMenu(
            preset_row,
            values=["(nenhum)"],
            variable=self.preset_choice_var,
            command=self._on_preset_selected,
            width=145,
        )
        self.preset_menu.grid(row=0, column=1, padx=(4, 4), sticky="ew")
        self.preset_name_entry = customtkinter.CTkEntry(
            preset_row, textvariable=self.preset_name_var,
            placeholder_text="nome do preset", width=125,
        )
        self.preset_name_entry.grid(row=0, column=2, padx=4, sticky="ew")
        self.preset_save_btn = customtkinter.CTkButton(
            preset_row, text="Salvar", width=58, height=26,
            command=self._on_preset_save,
        )
        self.preset_save_btn.grid(row=0, column=3, padx=2)
        self.preset_load_btn = customtkinter.CTkButton(
            preset_row, text="Carregar", width=64, height=26,
            command=self._on_preset_load,
        )
        self.preset_load_btn.grid(row=0, column=4, padx=2)
        self.preset_delete_btn = customtkinter.CTkButton(
            preset_row, text="Excluir", width=58, height=26,
            command=self._on_preset_delete,
        )
        self.preset_delete_btn.grid(row=0, column=5, padx=(2, 0))

        sections_row = customtkinter.CTkFrame(
            tab_session, fg_color="transparent"
        )
        sections_row.grid(row=1, column=0, padx=12, pady=2, sticky="ew")
        customtkinter.CTkLabel(
            sections_row, text="Seções:", width=54, anchor="w",
            font=("Segoe UI", 10),
        ).grid(row=0, column=0, sticky="w")
        self.section_labels = {
            "library": "Biblioteca",
            "voice": "Voz",
            "grains": "Grãos",
            "pitch": "Pitch",
            "output": "Saída",
        }
        for col, key in enumerate(SECTION_KEYS, start=1):
            customtkinter.CTkCheckBox(
                sections_row,
                text=self.section_labels[key],
                variable=self.section_vars[key],
                font=("Segoe UI", 10),
                checkbox_width=16,
                checkbox_height=16,
            ).grid(row=0, column=col, padx=(4, 7), sticky="w")

        library_row = customtkinter.CTkFrame(
            tab_session, fg_color="transparent"
        )
        library_row.grid(row=2, column=0, padx=12, pady=2, sticky="ew")
        library_row.grid_columnconfigure(1, weight=1)
        self.active_grains_label = customtkinter.CTkLabel(
            library_row,
            text=f"Grãos ativos: {self.active_grains_var.get()}",
            width=130, anchor="w", font=("Segoe UI", 11),
        )
        self.active_grains_label.grid(row=0, column=0, sticky="w")
        self.active_grains_slider = customtkinter.CTkSlider(
            library_row,
            from_=MIN_ACTIVE_GRAINS,
            to=MAX_ACTIVE_GRAINS,
            command=self._on_active_grains,
        )
        self.active_grains_slider.grid(
            row=0, column=1, padx=(8, 8), sticky="ew"
        )
        self.active_grains_slider.set(DEFAULT_ACTIVE_GRAINS)
        self.sources_btn = customtkinter.CTkButton(
            library_row, text="Fontes...", width=78, height=26,
            command=self._on_sources,
        )
        self.sources_btn.grid(row=0, column=2, padx=(0, 6))
        self.sources_label = customtkinter.CTkLabel(
            library_row, textvariable=self.sources_var, width=190,
            anchor="w", font=("Segoe UI", 10), text_color="gray70",
        )
        self.sources_label.grid(row=0, column=3, sticky="w")

        record_row = customtkinter.CTkFrame(
            tab_session, fg_color="transparent"
        )
        record_row.grid(row=3, column=0, padx=12, pady=(2, 8), sticky="ew")
        record_row.grid_columnconfigure(1, weight=1)
        self.record_btn = customtkinter.CTkButton(
            record_row, text="Gravar sessão", width=170, height=30,
            command=self._on_record_toggle, state="disabled",
        )
        self.record_btn.grid(row=0, column=0, sticky="w")
        self.record_label = customtkinter.CTkLabel(
            record_row, textvariable=self.record_status_var,
            anchor="w", font=("Segoe UI", 10), text_color="gray70",
        )
        self.record_label.grid(row=0, column=1, padx=(10, 0), sticky="w")

        self._refresh_preset_menu()
        self._refresh_sources_label()

        # Start/Stop
        self.toggle_btn = customtkinter.CTkButton(
            self, text="Iniciar Vocoolder", command=self._on_toggle,
            height=46, font=("Segoe UI", 16, "bold"),
        )
        self.toggle_btn.grid(row=3, column=0, padx=16, pady=8, sticky="ew")

        # Log
        self.log_box = customtkinter.CTkTextbox(self, height=80, font=("Consolas", 10))
        self.log_box.grid(row=4, column=0, padx=16, pady=(8, 16), sticky="nsew")
        self.grid_rowconfigure(4, weight=1)
        self._log("Bem-vindo ao Vocoolder.")
        self._log("Use 'Testar Áudio' para validar microfone e caixas antes de iniciar.")
        self._log("Dica: use fones de ouvido para evitar microfonia (feedback).")

    # ------------------------------------------------------------------
    # Devices
    # ------------------------------------------------------------------
    def _populate_devices(self) -> None:
        try:
            devices = list_devices()
        except Exception as exc:
            self._log(f"Erro ao listar dispositivos: {exc}")
            return

        self._devices = devices
        inputs = [d for d in devices if d["max_input_channels"] > 0]
        outputs = [d for d in devices if d["max_output_channels"] > 0]

        self._in_map = {_device_label(d): d["index"] for d in inputs}
        self._out_map = {_device_label(d): d["index"] for d in outputs}

        self.input_menu.configure(values=list(self._in_map.keys()) or ["nenhum"])
        self.output_menu.configure(values=list(self._out_map.keys()) or ["nenhum"])

        try:
            defaults = sd.default.device
            default_in, default_out = defaults[0], defaults[1]
            for dev in inputs:
                if dev["index"] == default_in:
                    self.input_menu.set(_device_label(dev))
                    break
            for dev in outputs:
                if dev["index"] == default_out:
                    self.output_menu.set(_device_label(dev))
                    break
            self._log(
                "Padrão do Windows: entrada #%s | saída #%s"
                % (default_in, default_out)
            )
        except Exception as exc:
            self._log(f"Não foi possível ler o padrão do Windows: {exc}")

        try:
            rec_in, rec_out = choose_devices()
            self._log(
                "Sugestão (não aplicada): entrada %s | saída %s"
                % (_device_label(rec_in), _device_label(rec_out))
            )
        except Exception:
            pass

        self._refresh_devices_var()

    def _refresh_devices_var(self) -> None:
        in_label = self.input_menu.get()
        out_label = self.output_menu.get()
        self.devices_var.set(f"Entrada: {in_label}  |  Saída: {out_label}")

    def _selected_input(self) -> Optional[int]:
        return self._in_map.get(self.input_menu.get())

    def _selected_output(self) -> Optional[int]:
        return self._out_map.get(self.output_menu.get())

    # ------------------------------------------------------------------
    # Sliders
    # ------------------------------------------------------------------
    def _on_density(self, value) -> None:
        density = int(round(float(value)))
        self.density_var.set(density)
        self.density_label.configure(text=f"Densidade do Ricochete: {density}")
        if self.engine:
            self.engine.set_parameters(ricochet_density=density)

    def _on_mutation(self, value) -> None:
        pct = float(value)
        self.mutation_var.set(pct)
        self.mutation_label.configure(text=f"Contraste Tímbrico: {pct:.0f}%")
        if self.engine:
            self.engine.set_parameters(mutation_rate=pct / 100.0)

    def _on_phonetic(self, value) -> None:
        pct = float(value)
        self.phonetic_var.set(pct)
        self.phonetic_label.configure(text=f"Resposta Fonética: {pct:.0f}%")
        if self.engine:
            self.engine.set_parameters(phonetic_response=pct / 100.0)

    def _on_mix(self, value) -> None:
        pct = float(value)
        self.mix_var.set(pct)
        self.mix_label.configure(
            text=f"Mix Voz/Grãos: {pct:.0f}% (0=grãos, 100=voz)"
        )
        if self.engine:
            self.engine.set_parameters(mix=pct / 100.0)

    def _on_pitch_mode(self, choice) -> None:
        self.pitch_mode_var.set(choice)
        mode = PITCH_MODE_OPTIONS.get(choice, DEFAULT_PITCH_MODE)
        if self.engine:
            self.engine.set_parameters(pitch_mode=mode)

    def _on_interval(self, choice) -> None:
        self.pitch_interval_var.set(choice)
        interval = PITCH_INTERVAL_OPTIONS.get(choice, DEFAULT_PITCH_INTERVAL)
        if self.engine:
            self.engine.set_parameters(pitch_interval=interval)

    def _on_tuning(self, choice) -> None:
        self.tuning_var.set(choice)
        tuning = TUNING_OPTIONS.get(choice, "24edo")
        if self.engine:
            self.engine.set_parameters(tuning=tuning)

    def _on_preserve(self, value) -> None:
        pct = float(value)
        self.preserve_var.set(pct)
        self.preserve_label.configure(
            text=f"Preservar Duração: {pct:.0f}% (0=varispeed, 100=WSOLA)"
        )
        if self.engine:
            self.engine.set_parameters(preserve_duration=pct / 100.0)

    def _on_rms(self, value) -> None:
        threshold = float(value)
        self.rms_var.set(threshold)
        self.rms_label.configure(text=f"Gate de Silêncio (RMS): {threshold:.4f}")
        if self.engine:
            self.engine.set_parameters(rms_threshold=threshold)

    def _on_active_grains(self, value) -> None:
        grains = int(round(float(value)))
        grains = max(MIN_ACTIVE_GRAINS, min(MAX_ACTIVE_GRAINS, grains))
        self.active_grains_var.set(grains)
        self.active_grains_label.configure(text=f"Grãos ativos: {grains}")

    # ------------------------------------------------------------------
    # Biblioteca / fontes
    # ------------------------------------------------------------------
    @staticmethod
    def _audio_files() -> List[str]:
        exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif")
        if not os.path.isdir(AMOSTRAS_DIR):
            return []
        return sorted(
            name for name in os.listdir(AMOSTRAS_DIR)
            if os.path.splitext(name)[1].lower() in exts
            and os.path.isfile(os.path.join(AMOSTRAS_DIR, name))
        )

    def _refresh_sources_label(self) -> None:
        if not self._selected_sources:
            self.sources_var.set("Todas as fontes")
        elif len(self._selected_sources) == 1:
            name = self._selected_sources[0]
            self.sources_var.set(name if len(name) <= 30 else name[:27] + "...")
        else:
            self.sources_var.set(f"{len(self._selected_sources)} fontes")

    def _on_sources(self) -> None:
        files = self._audio_files()
        if not files:
            self._log(f"Nenhuma amostra encontrada em {AMOSTRAS_DIR}.")
            return
        dialog = customtkinter.CTkToplevel(self)
        dialog.title("Fontes da biblioteca")
        dialog.geometry("500x500")
        dialog.transient(self)
        dialog.grab_set()
        customtkinter.CTkLabel(
            dialog,
            text="Marque as amostras que alimentam o conjunto ativo.\n"
                 "Nenhuma marcada = todas as fontes.",
            justify="left", anchor="w", font=("Segoe UI", 11),
        ).pack(fill="x", padx=14, pady=(12, 6))
        scroll = customtkinter.CTkScrollableFrame(dialog, width=440, height=340)
        scroll.pack(fill="both", expand=True, padx=12, pady=6)
        vars_by_name = {
            name: customtkinter.BooleanVar(
                value=(not self._selected_sources or name in self._selected_sources)
            )
            for name in files
        }
        for row, name in enumerate(files):
            customtkinter.CTkCheckBox(
                scroll, text=name, variable=vars_by_name[name],
                font=("Segoe UI", 11),
            ).grid(row=row, column=0, sticky="w", padx=8, pady=2)
        scroll.grid_columnconfigure(0, weight=1)

        buttons = customtkinter.CTkFrame(dialog, fg_color="transparent")
        buttons.pack(fill="x", padx=12, pady=(0, 12))
        for col in range(4):
            buttons.grid_columnconfigure(col, weight=1)

        def set_all(value: bool) -> None:
            for var in vars_by_name.values():
                var.set(value)

        def apply() -> None:
            chosen = [name for name in files if vars_by_name[name].get()]
            if len(chosen) == len(files):
                chosen = []
            self._selected_sources = chosen
            self._refresh_sources_label()
            if self.engine and self.engine.is_running:
                self._log(
                    "Fontes alteradas; pare e inicie o motor para aplicar."
                )
            dialog.destroy()

        customtkinter.CTkButton(
            buttons, text="Marcar todas", width=90, height=26,
            command=lambda: set_all(True),
        ).grid(row=0, column=0, padx=3)
        customtkinter.CTkButton(
            buttons, text="Desmarcar", width=90, height=26,
            command=lambda: set_all(False),
        ).grid(row=0, column=1, padx=3)
        customtkinter.CTkButton(
            buttons, text="Cancelar", width=90, height=26,
            command=dialog.destroy,
        ).grid(row=0, column=2, padx=3)
        customtkinter.CTkButton(
            buttons, text="Aplicar", width=90, height=26,
            command=apply,
        ).grid(row=0, column=3, padx=3)

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------
    def _refresh_preset_menu(self) -> None:
        names = list_presets()
        values = names or ["(nenhum)"]
        self.preset_menu.configure(values=values)
        current = self.preset_choice_var.get()
        if current not in values:
            self.preset_choice_var.set(values[0])

    def _on_preset_selected(self, choice: str) -> None:
        if choice and choice != "(nenhum)":
            self.preset_name_var.set(choice)

    def _device_snapshot(
        self, label: str, mapping: Dict[str, int]
    ) -> Optional[Dict[str, str]]:
        index = mapping.get(label)
        for dev in self._devices:
            if dev["index"] == index:
                return {"name": dev["name"], "hostapi": dev["hostapi"]}
        return None

    def _resolve_device_label(
        self,
        snapshot: Any,
        mapping: Dict[str, int],
    ) -> Optional[str]:
        if not snapshot:
            return None
        if isinstance(snapshot, str):
            name, hostapi = snapshot, ""
        elif isinstance(snapshot, dict):
            name = str(snapshot.get("name", ""))
            hostapi = str(snapshot.get("hostapi", ""))
        else:
            raise ValueError("Formato de dispositivo inválido no preset.")
        if not name:
            return None
        by_index = {dev["index"]: dev for dev in self._devices}
        for label, index in mapping.items():
            dev = by_index.get(index)
            if dev and dev["name"] == name and (
                not hostapi or dev["hostapi"] == hostapi
            ):
                return label
        for label, index in mapping.items():
            dev = by_index.get(index)
            if dev and dev["name"] == name:
                return label
        raise ValueError(f"Dispositivo do preset não encontrado: {name}")

    def _build_preset_payload(self) -> Dict[str, Any]:
        sections = {key: bool(self.section_vars[key].get()) for key in SECTION_KEYS}
        payload: Dict[str, Any] = {"sections": sections}
        if sections["library"]:
            payload["library"] = {
                "sources": relative_sources(self._selected_sources),
                "active_grains": int(self.active_grains_var.get()),
            }
        if sections["voice"]:
            payload["voice"] = {"mix": float(self.mix_var.get()) / 100.0}
        if sections["grains"]:
            payload["grains"] = {
                "density": int(self.density_var.get()),
                "mutation": float(self.mutation_var.get()) / 100.0,
                "phonetic": float(self.phonetic_var.get()) / 100.0,
            }
        if sections["pitch"]:
            payload["pitch"] = {
                "mode": PITCH_MODE_OPTIONS.get(
                    self.pitch_mode_var.get(), DEFAULT_PITCH_MODE
                ),
                "interval": PITCH_INTERVAL_OPTIONS.get(
                    self.pitch_interval_var.get(), DEFAULT_PITCH_INTERVAL
                ),
                "tuning": TUNING_OPTIONS.get(self.tuning_var.get(), "24edo"),
                "preserve": float(self.preserve_var.get()) / 100.0,
            }
        if sections["output"]:
            payload["output"] = {
                "rms_threshold": float(self.rms_var.get()),
                "input_device": self._device_snapshot(
                    self.input_menu.get(), self._in_map
                ),
                "output_device": self._device_snapshot(
                    self.output_menu.get(), self._out_map
                ),
            }
        return payload

    def _on_preset_save(self) -> None:
        name = self.preset_name_var.get().strip()
        if not name and self.preset_choice_var.get() != "(nenhum)":
            name = self.preset_choice_var.get()
        if not name:
            self._log("Digite um nome para salvar o preset.")
            return
        try:
            payload = self._build_preset_payload()
            path = save_preset(name, payload)
        except Exception as exc:
            self._log(f"Erro ao salvar preset: {exc}")
            return
        self._refresh_preset_menu()
        self.preset_choice_var.set(os.path.splitext(os.path.basename(path))[0])
        self.preset_name_var.set(self.preset_choice_var.get())
        self._log(f"Preset salvo: {path}")

    def _on_preset_load(self) -> None:
        choice = self.preset_choice_var.get()
        if not choice or choice == "(nenhum)":
            self._log("Selecione um preset para carregar.")
            return
        try:
            payload = load_preset(choice)
            self._apply_preset_payload(payload)
        except Exception as exc:
            self._log(f"Erro ao carregar preset: {exc}")
            return
        self.preset_name_var.set(choice)
        self._log(f"Preset aplicado: {choice}")

    def _on_preset_delete(self) -> None:
        choice = self.preset_choice_var.get()
        if not choice or choice == "(nenhum)":
            return
        if not messagebox.askyesno("Excluir preset", f"Excluir o preset '{choice}'?"):
            return
        try:
            delete_preset(choice)
        except Exception as exc:
            self._log(f"Erro ao excluir preset: {exc}")
            return
        self.preset_choice_var.set("(nenhum)")
        self.preset_name_var.set("")
        self._refresh_preset_menu()
        self._log(f"Preset excluído: {choice}")

    def _apply_preset_payload(self, payload: Dict[str, Any]) -> None:
        sections = payload.get("sections") or {}
        enabled = {key: bool(sections.get(key, True)) for key in SECTION_KEYS}
        running = bool(self.engine and self.engine.is_running)

        # Valida e resolve tudo antes de alterar qualquer controle.
        in_label = out_label = None
        if enabled["output"]:
            output = payload.get("output") or {}
            if output.get("input_device"):
                in_label = self._resolve_device_label(
                    output["input_device"], self._in_map
                )
            if output.get("output_device"):
                out_label = self._resolve_device_label(
                    output["output_device"], self._out_map
                )
        if running and enabled["library"]:
            library = payload.get("library") or {}
            new_sources = relative_sources(library.get("sources") or [])
            if new_sources != list(self._selected_sources):
                raise RuntimeError(
                    "Pare o motor para aplicar um preset que altera a biblioteca."
                )
            if "active_grains" in library and int(library["active_grains"]) != int(
                self.active_grains_var.get()
            ):
                raise RuntimeError(
                    "Pare o motor para alterar o conjunto ativo de grãos."
                )
        if running and enabled["output"]:
            if in_label and in_label != self.input_menu.get():
                raise RuntimeError(
                    "Pare o motor para trocar o dispositivo de entrada."
                )
            if out_label and out_label != self.output_menu.get():
                raise RuntimeError(
                    "Pare o motor para trocar o dispositivo de saída."
                )

        engine_params: Dict[str, Any] = {}

        if enabled["library"]:
            library = payload.get("library") or {}
            self._selected_sources = relative_sources(
                library.get("sources") or []
            )
            if "active_grains" in library:
                active = int(library["active_grains"])
                active = max(MIN_ACTIVE_GRAINS, min(MAX_ACTIVE_GRAINS, active))
                self.active_grains_var.set(active)
                self.active_grains_slider.set(active)
                self.active_grains_label.configure(
                    text=f"Grãos ativos: {active}"
                )
            self._refresh_sources_label()

        if enabled["voice"]:
            voice = payload.get("voice") or {}
            if "mix" in voice:
                pct = max(0.0, min(100.0, float(voice["mix"]) * 100.0))
                self.mix_var.set(pct)
                self.mix_slider.set(pct)
                self.mix_label.configure(
                    text=f"Mix Voz/Grãos: {pct:.0f}% (0=grãos, 100=voz)"
                )
                engine_params["mix"] = pct / 100.0

        if enabled["grains"]:
            grains = payload.get("grains") or {}
            if "density" in grains:
                density = int(grains["density"])
                density = max(
                    MIN_RICOCHET_DENSITY, min(MAX_RICOCHET_DENSITY, density)
                )
                self.density_var.set(density)
                self.density_slider.set(density)
                self.density_label.configure(
                    text=f"Densidade do Ricochete: {density}"
                )
                engine_params["ricochet_density"] = density
            if "mutation" in grains:
                pct = max(0.0, min(100.0, float(grains["mutation"]) * 100.0))
                self.mutation_var.set(pct)
                self.mutation_slider.set(pct)
                self.mutation_label.configure(
                    text=f"Contraste Tímbrico: {pct:.0f}%"
                )
                engine_params["mutation_rate"] = pct / 100.0
            if "phonetic" in grains:
                pct = max(0.0, min(100.0, float(grains["phonetic"]) * 100.0))
                self.phonetic_var.set(pct)
                self.phonetic_slider.set(pct)
                self.phonetic_label.configure(
                    text=f"Resposta Fonética: {pct:.0f}%"
                )
                engine_params["phonetic_response"] = pct / 100.0

        if enabled["pitch"]:
            pitch = payload.get("pitch") or {}
            if "mode" in pitch and pitch["mode"] in PITCH_MODE_LABELS:
                self.pitch_mode_var.set(PITCH_MODE_LABELS[pitch["mode"]])
                engine_params["pitch_mode"] = pitch["mode"]
            if (
                "interval" in pitch
                and pitch["interval"] in PITCH_INTERVAL_LABELS
            ):
                self.pitch_interval_var.set(
                    PITCH_INTERVAL_LABELS[pitch["interval"]]
                )
                engine_params["pitch_interval"] = pitch["interval"]
            if "tuning" in pitch and pitch["tuning"] in TUNING_LABELS:
                self.tuning_var.set(TUNING_LABELS[pitch["tuning"]])
                engine_params["tuning"] = pitch["tuning"]
            if "preserve" in pitch:
                pct = max(
                    0.0, min(100.0, float(pitch["preserve"]) * 100.0)
                )
                self.preserve_var.set(pct)
                self.preserve_slider.set(pct)
                self.preserve_label.configure(
                    text=f"Preservar Duração: {pct:.0f}% "
                         f"(0=varispeed, 100=WSOLA)"
                )
                engine_params["preserve_duration"] = pct / 100.0

        if enabled["output"]:
            output = payload.get("output") or {}
            if "rms_threshold" in output:
                threshold = max(
                    MIN_RMS_THRESHOLD,
                    min(MAX_RMS_THRESHOLD, float(output["rms_threshold"])),
                )
                self.rms_var.set(threshold)
                self.rms_slider.set(threshold)
                self.rms_label.configure(
                    text=f"Gate de Silêncio (RMS): {threshold:.4f}"
                )
                engine_params["rms_threshold"] = threshold
            if in_label:
                self.input_menu.set(in_label)
            if out_label:
                self.output_menu.set(out_label)
            self._refresh_devices_var()

        for key in SECTION_KEYS:
            self.section_vars[key].set(enabled[key])

        if self.engine and engine_params:
            self.engine.set_parameters(**engine_params)

    # ------------------------------------------------------------------
    # Gravação de sessão
    # ------------------------------------------------------------------
    def _on_record_toggle(self) -> None:
        if not self.engine or not self.engine.is_running:
            self._log("Inicie o motor antes de gravar a sessão.")
            return
        engine = self.engine
        recording = bool(
            engine.session_recorder and engine.session_recorder.is_recording
        )
        if recording:
            self._record_stopping = True
            self.record_btn.configure(state="disabled", text="Parando...")
            self.record_status_var.set("Finalizando gravação...")

            def run_stop() -> None:
                try:
                    info = engine.stop_recording()
                    error = ""
                except Exception as exc:
                    info = None
                    error = str(exc)
                self.after(0, self._on_record_stopped, info, error)

            threading.Thread(
                target=run_stop, daemon=True, name="VocoolderRecordStop"
            ).start()
            return
        try:
            info = engine.start_recording()
            self._log(f"Gravando áudio em: {info['audio_path']}")
            self._log(f"Eventos JSONL em: {info['event_path']}")
        except Exception as exc:
            self._log(f"Erro na gravação de sessão: {exc}")
        self._update_record_status()

    def _on_record_stopped(
        self, info: Optional[Dict[str, Any]], error: str
    ) -> None:
        self._record_stopping = False
        if error:
            self._log(f"Erro ao encerrar gravação: {error}")
        elif info:
            self._log(
                "Gravação encerrada: %.1f s | %s"
                % (info["duration_seconds"], info["audio_path"])
            )
        self._update_record_status()

    def _update_record_status(self) -> None:
        if self._record_stopping:
            self.record_btn.configure(state="disabled", text="Parando...")
            return
        if self.engine and self.engine.session_recorder:
            rec = self.engine.session_recorder
            self.record_status_var.set(
                f"Gravando {rec.duration_seconds:5.1f} s | "
                f"fila {rec.queue_depth} | perdas {rec.dropped_audio_blocks}"
            )
            self.record_btn.configure(text="Parar gravação", state="normal")
        else:
            self.record_status_var.set("Não gravando")
            self.record_btn.configure(
                text="Gravar sessão",
                state="normal" if self.engine and self.engine.is_running else "disabled",
            )

    # ------------------------------------------------------------------
    # Gate calibration
    # ------------------------------------------------------------------
    def _on_calibrate_gate(self) -> None:
        if self._busy:
            return
        if not self.engine or not self.engine.is_running:
            self._log("Inicie o motor antes de calibrar o gate.")
            return

        self._busy = True
        self.calibrate_btn.configure(state="disabled", text="Calibrando... (silêncio)")
        engine = self.engine

        def run() -> None:
            try:
                threshold = engine.calibrate_gate(0.8)
                self.after(0, self._on_gate_calibrated, threshold)
            except Exception as exc:
                self.after(0, self._log, f"Erro ao calibrar gate: {exc}")
                self.after(0, self._gate_calibration_done)

        threading.Thread(target=run, daemon=True, name="VocoolderGate").start()

    def _on_gate_calibrated(self, threshold: float) -> None:
        self.rms_var.set(threshold)
        self.rms_slider.set(threshold)
        self.rms_label.configure(text=f"Gate de Silêncio (RMS): {threshold:.4f}")
        self._log(f"Gate calibrado em {threshold:.4f} RMS.")
        self._gate_calibration_done()

    def _gate_calibration_done(self) -> None:
        self._busy = False
        self.calibrate_btn.configure(state="normal", text="Calibrar Gate (ficar em silêncio)")

    # ------------------------------------------------------------------
    # Test audio
    # ------------------------------------------------------------------
    def _on_test_audio(self) -> None:
        if self._testing or self._busy:
            return
        self._testing = True
        self.test_btn.configure(state="disabled", text="Testando...")

        in_dev = self._selected_input()
        out_dev = self._selected_output()

        def progress(msg: str) -> None:
            self.after(0, self._log, msg)

        def run() -> None:
            try:
                report = run_diagnostics(
                    input_device=in_dev,
                    output_device=out_dev,
                    progress=progress,
                )
                self.after(0, self._on_test_finished, report)
            except Exception as exc:
                self.after(0, self._log, f"ERRO no teste de áudio: {exc}")
                self.after(0, self._on_test_finished, None)

        threading.Thread(target=run, daemon=True, name="VocoolderAudioTest").start()

    def _on_test_finished(self, report: Optional[dict]) -> None:
        self._testing = False
        self.test_btn.configure(state="normal", text="Testar Áudio")
        if report:
            self._log(report.get("text", "Teste concluído."))
            if report.get("ok"):
                self._log("Teste concluído: entrada e saída parecem OK.")
            else:
                self._log("Teste concluído com avisos. Veja o relatório acima.")

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------
    def _on_toggle(self) -> None:
        if self._busy:
            return
        if self.engine and self.engine.is_running:
            self._stop_engine_async()
        else:
            self._start_engine_async()

    def _start_engine_async(self) -> None:
        self._busy = True
        self.toggle_btn.configure(state="disabled", text="Inicializando...")
        self.status_var.set("Inicializando motor...")
        self.status_label.configure(text_color="orange")

        in_dev = self._selected_input()
        out_dev = self._selected_output()
        density = int(round(self.density_var.get()))
        mutation = float(self.mutation_var.get()) / 100.0
        phonetic = float(self.phonetic_var.get()) / 100.0
        mix = float(self.mix_var.get()) / 100.0
        pitch_mode = PITCH_MODE_OPTIONS.get(
            self.pitch_mode_var.get(), DEFAULT_PITCH_MODE
        )
        pitch_interval = PITCH_INTERVAL_OPTIONS.get(
            self.pitch_interval_var.get(), DEFAULT_PITCH_INTERVAL
        )
        tuning = TUNING_OPTIONS.get(self.tuning_var.get(), "24edo")
        preserve = float(self.preserve_var.get()) / 100.0
        threshold = float(self.rms_var.get())
        active_grains = int(self.active_grains_var.get())
        sources = list(self._selected_sources) or None

        def progress(msg: str) -> None:
            self.after(0, self._log, msg)

        def run() -> None:
            try:
                engine = start_engine(
                    ricochet_density=density,
                    mutation_rate=mutation,
                    rms_threshold=threshold,
                    input_device=in_dev,
                    output_device=out_dev,
                    phonetic_response=phonetic,
                    mix=mix,
                    tuning=tuning,
                    pitch_mode=pitch_mode,
                    pitch_interval=pitch_interval,
                    preserve_duration=preserve,
                    active_grains=active_grains,
                    sources=sources,
                    auto_calibrate=True,
                    progress=progress,
                )
                self.after(0, self._on_engine_started, engine)
            except Exception as exc:
                self.after(0, self._on_engine_failed, str(exc))

        threading.Thread(target=run, daemon=True, name="VocoolderStart").start()

    def _on_engine_started(self, engine: VocoolderEngine) -> None:
        self.engine = engine
        self._busy = False
        self.toggle_btn.configure(state="normal", text="Parar")
        self.status_var.set("Ativo")
        self.status_label.configure(text_color="#2ecc71")
        self.input_menu.configure(state="disabled")
        self.output_menu.configure(state="disabled")
        self.refresh_btn.configure(state="disabled")
        self.test_btn.configure(state="disabled")
        self._refresh_devices_var()
        self.calibrate_btn.configure(state="normal")
        self._record_stopping = False
        self.record_btn.configure(state="normal")
        self.record_status_var.set("Não gravando")
        threshold = float(engine.rms_threshold)
        self.rms_var.set(threshold)
        self.rms_slider.set(threshold)
        self.rms_label.configure(text=f"Gate de Silêncio (RMS): {threshold:.4f}")
        self._log(f"Gate calibrado em {threshold:.4f} RMS (ruído ambiente).")
        if engine.corpus is not None:
            self._log(
                "Conjunto ativo: %d de %d grãos | fontes: %s"
                % (
                    len(engine.corpus.grains),
                    engine._library_total or len(engine.corpus.grains),
                    ", ".join(engine._active_sources) or "todas",
                )
            )
        self._log("Motor iniciado. Fale ou cante no microfone.")

    def _on_engine_failed(self, error: str) -> None:
        self.engine = None
        self._busy = False
        self.toggle_btn.configure(state="normal", text="Iniciar Vocoolder")
        self.record_btn.configure(state="disabled", text="Gravar sessão")
        self.status_var.set("Erro ao iniciar")
        self.status_label.configure(text_color="#e74c3c")
        self._log(f"FALHA ao iniciar o motor: {error}")

    def _stop_engine_async(self) -> None:
        if not self.engine:
            return
        self._busy = True
        self.toggle_btn.configure(state="disabled", text="Parando...")
        engine = self.engine

        def run() -> None:
            try:
                stop_engine(engine)
            finally:
                self.after(0, self._on_engine_stopped)

        threading.Thread(target=run, daemon=True, name="VocoolderStop").start()

    def _on_engine_stopped(self) -> None:
        self.engine = None
        self._busy = False
        self.toggle_btn.configure(state="normal", text="Iniciar Vocoolder")
        self.status_var.set("Parado")
        self.status_label.configure(text_color="gray")
        self.input_menu.configure(state="normal")
        self.output_menu.configure(state="normal")
        self.refresh_btn.configure(state="normal")
        self.test_btn.configure(state="normal")
        self._record_stopping = False
        self.record_btn.configure(state="disabled", text="Gravar sessão")
        self.record_status_var.set("Não gravando")
        self.meter.set(0.0)
        self.meter_var.set("Entrada: -- dBFS")
        self._log("Motor parado.")

    # ------------------------------------------------------------------
    # Update loop
    # ------------------------------------------------------------------
    def _update_loop(self) -> None:
        try:
            if self.engine and self.engine.is_running:
                import math

                rms = max(self.engine.input_rms, 1e-9)
                db = 20.0 * math.log10(rms)
                meter = max(0.0, min(1.0, (db + 60.0) / 60.0))
                self.meter.set(meter)
                self.meter_var.set(
                    f"Entrada: {db:.1f} dBFS | Saída: {self.engine.output_peak:.2f}"
                )
                self._update_record_status()
                if self.engine.last_error:
                    self._log(f"AVISO: {self.engine.last_error}")
                    self.engine.last_error = ""
        except Exception:
            pass
        self.after(60, self._update_loop)

    # ------------------------------------------------------------------
    # Log / close
    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        if not message:
            return
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")

    def destroy(self) -> None:
        try:
            if self.engine and self.engine.is_running:
                stop_engine(self.engine)
        except Exception:
            pass
        super().destroy()


def main() -> None:
    app = VocoolderGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

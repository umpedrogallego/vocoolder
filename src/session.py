"""Gravação assíncrona de sessão e log de eventos em JSONL.

O callback de áudio nunca escreve em disco. O motor empurra blocos já
limitados para uma fila limitada; uma thread dedicada grava o WAV.
"""
from __future__ import annotations

import json
import os
import queue
import threading
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "recordings")
DEFAULT_QUEUE_BLOCKS = 256


class SessionRecorder:
    def __init__(
        self,
        sample_rate: int,
        channels: int = 2,
        output_dir: str = RECORDINGS_DIR,
        max_queue_blocks: int = DEFAULT_QUEUE_BLOCKS,
        subtype: str = "PCM_24",
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = max(1, int(channels))
        self.output_dir = output_dir
        self.subtype = subtype
        self.max_queue_blocks = int(max(8, max_queue_blocks))
        self._queue: "queue.Queue[Any]" = queue.Queue(
            maxsize=self.max_queue_blocks
        )
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._closed = False
        self._error = ""
        self._frames_written = 0
        self._dropped_audio = 0
        self._dropped_events = 0
        self._started_at = datetime.now().astimezone()
        stamp = self._started_at.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        os.makedirs(self.output_dir, exist_ok=True)
        self.audio_path = os.path.join(
            self.output_dir, "session_%s.wav" % stamp
        )
        self.event_path = os.path.join(
            self.output_dir, "session_%s.jsonl" % stamp
        )
        self._thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="VocoolderSessionWriter",
        )
        self._thread.start()

    @property
    def is_recording(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._closed)

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    @property
    def duration_seconds(self) -> float:
        return self._frames_written / float(self.sample_rate or 1)

    @property
    def dropped_audio_blocks(self) -> int:
        return self._dropped_audio

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    @property
    def queue_depth(self) -> int:
        return int(self._queue.qsize())

    def _writer_loop(self) -> None:
        try:
            import soundfile as sf

            with sf.SoundFile(
                self.audio_path,
                mode="w",
                samplerate=self.sample_rate,
                channels=self.channels,
                subtype=self.subtype,
                format="WAV",
            ) as wav:
                with open(self.event_path, "w", encoding="utf-8") as log:
                    log.write(
                        json.dumps(
                            {
                                "type": "session_start",
                                "started_at": self._started_at.isoformat(),
                                "sample_rate": self.sample_rate,
                                "channels": self.channels,
                                "format": self.subtype,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    log.flush()
                    while True:
                        item = self._queue.get()
                        if item is None:
                            break
                        kind, payload = item
                        if kind == "audio":
                            wav.write(payload)
                            self._frames_written += len(payload)
                        elif kind == "event":
                            log.write(
                                json.dumps(
                                    payload,
                                    ensure_ascii=False,
                                    allow_nan=False,
                                )
                                + "\n"
                            )
                            log.flush()
        except Exception as exc:
            with self._lock:
                self._error = "Gravador de sessão: %s" % exc
        finally:
            with self._lock:
                self._closed = True

    def push(self, block: np.ndarray) -> None:
        if self._closed or self._stop_requested:
            return
        data = np.asarray(block, dtype=np.float32)
        if data.ndim == 1 and self.channels > 1:
            data = np.repeat(data[:, None], self.channels, axis=1)
        elif data.ndim == 2 and data.shape[1] != self.channels:
            if data.shape[1] > self.channels:
                data = data[:, : self.channels]
            else:
                data = np.pad(
                    data,
                    ((0, 0), (0, self.channels - data.shape[1])),
                )
        data = np.ascontiguousarray(data)
        try:
            self._queue.put_nowait(("audio", data.copy()))
        except queue.Full:
            self._dropped_audio += 1

    def log_event(self, event: Dict[str, Any]) -> None:
        if self._closed or self._stop_requested:
            return
        try:
            self._queue.put_nowait(("event", dict(event)))
        except queue.Full:
            self._dropped_events += 1

    def stop(self, timeout: float = 12.0) -> Dict[str, Any]:
        self._stop_requested = True
        if self._thread and self._thread.is_alive():
            try:
                self._queue.put(None, timeout=2.0)
            except queue.Full:
                # Se a fila estiver cheia, descarta o excedente e encerra.
                try:
                    while True:
                        self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(None)
                except queue.Full:
                    pass
            self._thread.join(timeout=timeout)
        return {
            "audio_path": self.audio_path,
            "event_path": self.event_path,
            "duration_seconds": self.duration_seconds,
            "dropped_audio_blocks": self._dropped_audio,
            "dropped_events": self._dropped_events,
            "error": self.error,
        }

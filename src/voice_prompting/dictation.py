"""Speech to text: record the microphone and transcribe it locally with faster-whisper."""

from __future__ import annotations

import ctypes
import importlib.util
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from voice_prompting.constants import (
    CUDA_LIBRARY_PACKAGES,
    SAMPLE_RATE,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
)
from voice_prompting.recorder import RecordingError, record
from voice_prompting.spinner import spinner

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from faster_whisper import WhisperModel

    from voice_prompting.config import Config


class DictationError(RuntimeError):
    """Raised when recording or transcription fails."""


def preload_cuda_libraries() -> bool:
    """Load cuBLAS and cuDNN from the ``gpu`` extra's wheels so CTranslate2 finds them by name.

    Returns:
        Whether every library loaded; False if the extra is not installed.
    """
    pending: list[Path] = []
    for package in CUDA_LIBRARY_PACKAGES:
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            return False
        pending += sorted(
            (Path(spec.submodule_search_locations[0]) / "lib").glob("lib*.so*")
        )
    # Libraries depend on each other; retry until a pass loads nothing new.
    while pending:
        failed = []
        for library in pending:
            try:
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(library)
        if len(failed) == len(pending):
            return False
        pending = failed
    return True


def resolve_device(device: str) -> str:
    """Turn ``auto`` into ``cuda`` when a GPU and the CUDA libraries are available, else ``cpu``.

    Raises:
        DictationError: If ``cuda`` was requested but the CUDA libraries cannot be loaded.
    """
    if device == "cpu":
        return device
    import ctranslate2  # noqa: PLC0415 - slow import, only needed when dictating

    usable = ctranslate2.get_cuda_device_count() > 0 and preload_cuda_libraries()
    if device == "cuda" and not usable:
        raise DictationError(
            "no usable GPU: install the gpu extra (`uv tool install 'voice-prompting[gpu]' ...`)"
        )
    return "cuda" if usable else "cpu"


def load_model(config: Config) -> tuple[WhisperModel, str]:
    """Load the configured faster-whisper model (downloaded on first use) and return it with its device."""
    from faster_whisper import WhisperModel  # noqa: PLC0415 - slow import

    device = resolve_device(config.whisper_device)
    model = WhisperModel(
        config.whisper_model, device=device, compute_type=WHISPER_COMPUTE_TYPE[device]
    )
    return model, device


def transcribe(
    model: WhisperModel,
    audio: np.ndarray,
    prompt: str = "",
    on_progress: Callable[[float], None] | None = None,
) -> str:
    """Transcribe 16 kHz float samples, detecting their language and skipping non-speech.

    Args:
        model: Loaded faster-whisper model.
        audio: Samples to transcribe.
        prompt: Text said just before this audio, given to Whisper as context.
        on_progress: Receives the seconds of ``audio`` transcribed so far, after each segment.
    """
    segments, _ = model.transcribe(
        audio,
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=True,
        initial_prompt=prompt or None,
    )
    texts = []
    for segment in segments:
        texts.append(segment.text.strip())
        if on_progress is not None:
            on_progress(segment.end)
    return " ".join(texts).strip()


class BatchTranscriber:
    """Load the model and transcribe batches of audio in order, on a background thread.

    Each batch gets the text of the batches before it as context and its own language detection.
    """

    def __init__(self, load: Callable[[], tuple[WhisperModel, str]]) -> None:
        """Start loading the model at once.

        Args:
            load: Returns the model and its device (``cpu`` or ``cuda``).
        """
        self.device: str | None = None
        self.submitted_s = 0.0
        self.done_s = 0.0
        self._load = load
        self._texts: list[str] = []
        self._error: Exception | None = None
        self._cancelled = False
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._thread = threading.Thread(target=self._work)
        self._thread.start()

    def submit(self, audio: np.ndarray) -> None:
        """Queue a batch for transcription."""
        self.submitted_s += len(audio) / SAMPLE_RATE
        self._queue.put(audio)

    @property
    def progress(self) -> float:
        """Fraction of the submitted audio transcribed so far (1.0 when nothing is pending)."""
        if not self.submitted_s:
            return 1.0
        return min(self.done_s / self.submitted_s, 1.0)

    def finish(self) -> str:
        """Wait for every submitted batch and return the joined text.

        Raises:
            DictationError: If loading the model or transcribing failed.
        """
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise DictationError(str(self._error)) from self._error
        return " ".join(self._texts)

    def cancel(self) -> None:
        """Skip the batches not yet started; the thread ends after the current step."""
        self._cancelled = True
        self._queue.put(None)

    def _work(self) -> None:
        try:
            model, self.device = self._load()
            while (audio := self._queue.get()) is not None:
                if self._cancelled:
                    continue
                start = self.done_s
                text = transcribe(
                    model,
                    audio,
                    prompt=" ".join(self._texts),
                    on_progress=lambda seconds: setattr(
                        self, "done_s", start + seconds
                    ),
                )
                if text:
                    self._texts.append(text)
                self.done_s = start + len(audio) / SAMPLE_RATE
        except (OSError, RuntimeError, ValueError) as exc:
            self._error = exc


def _record_batches(config: Config, note: str) -> BatchTranscriber | None:
    """Record, submitting each batch at its pause; None if recording never started."""
    started: list[BatchTranscriber] = []

    def start() -> None:
        started.append(BatchTranscriber(lambda: load_model(config)))

    def submit(audio: np.ndarray) -> None:
        started[0].submit(audio)

    try:
        record(
            config.speech_threshold,
            note=note,
            on_start=start,
            on_batch=submit,
            progress=lambda: started[0].progress if started else None,
        )
    except (RecordingError, KeyboardInterrupt):
        if started:
            started[0].cancel()
        raise
    return started[0] if started else None


def dictate(config: Config, note: str = "") -> str:
    """Record (Space starts and pauses, Ctrl+G finishes) and transcribe each batch at its pause.

    The model starts loading at the first Space, so finishing without ever starting doesn't wait
    for it. At Ctrl+G only the batches not yet transcribed remain; a spinner shows their progress.

    Args:
        config: Dictation settings.
        note: Shown on the recorder line before recording starts.

    Raises:
        DictationError: If the microphone cannot be recorded or the model fails.
    """
    try:
        transcriber = _record_batches(config, note)
    except RecordingError as exc:
        raise DictationError(str(exc)) from exc
    if transcriber is None:
        return ""
    if not transcriber.submitted_s:
        transcriber.cancel()
        return ""
    name = config.whisper_model

    def label() -> str:
        if transcriber.device is None:
            return f"Loading whisper {name}"
        return (
            f"Transcribing with whisper {name} on {transcriber.device.upper()}"
            f" · {transcriber.progress:.0%}"
        )

    with spinner(label):
        return transcriber.finish()

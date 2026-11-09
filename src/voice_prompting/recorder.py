"""Microphone recording with SoX: Space starts and pauses, Ctrl+G finishes, Ctrl+C cancels.

Each pause closes a batch of audio, handed over at once so it can be transcribed while you think.
"""

from __future__ import annotations

import math
import os
import select
import subprocess
import sys
import termios
import tty
from contextlib import contextmanager
from typing import IO, TYPE_CHECKING, TextIO, cast

import numpy as np

from voice_prompting.constants import (
    CHUNK_S,
    CLEAR_LINE,
    FINISH_KEY,
    LEVEL_BAR_WIDTH,
    LEVEL_CEIL_DB,
    LEVEL_FLOOR_DB,
    RECORDER_ARGV,
    RECORDING_DOTS,
    SAMPLE_RATE,
    TOGGLE_KEY,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

BYTES_PER_SAMPLE = 2
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_S) * BYTES_PER_SAMPLE


class RecordingError(RuntimeError):
    """Raised when the microphone cannot be recorded."""


def rms(samples: np.ndarray) -> float:
    """Root-mean-square level of float samples in [-1, 1]."""
    return float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0


def level_bar(level: float) -> str:
    """Draw ``level`` as a bar on a log scale between LEVEL_FLOOR_DB and LEVEL_CEIL_DB."""
    db = 20 * math.log10(level) if level > 0 else LEVEL_FLOOR_DB
    fraction = (db - LEVEL_FLOOR_DB) / (LEVEL_CEIL_DB - LEVEL_FLOOR_DB)
    filled = round(min(max(fraction, 0.0), 1.0) * LEVEL_BAR_WIDTH)
    return "▮" * filled + "▯" * (LEVEL_BAR_WIDTH - filled)


class Session:
    """Audio kept while recording is on; Space toggles it, chunks arriving while paused are dropped.

    Each pause closes a batch. Batches in which no chunk reached the speech threshold are dropped:
    Whisper turns background noise alone into phrases like "Thank you.".
    """

    def __init__(
        self,
        speech_threshold: float,
        on_start: Callable[[], None] | None = None,
        on_batch: Callable[[np.ndarray], None] | None = None,
        progress: Callable[[], float | None] | None = None,
        note: str = "",
    ) -> None:
        """Start paused, waiting for the first Space.

        Args:
            speech_threshold: RMS level a chunk must reach to count as speech.
            on_start: Called once, when recording starts for the first time.
            on_batch: Receives the samples of each closed batch that contains speech.
            progress: Fraction of the submitted audio already transcribed, shown while paused.
            note: Shown on the line before recording starts.
        """
        self.speech_threshold = speech_threshold
        self.note = note
        self.on_start = on_start
        self.on_batch = on_batch
        self.progress = progress
        self.recording = False
        self.started = False
        self.recorded_chunks = 0
        self.level = 0.0
        self.mic_ready = False
        self._batch: list[np.ndarray] = []
        self._batch_has_speech = False
        self._pending = b""

    def toggle(self) -> None:
        """Start, pause or resume recording; pausing closes the current batch."""
        if self.recording:
            self.close_batch()
        self.recording = not self.recording
        if not self.started and self.on_start is not None:
            self.on_start()
        self.started = True
        self.level = 0.0

    def close_batch(self) -> None:
        """Hand the audio recorded since the last pause to ``on_batch`` if it contains speech."""
        if self._batch_has_speech and self.on_batch is not None:
            self.on_batch(np.concatenate(self._batch))
        self._batch = []
        self._batch_has_speech = False

    def feed(self, data: bytes) -> None:
        """Split raw 16-bit PCM from the recorder into chunks; keep them if recording is on."""
        self.mic_ready = True
        self._pending += data
        while len(self._pending) >= CHUNK_BYTES:
            raw, self._pending = (
                self._pending[:CHUNK_BYTES],
                self._pending[CHUNK_BYTES:],
            )
            if self.recording:
                chunk = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
                self.level = rms(chunk)
                self._batch_has_speech |= self.level >= self.speech_threshold
                self._batch.append(chunk)
                self.recorded_chunks += 1

    @property
    def seconds(self) -> float:
        """Recorded duration."""
        return self.recorded_chunks * CHUNK_S

    def status(self) -> str:
        """One-line indicator for the terminal."""
        if not self.mic_ready:
            return "  ○  Starting microphone… · Ctrl+C to cancel"
        if not self.started:
            note = f" · {self.note}" if self.note else ""
            return (
                f"  ○  Press Space to talk{note} · Ctrl+G to finish · Ctrl+C to cancel"
            )
        if not self.recording:
            done = self.progress() if self.progress is not None else None
            transcribed = "" if done is None else f" · transcribed {done:.0%}"
            return (
                f"  ‖  Paused {self.seconds:.1f} s{transcribed}"
                " · Space to resume · Ctrl+G to finish"
            )
        dot = RECORDING_DOTS[int(self.seconds * 2) % len(RECORDING_DOTS)]
        return (
            f"  \033[31m{dot}\033[0m  Recording {level_bar(self.level)} {self.seconds:.1f} s"
            " · Space to pause · Ctrl+G to finish"
        )


@contextmanager
def _cbreak(keys: TextIO) -> Iterator[bool]:
    """Read single key presses from ``keys`` while inside; yield whether it is a terminal."""
    if not keys.isatty():
        yield False
        return
    fd = keys.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        yield True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _draw(out: TextIO, session: Session) -> None:
    if out.isatty():
        out.write(f"{CLEAR_LINE}{session.status()}")
        out.flush()


def _start(argv: Sequence[str]) -> subprocess.Popen[bytes]:
    try:
        return subprocess.Popen(  # noqa: S603 - fixed recorder command, no shell
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
    except FileNotFoundError as exc:
        raise RecordingError(
            f"recorder not found: {argv[0]} (install SoX, e.g. `sudo apt install sox`)"
        ) from exc


def _run(session: Session, audio_fd: int, key_fd: int | None, out: TextIO) -> None:
    """Feed recorder output to ``session`` and handle keys until Ctrl+G or the recorder exits."""
    sources = [audio_fd] if key_fd is None else [audio_fd, key_fd]
    _draw(out, session)
    while True:
        ready = select.select(sources, [], [])[0]
        if key_fd in ready:
            key = os.read(key_fd, 1).decode(errors="ignore")
            if key == FINISH_KEY:
                return
            if key == TOGGLE_KEY:
                session.toggle()
        if audio_fd in ready:
            data = os.read(audio_fd, CHUNK_BYTES)
            if not data:
                return
            session.feed(data)
        _draw(out, session)


def record(
    speech_threshold: float,
    *,
    note: str = "",
    on_start: Callable[[], None] | None = None,
    on_batch: Callable[[np.ndarray], None] | None = None,
    progress: Callable[[], float | None] | None = None,
    argv: Sequence[str] = RECORDER_ARGV,
    keys: TextIO | None = None,
    stream: TextIO | None = None,
) -> None:
    """Record the microphone: Space starts and pauses, Ctrl+G finishes.

    Every pause and the finish hand a batch to ``on_batch``. Without a terminal on ``keys``,
    recording starts at once and runs as one batch until the recorder exits.

    Args:
        speech_threshold: RMS level a chunk must reach to count as speech.
        note: Shown on the line before recording starts.
        on_start: Called once, when recording starts for the first time.
        on_batch: Receives the samples of each batch that contains speech.
        progress: Fraction of the submitted audio already transcribed, shown while paused.
        argv: Recorder command writing raw 16-bit mono PCM at SAMPLE_RATE to stdout.
        keys: Terminal to read keys from; defaults to stdin.
        stream: Terminal to draw the indicator on; defaults to stderr.

    Raises:
        RecordingError: If the recorder is missing or fails before producing audio.
        KeyboardInterrupt: If the user pressed Ctrl+C.
    """
    keys = keys or sys.stdin
    out = stream or sys.stderr
    proc = _start(argv)
    audio_fd = cast("IO[bytes]", proc.stdout).fileno()
    session = Session(speech_threshold, on_start, on_batch, progress, note)
    try:
        with _cbreak(keys) as interactive:
            if not interactive:
                session.toggle()
            _run(session, audio_fd, keys.fileno() if interactive else None, out)
            session.close_batch()
    finally:
        proc.terminate()
        _, stderr = proc.communicate()
        if out.isatty():
            out.write(CLEAR_LINE)
            out.flush()
    if not session.recorded_chunks and proc.returncode not in (0, -15):
        message = (
            stderr.decode(errors="replace").strip() or f"exit code {proc.returncode}"
        )
        raise RecordingError(f"recording failed: {message}")

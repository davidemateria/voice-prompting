from __future__ import annotations

import io
import os
import pty
import sys
import threading

import numpy as np
import pytest

from voice_prompting.constants import (
    CHUNK_S,
    FINISH_KEY,
    LEVEL_BAR_WIDTH,
    SAMPLE_RATE,
    TOGGLE_KEY,
)
from voice_prompting.recorder import (
    CHUNK_BYTES,
    RecordingError,
    Session,
    level_bar,
    record,
    rms,
)

TONE = (
    np.sin(2 * np.pi * 440 * np.arange(CHUNK_BYTES // 2) / SAMPLE_RATE) * 16000
).astype(np.int16)
LOUD = TONE.tobytes()
QUIET = bytes(CHUNK_BYTES)

# Raw 16-bit PCM like `rec`: optional pause, 0.5 s of tone, then silence, then optionally hangs.
FAKE_MIC = (
    "import math, struct, sys, time\n"
    "rate = {rate}\n"
    "time.sleep({delay})\n"
    "tone = [int(16000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(rate // 2)]\n"
    "sys.stdout.buffer.write(struct.pack(f'<{{len(tone)}}h', *tone) + bytes(2 * 3 * rate))\n"
    "sys.stdout.buffer.flush()\n"
    "time.sleep({hang})\n"
)


def fake_mic(delay=0.0, hang=0.0):
    return [
        sys.executable,
        "-c",
        FAKE_MIC.format(rate=SAMPLE_RATE, delay=delay, hang=hang),
    ]


def test_session_starts_paused_and_drops_audio():
    batches = []
    session = Session(0.05, on_batch=batches.append)
    assert "Starting microphone" in session.status()
    session.feed(LOUD)
    session.close_batch()

    assert batches == []
    assert "Press Space" in session.status()


def test_each_pause_closes_a_batch_and_silent_batches_are_dropped():
    starts, batches = [], []
    session = Session(0.05, on_start=lambda: starts.append(1), on_batch=batches.append)
    session.toggle()
    session.feed(LOUD + QUIET[:100])
    session.toggle()
    session.feed(QUIET[100:] + LOUD)
    assert "Paused" in session.status()
    session.toggle()
    session.feed(QUIET)
    session.toggle()
    session.toggle()
    session.feed(LOUD)
    session.close_batch()

    assert starts == [1]
    assert [len(b) for b in batches] == [CHUNK_BYTES // 2, CHUNK_BYTES // 2]
    assert session.seconds == pytest.approx(3 * CHUNK_S)


def test_paused_status_shows_transcription_progress():
    session = Session(0.05, progress=lambda: 0.5)
    session.feed(QUIET)
    session.toggle()
    session.toggle()

    assert "transcribed 50%" in session.status()


def test_level_bar_bounds():
    assert level_bar(0.0) == "▯" * LEVEL_BAR_WIDTH
    assert level_bar(1.0) == "▮" * LEVEL_BAR_WIDTH


def test_rms():
    assert rms(np.full(4, 0.5, np.float32)) == pytest.approx(0.5)
    assert rms(np.zeros(0, np.float32)) == 0.0


def test_record_without_terminal_is_one_batch_until_recorder_ends():
    batches = []
    record(0.05, on_batch=batches.append, argv=fake_mic(), keys=io.StringIO())

    assert len(batches) == 1
    assert batches[0].dtype == np.float32
    assert len(batches[0]) / SAMPLE_RATE == pytest.approx(3.5, abs=CHUNK_S)


def test_record_space_starts_and_pauses_ctrl_g_finishes():
    batches = []
    leader, follower = pty.openpty()
    keys = os.fdopen(follower, "r")
    threading.Timer(0.2, os.write, (leader, TOGGLE_KEY.encode())).start()
    threading.Timer(1.2, os.write, (leader, TOGGLE_KEY.encode())).start()
    threading.Timer(1.6, os.write, (leader, FINISH_KEY.encode())).start()
    try:
        record(
            0.05,
            on_batch=batches.append,
            argv=fake_mic(delay=0.6, hang=10),
            keys=keys,
            stream=io.StringIO(),
        )
    finally:
        keys.close()
        os.close(leader)

    assert [len(b) / SAMPLE_RATE for b in batches] == [pytest.approx(3.5, abs=CHUNK_S)]


def test_record_missing_recorder():
    with pytest.raises(RecordingError, match="install SoX"):
        record(0.05, argv=["no-such-recorder-xyz"], keys=io.StringIO())


def test_record_failing_recorder():
    script = "import sys; sys.stderr.write('no default audio device'); sys.exit(2)"

    with pytest.raises(RecordingError, match="no default audio device"):
        record(0.05, argv=[sys.executable, "-c", script], keys=io.StringIO())

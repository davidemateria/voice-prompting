from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from voice_prompting import dictation
from voice_prompting.config import Config
from voice_prompting.constants import SAMPLE_RATE
from voice_prompting.dictation import (
    BatchTranscriber,
    DictationError,
    dictate,
    preload_cuda_libraries,
    resolve_device,
)


class FakeWhisper:
    """Returns two segments per batch, half-way and at the end of the audio."""

    def __init__(self):
        self.prompts = []

    def transcribe(self, audio, **kwargs):
        self.prompts.append(kwargs["initial_prompt"])
        n = len(self.prompts)
        seconds = len(audio) / SAMPLE_RATE
        segments = [
            SimpleNamespace(text=f" part{n}", end=seconds / 2),
            SimpleNamespace(text=" end.", end=seconds),
        ]
        return iter(segments), None


def batch(seconds=1.0):
    return np.zeros(int(seconds * SAMPLE_RATE), np.float32)


def test_cpu_needs_no_gpu_check():
    assert resolve_device("cpu") == "cpu"


def test_preload_without_gpu_extra(monkeypatch):
    monkeypatch.setattr(dictation, "CUDA_LIBRARY_PACKAGES", ("no_such_package_xyz",))

    assert not preload_cuda_libraries()


def test_auto_falls_back_to_cpu_and_cuda_fails_without_libraries(monkeypatch):
    monkeypatch.setattr(dictation, "preload_cuda_libraries", lambda: False)

    assert resolve_device("auto") == "cpu"
    with pytest.raises(DictationError, match="gpu extra"):
        resolve_device("cuda")


def test_batches_are_transcribed_in_order_with_previous_text_as_context():
    model = FakeWhisper()
    transcriber = BatchTranscriber(lambda: (model, "cuda"))
    transcriber.submit(batch(2.0))
    transcriber.submit(batch(1.0))

    assert transcriber.finish() == "part1 end. part2 end."
    assert model.prompts == [None, "part1 end."]
    assert transcriber.device == "cuda"
    assert transcriber.done_s == pytest.approx(3.0)
    assert transcriber.progress == 1.0


def test_load_failure_is_raised_at_finish():
    def load():
        raise RuntimeError("CUDA driver too old")

    transcriber = BatchTranscriber(load)
    transcriber.submit(batch())

    with pytest.raises(DictationError, match="CUDA driver too old"):
        transcriber.finish()


def test_cancel_skips_batches_not_yet_started():
    model, release = FakeWhisper(), threading.Event()

    def load():
        release.wait()
        return model, "cpu"

    transcriber = BatchTranscriber(load)
    transcriber.submit(batch())
    transcriber.cancel()
    release.set()

    assert transcriber.finish() == ""
    assert model.prompts == []


def _fake_record(*batches, start=True):
    def record(_threshold, *, on_start, on_batch, **_):
        if start:
            on_start()
        for audio in batches:
            on_batch(audio)

    return record


def test_dictate_joins_the_batches(monkeypatch):
    monkeypatch.setattr(dictation, "load_model", lambda _: (FakeWhisper(), "cpu"))
    monkeypatch.setattr(dictation, "record", _fake_record(batch(), batch()))

    assert dictate(Config()) == "part1 end. part2 end."


@pytest.mark.parametrize("record", [_fake_record(start=False), _fake_record()])
def test_dictate_without_speech_returns_nothing(monkeypatch, record):
    monkeypatch.setattr(dictation, "load_model", lambda _: (FakeWhisper(), "cpu"))
    monkeypatch.setattr(dictation, "record", record)

    assert dictate(Config()) == ""

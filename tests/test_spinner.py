from __future__ import annotations

import io

from voice_prompting.constants import CLEAR_LINE, SPINNER_FRAMES
from voice_prompting.spinner import spinner


def test_draws_nothing_when_not_a_terminal():
    stream = io.StringIO()
    with spinner("Cleaning", stream):
        pass
    assert stream.getvalue() == ""


def test_draws_frame_and_label_then_clears_line(make_tty):
    stream = make_tty()
    with spinner("Cleaning", stream):
        pass
    output = stream.getvalue()
    assert any(f"{frame}  Cleaning" in output for frame in SPINNER_FRAMES)
    assert output.endswith(CLEAR_LINE)


def test_label_function_is_called_on_every_frame(make_tty):
    stream = make_tty()
    with spinner(lambda: "Transcribing · 42%", stream):
        pass
    assert "Transcribing · 42%" in stream.getvalue()

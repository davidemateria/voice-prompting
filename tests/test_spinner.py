from __future__ import annotations

import io

import pytest

from voice_prompting.constants import CLEAR_LINE, SPINNER_FRAMES
from voice_prompting.spinner import spinner


def test_draws_nothing_when_not_a_terminal():
    stream = io.StringIO()
    with spinner("Cleaning", stream):
        pass
    assert stream.getvalue() == ""


@pytest.mark.parametrize("label", ["Cleaning", lambda: "Cleaning"])
def test_draws_frame_and_label_then_clears_line(make_tty, label):
    stream = make_tty()
    with spinner(label, stream):
        pass
    output = stream.getvalue()
    assert any(f"{frame}  Cleaning" in output for frame in SPINNER_FRAMES)
    assert output.endswith(CLEAR_LINE)

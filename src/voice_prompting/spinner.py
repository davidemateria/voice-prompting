"""Animated status line shown while the tool works."""

from __future__ import annotations

import itertools
import sys
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, TextIO

from voice_prompting.constants import CLEAR_LINE, SPINNER_FRAMES, SPINNER_INTERVAL_S

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@contextmanager
def spinner(
    label: str | Callable[[], str], stream: TextIO | None = None
) -> Iterator[None]:
    """Show a rotating circle, ``label`` and the elapsed seconds until the block exits.

    Args:
        label: Text shown next to the circle, or a function returning it, called on every frame.
        stream: Terminal to draw on; defaults to stderr. Nothing is drawn if it is not a terminal.
    """
    out = stream or sys.stderr
    if not out.isatty():
        yield
        return
    stop = threading.Event()
    start = time.monotonic()

    def text() -> str:
        return label if isinstance(label, str) else label()

    def draw() -> None:
        for frame in itertools.cycle(SPINNER_FRAMES):
            out.write(
                f"{CLEAR_LINE}  {frame}  {text()} · {time.monotonic() - start:.1f} s"
            )
            out.flush()
            if stop.wait(SPINNER_INTERVAL_S):
                return

    thread = threading.Thread(target=draw, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        out.write(CLEAR_LINE)
        out.flush()

"""Command-line entry points: ``voice-prompting`` and ``voice-prompting-edit``."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from voice_prompting.cleanup import CleanupError, clean
from voice_prompting.config import ConfigError, load_config
from voice_prompting.constants import EDIT_COMMAND_NAME, FALLBACK_EDITOR
from voice_prompting.dictation import DictationError, dictate
from voice_prompting.spinner import spinner

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from voice_prompting.cleanup import Runner
    from voice_prompting.config import Config


def llm_label(command: Sequence[str]) -> str:
    """Return the ``--model`` value of ``command``, else its program name."""
    if "--model" in command[:-1]:
        return command[list(command).index("--model") + 1]
    return command[0]


def is_prompt_file(path: Path) -> bool:
    """Return whether ``path`` lies in the temp directory, where harnesses write the prompt."""
    return path.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())


def open_real_editor(path: Path) -> int:
    """Open ``path`` in ``$EDITOR`` (``vi`` if unset or pointing back here) and return its exit code."""
    editor = os.environ.get("EDITOR", "")
    if not editor or EDIT_COMMAND_NAME in editor:
        editor = FALLBACK_EDITOR
    argv = [*shlex.split(editor), str(path)]
    return subprocess.run(argv, check=False).returncode  # noqa: S603 - runs the user's own $EDITOR


def report_failure(exc: Exception) -> int:
    """Show why the prompt was left unchanged, wait for Enter on a terminal, and return 1."""
    sys.stderr.write(
        f"\nvoice-prompting: {exc}\nYour prompt is unchanged. Press Enter to go back."
    )
    sys.stderr.flush()
    if sys.stdin.isatty():
        sys.stdin.readline()
    return 1


def box_note(typed: str) -> str:
    """Describe the text already in the prompt box for the recorder line."""
    words = len(typed.split())
    if not words:
        return ""
    return f"{words} word{'s' if words > 1 else ''} in the box"


def edit_file(
    path: Path,
    config: Config,
    run: Runner = subprocess.run,
    listen: Callable[[Config, str], str] = dictate,
) -> int:
    """Dictate into the prompt stored in ``path``: the cleaned dictation goes after the text in it.

    The text already in the file is never changed; it is passed to the cleanup as context only.

    Args:
        path: Prompt file handed over by the harness.
        config: Dictation and cleanup settings.
        run: Process runner, injectable for tests.
        listen: Records and transcribes speech, given a note about the box; injectable for tests.

    Returns:
        0 on success or when nothing was said, 1 if dictation or cleanup failed or was cancelled
        (the file is then left untouched).
    """
    typed = path.read_text(encoding="utf-8")
    try:
        spoken = listen(config, box_note(typed))
    except DictationError as exc:
        return report_failure(exc)
    except KeyboardInterrupt:
        return 1
    if not spoken:
        return 0
    try:
        with spinner(f"Cleaning with {llm_label(config.llm_command)}"):
            cleaned = clean(spoken, config, run, typed)
    except CleanupError as exc:
        return report_failure(exc)
    if typed.strip() and not typed.endswith("\n"):
        typed += "\n"
    path.write_text(f"{typed if typed.strip() else ''}{cleaned}\n", encoding="utf-8")
    return 0


def edit_main(
    argv: Sequence[str] | None = None,
    run: Runner = subprocess.run,
    listen: Callable[[Config, str], str] = dictate,
) -> int:
    """Act as a harness's external editor: dictate into the prompt file it hands over.

    Files outside the temp directory (e.g. a ``CLAUDE.md`` opened by ``/memory``) go to the real editor.
    Without a terminal (a process started by the agent) nothing is touched.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
        run: Process runner, injectable for tests.
        listen: Records and transcribes speech, injectable for tests.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog=EDIT_COMMAND_NAME,
        description="External-editor command for coding agents: dictates into the prompt in FILE.",
    )
    parser.add_argument("file", type=Path, help="prompt file written by the harness")
    path = parser.parse_args(argv).file
    if not sys.stdin.isatty():
        sys.stderr.write(
            f"{EDIT_COMMAND_NAME}: no terminal attached, leaving {path} unchanged\n"
        )
        return 1
    if not is_prompt_file(path):
        return open_real_editor(path)
    try:
        config = load_config()
    except ConfigError as exc:
        return report_failure(exc)
    return edit_file(path, config, run, listen)


def main(
    argv: Sequence[str] | None = None,
    run: Runner = subprocess.run,
    listen: Callable[[Config, str], str] = dictate,
) -> int:
    """Run ``voice-prompting clean`` (text on stdin) or ``dictate`` (microphone); print the cleaned prompt.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
        run: Process runner, injectable for tests.
        listen: Records and transcribes speech, injectable for tests.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="voice-prompting", description="Clean dictated prompts with your LLM CLI."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "clean", help="read dictated text on stdin and print the cleaned English prompt"
    )
    commands.add_parser(
        "dictate", help="record the microphone and print the cleaned English prompt"
    )
    command = parser.parse_args(argv).command
    try:
        config = load_config()
        text = listen(config, "") if command == "dictate" else sys.stdin.read()
        with spinner(f"Cleaning with {llm_label(config.llm_command)}"):
            cleaned = clean(text, config, run)
    except (CleanupError, ConfigError, DictationError) as exc:
        sys.stderr.write(f"voice-prompting: {exc}\n")
        return 1
    if cleaned.strip():
        sys.stdout.write(f"{cleaned}\n")
    return 0

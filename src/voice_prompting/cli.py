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

from voice_prompting.cleanup import clean
from voice_prompting.config import load_config
from voice_prompting.constants import EDIT_COMMAND_NAME, FALLBACK_EDITOR
from voice_prompting.errors import VoicePromptingError
from voice_prompting.setup_agents import setup_agents
from voice_prompting.spinner import spinner

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from voice_prompting.cleanup import Runner
    from voice_prompting.config import Config

    Listener = Callable[[Config, str], str]


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
    return f"{words} word{'s' if words > 1 else ''} in the box" if words else ""


def dictate(config: Config, note: str) -> str:
    """Record and transcribe speech; imports the heavy dictation stack only when called."""
    from voice_prompting import dictation  # noqa: PLC0415 - numpy takes 0.6 s

    return dictation.dictate(config, note)


def clean_with_spinner(text: str, config: Config, run: Runner, typed: str = "") -> str:
    """Clean ``text`` while showing which model is working."""
    with spinner(f"Cleaning with {llm_label(config.llm_command)}"):
        return clean(text, config, run, typed)


def edit_file(
    path: Path,
    config: Config,
    run: Runner = subprocess.run,
    listen: Listener = dictate,
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
        if not spoken:
            return 0
        cleaned = clean_with_spinner(spoken, config, run, typed)
    except VoicePromptingError as exc:
        return report_failure(exc)
    except KeyboardInterrupt:
        return 1
    if not typed.strip():
        typed = ""
    elif not typed.endswith("\n"):
        typed += "\n"
    path.write_text(f"{typed}{cleaned}\n", encoding="utf-8")
    return 0


def edit_main(
    argv: Sequence[str] | None = None,
    run: Runner = subprocess.run,
    listen: Listener = dictate,
) -> int:
    """Act as a harness's external editor: dictate into the prompt file it hands over.

    Files outside the temp directory (e.g. a ``CLAUDE.md`` opened by ``/memory``) go to the real editor.
    Without a terminal (e.g. ``git rebase --continue`` run by the agent) the file is accepted as is.

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
        return 0
    if not is_prompt_file(path):
        return open_real_editor(path)
    try:
        config = load_config()
    except VoicePromptingError as exc:
        return report_failure(exc)
    return edit_file(path, config, run, listen)


def main(
    argv: Sequence[str] | None = None,
    run: Runner = subprocess.run,
    listen: Listener = dictate,
) -> int:
    """Run ``voice-prompting clean`` (text on stdin), ``dictate`` (microphone) or ``setup`` (agent settings).

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
    commands.add_parser(
        "setup", help="make Ctrl+G in Claude Code and pi start voice-prompting"
    )
    command = parser.parse_args(argv).command
    try:
        if command == "setup":
            sys.stdout.write("\n".join(setup_agents()) + "\n")
            return 0
        config = load_config()
        text = listen(config, "") if command == "dictate" else sys.stdin.read()
        cleaned = clean_with_spinner(text, config, run)
    except VoicePromptingError as exc:
        sys.stderr.write(f"voice-prompting: {exc}\n")
        return 1
    except KeyboardInterrupt:
        return 130
    if cleaned.strip():
        sys.stdout.write(f"{cleaned}\n")
    return 0

"""Clean a dictated transcript with the user's own LLM command-line tool."""

from __future__ import annotations

import subprocess
from importlib.resources import files
from typing import TYPE_CHECKING

from voice_prompting.constants import GLOSSARY_HEADING, SYSTEM_PROMPT_PLACEHOLDER
from voice_prompting.errors import CleanupError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from voice_prompting.config import Config

    Runner = Callable[..., subprocess.CompletedProcess[str]]


def system_prompt(glossary_file: Path | None = None) -> str:
    """Return the cleanup instructions, followed by the user's glossary if one is set.

    Args:
        glossary_file: Text file listing terms the speaker uses.
    """
    prompt = (files("voice_prompting") / "system_prompt.md").read_text(encoding="utf-8")
    if glossary_file is None:
        return prompt
    glossary = glossary_file.read_text(encoding="utf-8")
    return f"{prompt}\n{GLOSSARY_HEADING}\n{glossary}"


def build_invocation(
    command: Sequence[str], prompt: str, text: str, typed: str = ""
) -> tuple[list[str], str]:
    """Return the argv and stdin for one cleanup call.

    Args:
        command: LLM command template from the configuration.
        prompt: Cleanup instructions and glossary.
        text: Dictated text to clean.
        typed: Text already in the prompt box, sent as context only.
    """
    transcript = f"<transcript>{text}</transcript>"
    if typed:
        transcript = f"<typed>{typed}</typed>\n{transcript}"
    if any(SYSTEM_PROMPT_PLACEHOLDER in arg for arg in command):
        return [
            arg.replace(SYSTEM_PROMPT_PLACEHOLDER, prompt) for arg in command
        ], transcript
    return list(command), f"{prompt}\n\n{transcript}"


def clean(
    text: str, config: Config, run: Runner = subprocess.run, typed: str = ""
) -> str:
    """Rewrite dictated text as a clean English prompt.

    Args:
        text: Raw dictated text, in any language.
        config: LLM command, glossary and timeout.
        run: Process runner, injectable for tests.
        typed: Text already in the prompt box; context for the cleanup, never part of the output.

    Returns:
        The cleaned prompt, or ``text`` unchanged when it is blank.

    Raises:
        CleanupError: If the command is missing, times out, fails or prints nothing.
    """
    if not text.strip():
        return text
    argv, stdin = build_invocation(
        config.llm_command,
        system_prompt(config.glossary_file),
        text.strip(),
        typed.strip(),
    )
    try:
        proc = run(
            argv,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=config.timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise CleanupError(f"command not found: {argv[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CleanupError(
            f"{argv[0]} did not answer within {config.timeout_s:g} s"
        ) from exc
    if proc.returncode != 0:
        raise CleanupError(
            f"{argv[0]} exited with code {proc.returncode}: {proc.stderr.strip()}"
        )
    if not (cleaned := proc.stdout.strip()):
        raise CleanupError(f"{argv[0]} returned no text")
    return cleaned

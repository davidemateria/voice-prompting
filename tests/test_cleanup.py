from __future__ import annotations

import subprocess

import pytest

from voice_prompting.cleanup import CleanupError, clean, system_prompt
from voice_prompting.config import Config
from voice_prompting.constants import GLOSSARY_HEADING, SYSTEM_PROMPT_PLACEHOLDER


def test_default_command_passes_prompt_as_argument_and_transcript_on_stdin(make_runner):
    runner = make_runner()

    assert clean("um run the tests", Config(), runner) == "Clean prompt."

    argv, kwargs = runner.calls[0]
    assert argv[:2] == ["claude", "-p"]
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5-5"
    assert argv[argv.index("--effort") + 1] == "low"
    assert argv[argv.index("--system-prompt") + 1] == system_prompt()
    assert SYSTEM_PROMPT_PLACEHOLDER not in argv
    assert kwargs["input"] == "<transcript>um run the tests</transcript>"


def test_command_without_placeholder_gets_prompt_on_stdin(make_runner):
    runner = make_runner()

    clean("hello", Config(llm_command=("mycli", "run")), runner)

    argv, kwargs = runner.calls[0]
    assert argv == ["mycli", "run"]
    assert kwargs["input"] == f"{system_prompt()}\n\n<transcript>hello</transcript>"


def test_no_glossary_by_default():
    assert GLOSSARY_HEADING not in system_prompt()


def test_glossary_file_is_appended(tmp_path):
    glossary = tmp_path / "terms.txt"
    glossary.write_text("FooBarQR\n", encoding="utf-8")

    prompt = system_prompt(glossary)

    assert prompt.endswith(f"{GLOSSARY_HEADING}\nFooBarQR\n")


def test_blank_text_skips_llm(make_runner):
    runner = make_runner()

    assert clean("  \n", Config(), runner) == "  \n"
    assert runner.calls == []


@pytest.mark.parametrize(
    ("runner_kwargs", "message"),
    [
        (
            {"returncode": 1, "stderr": "not logged in"},
            "exited with code 1: not logged in",
        ),
        ({"stdout": "  \n"}, "returned no text"),
        ({"exc": FileNotFoundError()}, "command not found: claude"),
        (
            {"exc": subprocess.TimeoutExpired("claude", 60)},
            "did not answer within 60 s",
        ),
    ],
)
def test_failures_raise_cleanup_error(make_runner, runner_kwargs, message):
    with pytest.raises(CleanupError, match=message):
        clean("hello", Config(), make_runner(**runner_kwargs))

from __future__ import annotations

import io
import sys
import tempfile

import pytest

from voice_prompting.cli import (
    box_note,
    edit_file,
    edit_main,
    is_prompt_file,
    llm_label,
    main,
)
from voice_prompting.config import Config
from voice_prompting.constants import DEFAULT_LLM_COMMAND
from voice_prompting.dictation import DictationError

TYPED = "Fix `parse_args` in cli.py\n"


@pytest.fixture
def prompt_file(tmp_path):
    path = tmp_path / "prompt.md"
    path.write_text(TYPED, encoding="utf-8")
    return path


def say(text):
    return lambda _config, _note: text


def _fail(exc):
    def listen(_config, _note):
        raise exc

    return listen


def test_llm_label():
    assert llm_label(DEFAULT_LLM_COMMAND) == "claude-sonnet-5-5"
    assert llm_label(("mycli", "run")) == "mycli"


def test_is_prompt_file(tmp_path, monkeypatch):
    assert is_prompt_file(tmp_path / "prompt.md")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "elsewhere"))
    assert not is_prompt_file(tmp_path / "CLAUDE.md")


def test_box_note():
    assert box_note(TYPED) == "4 words in the box"
    assert box_note("one") == "1 word in the box"
    assert box_note(" \n") == ""


def test_dictation_goes_on_a_new_line_after_untouched_typed_text(
    prompt_file, make_runner
):
    runner = make_runner()
    notes = []

    def listen(_config, note):
        notes.append(note)
        return "and add a test"

    assert edit_file(prompt_file, Config(), runner, listen) == 0
    assert prompt_file.read_text(encoding="utf-8") == TYPED + "Clean prompt.\n"
    assert notes == ["4 words in the box"]
    assert runner.calls[0][1]["input"] == (
        "<typed>Fix `parse_args` in cli.py</typed>\n<transcript>and add a test</transcript>"
    )


@pytest.mark.parametrize(
    ("typed", "kept", "context"),
    [("\n", "", ""), ("Fix it", "Fix it\n", "<typed>Fix it</typed>\n")],
)
def test_blank_or_unterminated_typed_text(tmp_path, make_runner, typed, kept, context):
    path, runner = tmp_path / "prompt.md", make_runner()
    path.write_text(typed, encoding="utf-8")

    assert edit_file(path, Config(), runner, say("now")) == 0
    assert path.read_text(encoding="utf-8") == f"{kept}Clean prompt.\n"
    assert runner.calls[0][1]["input"] == f"{context}<transcript>now</transcript>"


@pytest.mark.parametrize(
    "listen",
    [say(""), _fail(DictationError("no microphone")), _fail(KeyboardInterrupt())],
)
def test_nothing_said_or_failure_leaves_typed_text_untouched(
    prompt_file, make_runner, listen
):
    runner = make_runner()

    assert edit_file(prompt_file, Config(), runner, listen) in (0, 1)
    assert prompt_file.read_text(encoding="utf-8") == TYPED
    assert runner.calls == []


def test_cleanup_failure_leaves_typed_text_untouched(prompt_file, make_runner, capsys):
    runner = make_runner(returncode=1, stderr="boom")

    assert edit_file(prompt_file, Config(), runner, say("and more")) == 1
    assert prompt_file.read_text(encoding="utf-8") == TYPED
    assert "unchanged" in capsys.readouterr().err


def test_edit_main_without_terminal_touches_nothing(
    prompt_file, make_runner, monkeypatch
):
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    runner = make_runner()

    assert edit_main([str(prompt_file)], runner) == 0
    assert prompt_file.read_text(encoding="utf-8") == TYPED
    assert runner.calls == []


def test_edit_main_dictates_into_prompt_in_temp_dir(
    prompt_file, make_runner, make_tty, monkeypatch
):
    monkeypatch.setattr(sys, "stdin", make_tty())

    assert edit_main([str(prompt_file)], make_runner(), say("more")) == 0
    assert prompt_file.read_text(encoding="utf-8") == TYPED + "Clean prompt.\n"


def test_edit_main_sends_other_files_to_real_editor(
    tmp_path, make_runner, make_tty, monkeypatch
):
    user_file = tmp_path / "CLAUDE.md"
    user_file.write_text("# notes\n", encoding="utf-8")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "elsewhere"))
    monkeypatch.setattr(sys, "stdin", make_tty())
    monkeypatch.setenv("EDITOR", "true")
    runner = make_runner()

    assert edit_main([str(user_file)], runner) == 0
    assert user_file.read_text(encoding="utf-8") == "# notes\n"
    assert runner.calls == []


def test_main_clean_reads_stdin(make_runner, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("um hello"))

    assert main(["clean"], make_runner()) == 0
    assert capsys.readouterr().out == "Clean prompt.\n"


def test_main_clean_reports_failure(make_runner, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("um hello"))

    assert main(["clean"], make_runner(returncode=2, stderr="quota")) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "quota" in captured.err


def test_main_dictate_prints_cleaned_prompt(make_runner, capsys):
    runner = make_runner()

    assert main(["dictate"], runner, say("um hello")) == 0
    assert capsys.readouterr().out == "Clean prompt.\n"
    assert runner.calls[0][1]["input"] == "<transcript>um hello</transcript>"


def test_main_dictate_ctrl_c_exits_quietly(make_runner, capsys):
    assert main(["dictate"], make_runner(), _fail(KeyboardInterrupt())) == 130
    assert capsys.readouterr().err == ""


def test_ctrl_c_during_cleanup_keeps_prompt(prompt_file, make_runner):
    runner = make_runner(exc=KeyboardInterrupt())

    assert edit_file(prompt_file, Config(), runner, say("more")) == 1
    assert prompt_file.read_text(encoding="utf-8") == TYPED

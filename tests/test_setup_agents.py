from __future__ import annotations

import json

import pytest

from voice_prompting.errors import SetupError
from voice_prompting.setup_agents import setup_agents


def test_sets_up_installed_agents_and_keeps_other_settings(tmp_path):
    claude = tmp_path / ".claude" / "settings.json"
    claude.parent.mkdir()
    claude.write_text('{"model": "opus", "env": {"DEBUG": "1"}}', encoding="utf-8")
    (tmp_path / ".pi" / "agent").mkdir(parents=True)

    report = setup_agents(tmp_path)

    assert json.loads(claude.read_text(encoding="utf-8")) == {
        "model": "opus",
        "env": {"DEBUG": "1", "VISUAL": "voice-prompting-edit"},
    }
    assert json.loads(claude.with_name("settings.json.bak").read_text()) == {
        "model": "opus",
        "env": {"DEBUG": "1"},
    }
    pi = tmp_path / ".pi" / "agent" / "settings.json"
    assert json.loads(pi.read_text()) == {"externalEditor": "voice-prompting-edit"}
    assert all("set up in" in line for line in report)


def test_second_run_and_missing_agents_change_nothing(tmp_path):
    (tmp_path / ".claude").mkdir()
    setup_agents(tmp_path)
    backup = tmp_path / ".claude" / "settings.json.bak"

    assert setup_agents(tmp_path) == [
        "Claude Code: already set up",
        "pi: not installed, skipped",
    ]
    assert not backup.exists()


def test_rejects_settings_that_are_not_an_object(tmp_path):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.json").write_text("[]", encoding="utf-8")

    with pytest.raises(SetupError, match="not a JSON object"):
        setup_agents(tmp_path)

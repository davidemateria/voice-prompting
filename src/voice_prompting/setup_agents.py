"""Register ``voice-prompting-edit`` as the external editor of the installed coding agents."""

from __future__ import annotations

import json
from pathlib import Path

from voice_prompting.constants import EDIT_COMMAND_NAME
from voice_prompting.errors import SetupError


def _claude(settings: dict) -> None:
    settings.setdefault("env", {})["VISUAL"] = EDIT_COMMAND_NAME


def _pi(settings: dict) -> None:
    settings["externalEditor"] = EDIT_COMMAND_NAME


AGENTS = {
    "Claude Code": (Path(".claude/settings.json"), _claude),
    "pi": (Path(".pi/agent/settings.json"), _pi),
}


def setup_agents(home: Path | None = None) -> list[str]:
    """Add the external-editor setting to every agent whose settings folder exists.

    A settings file that changes is first copied to ``<name>.bak``.

    Args:
        home: Home directory; defaults to the user's.

    Returns:
        One line per agent, saying what was done.

    Raises:
        SetupError: If a settings file is not valid JSON or not a JSON object.
    """
    home = home or Path.home()
    report = []
    for agent, (relative, apply) in AGENTS.items():
        path = home / relative
        if not path.parent.is_dir():
            report.append(f"{agent}: not installed, skipped")
            continue
        old = path.read_text(encoding="utf-8") if path.is_file() else "{}"
        try:
            settings = json.loads(old)
        except json.JSONDecodeError as exc:
            raise SetupError(f"{path}: {exc}") from exc
        if not isinstance(settings, dict):
            raise SetupError(f"{path} is not a JSON object")
        before = json.dumps(settings)
        apply(settings)
        if json.dumps(settings) == before:
            report.append(f"{agent}: already set up")
            continue
        if path.is_file():
            path.with_name(f"{path.name}.bak").write_text(old, encoding="utf-8")
        path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        report.append(f"{agent}: set up in {path} (restart it)")
    return report

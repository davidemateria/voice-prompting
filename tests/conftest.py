from __future__ import annotations

import io
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path


@dataclass
class FakeRunner:
    """Stands in for ``subprocess.run``: records calls and returns a canned result or raises."""

    stdout: str = "Clean prompt.\n"
    returncode: int = 0
    stderr: str = ""
    exc: Exception | None = None
    calls: list[tuple[list[str], dict[str, Any]]] = field(default_factory=list)

    def __call__(
        self, argv: Sequence[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(argv), kwargs))
        if self.exc is not None:
            raise self.exc
        return subprocess.CompletedProcess(
            list(argv), self.returncode, self.stdout, self.stderr
        )


class FakeTTY(io.StringIO):
    """In-memory stream that reports itself as a terminal."""

    def isatty(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point XDG_CONFIG_HOME at an empty directory so tests never read the user's config."""
    config_home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    return config_home


@pytest.fixture
def make_runner() -> Callable[..., FakeRunner]:
    return FakeRunner


@pytest.fixture
def make_tty() -> Callable[..., FakeTTY]:
    return FakeTTY

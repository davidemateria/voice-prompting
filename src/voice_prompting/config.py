"""User configuration, read from ``$XDG_CONFIG_HOME/voice-prompting/config.toml``."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from voice_prompting.constants import (
    CONFIG_DIR_NAME,
    CONFIG_FILE_NAME,
    DEFAULT_LLM_COMMAND,
    DEFAULT_SPEECH_THRESHOLD,
    DEFAULT_TIMEOUT_S,
    DEFAULT_WHISPER_DEVICE,
    DEFAULT_WHISPER_MODEL,
    WHISPER_DEVICES,
)
from voice_prompting.errors import ConfigError


@dataclass(frozen=True)
class Config:
    """Settings for dictation and cleanup.

    Attributes:
        llm_command: Command that runs the LLM non-interactively. ``{system_prompt}`` inside any
            argument is replaced by the cleanup instructions; without it, the instructions are sent
            on stdin before the transcript.
        glossary_file: Text file listing terms the speaker uses, appended to the cleanup instructions.
        timeout_s: Seconds to wait for the LLM before giving up.
        whisper_model: faster-whisper model name or path used to transcribe dictation.
        whisper_device: ``auto`` (GPU when usable, else CPU), ``cpu`` or ``cuda``.
        speech_threshold: Microphone RMS level (full scale = 1) that counts as speech; a recording
            that never reaches it is dropped instead of transcribed.
    """

    llm_command: tuple[str, ...] = DEFAULT_LLM_COMMAND
    glossary_file: Path | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    whisper_model: str = DEFAULT_WHISPER_MODEL
    whisper_device: str = DEFAULT_WHISPER_DEVICE
    speech_threshold: float = DEFAULT_SPEECH_THRESHOLD


def default_config_path() -> Path:
    """Return the config file path, honouring ``XDG_CONFIG_HOME``."""
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / CONFIG_DIR_NAME / CONFIG_FILE_NAME


def load_config(path: Path | None = None) -> Config:
    """Load the configuration; a missing file gives the defaults.

    Args:
        path: Config file to read; defaults to :func:`default_config_path`.

    Returns:
        The parsed configuration. A relative ``glossary_file`` is resolved against the config directory.

    Raises:
        ConfigError: If the file is not valid TOML, has unknown keys or values of the wrong type,
            or names a glossary file that does not exist.
    """
    path = path or default_config_path()
    if not path.is_file():
        return Config()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if unknown := data.keys() - {f.name for f in fields(Config)}:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")

    command = data.get("llm_command", list(DEFAULT_LLM_COMMAND))
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(arg, str) for arg in command)
    ):
        raise ConfigError(f"{path}: llm_command must be a non-empty list of strings")
    glossary = data.get("glossary_file")
    if glossary is not None and not isinstance(glossary, str):
        raise ConfigError(f"{path}: glossary_file must be a path string")
    glossary_file = path.parent / Path(glossary).expanduser() if glossary else None
    if glossary_file is not None and not glossary_file.is_file():
        raise ConfigError(f"{path}: glossary_file {glossary_file} does not exist")
    return Config(
        llm_command=tuple(command),
        glossary_file=glossary_file,
        timeout_s=_positive(data, "timeout_s", DEFAULT_TIMEOUT_S, path),
        whisper_model=_string(data, "whisper_model", DEFAULT_WHISPER_MODEL, path),
        whisper_device=_device(data, path),
        speech_threshold=_positive(
            data, "speech_threshold", DEFAULT_SPEECH_THRESHOLD, path
        ),
    )


def _positive(data: dict[str, object], key: str, default: float, path: Path) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ConfigError(f"{path}: {key} must be a positive number")
    return float(value)


def _device(data: dict[str, object], path: Path) -> str:
    value = data.get("whisper_device", DEFAULT_WHISPER_DEVICE)
    if value not in WHISPER_DEVICES:
        raise ConfigError(
            f"{path}: whisper_device must be one of {sorted(WHISPER_DEVICES)}"
        )
    return str(value)


def _string(data: dict[str, object], key: str, default: str, path: Path) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{path}: {key} must be a non-empty string")
    return value

"""Errors shown to the user as a message instead of a traceback."""

from __future__ import annotations


class VoicePromptingError(Exception):
    """Base of every expected failure."""


class ConfigError(VoicePromptingError):
    """Raised when the configuration file is invalid."""


class RecordingError(VoicePromptingError):
    """Raised when the microphone cannot be recorded."""


class DictationError(VoicePromptingError):
    """Raised when recording or transcription fails."""


class CleanupError(VoicePromptingError):
    """Raised when the LLM command fails; callers keep the original text."""


class SetupError(VoicePromptingError):
    """Raised when an agent's settings file cannot be updated."""

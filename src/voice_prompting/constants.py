"""Package-wide constants."""

from __future__ import annotations

CONFIG_DIR_NAME = "voice-prompting"
CONFIG_FILE_NAME = "config.toml"
CONFIG_KEYS = frozenset(
    {
        "llm_command",
        "glossary_file",
        "timeout_s",
        "whisper_model",
        "whisper_device",
        "speech_threshold",
    }
)

SYSTEM_PROMPT_PLACEHOLDER = "{system_prompt}"
GLOSSARY_HEADING = "Glossary of terms the speaker uses (correct spelling):"
DEFAULT_LLM_COMMAND = (
    "claude",
    "-p",
    "--model",
    "claude-sonnet-5-5",
    "--effort",
    "low",
    "--tools",
    "",
    "--strict-mcp-config",
    "--setting-sources",
    "",
    "--no-session-persistence",
    "--disable-slash-commands",
    "--system-prompt",
    SYSTEM_PROMPT_PLACEHOLDER,
)
DEFAULT_TIMEOUT_S = 60.0

DEFAULT_WHISPER_MODEL = "small"
DEFAULT_WHISPER_DEVICE = "auto"
WHISPER_DEVICES = frozenset({"auto", "cpu", "cuda"})
CUDA_LIBRARY_PACKAGES = ("nvidia.cublas", "nvidia.cudnn")
WHISPER_COMPUTE_TYPE = {"cpu": "int8", "cuda": "int8_float16"}
WHISPER_BEAM_SIZE = 1
DEFAULT_SPEECH_THRESHOLD = 0.03

SAMPLE_RATE = 16000
CHUNK_S = 0.1
RECORDER_ARGV = (
    "rec",
    "-q",
    "-t",
    "raw",
    "-b",
    "16",
    "-e",
    "signed-integer",
    "-c",
    "1",
    "-r",
    str(SAMPLE_RATE),
    "-",
)
TOGGLE_KEY = " "
FINISH_KEY = "\x07"  # Ctrl+G, the same key that opened the recorder
LEVEL_BAR_WIDTH = 8
LEVEL_FLOOR_DB = -60.0
LEVEL_CEIL_DB = -10.0
RECORDING_DOTS = "●○"

EDIT_COMMAND_NAME = "voice-prompting-edit"
FALLBACK_EDITOR = "vi"

SPINNER_FRAMES = "◐◓◑◒"
SPINNER_INTERVAL_S = 0.12
CLEAR_LINE = "\r\033[2K"

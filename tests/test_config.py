from __future__ import annotations

import pytest

from voice_prompting.config import Config, ConfigError, default_config_path, load_config


def test_default_path_honours_xdg_config_home(isolated_config):
    assert default_config_path() == isolated_config / "voice-prompting" / "config.toml"


def test_missing_file_gives_defaults(tmp_path):
    assert load_config(tmp_path / "absent.toml") == Config()


def test_full_file_is_parsed(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        'llm_command = ["mycli", "-p", "{system_prompt}"]\nglossary_file = "terms.txt"\ntimeout_s = 30\n'
        'whisper_model = "base"\nwhisper_device = "cuda"\nspeech_threshold = 0.05\n',
        encoding="utf-8",
    )

    assert load_config(path) == Config(
        llm_command=("mycli", "-p", "{system_prompt}"),
        glossary_file=tmp_path / "terms.txt",
        timeout_s=30.0,
        whisper_model="base",
        whisper_device="cuda",
        speech_threshold=0.05,
    )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("model = 'x'\n", "unknown keys"),
        ("llm_command = 'claude -p'\n", "llm_command"),
        ("llm_command = []\n", "llm_command"),
        ("glossary_file = 3\n", "glossary_file"),
        ("timeout_s = -1\n", "timeout_s"),
        ("timeout_s = true\n", "timeout_s"),
        ("llm_command = [\n", "config.toml"),
        ("whisper_model = ''\n", "whisper_model"),
        ("whisper_device = 1\n", "whisper_device"),
        ("whisper_device = 'gpu'\n", "whisper_device"),
        ("speech_threshold = 0\n", "speech_threshold"),
        ("speech_threshold = '0.1'\n", "speech_threshold"),
    ],
)
def test_invalid_file_raises(tmp_path, content, message):
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(path)

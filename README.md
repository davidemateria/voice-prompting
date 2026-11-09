# voice-prompting

Dictate prompts to a terminal coding agent (Claude Code, pi) and get clean English text in its prompt box. Speech is transcribed on your machine with [faster-whisper](https://github.com/SYSTRAN/faster-whisper); the transcript is then cleaned by your own LLM command-line tool (by default `claude -p`): punctuation fixed, filler words and false starts removed, self-corrections applied, any language translated to English. Nothing is sent to the agent until you press Enter.

It hooks into the key your agent already has for "edit the prompt in an external editor" (Ctrl+G), so it needs no plugin.

## Install

Follow the steps for your platform, then the common steps. Every step ends with a check.

### 1. Platform

**Windows 11 (through WSL2)**

1. In PowerShell as administrator: `wsl --install -d Ubuntu-24.04`, reboot, open "Ubuntu" from the Start menu and create your user.
2. Microphone: Windows Settings › Privacy & security › Microphone: turn on **Microphone access** and **Let desktop apps access your microphone**.
3. In Ubuntu: `sudo apt update && sudo apt install -y sox libsox-fmt-pulse git curl`
4. Check: `ls /mnt/wslg/PulseServer` prints the path (WSLg, the audio bridge to Windows, is running).
5. NVIDIA GPU (optional, makes transcription about 2× faster): install the normal **Windows** NVIDIA driver; never install a Linux NVIDIA driver inside WSL. Check in Ubuntu: `nvidia-smi` lists your GPU.

**Ubuntu / Debian desktop**

1. `sudo apt update && sudo apt install -y sox libsox-fmt-pulse git curl`
2. NVIDIA GPU (optional): `sudo ubuntu-drivers install`, reboot. Check: `nvidia-smi` lists your GPU.

**macOS** (untested)

1. Install [Homebrew](https://brew.sh), then `brew install sox git`.
2. The first recording makes macOS ask whether your terminal may use the microphone: allow it.

### 2. Common

1. **Microphone check.** Run `rec -q /tmp/t.wav trim 0 3` and speak for 3 seconds, then `sox /tmp/t.wav -n stat 2>&1 | grep RMS`. While speaking, the RMS amplitude should be about 0.01 or more; values like 0.00001 mean silence reaches Linux (see [Troubleshooting](#troubleshooting)).
2. **uv** (Python tool manager; it also downloads the right Python): `curl -LsSf https://astral.sh/uv/install.sh | sh`, then open a new terminal. Check: `uv --version`.
3. **Claude Code**: `curl -fsSL https://claude.ai/install.sh | bash`, then run `claude` once and log in. A Claude subscription or API key is needed. Check: `claude -p "say ok"` prints an answer. (Another LLM CLI works too; see [Configuration](#configuration).)
4. **voice-prompting**:
   ```bash
   # With an NVIDIA GPU (adds CUDA libraries, about 1 GB):
   uv tool install 'voice-prompting[gpu] @ git+https://github.com/davidemateria/voice-prompting'
   # Without:
   uv tool install git+https://github.com/davidemateria/voice-prompting
   ```
   If the shell then can't find `voice-prompting`, run `uv tool update-shell` and open a new terminal.
5. **Test it**: `voice-prompting dictate`. Wait for `Press Space to talk`, press Space, say a sentence, press Ctrl+G. The cleaned English sentence is printed. The first run downloads the Whisper model (464 MB) to `~/.cache/huggingface`.

### 3. Connect your agent

Add one setting, merged into the JSON object already in the file (create the file if it doesn't exist), then restart the agent.

| Agent | File | Setting |
|---|---|---|
| Claude Code | `~/.claude/settings.json` | `"env": { "VISUAL": "voice-prompting-edit" }` |
| pi | `~/.pi/agent/settings.json` | `"externalEditor": "voice-prompting-edit"` |

Check: in the agent, press Ctrl+G. You should see `○  Starting microphone…`, then `○  Press Space to talk`. Press Ctrl+G again to go back.

Codex CLI, Gemini CLI and opencode use the same mechanism through `$VISUAL` / `$EDITOR` (e.g. `alias codex='VISUAL=voice-prompting-edit codex'`), but are untested.

## Use

In the agent's prompt box:

| Key | Effect |
|---|---|
| **Ctrl+G** | Opens the recorder. Wait for `Press Space to talk` (the microphone needs about 1 s). |
| **Space** | Starts recording: `●  Recording ▮▮▮▮▯▯▯▯ 4.1 s` |
| **Space** | Pauses: `‖  Paused 4.1 s · transcribed 100%`. Take your time; what you said is transcribed meanwhile. |
| **Space** | Resumes. Pause and resume as often as you like. |
| **Ctrl+G** | Finishes: transcribes the rest, cleans it, and puts it in the prompt box. |
| **Ctrl+C** | Cancels; the prompt box stays as it was. |

Then review the text and press Enter to send it.

- Speak any language, and switch between pauses if you like; the prompt comes out in English.
- Text you had already typed is never changed: the dictation is added on a new line after it, and the cleanup uses your text only as context.
- If nothing louder than room noise was heard, nothing is added.

## Configuration

Optional, in `~/.config/voice-prompting/config.toml`. Defaults:

```toml
llm_command = [                       # "{system_prompt}" is replaced by the cleanup instructions
  "claude", "-p", "--model", "claude-sonnet-5-5", "--effort", "low",
  "--tools", "", "--strict-mcp-config", "--setting-sources", "",
  "--no-session-persistence", "--disable-slash-commands",
  "--system-prompt", "{system_prompt}",
]
timeout_s = 60                        # seconds to wait for the LLM
# glossary_file = "~/terms.md"        # terms you use (names, jargon), spelled correctly
whisper_model = "small"               # tiny, base, small, medium, large-v3-turbo
whisper_device = "auto"               # auto (GPU if usable), cpu or cuda
speech_threshold = 0.03               # microphone level that counts as speech
```

- **Another LLM CLI**: any command that answers one prompt non-interactively. If no argument contains `"{system_prompt}"`, the instructions are sent on stdin before the transcript.
- **Glossary**: free text; listing project, package and function names helps the cleanup spell them right.
- **Speed**: on a laptop (Intel Core Ultra 7, NVIDIA RTX 500), `small` transcribes 11 s of speech in 3.9 s on the CPU and 1.9 s on the GPU; the Sonnet cleanup adds about 3 s.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Ctrl+G opens nano or vi | The setting isn't active: check the JSON, restart the agent. |
| You spoke but nothing appears | Microphone too quiet or silent: redo the microphone check; lower `speech_threshold`. |
| WSL: microphone silent after a reboot | In PowerShell: `wsl --shutdown`, then reopen Ubuntu. Check the Windows microphone settings from step 1. |
| `recorder not found: rec` | Install SoX (step 1). |
| Transcribing says CPU despite an NVIDIA GPU | Install with `[gpu]`; set `whisper_device = "cuda"` to see the reason as an error. |
| `exited with code 1` mentioning login | Run `claude` and log in. |

Other files the agent opens with Ctrl+G (for example `CLAUDE.md` from `/memory`) go to `$EDITOR`, or `vi`. Commands the agent runs, which have no terminal, are left untouched.

## Uninstall

```bash
uv tool uninstall voice-prompting
rm -rf ~/.cache/huggingface/hub/models--*faster-whisper* ~/.config/voice-prompting
```

Then remove the setting from step 3.

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

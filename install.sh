#!/bin/sh
# Installs voice-prompting and what it needs: SoX, uv, the tool itself, and the agent settings.
set -eu
REPO="git+https://github.com/davidemateria/voice-prompting"

if command -v apt-get >/dev/null; then
  sudo apt-get update && sudo apt-get install -y sox libsox-fmt-pulse git curl
elif command -v brew >/dev/null; then
  brew install sox git
else
  echo "Install SoX and git with your package manager, then rerun." >&2; exit 1
fi

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

SPEC="$REPO"
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  SPEC="voice-prompting[gpu] @ $REPO"
fi
uv tool install --force "$SPEC"
uv tool update-shell >/dev/null 2>&1 || true

voice-prompting setup
command -v claude >/dev/null || echo "Claude Code not found: curl -fsSL https://claude.ai/install.sh | bash, then run 'claude' and log in."
echo "Done. Open a new terminal and test with: voice-prompting dictate"

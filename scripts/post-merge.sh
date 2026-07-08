#!/bin/bash
set -eo pipefail

# Sync Python dependencies into .pythonlibs (Replit-managed, no venv)
if [ -f pyproject.toml ]; then
  uv pip install --target .pythonlibs/lib/python3.11/site-packages -r pyproject.toml 2>&1 | tail -n 3
fi

# Sync Node dependencies (server)
# npm can leave node_modules half-renamed (ENOTEMPTY) if a previous install
# was interrupted; on failure, wipe node_modules and retry once from scratch.
if [ -f package-lock.json ]; then
  if ! npm install --no-audit --no-fund 2>&1 | tail -n 3; then
    echo "npm install failed — wiping node_modules and retrying"
    rm -rf node_modules
    npm install --no-audit --no-fund 2>&1 | tail -n 3
  fi
fi

# Fail fast if the bot has a syntax error after merge
python -m py_compile main.py trade_modes.py trading_engine.py

echo "post-merge setup OK"

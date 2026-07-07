#!/bin/bash
set -eo pipefail

# Sync Python dependencies into .pythonlibs (Replit-managed, no venv)
if [ -f pyproject.toml ]; then
  uv pip install --target .pythonlibs/lib/python3.11/site-packages -r pyproject.toml 2>&1 | tail -n 3
fi

# Sync Node dependencies (server)
if [ -f package-lock.json ]; then
  npm install --no-audit --no-fund 2>&1 | tail -n 3
fi

# Fail fast if the bot has a syntax error after merge
python -m py_compile main.py trade_modes.py trading_engine.py

echo "post-merge setup OK"

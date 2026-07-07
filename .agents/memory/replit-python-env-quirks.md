---
name: Replit Python env + Telegram polling quirks
description: uv sync fails here (no venv in .pythonlibs); how to install deps in scripts. Also why telegram.error.Conflict appears while parallel task agents run.
---

**uv sync is broken in this workspace.** `.pythonlibs` is not a real venv (no pyvenv.cfg), so `uv sync` and `uv pip install --python .pythonlibs/bin/python` fall through to the read-only nix-store python and die with "Permission denied". 

**How to apply:** in shell scripts (e.g. the post-merge script at `scripts/post-merge.sh`), install Python deps with:
`uv pip install --target .pythonlibs/lib/python3.11/site-packages -r pyproject.toml`
For interactive installs, prefer the Replit package-management tools instead.

**telegram.error.Conflict ("terminated by other getUpdates request") is usually NOT a local bug.** Task agents run in isolated environments but share the same TG_TOKEN secret; if a task agent starts the bot there while the main bot runs here, both poll getUpdates and Telegram rejects one. 

**How to apply:** first check `ps aux` for duplicate local `python main.py` processes and deployment logs; if both are clean and a project task is IN_PROGRESS, the conflict is the task agent's instance and resolves itself when that task finishes. Outgoing signals/messages are unaffected (sendMessage doesn't conflict); only inbound command polling is contested intermittently.

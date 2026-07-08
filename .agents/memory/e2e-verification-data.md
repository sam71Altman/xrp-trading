---
name: E2E verification runs and gitignored CSV data
description: Why task-agent e2e test trades never appear in the main environment, and how synthetic trades are tagged/excluded
---
- `paper_trades.csv` and `trades.csv` are gitignored, so e2e verification trades created inside isolated task-agent environments never merge into the main environment's data files. Checking the local CSVs after a merge shows only locally-generated rows.
- Synthetic verification trades are tagged by reserved substrings `E2E` / `TEST` / `VERIFY` (case-insensitive) in `entry_reason`/`exit_reason` and are excluded from paper stats, Kill Switch layers, and balance (live + boot restore). Never use these substrings in real reason names (also documented in replit.md).
- **Why:** a losing "E2E SL test" trade would otherwise count toward loss streak / drawdown / win-rate and could trip the Kill Switch or skew the paper balance.
- **How to apply:** when writing future e2e verification hooks, always put a reserved marker in the entry reason; when running unit tests that call `finalize_trade`/`log_trade`, back up AND restore both CSVs (log_trade appends to trades.csv too — easy to forget).
- Live-drill pattern for boot-time safety gates (e.g. daily loss limiter): temporarily lower the threshold constant so today's REAL data trips it, restart, grep logs for the one-shot alert + rejection reason, then restore the constant and restart to confirm clean boot. Zero synthetic trades needed.
- The daily loss limiter bootstraps today's net PnL from `paper_trades.csv` at boot — a user "clear history" action via Telegram wipes the CSV, so a later restart forgets today's losses and weakens the 2% limit for the rest of the day (follow-up proposed to persist limiter state separately).

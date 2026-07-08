---
name: paper_trades.csv had no per-trade mode column
description: How trading mode (DEFAULT/FAST_SCALP/BOUNCE) is captured per trade for economics/reporting, and why entry_reason alone is not a reliable mode signal.
---

`paper_trades.csv` originally had no explicit "mode" column. Grouping trades by
mode cannot reliably use `entry_reason` alone: both a dedicated BOUNCE-mode
entry and a DEFAULT-mode HARD_MARKET dip-buy set `state.hold_active = True`,
so both get logged with `entry_reason_code = "BOUNCE_ENTRY"` — this conflates
two different modes under one label.

**Why:** the entry_reason labeling logic (in the TradingEngine entry path)
only distinguishes `state.hold_active` vs `FAST_SCALP` vs else="TREND_ENTRY";
it was never designed to be a mode discriminator, just a human-readable
reason string.

**How to apply:** a `paper_state.trade_mode` field now snapshots
`get_current_mode()` at BUY time and is carried through to the EXIT CSV row
via a `mode` column (with CSV header migration for old files, and
`infer_trade_mode()` as a best-effort fallback classifier from entry_reason
for legacy rows lacking the column). Any new per-mode reporting must read
this `mode` column rather than re-deriving mode from entry_reason.

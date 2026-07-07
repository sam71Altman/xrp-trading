---
name: PriceEngine single source of truth
description: PriceEngine/TradingGuard/FailSafeSystem live ONLY in price_engine.py; never redefine them in main.py. Also the 0-10 vs 0-1 score-scale rule.
---
The rule: `price_engine.py` is the ONLY home of `PriceEngine`, `TradingGuard`, and `FailSafeSystem`. main.py and trading_engine.py both import from it. Never define a local class with any of these names in main.py — a local duplicate shadows the imported one.

**Why:** main.py used to define its own `PriceEngine` (fed by the aggTrade websocket) shadowing the module one that `TradingGuard.enforce_guard("TRADE")` reads. The module engine's `last_price` stayed `None` forever and EVERY trade was silently blocked with "Waiting for price data for TRADE". A mirroring hack patched it; the classes were later unified into price_engine.py and the duplicates deleted.

**How to apply:** if trades are blocked with "Waiting for price data" / `BLOCKED_SYSTEM_ERROR | score=None`, check for a reintroduced duplicate class (`grep -c "class PriceEngine"` must be 1, in price_engine.py). Guard semantics: stale >2s or latency_ms >500 blocks; `update_price()` auto-resumes the failsafe only when the block reason is exactly "WebSocket disconnected".

Related scale rule: the bot signal score (calculate_signal_score, 0–10) and the AI score (0–1) are different scales. `analysis['score']` must be enriched once per cycle in signal_loop before check_buy_signal, or it silently defaults to 0 and all score-gated entries are rejected.

---
name: PriceEngine class shadowing
description: Two PriceEngine classes exist (price_engine.py module + a local one in main.py); guards read the module one.
---
The rule: any real-time price update must reach `price_engine.PriceEngine.last_price`, because `trading_engine`'s `TradingGuard.enforce_guard("TRADE")` reads that module-level class — not the local `PriceEngine` defined in main.py that the websocket actually feeds.

**Why:** main.py defines its own `PriceEngine` (aggTrade websocket) which shadows the imported one. The module-level engine was never started, so its `last_price` stayed `None` and EVERY trade was silently blocked with "Waiting for price data for TRADE". Fixed by mirroring updates from the local class into the module class inside `update_price()`.

**How to apply:** if trades are blocked with `BLOCKED_SYSTEM_ERROR | score=None` plus "Waiting for price data", check that the mirror in main.py's `PriceEngine.update_price()` is intact, and beware the guard's 0.5s staleness window (aggTrade stream is fine; the module's own 1s ticker stream would be too slow).

Related scale rule: the bot signal score (calculate_signal_score, 0–10) and the AI score (0–1) are different scales. `analysis['score']` must be enriched once per cycle in signal_loop before check_buy_signal, or it silently defaults to 0 and all score-gated entries are rejected.

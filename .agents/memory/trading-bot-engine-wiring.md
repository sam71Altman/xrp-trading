---
name: TradingEngine wiring pitfalls
description: Broker/telegram must be wired at startup; guards must match stream cadence; E2E verification pattern for the paper bot.
---

**The engine's collaborators (broker, telegram sender) must be wired at startup in main(), not lazily inside one entry path.**
**Why:** broker was once created lazily only inside the DEFAULT entry branch, so FAST_DOWN entries failed with "No broker configured"; `execution_engine.telegram` was never assigned at all, so open/close notifications silently never sent (`_notify_event` no-ops when telegram is None — no error appears).
**How to apply:** when adding a new engine dependency, wire it right after `application.initialize()` alongside broker/telegram, and remember silent no-op paths hide missing wiring — grep for `is None` early-returns when a notification "doesn't arrive".

**Price-staleness guards must be looser than the price stream's update cadence.** Binance `@ticker` pushes ~1 update/sec; a 0.5s staleness ceiling blocked ~half of real trade attempts with "Price stale - Blocking TRADE". Use 2s (matches main.py's own guard).

**Guards on locals-vs-globals:** a check like `'candles' in globals()` inside the signal loop is always False (candles is a loop local) — it silently disabled the generic TP/SL manage block. Prefer the live websocket price (`PriceEngine.last_price`) for exit management.

**E2E verification pattern that worked:** a one-shot trigger file consumed only *after* a successful open, bypassing only the entry-signal condition, drives the full real path (engine → broker → single Telegram open msg → [MANAGE] TP/SL close → CSV sync) on live prices within ~2 min. Remove the hook after verification.

**Benign noise:** `[STATE_DRIFT] Broker=False, Engine=True` warnings come from reconcile_state comparing safety_core (never updated on engine-path opens) with the engine; it is log-only ("no local state to fix").

---
name: TradingEngine wiring pitfalls
description: Broker/telegram must be wired at startup; guards must match stream cadence; E2E verification pattern for the paper bot.
---

**The engine's collaborators (broker, telegram sender) must be wired at startup in main(), not lazily inside one entry path.**
**Why:** broker was once created lazily only inside the DEFAULT entry branch, so FAST_DOWN entries failed with "No broker configured"; `execution_engine.telegram` was never assigned at all, so open/close notifications silently never sent (`_notify_event` no-ops when telegram is None — no error appears).
**How to apply:** when adding a new engine dependency, wire it right after `application.initialize()` alongside broker/telegram, and remember silent no-op paths hide missing wiring — grep for `is None` early-returns when a notification "doesn't arrive".

**Price-staleness guards must be looser than the price stream's update cadence.** Binance `@ticker` pushes ~1 update/sec; a 0.5s staleness ceiling blocked ~half of real trade attempts with "Price stale - Blocking TRADE". Use 2s (matches main.py's own guard).

**Guards on locals-vs-globals:** a check like `'candles' in globals()` inside the signal loop is always False (candles is a loop local) — it silently disabled the generic TP/SL manage block. Prefer the live websocket price (`PriceEngine.last_price`) for exit management.

**E2E verification pattern that worked:** a one-shot trigger file consumed only *after* a successful open, bypassing only the entry-signal condition, drives the full real path (engine → broker → single Telegram open msg → [MANAGE] TP/SL close → CSV sync) on live prices within ~2 min. Remove the hook after verification. To exercise the SL branch deterministically, record the entry a bit above the real fill (e.g. +0.5%) in both the engine and paper bookkeeping so PNL is instantly below the SL threshold — TP and SL close paths verified this way end-to-end (one message each, EXIT row with fees).

**Every entry path must set the paper bookkeeping fields (position qty + USDT size) at open, or close-side accounting silently falls back to the legacy fixed size.** The engine only tracks entry price; paper_state carries size. The DEFAULT entry path historically set neither — closes were sized by a hardcoded fallback and nobody noticed because the fallback "worked". When adding/altering an entry path, verify qty/size/entry_reason are stored and that they are cleared on close (stale qty from a prior trade corrupts the next close).

**Benign noise:** `[STATE_DRIFT] Broker=False, Engine=True` warnings come from reconcile_state comparing safety_core (never updated on engine-path opens) with the engine; it is log-only ("no local state to fix").

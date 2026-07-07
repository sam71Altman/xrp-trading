---
name: Mode system unification (state.mode vs mode_state)
description: The bot had two independent mode systems; state.mode is now a property delegating to trade_modes' mode_state. Legacy "AGGRESSIVE" == TradeMode "DEFAULT".
---

The legacy `LegacyBotState.mode` used to be hardcoded `"AGGRESSIVE"` and never updated, while the real mode manager was `mode_state.current_mode` in trade_modes.py (valid values: `DEFAULT`, `FAST_SCALP`, `BOUNCE` — `"AGGRESSIVE"` is NOT valid there). This made kill-switch display and all `state.mode` checks permanently wrong.

**The rule:** `state.mode` is now a read/write property that delegates to `get_current_mode()` / `change_trade_mode()`. Never reintroduce a stored `self.mode` attribute or compare against `"AGGRESSIVE"` — the aggressive mode is named `"DEFAULT"` (`TradeMode.DEFAULT_CLEAN_AGGRESSIVE`).

**Why:** Two sources of truth caused the kill switch to always show "disabled" and mode switching to appear broken. DEFAULT bypassing the Kill Switch is intentional (user-confirmed), same as legacy AGGRESSIVE.

**How to apply:**
- Mode comparisons must use `"DEFAULT"` / `"FAST_SCALP"` / `"BOUNCE"`.
- Current mode persists in `mode_state.json` (restored on startup by ModeStateManager); fast mode persists in `fast_mode_state.json`. If persistence is touched, keep both save+restore in lockstep or restarts silently revert to the kill-switch-bypassing DEFAULT mode.
- `change_trade_mode` has rate limits (300s interval, 3/hour) and rejects same-mode changes — writes via `state.mode = X` can be silently rejected.
- httpx INFO logging is suppressed in main.py because it leaks the Telegram bot token in request URLs — don't remove that suppression.

---
name: Bounce entry design (score-0 entries are intentional)
description: Why the bot opens BUY trades with signal score 0 during downtrends, and the hold_active leak pitfall
---

# Bounce (dip-buy) entries in DEFAULT mode

**The rule:** In DEFAULT mode + HARD_MARKET (EMAs not bullishly aligned), the ONLY allowed entry path is the bounce entry, which *requires* a LOW signal score (<= 5 on the 0–10 scale) plus RSI <= 35, a 15-candle local low, and a 1.8x volume spike. A score-0 BUY during a bearish alert is therefore by design, not a bug.

**Why:** User reported a "contradiction" (drop alert + buy entry + score 0/10 diagnostics at the same time). Root cause was labeling, not logic: the trade was logged with a hardcoded misleading reason ("AI_EXECUTION" while AI was OFF) and the Telegram open message showed no reason. Fix was transparency: derive the real entry reason (BOUNCE_ENTRY / FAST_SCALP_ENTRY / TREND_ENTRY) and show it in both the trade log and the "دخول صفقة" message.

## AI "enabled" key pitfall

**The rule:** The AI status dict (from AIState.get_status) exposes `mode` (OFF/LEARN/FULL) — there is NO `enabled` key. Any `status.get("enabled", True)` check is always True, silently keeping AI scoring active while the user believes AI is off. Derive enabled-ness from `mode != "OFF"`.

**Why:** User turned AI off but logs still showed ai_enabled=True and AI scores being computed/injected every cycle; combined with the misleading "AI_EXECUTION" trade label this looked like AI was opening trades. AI never opens trades — it only filters; OFF mode passes strategy trades through untouched.

**How to apply:**
- Never "fix" the score<=5 bounce condition by inverting it — low score is a required input to the contrarian dip-buy.
- Any new entry path must pass a real entry reason to the engine so the single OPEN notification stays self-explanatory.
- `state.hold_active` (set at bounce entry, makes exits ignore EMA/trailing) must be cleared on EVERY close path. Closes that bypass the legacy finalize funnel (engine-internal TP, force-close) leak it — the reconcile loop clears it as a safety net whenever no position is open. If new close paths are added, verify hold state clearing.

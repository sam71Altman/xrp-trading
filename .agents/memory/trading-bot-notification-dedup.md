---
name: Trading bot notification dedup and single-engine execution
description: Architectural rule for this XRP/USDT trading bot to avoid duplicate trades/messages
---

## Rule
Exactly ONE component (the TradingEngine's atomic open/close methods) is allowed
to talk to the broker and send trade-lifecycle Telegram messages. Any other
function — especially ones named `check_*`, `generate_*_signal`, or documented
as "pure"/"NO side effects" — must only return a decision, never send a
notification or touch the broker itself.

**Why:** This codebase previously had two parallel execution paths (a real
engine with a broker, and a second "AI engine" instance whose broker was never
configured) plus multiple signal-check helper functions that fired their own
Telegram sends "for convenience." The result was duplicate broker orders and
duplicate Telegram messages for a single logical trade, and one broken path
would silently call `reset_position_state()` and undo a valid trade.

**How to apply:** When adding a new strategy/signal path, make sure it ends by
calling the single engine's execute/close method and nothing else sends a
message for that event. If you need a dedup guard for a notification that
isn't naturally serialized through the engine (e.g. legacy/manual exit paths,
periodic alerts, resume messages), gate the `bot.send_message`/`telegram.send`
call with a small time-windowed "have I already sent this exact event key"
check rather than adding another ad-hoc send site.

## Related gotcha: engine state vs. legacy bookkeeping state
If a codebase has both a modern "engine" that owns execution state and a
legacy/paper bookkeeping system (balance, CSV, PnL) that predates it, closing
a position through the engine does NOT automatically update the legacy
bookkeeping unless you explicitly wire it. Watch for "DEPRECATED" comments
claiming a function is unused — verify with a real call-site search, since in
practice such functions are often still the only place balance/CSV updates
happen, and the deprecation was left incomplete.

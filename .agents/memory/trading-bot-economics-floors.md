---
name: Trade economics floors
description: Non-negotiable TP/SL/R-R floors for the paper-trading bot and the rounding pitfall that silently breaks them
---

# Trade economics floors

Rule: every TP/SL target (any mode, any entry path) must satisfy TP >= 2× round-trip cost (0.24% both sides → TP >= 0.48%) and R/R >= 1.5 (SL <= TP/1.5). Floors are enforced centrally in the dynamic-targets function — never hand-set raw percentages elsewhere.

**Why:** With ~0.24% round-trip fees+slippage, smaller TPs are guaranteed net losers even at high win rates; also rounding SL to 2 decimals AFTER clamping to TP/1.5 can round SL *up* and silently break R/R >= 1.5. Round SL to 3 decimals (or round down).

**How to apply:** When touching targets, re-run the all-modes × ATR-regimes assertion sweep (TP >= 0.48, R/R >= 1.5, BOUNCE SL widest). Per-trade targets are stored on state + engine at entry and read by ALL exit paths — changing one side without the other reintroduces hardcoded-percentage drift. Risk-free SL after TP trigger must lock breakeven AFTER fees (+0.30%), not a nominal +0.10%.

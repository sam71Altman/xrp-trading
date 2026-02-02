# XRP/USDT Telegram Signals Bot v4.6.PRO

## Overview

A Python Telegram trading signals bot for XRP/USDT that runs paper trading simulations with a multi-layer Kill Switch protection system, Smart Adaptive Trading System, and Governed AI Intelligence layer. It fetches price data from Binance API (read-only), analyzes using EMA crossover and breakout strategies, and sends buy/exit signals to Telegram.

### Institutional Production-Grade Architecture (v4.6.PRO)

**SYSTEM PHILOSOPHY (ABSOLUTE – NON-NEGOTIABLE)**
- TP = EXECUTION EVENT (إغلاق فوري)
- SL = FINAL SAFETY EXIT
- EMA Exit = CONFIRMED FAILURE JUDGMENT فقط
- ENTRY LOGIC = UNTOUCHED
- EXECUTION ENGINE = SINGLE SOURCE OF TRUTH
- UI = VIEW ONLY
- FAILURE IS ISOLATED PER STRATEGY
- SAFETY > AVAILABILITY

**v4.6.PRO GUARANTEES**
- ZERO duplicate trades
- ZERO duplicate telegram messages
- ZERO race conditions
- ZERO state mismatch (UI == Engine == DB)
- ONE execution path only
- ONE state source only
- Deterministic behavior
- Production/institutional reliability

### v4.6.PRO Infrastructure Components

**TradingEngine** (Single Writer Principle)
- `_trade_lock`: asyncio.Lock for atomic operations
- `execute_trade_atomically()`: ONLY method that may call broker, update state, send telegram
- Pipeline isolation via asyncio.Queue(maxsize=1)

**TradingSnapshot** (Immutable)
- Created once per cycle, never modified
- Contains: timestamp, position_open, price, entry_price, indicators, candles, mode, balance

**TradeSignal** (Pure)
- Pure signal from strategies - NO side effects
- Contains: action, confidence, reasons, suggested_tp, suggested_sl, source

**CircuitBreaker**
- Failure threshold: 3 failures
- Recovery timeout: 60 seconds
- Auto-recovery when healthy

**RateLimiter**
- Max 2 trades per minute
- Max 10 trades per hour

**AuditTrail** (Non-blocking)
- In-memory buffer with async flush
- Writes to audit_trail.jsonl

**StateGuard** (Dual Failsafe)
- Real-time state consistency verification
- Emergency halt on state mismatch
- Backup guard for failsafe

**Atomic Close System** (State Drift Fix)
- `close_trade_atomically()`: Single close path with guaranteed order
- Order: broker close -> state update -> DB/UI -> telegram (non-blocking)
- `force_close_trade()`: Now calls `reset_position_state()` to prevent drift
- `reconcile_state()`: Runs every 2 seconds as safety net
- Auto-detects and fixes state drift (Broker != Engine mismatch)

**Multi-Strategy Architecture**
- SCALP_FAST: 1m/5m frames with isolated state
- SCALP_PULLBACK: 5m frame with isolated state
- BREAKOUT: 15m frame with isolated state

**Execution Priority (IMMUTABLE ORDER)**
1. TAKE_PROFIT (Tick-level, HARD STOP)
2. STOP_LOSS
3. EMERGENCY_CLOSE
4. EMA_FAILURE_EXIT
5. MAX_TIME_ESCAPE

**Smart Trailing SL**
- 1M: Activate @ +0.20%, Lock @ +0.10%, Step = 0.05%
- 5M: Activate @ +0.35%, Lock @ +0.18%, Step = 0.10%

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Trading Strategy Engine
- Uses EMA (20/50) crossover for trend identification
- Breakout confirmation using 5-candle high breaks
- Signal scoring system (0-10) based on multiple indicators
- Adaptive cooldown based on win/loss streaks

### Kill Switch V3.2 (Multi-Layer Protection)
1. **Data Maturity Gate**: No Kill Switch before 5 closed trades
2. **Loss Streak Protection**: 3 consecutive losses = stop trading
3. **Drawdown Protection**: >3% drop from peak balance = stop trading
4. **Recent Performance**: Win Rate <40% in last 10 trades = stop trading
- Auto-resume after 30 minutes
- Manual resume via `/استئناف` command

### AI Filtering System v4.5.PRO-AI
**Core Files**: `ai_state.py`, `ai_filter.py`, `trading_engine.py`, `telegram_interface.py`, `monitor.py`, `ai_integration.py`

**Weight Values (Fixed)**: 0.0 (OFF), 0.3 (LOW), 0.6 (MEDIUM), 1.0 (HIGH)

**Modes**:
- OFF: All trades pass through
- LEARN: Analyze and score, but always allow
- FULL: Real filtering (block if score < weight)

**Design Principles**:
- Single Entry Point: All trades through `check_and_execute_trade()`
- Dependency Injection: No monkey patching
- No Global State: Each symbol has independent AIState
- Safety First: Return None on any error
- Cooldown applies in all modes (30s default)
- Daily intervention limit (50), fallback to allow when reached

**Telegram Commands**: `/ai` for control panel

### Data Source
- Binance REST API for OHLCV candlestick data (no WebSocket)
- Polling interval: every 10 seconds
- Supported timeframes: 1m (default), 5m

### Paper Trading System
- Virtual position tracking (one position at a time)
- Fixed trade size: 100 USDT from 1000 USDT starting balance
- Take profit: 0.40%, Stop loss: 0.30%
- Trailing stop trigger at 0.25%

### Telegram Interface
- Commands in both Arabic and English
- Arabic: `/الحالة`, `/احصائيات`, `/استئناف`
- Inline keyboard buttons for common actions
- Status, timeframe switching, and on/off controls

## External Dependencies

### Python Dependencies
- `python-telegram-bot` (v20+) - Telegram Bot API wrapper
- `requests` - HTTP client for Binance API calls
- Binance REST API (`api.binance.com`) - Price data source (read-only, no API key required)

### Environment Variables (Secrets)
| Variable | Purpose |
|----------|---------|
| `TG_TOKEN` | Telegram Bot token from @BotFather |
| `TG_CHAT_ID` | Target Telegram chat/channel ID |

### Data Storage
- `paper_trades.csv` - Paper trading history with Kill Switch columns
- `trades.csv` - Signal history log

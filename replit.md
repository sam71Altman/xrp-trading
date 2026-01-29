# XRP/USDT Telegram Signals Bot v4.2.PRO-AI

## Overview

A Python Telegram trading signals bot for XRP/USDT that runs paper trading simulations with a multi-layer Kill Switch protection system, Smart Adaptive Trading System, and Governed AI Intelligence layer. It fetches price data from Binance API (read-only), analyzes using EMA crossover and breakout strategies, and sends buy/exit signals to Telegram.

### Smart Adaptive Trading System
Three distinct trading modes that adapt to market conditions:
- **DEFAULT (الوضع الذكي)**: Balanced approach with all filters active
- **FAST_SCALP (سكالب سريع)**: Aggressive scalping with relaxed entry filters
- **BOUNCE (اصطياد الارتدادات)**: Conservative, enters only on RSI oversold bounces

### AI System v4.2.PRO-AI (NEW)
Governed intelligence layer with three modes:
- **OFF**: AI disabled, static logic only
- **LEARN**: Suggestions only, no modifications
- **FULL**: Smart adjustments within safe limits

#### Key AI Features:
- **Impact Cap**: Daily adjustment limits (LOW: 15, MEDIUM: 25, HIGH: 40)
- **One-click disable**: Emergency shutdown available
- **Open trades protected**: No interference with active positions
- **Next candle only**: Changes apply from next candle
- **Full transparency**: All actions visible and logged

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

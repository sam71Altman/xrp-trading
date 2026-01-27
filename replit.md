# XRP/USDT Telegram Signals Bot

## Overview

This is a hybrid project containing two distinct applications:

1. **Primary: Python Telegram Bot** - A trading signals bot for XRP/USDT that runs paper trading simulations. It fetches price data from Binance API, analyzes using EMA crossover and breakout strategies, and sends buy/exit signals to Telegram.

2. **Secondary: React Native/Expo Mobile App** - A cross-platform mobile application scaffold with Express.js backend. This appears to be a template that was included but is not the main focus of the project.

The main functionality is the Python bot (`main.py`) which operates in paper trading mode, requiring no real exchange API keys. It uses Telegram for all user interaction via bot commands.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Python Telegram Bot (Primary Application)

**Trading Strategy Engine**
- Uses EMA (20/50) crossover for trend identification
- Breakout confirmation using 5-candle high breaks
- Signal scoring system (0-10) based on multiple indicators
- Adaptive cooldown based on win/loss streaks
- Kill switch when win rate drops below 45% (after 5 trades minimum)

**Data Source**
- Binance REST API for OHLCV candlestick data (no WebSocket)
- Polling interval: every 10 seconds
- Supported timeframes: 1m (default), 5m

**Paper Trading System**
- Virtual position tracking (one position at a time)
- Fixed trade size: 100 USDT from 1000 USDT starting balance
- Take profit: 0.40%, Stop loss: 0.30%
- Trailing stop trigger at 0.25%

**Telegram Interface**
- Commands are in Arabic for target audience
- Inline keyboard buttons for common actions
- Status, timeframe switching, and on/off controls

### React Native/Expo App (Secondary)

**Frontend Architecture**
- Expo SDK 54 with React Native 0.81
- React Navigation (native stack + bottom tabs)
- React Query for server state management
- Reanimated for animations
- Themed component system with light/dark mode support

**Backend Architecture**
- Express.js server with TypeScript
- Drizzle ORM for database operations
- In-memory storage as default (can be extended to PostgreSQL)

**Path Aliases**
- `@/` maps to `./client/`
- `@shared/` maps to `./shared/`

## External Dependencies

### Python Bot Dependencies
- `python-telegram-bot` (v20+) - Telegram Bot API wrapper
- `requests` - HTTP client for Binance API calls
- Binance REST API (`api.binance.com`) - Price data source (read-only, no API key required)

### Environment Variables (Secrets)
| Variable | Purpose |
|----------|---------|
| `TG_TOKEN` | Telegram Bot token from @BotFather |
| `TG_CHAT_ID` | Target Telegram chat/channel ID |
| `DATABASE_URL` | PostgreSQL connection string (for Expo app, optional) |

### Node.js Dependencies (Expo App)
- PostgreSQL via `pg` driver
- Drizzle ORM with PostgreSQL dialect
- Express.js v5 for API server
- TanStack React Query for data fetching

### Data Storage
- `trades.csv` - Persistent trade history log for the Python bot
- PostgreSQL database (optional, for the Expo app user management)
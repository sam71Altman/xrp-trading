# XRP/USDT Telegram Signals Bot

A Python-based Telegram bot that sends XRP/USDT trading signals based on EMA crossover and breakout strategy. **Signals only - no auto trading.**

## Features

- Real-time price data from Binance public API
- EMA20/EMA50 crossover strategy with breakout confirmation
- Virtual position tracking with TP/SL management
- Arabic-formatted Telegram messages
- Anti-spam protection (1 message per minute cooldown)
- Runtime configuration via Telegram commands

## Strategy

### Entry (BUY) Signal
- EMA20 > EMA50 (bullish trend)
- Close price breaks above the highest high of the previous 5 candles

### Exit Signal
One of the following conditions:
- Take Profit: +0.40% from entry
- Stop Loss: -0.30% from entry
- Protective Exit: Close drops below EMA20

## Setup

### 1. Create a Telegram Bot

1. Open Telegram and search for [@BotFather](https://t.me/BotFather)
2. Send `/newbot` and follow the instructions
3. Copy the **API token** (looks like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

### 2. Get Your Chat ID

1. Start a chat with your new bot
2. Send any message to the bot
3. Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
4. Find `"chat":{"id":123456789}` - this is your Chat ID

### 3. Set Environment Variables

In Replit:
1. Click on **Secrets** (lock icon in the left sidebar)
2. Add the following secrets:

| Key | Value |
|-----|-------|
| `TG_TOKEN` | Your Telegram bot token |
| `TG_CHAT_ID` | Your Telegram chat ID |

### 4. Run the Bot

In the Replit Shell, run:

```bash
python main.py
```

The bot will start and display:
```
==================================================
XRP/USDT Signals Bot Started
Symbol: XRPUSDT
Timeframe: 1m
Strategy: EMA20/EMA50 + Breakout
TP: +0.4% | SL: -0.3%
==================================================
```

Send `/start` to your bot on Telegram to confirm it's working.

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Confirm bot is running |
| `/status` | Show current state (position, entry price, last close) |
| `/settf 1m` | Change timeframe to 1 minute |
| `/settf 5m` | Change timeframe to 5 minutes |
| `/on` | Enable signal sending |
| `/off` | Disable signal sending (bot keeps running) |

## Configuration

Edit the variables at the top of `main.py` to customize:

```python
TIMEFRAME = "1m"        # Default timeframe: "1m" or "5m"
TAKE_PROFIT_PCT = 0.40  # Take profit percentage
STOP_LOSS_PCT = 0.30    # Stop loss percentage
POLL_INTERVAL = 60      # Seconds between market checks
COOLDOWN_SECONDS = 60   # Minimum seconds between messages
```

## Message Format (Arabic)

### BUY Signal
```
🟢 **إشارة شراء - XRPUSDT**

📊 **الإطار الزمني**: 1m
💰 **سعر الدخول**: 2.1450
🎯 **جني الأرباح**: 2.1536 (+0.40%)
🛑 **وقف الخسارة**: 2.1386 (-0.30%)

📈 **السبب**: EMA20 > EMA50 + اختراق أعلى سعر
```

### EXIT Signal
```
🔴 **إغلاق المركز - XRPUSDT**

💰 **سعر الخروج**: 2.1540
📊 **الربح/الخسارة**: +0.42%
✅ **السبب**: وصول الهدف
```

## Error Handling

- Network/API errors are logged and do not crash the bot
- After 5 consecutive API failures, a warning is sent to Telegram
- The bot automatically retries on the next polling cycle

## Dependencies

- `python-telegram-bot>=20.0` - Telegram bot framework
- `requests` - HTTP client for Binance API
- `pandas` - Data manipulation and EMA calculation
- `numpy` - Numerical operations

## License

MIT License - Feel free to modify and use as needed.

## Notes

- The bot uses multiple Binance API endpoints (including Binance US) to handle geographic restrictions
- If running from Replit, the bot will automatically try alternative endpoints if the main API is blocked

## Disclaimer

This bot is for educational purposes only. Trading cryptocurrencies involves significant risk. Use at your own risk and never trade more than you can afford to lose.

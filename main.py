#!/usr/bin/env python3
"""
XRP/USDT Telegram Signals Bot
Sends trading signals based on EMA crossover and breakout strategy.
No auto-trading - signals only.
"""

import os
import time
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import requests
import numpy as np
import pandas as pd
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

# ============================================================================
# CONFIGURATION
# ============================================================================

# Timeframe setting - change this variable to switch between 1m/5m
TIMEFRAME = "1m"  # Options: "1m", "5m"

# Trading pair
SYMBOL = "XRPUSDT"

# Strategy parameters
EMA_SHORT = 20
EMA_LONG = 50
BREAKOUT_CANDLES = 5  # Number of candles to check for breakout

# Risk management (percentages)
TAKE_PROFIT_PCT = 0.40  # +0.40%
STOP_LOSS_PCT = 0.30    # -0.30%

# Anti-spam settings
COOLDOWN_SECONDS = 60  # Minimum seconds between messages

# Polling interval (seconds)
POLL_INTERVAL = 60

# Binance API endpoint
BINANCE_API = "https://api.binance.com/api/v3/klines"

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============================================================================
# BOT STATE
# ============================================================================

class BotState:
    """Tracks the bot's internal state."""
    
    def __init__(self):
        self.position_open: bool = False
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[datetime] = None
        self.last_message_time: float = 0
        self.signals_enabled: bool = True
        self.timeframe: str = TIMEFRAME
        self.last_close: Optional[float] = None
        self.consecutive_errors: int = 0
        self.error_alerted: bool = False

state = BotState()

# ============================================================================
# BINANCE API FUNCTIONS
# ============================================================================

def get_klines(symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV candles from Binance public API.
    Returns DataFrame with columns: open, high, low, close, volume
    """
    try:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        response = requests.get(BINANCE_API, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        # Convert to numeric
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
        
    except requests.RequestException as e:
        logger.error(f"Binance API error: {e}")
        return None

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()

# ============================================================================
# STRATEGY LOGIC
# ============================================================================

def analyze_market(df: pd.DataFrame) -> dict:
    """
    Analyze market data and generate signals.
    Returns dict with analysis results.
    """
    if df is None or len(df) < EMA_LONG + BREAKOUT_CANDLES:
        return {"error": "Insufficient data"}
    
    # Calculate EMAs
    df["ema_short"] = calculate_ema(df["close"], EMA_SHORT)
    df["ema_long"] = calculate_ema(df["close"], EMA_LONG)
    
    # Current values (latest completed candle)
    current = df.iloc[-1]
    current_close = current["close"]
    current_ema_short = current["ema_short"]
    current_ema_long = current["ema_long"]
    
    # Previous candles for breakout check (excluding current candle)
    prev_candles = df.iloc[-(BREAKOUT_CANDLES + 1):-1]
    highest_high = prev_candles["high"].max()
    
    # Store last close in state
    state.last_close = current_close
    
    return {
        "close": current_close,
        "ema_short": current_ema_short,
        "ema_long": current_ema_long,
        "highest_high": highest_high,
        "ema_bullish": current_ema_short > current_ema_long,
        "breakout": current_close > highest_high,
    }

def check_buy_signal(analysis: dict) -> bool:
    """Check if BUY signal conditions are met."""
    if "error" in analysis:
        return False
    
    # BUY: EMA20 > EMA50 AND close breaks above highest high of previous 5 candles
    return analysis["ema_bullish"] and analysis["breakout"]

def check_exit_signal(analysis: dict) -> Optional[str]:
    """
    Check if EXIT signal conditions are met.
    Returns exit reason or None.
    """
    if "error" in analysis or not state.position_open or state.entry_price is None:
        return None
    
    current_close = analysis["close"]
    entry = state.entry_price
    
    # Calculate PnL percentage
    pnl_pct = ((current_close - entry) / entry) * 100
    
    # Check Take Profit
    if pnl_pct >= TAKE_PROFIT_PCT:
        return "tp"  # Take Profit hit
    
    # Check Stop Loss
    if pnl_pct <= -STOP_LOSS_PCT:
        return "sl"  # Stop Loss hit
    
    # Check protective exit (close drops below EMA20)
    if current_close < analysis["ema_short"]:
        return "ema"  # EMA protective exit
    
    return None

def calculate_targets(entry_price: float) -> tuple:
    """Calculate TP and SL prices from entry."""
    tp = entry_price * (1 + TAKE_PROFIT_PCT / 100)
    sl = entry_price * (1 - STOP_LOSS_PCT / 100)
    return tp, sl

def calculate_pnl(entry: float, exit_price: float) -> float:
    """Calculate PnL percentage."""
    return ((exit_price - entry) / entry) * 100

# ============================================================================
# MESSAGE FORMATTING (Arabic)
# ============================================================================

def format_buy_message(entry: float, tp: float, sl: float, timeframe: str) -> str:
    """Format BUY signal message in Arabic."""
    return (
        f"🟢 **إشارة شراء - {SYMBOL}**\n\n"
        f"📊 **الإطار الزمني**: {timeframe}\n"
        f"💰 **سعر الدخول**: {entry:.4f}\n"
        f"🎯 **جني الأرباح**: {tp:.4f} (+{TAKE_PROFIT_PCT}%)\n"
        f"🛑 **وقف الخسارة**: {sl:.4f} (-{STOP_LOSS_PCT}%)\n\n"
        f"📈 **السبب**: EMA{EMA_SHORT} > EMA{EMA_LONG} + اختراق أعلى سعر"
    )

def format_exit_message(exit_price: float, pnl: float, reason: str) -> str:
    """Format EXIT signal message in Arabic."""
    reason_text = {
        "tp": "وصول الهدف",
        "sl": "وصول وقف الخسارة",
        "ema": f"السعر أسفل EMA{EMA_SHORT}"
    }.get(reason, "خروج يدوي")
    
    pnl_sign = "+" if pnl >= 0 else ""
    status_emoji = "✅" if pnl >= 0 else "❌"
    
    return (
        f"🔴 **إغلاق المركز - {SYMBOL}**\n\n"
        f"💰 **سعر الخروج**: {exit_price:.4f}\n"
        f"📊 **الربح/الخسارة**: {pnl_sign}{pnl:.2f}%\n"
        f"{status_emoji} **السبب**: {reason_text}"
    )

def format_status_message() -> str:
    """Format status response message in Arabic."""
    status = "مُفعَّل" if state.signals_enabled else "مُعطَّل"
    position = "مفتوح" if state.position_open else "مُغلق"
    
    msg = (
        f"ℹ️ **حالة البوت**\n\n"
        f"📊 **الإطار الزمني**: {state.timeframe}\n"
        f"🔔 **الإشارات**: {status}\n"
        f"📈 **المركز**: {position}\n"
    )
    
    if state.position_open and state.entry_price:
        msg += f"💰 **سعر الدخول**: {state.entry_price:.4f}\n"
        if state.last_close:
            pnl = calculate_pnl(state.entry_price, state.last_close)
            pnl_sign = "+" if pnl >= 0 else ""
            msg += f"📉 **الربح الحالي**: {pnl_sign}{pnl:.2f}%\n"
    
    if state.last_close:
        msg += f"🕯️ **آخر إغلاق**: {state.last_close:.4f}"
    
    return msg

# ============================================================================
# ANTI-SPAM & MESSAGE SENDING
# ============================================================================

def can_send_message() -> bool:
    """Check if cooldown period has passed."""
    return (time.time() - state.last_message_time) >= COOLDOWN_SECONDS

async def send_telegram_message(bot: Bot, chat_id: str, message: str) -> bool:
    """Send message to Telegram chat with cooldown check."""
    if not can_send_message():
        logger.info("Message skipped due to cooldown")
        return False
    
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode="Markdown"
        )
        state.last_message_time = time.time()
        logger.info("Message sent successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        return False

# ============================================================================
# TELEGRAM COMMAND HANDLERS
# ============================================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    status = "نشط" if state.signals_enabled else "متوقف"
    await update.message.reply_text(
        f"✅ البوت يعمل | الحالة: {status}",
        parse_mode="Markdown"
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command."""
    await update.message.reply_text(
        format_status_message(),
        parse_mode="Markdown"
    )

async def cmd_settf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /settf command to change timeframe."""
    if not context.args:
        await update.message.reply_text(
            "❌ استخدم: /settf 1m أو /settf 5m",
            parse_mode="Markdown"
        )
        return
    
    new_tf = context.args[0].lower()
    if new_tf not in ["1m", "5m"]:
        await update.message.reply_text(
            "❌ الإطار الزمني غير صحيح. استخدم 1m أو 5m",
            parse_mode="Markdown"
        )
        return
    
    state.timeframe = new_tf
    await update.message.reply_text(
        f"✅ تم تغيير الإطار الزمني إلى {new_tf}",
        parse_mode="Markdown"
    )

async def cmd_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /on command to enable signals."""
    state.signals_enabled = True
    await update.message.reply_text(
        "✅ تم تفعيل إرسال الإشارات",
        parse_mode="Markdown"
    )

async def cmd_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /off command to disable signals."""
    state.signals_enabled = False
    await update.message.reply_text(
        "⏸️ تم إيقاف إرسال الإشارات مؤقتاً",
        parse_mode="Markdown"
    )

# ============================================================================
# MAIN SIGNAL LOOP
# ============================================================================

async def signal_loop(bot: Bot, chat_id: str) -> None:
    """
    Main loop that polls market data and sends signals.
    Runs every POLL_INTERVAL seconds.
    """
    logger.info(f"Signal loop started - Polling every {POLL_INTERVAL}s")
    
    while True:
        try:
            # Skip if signals are disabled
            if not state.signals_enabled:
                await asyncio.sleep(POLL_INTERVAL)
                continue
            
            # Fetch market data
            df = get_klines(SYMBOL, state.timeframe)
            
            if df is None:
                state.consecutive_errors += 1
                logger.warning(f"Failed to fetch data (errors: {state.consecutive_errors})")
                
                # Alert after 5 consecutive errors
                if state.consecutive_errors >= 5 and not state.error_alerted:
                    await send_telegram_message(
                        bot, chat_id,
                        "⚠️ مشكلة في الاتصال بـBinance"
                    )
                    state.error_alerted = True
                
                await asyncio.sleep(POLL_INTERVAL)
                continue
            
            # Reset error counter on success
            state.consecutive_errors = 0
            state.error_alerted = False
            
            # Analyze market
            analysis = analyze_market(df)
            
            if "error" in analysis:
                logger.warning(f"Analysis error: {analysis['error']}")
                await asyncio.sleep(POLL_INTERVAL)
                continue
            
            # Check for EXIT signal first (if position is open)
            if state.position_open:
                exit_reason = check_exit_signal(analysis)
                if exit_reason:
                    exit_price = analysis["close"]
                    pnl = calculate_pnl(state.entry_price, exit_price)
                    
                    # Send EXIT message
                    msg = format_exit_message(exit_price, pnl, exit_reason)
                    await send_telegram_message(bot, chat_id, msg)
                    
                    # Close virtual position
                    state.position_open = False
                    state.entry_price = None
                    state.entry_time = None
                    
                    logger.info(f"Position closed: {exit_reason} @ {exit_price:.4f} (PnL: {pnl:.2f}%)")
            
            # Check for BUY signal (if no position is open)
            elif not state.position_open:
                if check_buy_signal(analysis):
                    entry_price = analysis["close"]
                    tp, sl = calculate_targets(entry_price)
                    
                    # Send BUY message
                    msg = format_buy_message(entry_price, tp, sl, state.timeframe)
                    sent = await send_telegram_message(bot, chat_id, msg)
                    
                    if sent:
                        # Open virtual position
                        state.position_open = True
                        state.entry_price = entry_price
                        state.entry_time = datetime.now(timezone.utc)
                        
                        logger.info(f"Position opened @ {entry_price:.4f}")
            
        except Exception as e:
            logger.error(f"Error in signal loop: {e}")
        
        await asyncio.sleep(POLL_INTERVAL)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main() -> None:
    """Main function to start the bot."""
    
    # Get credentials from environment
    tg_token = os.environ.get("TG_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    
    if not tg_token:
        logger.error("TG_TOKEN environment variable not set!")
        print("ERROR: Please set TG_TOKEN in Replit Secrets")
        return
    
    if not chat_id:
        logger.error("TG_CHAT_ID environment variable not set!")
        print("ERROR: Please set TG_CHAT_ID in Replit Secrets")
        return
    
    logger.info(f"Starting XRP/USDT Signals Bot - {SYMBOL} on {state.timeframe}")
    
    # Create application
    application = Application.builder().token(tg_token).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("settf", cmd_settf))
    application.add_handler(CommandHandler("on", cmd_on))
    application.add_handler(CommandHandler("off", cmd_off))
    
    # Get bot instance for signal loop
    bot = application.bot
    
    # Initialize application
    await application.initialize()
    await application.start()
    
    # Start polling for commands in the background
    await application.updater.start_polling(drop_pending_updates=True)
    
    logger.info("Bot is running! Press Ctrl+C to stop.")
    print("=" * 50)
    print(f"XRP/USDT Signals Bot Started")
    print(f"Symbol: {SYMBOL}")
    print(f"Timeframe: {state.timeframe}")
    print(f"Strategy: EMA{EMA_SHORT}/EMA{EMA_LONG} + Breakout")
    print(f"TP: +{TAKE_PROFIT_PCT}% | SL: -{STOP_LOSS_PCT}%")
    print("=" * 50)
    
    # Run signal loop
    try:
        await signal_loop(bot, chat_id)
    except asyncio.CancelledError:
        logger.info("Bot stopped")
    finally:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

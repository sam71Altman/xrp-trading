#!/usr/bin/env python3
"""
XRP/USDT Telegram Signals Bot V3
بوت إشارات تداول يرسل إشارات دخول/خروج لزوج XRP/USDT
V3: Backtesting + Signal Score + Adaptive Cooldown + Session Awareness + سجل الأداء
"""

import os
import csv
import time
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict

import requests
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ============================================================================
# CONFIGURATION
# ============================================================================

TIMEFRAME = "1m"
SYMBOL = "XRPUSDT"
SYMBOL_DISPLAY = "XRP/USDT"

EMA_SHORT = 20
EMA_LONG = 50
BREAKOUT_CANDLES = 5

TAKE_PROFIT_PCT = 0.40
STOP_LOSS_PCT = 0.30
TRAILING_TRIGGER_PCT = 0.25

RANGE_FILTER_THRESHOLD = 0.001
VOLUME_LOOKBACK = 20
TREND_LOOKBACK = 30

COOLDOWN_NORMAL = 60
COOLDOWN_AFTER_SL = 180
COOLDOWN_STREAK_WIN = 30
COOLDOWN_PAUSE_MINUTES = 10

MIN_WIN_RATE = 45.0
MIN_SIGNAL_SCORE = 6

POLL_INTERVAL = 10
KLINE_LIMIT = 200
BACKTEST_DAYS = 30

TRADES_FILE = "trades.csv"

BINANCE_APIS = [
    "https://api.binance.us/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api2.binance.com/api/v3/klines",
    "https://api3.binance.com/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]

# ============================================================================
# LOGGING
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
    def __init__(self):
        self.position_open: bool = False
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[datetime] = None
        self.entry_timeframe: Optional[str] = None
        self.last_message_time: float = 0
        self.signals_enabled: bool = True
        self.timeframe: str = TIMEFRAME
        self.last_close: Optional[float] = None
        self.last_signal_type: Optional[str] = None
        self.consecutive_errors: int = 0
        self.error_alerted: bool = False
        self.trailing_activated: bool = False
        self.candles_below_ema: int = 0
        self.last_exit_type: Optional[str] = None
        self.current_cooldown: int = COOLDOWN_NORMAL
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0
        self.pause_until: Optional[datetime] = None
        self.pause_alerted: bool = False
        self.backtest_warned: bool = False
        self.last_signal_score: int = 0
        self.last_signal_reasons: List[str] = []
        self.backtest_stats: Dict = {}

state = BotState()

# ============================================================================
# TRADES LOG
# ============================================================================

def init_trades_file():
    """إنشاء ملف السجل إذا لم يكن موجوداً"""
    if not os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['التاريخ', 'النوع', 'السبب', 'السعر', 'النتيجة%'])

def log_trade(trade_type: str, reason: str, price: float, result_pct: Optional[float] = None):
    """تسجيل صفقة في الملف"""
    init_trades_file()
    with open(TRADES_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        result_str = f"{result_pct:.2f}" if result_pct is not None else ""
        writer.writerow([
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            trade_type,
            reason,
            f"{price:.4f}",
            result_str
        ])

def get_trade_stats() -> Dict:
    """الحصول على إحصائيات الصفقات"""
    stats = {
        'total': 0,
        'wins': 0,
        'losses': 0,
        'win_rate': 0.0,
        'last_5': []
    }
    
    if not os.path.exists(TRADES_FILE):
        return stats
    
    trades = []
    with open(TRADES_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 5 and row[1] == 'EXIT' and row[4]:
                try:
                    result = float(row[4])
                    trades.append({
                        'date': row[0],
                        'reason': row[2],
                        'price': row[3],
                        'result': result
                    })
                    if result >= 0:
                        stats['wins'] += 1
                    else:
                        stats['losses'] += 1
                except ValueError:
                    pass
    
    stats['total'] = stats['wins'] + stats['losses']
    if stats['total'] > 0:
        stats['win_rate'] = (stats['wins'] / stats['total']) * 100
    
    stats['last_5'] = trades[-5:][::-1]
    
    return stats

# ============================================================================
# BINANCE API
# ============================================================================

def get_klines(symbol: str, interval: str, limit: int = KLINE_LIMIT) -> Optional[List[dict]]:
    """جلب بيانات الشموع من Binance API"""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    for api_url in BINANCE_APIS:
        try:
            response = requests.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            candles = []
            for c in data:
                candles.append({
                    "open_time": int(c[0]),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                })
            return candles
            
        except requests.RequestException as e:
            logger.debug(f"API {api_url} failed: {e}")
            continue
    
    logger.error("All Binance API endpoints failed")
    return None

def get_historical_klines(symbol: str, interval: str, days: int = BACKTEST_DAYS) -> Optional[List[dict]]:
    """جلب بيانات تاريخية للـ Backtest"""
    if interval == "1m":
        limit = min(days * 24 * 60, 1000)
    elif interval == "5m":
        limit = min(days * 24 * 12, 1000)
    else:
        limit = 500
    
    return get_klines(symbol, interval, limit)

# ============================================================================
# EMA CALCULATION (بدون pandas)
# ============================================================================

def calculate_ema(prices: List[float], period: int) -> List[float]:
    """حساب EMA بدون مكتبات خارجية"""
    if len(prices) < period:
        return []
    
    ema_values = []
    multiplier = 2 / (period + 1)
    
    sma = sum(prices[:period]) / period
    ema_values.append(sma)
    
    for i in range(period, len(prices)):
        ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
        ema_values.append(ema)
    
    return ema_values

# ============================================================================
# BACKTESTING
# ============================================================================

def run_backtest(candles: List[dict]) -> Dict:
    """تشغيل Backtest على البيانات التاريخية"""
    if len(candles) < EMA_LONG + BREAKOUT_CANDLES + 10:
        return {"error": "بيانات غير كافية للـ Backtest"}
    
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    volumes = [c["volume"] for c in candles]
    
    ema_short_vals = calculate_ema(closes, EMA_SHORT)
    ema_long_vals = calculate_ema(closes, EMA_LONG)
    
    if len(ema_short_vals) < 50 or len(ema_long_vals) < 50:
        return {"error": "فشل حساب EMA للـ Backtest"}
    
    trades = []
    position_open = False
    entry_price = 0
    entry_idx = 0
    
    offset = len(closes) - len(ema_short_vals)
    ema_long_offset = len(closes) - len(ema_long_vals)
    
    for i in range(EMA_LONG + BREAKOUT_CANDLES, len(closes) - 1):
        ema_s_idx = i - offset
        ema_l_idx = i - ema_long_offset
        
        if ema_s_idx < 0 or ema_l_idx < 0 or ema_s_idx >= len(ema_short_vals) or ema_l_idx >= len(ema_long_vals):
            continue
        
        current_close = closes[i]
        ema_short = ema_short_vals[ema_s_idx]
        ema_long = ema_long_vals[ema_l_idx]
        
        prev_highs = highs[i - BREAKOUT_CANDLES:i]
        highest_high = max(prev_highs) if prev_highs else current_close
        
        if position_open:
            pnl_pct = ((current_close - entry_price) / entry_price) * 100
            
            if pnl_pct >= TAKE_PROFIT_PCT:
                trades.append(pnl_pct)
                position_open = False
            elif pnl_pct <= -STOP_LOSS_PCT:
                trades.append(pnl_pct)
                position_open = False
            elif current_close < ema_short:
                trades.append(pnl_pct)
                position_open = False
        else:
            ema_bullish = ema_short > ema_long
            breakout = current_close > highest_high
            ema_diff = abs(ema_short - ema_long) / ema_long if ema_long != 0 else 0
            range_ok = ema_diff >= RANGE_FILTER_THRESHOLD
            
            vol_start = max(0, i - VOLUME_LOOKBACK)
            avg_vol = sum(volumes[vol_start:i]) / VOLUME_LOOKBACK if i > VOLUME_LOOKBACK else volumes[i]
            volume_ok = volumes[i] > avg_vol
            
            if ema_bullish and breakout and range_ok and volume_ok:
                position_open = True
                entry_price = current_close
                entry_idx = i
    
    if len(trades) == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0
        }
    
    wins = sum(1 for t in trades if t >= 0)
    losses = len(trades) - wins
    win_rate = (wins / len(trades)) * 100
    expectancy = sum(trades) / len(trades)
    
    cumulative = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cumulative += t
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "max_drawdown": max_dd
    }

# ============================================================================
# SIGNAL SCORE
# ============================================================================

def calculate_signal_score(analysis: dict, candles: List[dict]) -> tuple:
    """حساب نقاط الإشارة من 10"""
    score = 0
    reasons = []
    
    if analysis.get("ema_bullish"):
        score += 3
        reasons.append(f"✅ EMA{EMA_SHORT} > EMA{EMA_LONG} (+3)")
    
    if analysis.get("breakout"):
        score += 3
        reasons.append(f"✅ كسر قمة {BREAKOUT_CANDLES} شموع (+3)")
    
    if analysis.get("volume_confirmed"):
        score += 2
        reasons.append("✅ حجم أعلى من المتوسط (+2)")
    
    if len(candles) >= TREND_LOOKBACK:
        closes = [c["close"] for c in candles[-TREND_LOOKBACK:]]
        if closes[-1] > closes[0]:
            score += 2
            reasons.append(f"✅ اتجاه صاعد (آخر {TREND_LOOKBACK} شمعة) (+2)")
    
    return score, reasons

# ============================================================================
# SESSION AWARENESS
# ============================================================================

def is_low_liquidity_session() -> bool:
    """فحص إذا كانت الجلسة منخفضة السيولة"""
    now = datetime.now(timezone.utc)
    hour = now.hour
    
    if 21 <= hour or hour < 1:
        return True
    if 5 <= hour < 7:
        return True
    
    return False

# ============================================================================
# STRATEGY LOGIC V3
# ============================================================================

def analyze_market(candles: List[dict]) -> dict:
    """تحليل بيانات السوق وتوليد الإشارات"""
    if not candles or len(candles) < EMA_LONG + BREAKOUT_CANDLES:
        return {"error": "بيانات غير كافية"}
    
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    volumes = [c["volume"] for c in candles]
    
    ema_short_vals = calculate_ema(closes, EMA_SHORT)
    ema_long_vals = calculate_ema(closes, EMA_LONG)
    
    if not ema_short_vals or not ema_long_vals:
        return {"error": "فشل حساب EMA"}
    
    current_close = closes[-1]
    prev_close = closes[-2] if len(closes) >= 2 else current_close
    current_ema_short = ema_short_vals[-1]
    prev_ema_short = ema_short_vals[-2] if len(ema_short_vals) >= 2 else current_ema_short
    current_ema_long = ema_long_vals[-1]
    
    prev_highs = highs[-(BREAKOUT_CANDLES + 1):-1]
    highest_high = max(prev_highs) if prev_highs else current_close
    
    current_volume = volumes[-1]
    avg_volume = sum(volumes[-VOLUME_LOOKBACK:]) / VOLUME_LOOKBACK if len(volumes) >= VOLUME_LOOKBACK else current_volume
    
    ema_diff_pct = abs(current_ema_short - current_ema_long) / current_ema_long if current_ema_long != 0 else 0
    
    state.last_close = current_close
    
    return {
        "close": current_close,
        "prev_close": prev_close,
        "ema_short": current_ema_short,
        "prev_ema_short": prev_ema_short,
        "ema_long": current_ema_long,
        "highest_high": highest_high,
        "ema_bullish": current_ema_short > current_ema_long,
        "breakout": current_close > highest_high,
        "current_volume": current_volume,
        "avg_volume": avg_volume,
        "volume_confirmed": current_volume > avg_volume,
        "ema_diff_pct": ema_diff_pct,
        "range_confirmed": ema_diff_pct >= RANGE_FILTER_THRESHOLD,
    }

def check_buy_signal(analysis: dict, candles: List[dict]) -> bool:
    """فحص شروط الدخول مع Score"""
    if "error" in analysis:
        return False
    
    if is_low_liquidity_session():
        logger.debug("تخطي - جلسة منخفضة السيولة")
        return False
    
    if state.pause_until and datetime.now(timezone.utc) < state.pause_until:
        return False
    
    if not analysis["ema_bullish"]:
        return False
    if not analysis["breakout"]:
        return False
    if not analysis["range_confirmed"]:
        return False
    if not analysis["volume_confirmed"]:
        return False
    
    score, reasons = calculate_signal_score(analysis, candles)
    state.last_signal_score = score
    state.last_signal_reasons = reasons
    
    if score < MIN_SIGNAL_SCORE:
        logger.debug(f"تخطي - Score منخفض: {score}/10")
        return False
    
    return True

def check_exit_signal(analysis: dict) -> Optional[str]:
    """فحص شروط الخروج"""
    if "error" in analysis or not state.position_open or state.entry_price is None:
        return None
    
    current_close = analysis["close"]
    entry = state.entry_price
    pnl_pct = ((current_close - entry) / entry) * 100
    
    if pnl_pct >= TAKE_PROFIT_PCT:
        return "tp"
    
    if pnl_pct <= -STOP_LOSS_PCT:
        return "sl"
    
    if not state.trailing_activated:
        if pnl_pct >= TRAILING_TRIGGER_PCT:
            state.trailing_activated = True
            logger.info(f"تم تفعيل Trailing Stop @ {current_close:.4f}")
    
    if state.trailing_activated:
        if current_close <= entry:
            return "trailing_sl"
    
    if current_close < analysis["ema_short"]:
        state.candles_below_ema += 1
    else:
        state.candles_below_ema = 0
    
    if state.candles_below_ema >= 2:
        return "ema_confirmation"
    
    return None

def calculate_targets(entry_price: float) -> tuple:
    tp = entry_price * (1 + TAKE_PROFIT_PCT / 100)
    sl = entry_price * (1 - STOP_LOSS_PCT / 100)
    return tp, sl

def calculate_pnl(entry: float, exit_price: float) -> float:
    return ((exit_price - entry) / entry) * 100

def get_trade_duration_minutes() -> int:
    if state.entry_time is None:
        return 0
    now = datetime.now(timezone.utc)
    duration = now - state.entry_time
    return int(duration.total_seconds() / 60)

def reset_position_state():
    state.position_open = False
    state.entry_price = None
    state.entry_time = None
    state.entry_timeframe = None
    state.trailing_activated = False
    state.candles_below_ema = 0

def update_cooldown_after_exit(exit_type: str):
    state.last_exit_type = exit_type
    
    if exit_type == "sl":
        state.consecutive_losses += 1
        state.consecutive_wins = 0
        state.current_cooldown = COOLDOWN_AFTER_SL
        
        if state.consecutive_losses >= 2:
            state.pause_until = datetime.now(timezone.utc) + timedelta(minutes=COOLDOWN_PAUSE_MINUTES)
            state.pause_alerted = False
            logger.info(f"إيقاف مؤقت بعد {state.consecutive_losses} خسائر متتالية")
    
    elif exit_type == "tp":
        state.consecutive_wins += 1
        state.consecutive_losses = 0
        
        if state.consecutive_wins >= 3:
            state.current_cooldown = COOLDOWN_STREAK_WIN
        else:
            state.current_cooldown = COOLDOWN_NORMAL
    
    else:
        state.consecutive_losses = 0
        state.consecutive_wins = 0
        state.current_cooldown = COOLDOWN_NORMAL

# ============================================================================
# MESSAGE FORMATTING (عربي احترافي)
# ============================================================================

def get_current_time_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def format_buy_message(entry: float, tp: float, sl: float, timeframe: str, score: int) -> str:
    score_label = "قوية 🔥" if score >= 8 else "عادية"
    reasons_text = "\n".join(state.last_signal_reasons[:3]) if state.last_signal_reasons else ""
    
    return (
        f"📥 *إشارة دخول (BUY)*\n\n"
        f"📈 *الزوج:* {SYMBOL_DISPLAY}\n"
        f"📊 *الفريم:* {timeframe}\n"
        f"💰 *سعر الدخول:* {entry:.4f}\n"
        f"🎯 *جني الأرباح:* {tp:.4f} (+{TAKE_PROFIT_PCT}%)\n"
        f"🛑 *وقف الخسارة:* {sl:.4f} (-{STOP_LOSS_PCT}%)\n\n"
        f"⭐ *Score:* {score}/10 ({score_label})\n\n"
        f"📋 *أسباب الدخول:*\n{reasons_text}\n\n"
        f"🕐 *الوقت:* {get_current_time_str()}"
    )

def format_exit_message(entry: float, exit_price: float, pnl: float, reason: str, duration_min: int) -> str:
    reason_text = {
        "tp": "وصول الهدف (TP) ✅",
        "sl": "وقف الخسارة (SL) ❌",
        "trailing_sl": "Trailing Stop Loss 🔒",
        "ema_confirmation": f"تأكيد كسر EMA{EMA_SHORT} 📊",
    }.get(reason, "خروج يدوي")
    
    pnl_sign = "+" if pnl >= 0 else ""
    status_emoji = "✅" if pnl >= 0 else "❌"
    
    return (
        f"📤 *إشارة خروج*\n\n"
        f"📈 *الزوج:* {SYMBOL_DISPLAY}\n"
        f"💰 *سعر الدخول:* {entry:.4f}\n"
        f"💵 *سعر الخروج:* {exit_price:.4f}\n"
        f"📊 *النتيجة:* {pnl_sign}{pnl:.2f}%\n\n"
        f"{status_emoji} *السبب:* {reason_text}\n"
        f"⏱️ *مدة الصفقة:* {duration_min} دقيقة\n"
        f"🕐 *الوقت:* {get_current_time_str()}"
    )

def format_status_message() -> str:
    status = "✅ نشط" if state.signals_enabled else "⏸️ متوقف"
    position = "📈 مفتوح" if state.position_open else "📉 مغلق"
    
    msg = (
        f"ℹ️ *حالة البوت V3*\n\n"
        f"🔔 *الإشارات:* {status}\n"
        f"📊 *الفريم:* {state.timeframe}\n"
        f"📈 *المركز:* {position}\n"
    )
    
    if state.position_open and state.entry_price:
        msg += f"💰 *سعر الدخول:* {state.entry_price:.4f}\n"
        if state.trailing_activated:
            msg += f"🔒 *Trailing:* مفعّل\n"
        if state.last_close:
            pnl = calculate_pnl(state.entry_price, state.last_close)
            pnl_sign = "+" if pnl >= 0 else ""
            msg += f"📉 *الربح الحالي:* {pnl_sign}{pnl:.2f}%\n"
        duration = get_trade_duration_minutes()
        msg += f"⏱️ *مدة الصفقة:* {duration} دقيقة\n"
    
    if state.pause_until and datetime.now(timezone.utc) < state.pause_until:
        remaining = (state.pause_until - datetime.now(timezone.utc)).seconds // 60
        msg += f"⏳ *إيقاف مؤقت:* {remaining} دقيقة متبقية\n"
    
    if state.last_close:
        msg += f"🕯️ *آخر سعر:* {state.last_close:.4f}\n"
    
    msg += f"⏳ *Cooldown:* {state.current_cooldown} ثانية\n"
    
    if state.backtest_stats:
        msg += f"\n📊 *Backtest:* Win Rate {state.backtest_stats.get('win_rate', 0):.1f}%\n"
    
    msg += f"🕐 *التحديث:* {get_current_time_str()}"
    
    return msg

def format_welcome_message() -> str:
    return (
        f"🤖 *مرحباً بك في بوت إشارات {SYMBOL_DISPLAY} V3*\n\n"
        f"📊 *الاستراتيجية:* EMA{EMA_SHORT}/EMA{EMA_LONG} + Breakout\n"
        f"🎯 *الهدف:* +{TAKE_PROFIT_PCT}%\n"
        f"🛑 *وقف الخسارة:* -{STOP_LOSS_PCT}%\n"
        f"🔒 *Trailing:* +{TRAILING_TRIGGER_PCT}%\n\n"
        f"✨ *ميزات V3:*\n"
        f"• Backtest تلقائي قبل الإشارات\n"
        f"• Signal Score من 10\n"
        f"• Cooldown تكيفي ذكي\n"
        f"• فلتر جلسات السيولة\n"
        f"• سجل الأداء والإحصائيات\n\n"
        f"استخدم الأزرار للتحكم 👇\n"
    )

def format_rules_message() -> str:
    return (
        f"📜 *قواعد التداول V3*\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*شروط الدخول (BUY):*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"1️⃣ EMA{EMA_SHORT} > EMA{EMA_LONG} (+3 نقاط)\n"
        f"2️⃣ كسر قمة {BREAKOUT_CANDLES} شموع (+3 نقاط)\n"
        f"3️⃣ حجم > متوسط {VOLUME_LOOKBACK} شمعة (+2 نقاط)\n"
        f"4️⃣ اتجاه {TREND_LOOKBACK} شمعة صاعد (+2 نقاط)\n"
        f"⭐ الحد الأدنى للدخول: {MIN_SIGNAL_SCORE}/10\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*شروط الخروج (EXIT):*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ *TP:* +{TAKE_PROFIT_PCT}%\n"
        f"❌ *SL:* -{STOP_LOSS_PCT}%\n"
        f"🔒 *Trailing:* عند +{TRAILING_TRIGGER_PCT}%\n"
        f"📊 *EMA:* شمعتين تحت EMA{EMA_SHORT}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*Backtest:*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"• الحد الأدنى Win Rate: {MIN_WIN_RATE}%\n"
        f"• إيقاف تلقائي عند ضعف الأداء\n"
    )

def format_stats_message() -> str:
    stats = get_trade_stats()
    
    msg = (
        f"📈 *إحصائيات الأداء*\n\n"
        f"📊 *إجمالي الصفقات:* {stats['total']}\n"
        f"✅ *الصفقات الرابحة:* {stats['wins']}\n"
        f"❌ *الصفقات الخاسرة:* {stats['losses']}\n"
        f"📉 *Win Rate:* {stats['win_rate']:.1f}%\n\n"
    )
    
    if stats['last_5']:
        msg += "━━━━━━━━━━━━━━━━━━━━\n"
        msg += "*آخر 5 صفقات:*\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n"
        for t in stats['last_5']:
            result_sign = "+" if t['result'] >= 0 else ""
            emoji = "✅" if t['result'] >= 0 else "❌"
            msg += f"{emoji} {result_sign}{t['result']:.2f}% | {t['reason']}\n"
    
    if state.backtest_stats:
        msg += f"\n📊 *آخر Backtest:*\n"
        msg += f"• صفقات: {state.backtest_stats.get('trades', 0)}\n"
        msg += f"• Win Rate: {state.backtest_stats.get('win_rate', 0):.1f}%\n"
        msg += f"• Expectancy: {state.backtest_stats.get('expectancy', 0):.2f}%\n"
    
    msg += f"\n🕐 *التحديث:* {get_current_time_str()}"
    
    return msg

def format_signal_reasons_message() -> str:
    if not state.last_signal_reasons:
        return "❓ لا توجد إشارة حديثة لعرض أسبابها"
    
    msg = (
        f"🧠 *لماذا هذه الإشارة؟*\n\n"
        f"⭐ *Score:* {state.last_signal_score}/10\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*تفاصيل النقاط:*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
    )
    
    for reason in state.last_signal_reasons:
        msg += f"{reason}\n"
    
    return msg

# ============================================================================
# INLINE KEYBOARD (عربي)
# ============================================================================

def get_main_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("▶️ تشغيل", callback_data="on"),
            InlineKeyboardButton("⏸ إيقاف", callback_data="off"),
        ],
        [
            InlineKeyboardButton("📊 الحالة", callback_data="status"),
            InlineKeyboardButton("📈 الإحصائيات", callback_data="stats"),
        ],
        [
            InlineKeyboardButton("⏱ 1 دقيقة", callback_data="tf_1m"),
            InlineKeyboardButton("⏱ 5 دقائق", callback_data="tf_5m"),
        ],
        [
            InlineKeyboardButton("📜 القواعد", callback_data="rules"),
            InlineKeyboardButton("🧠 لماذا؟", callback_data="why"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)

# ============================================================================
# ANTI-SPAM & MESSAGE SENDING
# ============================================================================

def can_send_message() -> bool:
    return (time.time() - state.last_message_time) >= state.current_cooldown

async def send_signal_message(bot: Bot, chat_id: str, message: str, signal_type: str) -> bool:
    if not can_send_message():
        logger.info(f"تخطي بسبب cooldown ({signal_type})")
        return False
    
    if state.last_signal_type == signal_type and signal_type == "buy" and state.position_open:
        return False
    
    try:
        await bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
        state.last_message_time = time.time()
        state.last_signal_type = signal_type
        logger.info(f"تم إرسال إشارة {signal_type}")
        return True
    except Exception as e:
        logger.error(f"فشل إرسال الرسالة: {e}")
        return False

# ============================================================================
# COMMAND HANDLERS (عربي)
# ============================================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        format_welcome_message(),
        reply_markup=get_main_keyboard(),
        parse_mode="Markdown"
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        format_status_message(),
        reply_markup=get_main_keyboard(),
        parse_mode="Markdown"
    )

async def cmd_rules(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        format_rules_message(),
        reply_markup=get_main_keyboard(),
        parse_mode="Markdown"
    )

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        format_stats_message(),
        reply_markup=get_main_keyboard(),
        parse_mode="Markdown"
    )

async def cmd_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state.signals_enabled = True
    state.backtest_warned = False
    await update.message.reply_text(
        "✅ تم تشغيل الإشارات",
        reply_markup=get_main_keyboard()
    )

async def cmd_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state.signals_enabled = False
    await update.message.reply_text(
        "⏸️ تم إيقاف الإشارات",
        reply_markup=get_main_keyboard()
    )

async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("❌ استخدم: /الفريم 1m أو /الفريم 5m")
        return
    
    new_tf = context.args[0].lower()
    if new_tf not in ["1m", "5m"]:
        await update.message.reply_text("❌ الفريم غير صحيح. استخدم 1m أو 5m")
        return
    
    state.timeframe = new_tf
    state.backtest_warned = False
    await update.message.reply_text(
        f"✅ تم تغيير الفريم إلى {new_tf}",
        reply_markup=get_main_keyboard()
    )

# ============================================================================
# CALLBACK QUERY HANDLER
# ============================================================================

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data == "on":
        state.signals_enabled = True
        state.backtest_warned = False
        await query.edit_message_text(
            "✅ تم تشغيل الإشارات\n\n" + format_status_message(),
            reply_markup=get_main_keyboard(),
            parse_mode="Markdown"
        )
    
    elif data == "off":
        state.signals_enabled = False
        await query.edit_message_text(
            "⏸️ تم إيقاف الإشارات\n\n" + format_status_message(),
            reply_markup=get_main_keyboard(),
            parse_mode="Markdown"
        )
    
    elif data == "status":
        await query.edit_message_text(
            format_status_message(),
            reply_markup=get_main_keyboard(),
            parse_mode="Markdown"
        )
    
    elif data == "stats":
        await query.edit_message_text(
            format_stats_message(),
            reply_markup=get_main_keyboard(),
            parse_mode="Markdown"
        )
    
    elif data == "rules":
        await query.edit_message_text(
            format_rules_message(),
            reply_markup=get_main_keyboard(),
            parse_mode="Markdown"
        )
    
    elif data == "why":
        await query.edit_message_text(
            format_signal_reasons_message(),
            reply_markup=get_main_keyboard(),
            parse_mode="Markdown"
        )
    
    elif data == "tf_1m":
        state.timeframe = "1m"
        state.backtest_warned = False
        await query.edit_message_text(
            f"✅ تم تغيير الفريم إلى 1m\n\n" + format_status_message(),
            reply_markup=get_main_keyboard(),
            parse_mode="Markdown"
        )
    
    elif data == "tf_5m":
        state.timeframe = "5m"
        state.backtest_warned = False
        await query.edit_message_text(
            f"✅ تم تغيير الفريم إلى 5m\n\n" + format_status_message(),
            reply_markup=get_main_keyboard(),
            parse_mode="Markdown"
        )

# ============================================================================
# BACKGROUND SIGNAL LOOP
# ============================================================================

async def signal_loop(bot: Bot, chat_id: str) -> None:
    """حلقة فحص الإشارات V3"""
    logger.info(f"بدء حلقة الإشارات V3 - التحديث كل {POLL_INTERVAL} ثانية")
    
    init_trades_file()
    
    while True:
        try:
            if not state.signals_enabled:
                await asyncio.sleep(POLL_INTERVAL)
                continue
            
            if state.pause_until and datetime.now(timezone.utc) < state.pause_until:
                if not state.pause_alerted:
                    try:
                        await bot.send_message(
                            chat_id=chat_id,
                            text="🛑 تم إيقاف الإشارات مؤقتًا بعد خسارتين متتاليتين",
                            parse_mode="Markdown"
                        )
                        state.pause_alerted = True
                    except:
                        pass
                await asyncio.sleep(POLL_INTERVAL)
                continue
            else:
                state.pause_until = None
                state.pause_alerted = False
            
            candles = get_klines(SYMBOL, state.timeframe)
            
            if candles is None:
                state.consecutive_errors += 1
                if state.consecutive_errors >= 5 and not state.error_alerted:
                    try:
                        await bot.send_message(
                            chat_id=chat_id,
                            text="⚠️ مشكلة في الاتصال بـ Binance API",
                            parse_mode="Markdown"
                        )
                        state.error_alerted = True
                    except:
                        pass
                await asyncio.sleep(POLL_INTERVAL)
                continue
            
            state.consecutive_errors = 0
            state.error_alerted = False
            
            if not state.position_open and not state.backtest_warned:
                hist_candles = get_historical_klines(SYMBOL, state.timeframe, BACKTEST_DAYS)
                if hist_candles:
                    bt_stats = run_backtest(hist_candles)
                    state.backtest_stats = bt_stats
                    
                    if "error" not in bt_stats and bt_stats.get("win_rate", 0) < MIN_WIN_RATE:
                        try:
                            await bot.send_message(
                                chat_id=chat_id,
                                text=f"⚠️ تم إيقاف الإشارات مؤقتًا بسبب ضعف الأداء الإحصائي\n"
                                     f"(Win Rate {bt_stats['win_rate']:.1f}% أقل من {MIN_WIN_RATE}%)",
                                parse_mode="Markdown"
                            )
                            state.backtest_warned = True
                        except:
                            pass
                        await asyncio.sleep(POLL_INTERVAL)
                        continue
            
            if state.backtest_warned:
                await asyncio.sleep(POLL_INTERVAL)
                continue
            
            analysis = analyze_market(candles)
            
            if "error" in analysis:
                await asyncio.sleep(POLL_INTERVAL)
                continue
            
            if state.position_open:
                exit_reason = check_exit_signal(analysis)
                if exit_reason:
                    exit_price = analysis["close"]
                    pnl = calculate_pnl(state.entry_price, exit_price)
                    duration = get_trade_duration_minutes()
                    
                    log_trade("EXIT", exit_reason.upper(), exit_price, pnl)
                    
                    msg = format_exit_message(state.entry_price, exit_price, pnl, exit_reason, duration)
                    sent = await send_signal_message(bot, chat_id, msg, "exit")
                    
                    if sent:
                        update_cooldown_after_exit(exit_reason)
                        reset_position_state()
                        logger.info(f"إغلاق المركز: {exit_reason} @ {exit_price:.4f} (PnL: {pnl:.2f}%)")
            
            else:
                if check_buy_signal(analysis, candles):
                    entry_price = analysis["close"]
                    tp, sl = calculate_targets(entry_price)
                    
                    log_trade("BUY", "SIGNAL", entry_price, None)
                    
                    msg = format_buy_message(entry_price, tp, sl, state.timeframe, state.last_signal_score)
                    sent = await send_signal_message(bot, chat_id, msg, "buy")
                    
                    if sent:
                        state.position_open = True
                        state.entry_price = entry_price
                        state.entry_time = datetime.now(timezone.utc)
                        state.entry_timeframe = state.timeframe
                        state.trailing_activated = False
                        state.candles_below_ema = 0
                        logger.info(f"فتح مركز @ {entry_price:.4f} (Score: {state.last_signal_score}/10)")
        
        except Exception as e:
            logger.error(f"خطأ في حلقة الإشارات: {e}")
        
        await asyncio.sleep(POLL_INTERVAL)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main() -> None:
    tg_token = os.environ.get("TG_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    
    if not tg_token:
        logger.error("TG_TOKEN غير موجود!")
        print("❌ الرجاء تعيين TG_TOKEN في Replit Secrets")
        return
    
    if not chat_id:
        logger.error("TG_CHAT_ID غير موجود!")
        print("❌ الرجاء تعيين TG_CHAT_ID في Replit Secrets")
        return
    
    logger.info(f"بدء بوت إشارات {SYMBOL_DISPLAY} V3")
    
    application = Application.builder().token(tg_token).build()
    
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("on", cmd_on))
    application.add_handler(CommandHandler("off", cmd_off))
    application.add_handler(CommandHandler("rules", cmd_rules))
    application.add_handler(CommandHandler("stats", cmd_stats))
    application.add_handler(CommandHandler("settf", cmd_timeframe))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    bot = application.bot
    
    await application.initialize()
    await application.start()
    await application.updater.start_polling(drop_pending_updates=True)
    
    print("=" * 50)
    print(f"🚀 بوت إشارات {SYMBOL_DISPLAY} V3")
    print(f"📊 الفريم: {state.timeframe}")
    print(f"📈 الاستراتيجية: EMA{EMA_SHORT}/EMA{EMA_LONG} + Score")
    print(f"🎯 TP: +{TAKE_PROFIT_PCT}% | SL: -{STOP_LOSS_PCT}%")
    print(f"⭐ Min Score: {MIN_SIGNAL_SCORE}/10")
    print(f"📉 Min Win Rate: {MIN_WIN_RATE}%")
    print("=" * 50)
    
    try:
        await signal_loop(bot, chat_id)
    except asyncio.CancelledError:
        logger.info("تم إيقاف البوت")
    finally:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

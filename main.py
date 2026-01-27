#!/usr/bin/env python3
"""
XRP/USDT Telegram Signals Bot V3 + Paper Trading
بوت إشارات تداول يرسل إشارات دخول/خروج لزوج XRP/USDT
V3: Backtesting + Signal Score + Adaptive Cooldown + Session Awareness
Paper Trading: تداول افتراضي بدون اتصال بمنصة حقيقية
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

MODE = "PAPER"

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

START_BALANCE = 1000.0
FIXED_TRADE_SIZE = 100.0

TRADES_FILE = "trades.csv"
PAPER_TRADES_FILE = "paper_trades.csv"

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
# PAPER TRADING STATE
# ============================================================================

class PaperTradingState:
    def __init__(self):
        self.balance: float = START_BALANCE
        self.position_qty: float = 0.0
        self.entry_reason: str = ""
        self.load_balance()
    
    def load_balance(self):
        """تحميل الرصيد من آخر صفقة"""
        if os.path.exists(PAPER_TRADES_FILE):
            try:
                with open(PAPER_TRADES_FILE, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    rows = list(reader)
                    if rows:
                        last_row = rows[-1]
                        if len(last_row) >= 7 and last_row[6]:
                            self.balance = float(last_row[6])
            except:
                self.balance = START_BALANCE
    
    def reset(self):
        """إعادة تعيين الرصيد"""
        self.balance = START_BALANCE
        self.position_qty = 0.0
        self.entry_reason = ""
        if os.path.exists(PAPER_TRADES_FILE):
            os.remove(PAPER_TRADES_FILE)
        init_paper_trades_file()

paper_state = PaperTradingState()

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
        self.pending_reset: bool = False

state = BotState()

# ============================================================================
# PAPER TRADES LOG
# ============================================================================

def init_paper_trades_file():
    """إنشاء ملف سجل Paper Trading"""
    if not os.path.exists(PAPER_TRADES_FILE):
        with open(PAPER_TRADES_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'action', 'entry_price', 'exit_price',
                'pnl_percent', 'pnl_usdt', 'balance_after', 'score',
                'entry_reason', 'exit_reason', 'duration_minutes'
            ])

def log_paper_trade(action: str, entry_price: float, exit_price: Optional[float],
                    pnl_pct: Optional[float], pnl_usdt: Optional[float],
                    balance_after: float, score: int, entry_reason: str,
                    exit_reason: str, duration_min: int):
    """تسجيل صفقة Paper Trading"""
    init_paper_trades_file()
    with open(PAPER_TRADES_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            action,
            f"{entry_price:.4f}" if entry_price else "",
            f"{exit_price:.4f}" if exit_price else "",
            f"{pnl_pct:.2f}" if pnl_pct is not None else "",
            f"{pnl_usdt:.2f}" if pnl_usdt is not None else "",
            f"{balance_after:.2f}",
            score,
            entry_reason,
            exit_reason,
            duration_min
        ])

def get_paper_trades(limit: int = 5) -> List[Dict]:
    """الحصول على آخر الصفقات"""
    trades = []
    if not os.path.exists(PAPER_TRADES_FILE):
        return trades
    
    with open(PAPER_TRADES_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        rows = list(reader)
        
        exit_trades = [r for r in rows if len(r) >= 11 and r[1] == 'EXIT']
        
        for row in exit_trades[-limit:][::-1]:
            try:
                trades.append({
                    'timestamp': row[0],
                    'entry_price': row[2],
                    'exit_price': row[3],
                    'pnl_pct': float(row[4]) if row[4] else 0,
                    'pnl_usdt': float(row[5]) if row[5] else 0,
                    'balance': float(row[6]) if row[6] else 0,
                    'exit_reason': row[9]
                })
            except:
                pass
    
    return trades

def get_paper_stats() -> Dict:
    """إحصائيات Paper Trading"""
    stats = {
        'total': 0,
        'wins': 0,
        'losses': 0,
        'win_rate': 0.0,
        'total_pnl': 0.0,
        'balance': paper_state.balance
    }
    
    if not os.path.exists(PAPER_TRADES_FILE):
        return stats
    
    with open(PAPER_TRADES_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 6 and row[1] == 'EXIT' and row[5]:
                try:
                    pnl = float(row[5])
                    stats['total'] += 1
                    stats['total_pnl'] += pnl
                    if pnl >= 0:
                        stats['wins'] += 1
                    else:
                        stats['losses'] += 1
                except:
                    pass
    
    if stats['total'] > 0:
        stats['win_rate'] = (stats['wins'] / stats['total']) * 100
    
    return stats

# ============================================================================
# TRADES LOG (Original)
# ============================================================================

def init_trades_file():
    """إنشاء ملف السجل"""
    if not os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['التاريخ', 'النوع', 'السبب', 'السعر', 'النتيجة%'])

def log_trade(trade_type: str, reason: str, price: float, result_pct: Optional[float] = None):
    """تسجيل صفقة"""
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
    """إحصائيات الصفقات"""
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
                except:
                    pass
    
    stats['total'] = stats['wins'] + stats['losses']
    if stats['total'] > 0:
        stats['win_rate'] = (stats['wins'] / stats['total']) * 100
    
    stats['last_5'] = trades[-5:][::-1]
    
    return stats

# ============================================================================
# BINANCE API (قراءة فقط)
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
    """جلب بيانات تاريخية"""
    if interval == "1m":
        limit = min(days * 24 * 60, 1000)
    elif interval == "5m":
        limit = min(days * 24 * 12, 1000)
    else:
        limit = 500
    
    return get_klines(symbol, interval, limit)

# ============================================================================
# EMA CALCULATION
# ============================================================================

def calculate_ema(prices: List[float], period: int) -> List[float]:
    """حساب EMA"""
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
    """تشغيل Backtest"""
    if len(candles) < EMA_LONG + BREAKOUT_CANDLES + 10:
        return {"error": "بيانات غير كافية"}
    
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    volumes = [c["volume"] for c in candles]
    
    ema_short_vals = calculate_ema(closes, EMA_SHORT)
    ema_long_vals = calculate_ema(closes, EMA_LONG)
    
    if len(ema_short_vals) < 50 or len(ema_long_vals) < 50:
        return {"error": "فشل حساب EMA"}
    
    trades = []
    position_open = False
    entry_price = 0
    
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
    
    if len(trades) == 0:
        return {
            "trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "expectancy": 0.0, "max_drawdown": 0.0
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
        "trades": len(trades), "wins": wins, "losses": losses,
        "win_rate": win_rate, "expectancy": expectancy, "max_drawdown": max_dd
    }

# ============================================================================
# SIGNAL SCORE
# ============================================================================

def calculate_signal_score(analysis: dict, candles: List[dict]) -> tuple:
    """حساب نقاط الإشارة"""
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
            reasons.append(f"✅ اتجاه صاعد (+2)")
    
    return score, reasons

# ============================================================================
# SESSION AWARENESS
# ============================================================================

def is_low_liquidity_session() -> bool:
    """فحص جلسة منخفضة السيولة"""
    now = datetime.now(timezone.utc)
    hour = now.hour
    
    if 21 <= hour or hour < 1:
        return True
    if 5 <= hour < 7:
        return True
    
    return False

# ============================================================================
# STRATEGY LOGIC
# ============================================================================

def analyze_market(candles: List[dict]) -> dict:
    """تحليل السوق"""
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
    """فحص شروط الدخول"""
    if "error" in analysis:
        return False
    
    if is_low_liquidity_session():
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
    paper_state.position_qty = 0.0
    paper_state.entry_reason = ""

def update_cooldown_after_exit(exit_type: str):
    state.last_exit_type = exit_type
    
    if exit_type == "sl":
        state.consecutive_losses += 1
        state.consecutive_wins = 0
        state.current_cooldown = COOLDOWN_AFTER_SL
        
        if state.consecutive_losses >= 2:
            state.pause_until = datetime.now(timezone.utc) + timedelta(minutes=COOLDOWN_PAUSE_MINUTES)
            state.pause_alerted = False
    
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
# PAPER TRADING EXECUTION
# ============================================================================

def execute_paper_buy(price: float, score: int, reasons: List[str]) -> float:
    """تنفيذ شراء افتراضي"""
    qty = FIXED_TRADE_SIZE / price
    paper_state.position_qty = qty
    paper_state.entry_reason = ", ".join([r.split(" (+")[0].replace("✅ ", "") for r in reasons[:2]])
    
    log_paper_trade(
        action="BUY",
        entry_price=price,
        exit_price=None,
        pnl_pct=None,
        pnl_usdt=None,
        balance_after=paper_state.balance,
        score=score,
        entry_reason=paper_state.entry_reason,
        exit_reason="",
        duration_min=0
    )
    
    return qty

def execute_paper_exit(entry_price: float, exit_price: float, exit_reason: str, 
                       score: int, duration_min: int) -> tuple:
    """تنفيذ خروج افتراضي"""
    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
    pnl_usdt = paper_state.position_qty * (exit_price - entry_price)
    
    paper_state.balance += pnl_usdt
    
    log_paper_trade(
        action="EXIT",
        entry_price=entry_price,
        exit_price=exit_price,
        pnl_pct=pnl_pct,
        pnl_usdt=pnl_usdt,
        balance_after=paper_state.balance,
        score=score,
        entry_reason=paper_state.entry_reason,
        exit_reason=exit_reason,
        duration_min=duration_min
    )
    
    return pnl_pct, pnl_usdt, paper_state.balance

# ============================================================================
# MESSAGE FORMATTING (Paper Trading)
# ============================================================================

def get_current_time_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def format_buy_message(entry: float, tp: float, sl: float, timeframe: str, score: int, qty: float) -> str:
    score_label = "قوية 🔥" if score >= 8 else "عادية"
    reasons_text = "\n".join(state.last_signal_reasons[:3]) if state.last_signal_reasons else ""
    
    return (
        f"📥 *إشارة دخول (Paper)*\n\n"
        f"📈 *الزوج:* {SYMBOL_DISPLAY}\n"
        f"📊 *الفريم:* {timeframe}\n"
        f"💰 *سعر الدخول:* {entry:.4f}\n"
        f"📦 *الحجم:* {FIXED_TRADE_SIZE:.0f} USDT ({qty:.2f} XRP)\n"
        f"🎯 *جني الأرباح:* {tp:.4f} (+{TAKE_PROFIT_PCT}%)\n"
        f"🛑 *وقف الخسارة:* {sl:.4f} (-{STOP_LOSS_PCT}%)\n\n"
        f"⭐ *Score:* {score}/10 ({score_label})\n\n"
        f"📋 *أسباب الدخول:*\n{reasons_text}\n\n"
        f"💵 *الرصيد:* {paper_state.balance:.2f} USDT\n"
        f"🕐 *الوقت:* {get_current_time_str()}"
    )

def format_exit_message(entry: float, exit_price: float, pnl_pct: float, pnl_usdt: float,
                        reason: str, duration_min: int, balance: float) -> str:
    reason_text = {
        "tp": "وصول الهدف (TP) ✅",
        "sl": "وقف الخسارة (SL) ❌",
        "trailing_sl": "Trailing Stop Loss 🔒",
        "ema_confirmation": f"تأكيد كسر EMA{EMA_SHORT} 📊",
    }.get(reason, "خروج يدوي")
    
    pnl_sign = "+" if pnl_pct >= 0 else ""
    usdt_sign = "+" if pnl_usdt >= 0 else ""
    status_emoji = "✅" if pnl_pct >= 0 else "❌"
    
    return (
        f"📤 *إشارة خروج (Paper)*\n\n"
        f"📈 *الزوج:* {SYMBOL_DISPLAY}\n"
        f"💰 *سعر الدخول:* {entry:.4f}\n"
        f"💵 *سعر الخروج:* {exit_price:.4f}\n"
        f"📊 *النتيجة:* {pnl_sign}{pnl_pct:.2f}% ({usdt_sign}{pnl_usdt:.2f} USDT)\n\n"
        f"{status_emoji} *السبب:* {reason_text}\n"
        f"⏱️ *مدة الصفقة:* {duration_min} دقيقة\n"
        f"💵 *الرصيد بعد الصفقة:* {balance:.2f} USDT\n"
        f"🕐 *الوقت:* {get_current_time_str()}"
    )

def format_status_message() -> str:
    """تنسيق رسالة الحالة"""
    stats = get_paper_stats()
    status = "✅ نشط" if state.signals_enabled else "⏸️ متوقف (Kill Switch)"
    position = "📈 مفتوح" if state.position_open else "📉 مغلق"
    
    # إضافة سطر نضج البيانات (V3.1)
    maturity_str = ""
    if stats['total'] < 5:
        maturity_str = f"\n🧪 وضع التعلّم: لم يكتمل نضج البيانات بعد ({stats['total']} / 5)"
    
    msg = (
        f"ℹ️ *حالة البوت (V3.1 - Paper Trading)*\n\n"
        f"🔔 *الإشارات:* {status}{maturity_str}\n"
        f"📊 *الفريم:* {state.timeframe}\n"
        f"📈 *المركز:* {position}\n"
        f"💵 *الرصيد:* {paper_state.balance:.2f} USDT\n"
    )
    
    if state.position_open and state.entry_price:
        msg += f"💰 *سعر الدخول:* {state.entry_price:.4f}\n"
        msg += f"📦 *الكمية:* {paper_state.position_qty:.2f} XRP\n"
        if state.trailing_activated:
            msg += f"🔒 *Trailing:* مفعّل\n"
        if state.last_close:
            pnl = calculate_pnl(state.entry_price, state.last_close)
            pnl_sign = "+" if pnl >= 0 else ""
            pnl_usdt = paper_state.position_qty * (state.last_close - state.entry_price)
            msg += f"📉 *الربح الحالي:* {pnl_sign}{pnl:.2f}% ({pnl_usdt:+.2f} USDT)\n"
        duration = get_trade_duration_minutes()
        msg += f"⏱️ *مدة الصفقة:* {duration} دقيقة\n"
    
    if state.pause_until and datetime.now(timezone.utc) < state.pause_until:
        remaining = (state.pause_until - datetime.now(timezone.utc)).seconds // 60
        msg += f"⏳ *إيقاف مؤقت:* {remaining} دقيقة\n"
    
    if state.last_close:
        msg += f"🕯️ *آخر سعر:* {state.last_close:.4f}\n"
    
    msg += f"\n🕐 *التحديث:* {get_current_time_str()}"
    
    return msg

def format_welcome_message() -> str:
    return (
        f"🤖 *بوت إشارات {SYMBOL_DISPLAY} - Paper Trading*\n\n"
        f"📊 *الاستراتيجية:* EMA{EMA_SHORT}/EMA{EMA_LONG} + Breakout\n"
        f"🎯 *الهدف:* +{TAKE_PROFIT_PCT}%\n"
        f"🛑 *وقف الخسارة:* -{STOP_LOSS_PCT}%\n\n"
        f"💵 *رأس المال:* {START_BALANCE:.0f} USDT\n"
        f"📦 *حجم الصفقة:* {FIXED_TRADE_SIZE:.0f} USDT\n"
        f"💰 *الرصيد الحالي:* {paper_state.balance:.2f} USDT\n\n"
        f"⚠️ *ملاحظة:* هذا تداول افتراضي للتعلم والاختبار\n\n"
        f"استخدم الأزرار للتحكم 👇\n"
    )

def format_rules_message() -> str:
    return (
        f"📜 *قواعد التداول*\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*شروط الدخول (BUY):*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"1️⃣ EMA{EMA_SHORT} > EMA{EMA_LONG} (+3)\n"
        f"2️⃣ كسر قمة {BREAKOUT_CANDLES} شموع (+3)\n"
        f"3️⃣ حجم > متوسط {VOLUME_LOOKBACK} (+2)\n"
        f"4️⃣ اتجاه صاعد (+2)\n"
        f"⭐ الحد الأدنى: {MIN_SIGNAL_SCORE}/10\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*شروط الخروج:*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ TP: +{TAKE_PROFIT_PCT}%\n"
        f"❌ SL: -{STOP_LOSS_PCT}%\n"
        f"🔒 Trailing: +{TRAILING_TRIGGER_PCT}%\n"
        f"📊 EMA: شمعتين تحت EMA{EMA_SHORT}\n"
    )

def format_balance_message() -> str:
    stats = get_paper_stats()
    pnl_total = paper_state.balance - START_BALANCE
    pnl_sign = "+" if pnl_total >= 0 else ""
    
    return (
        f"💵 *الرصيد الافتراضي*\n\n"
        f"💰 *الرصيد الحالي:* {paper_state.balance:.2f} USDT\n"
        f"📈 *الربح/الخسارة:* {pnl_sign}{pnl_total:.2f} USDT\n"
        f"📊 *نسبة التغير:* {pnl_sign}{(pnl_total/START_BALANCE)*100:.2f}%\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*الإحصائيات:*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *إجمالي الصفقات:* {stats['total']}\n"
        f"✅ *رابحة:* {stats['wins']}\n"
        f"❌ *خاسرة:* {stats['losses']}\n"
        f"📉 *Win Rate:* {stats['win_rate']:.1f}%\n"
    )

def format_trades_message() -> str:
    trades = get_paper_trades(5)
    
    if not trades:
        return "📋 لا توجد صفقات مسجلة بعد"
    
    msg = "📋 *آخر 5 صفقات (Paper)*\n\n"
    
    for i, t in enumerate(trades, 1):
        pnl_sign = "+" if t['pnl_pct'] >= 0 else ""
        emoji = "✅" if t['pnl_pct'] >= 0 else "❌"
        msg += (
            f"{emoji} *{i}.* {pnl_sign}{t['pnl_pct']:.2f}% "
            f"({pnl_sign}{t['pnl_usdt']:.2f} USDT)\n"
            f"   {t['exit_reason']} | الرصيد: {t['balance']:.2f}\n\n"
        )
    
    return msg

def format_stats_message() -> str:
    stats = get_trade_stats()
    paper_stats = get_paper_stats()
    
    msg = (
        f"📈 *إحصائيات الأداء (Paper)*\n\n"
        f"💵 *الرصيد:* {paper_state.balance:.2f} USDT\n"
        f"📊 *إجمالي الصفقات:* {paper_stats['total']}\n"
        f"✅ *رابحة:* {paper_stats['wins']}\n"
        f"❌ *خاسرة:* {paper_stats['losses']}\n"
        f"📉 *Win Rate:* {paper_stats['win_rate']:.1f}%\n"
        f"💰 *إجمالي PnL:* {paper_stats['total_pnl']:+.2f} USDT\n"
    )
    
    trades = get_paper_trades(5)
    if trades:
        msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
        msg += "*آخر 5 صفقات:*\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n"
        for t in trades:
            pnl_sign = "+" if t['pnl_pct'] >= 0 else ""
            emoji = "✅" if t['pnl_pct'] >= 0 else "❌"
            msg += f"{emoji} {pnl_sign}{t['pnl_pct']:.2f}% | {t['exit_reason']}\n"
    
    msg += f"\n🕐 *التحديث:* {get_current_time_str()}"
    
    return msg

def format_signal_reasons_message() -> str:
    if not state.last_signal_reasons:
        return "❓ لا توجد إشارة حديثة"
    
    msg = (
        f"🧠 *لماذا هذه الإشارة؟*\n\n"
        f"⭐ *Score:* {state.last_signal_score}/10\n\n"
    )
    
    for reason in state.last_signal_reasons:
        msg += f"{reason}\n"
    
    return msg

# ============================================================================
# INLINE KEYBOARD
# ============================================================================

def get_main_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("▶️ تشغيل", callback_data="on"),
            InlineKeyboardButton("⏸ إيقاف", callback_data="off"),
        ],
        [
            InlineKeyboardButton("📊 الحالة", callback_data="status"),
            InlineKeyboardButton("💵 الرصيد", callback_data="balance"),
        ],
        [
            InlineKeyboardButton("📋 الصفقات", callback_data="trades"),
            InlineKeyboardButton("📈 الإحصائيات", callback_data="stats"),
        ],
        [
            InlineKeyboardButton("⏱ 1 دقيقة", callback_data="tf_1m"),
            InlineKeyboardButton("⏱ 5 دقائق", callback_data="tf_5m"),
        ],
        [
            InlineKeyboardButton("📜 القواعد", callback_data="rules"),
            InlineKeyboardButton("🔄 تصفير", callback_data="reset"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)

def get_confirm_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("✅ نعم، تصفير", callback_data="confirm_reset"),
            InlineKeyboardButton("❌ إلغاء", callback_data="cancel_reset"),
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
        return False
    
    if state.last_signal_type == signal_type and signal_type == "buy" and state.position_open:
        return False
    
    try:
        await bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
        state.last_message_time = time.time()
        state.last_signal_type = signal_type
        return True
    except Exception as e:
        logger.error(f"فشل إرسال الرسالة: {e}")
        return False

# ============================================================================
# COMMAND HANDLERS
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

async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        format_balance_message(),
        reply_markup=get_main_keyboard(),
        parse_mode="Markdown"
    )

async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        format_trades_message(),
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

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "⚠️ *هل تريد تصفير الرصيد والسجل؟*\n\n"
        "سيتم إعادة الرصيد إلى 1000 USDT\nوحذف جميع الصفقات المسجلة",
        reply_markup=get_confirm_keyboard(),
        parse_mode="Markdown"
    )

async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("❌ استخدم: /settf 1m أو /settf 5m")
        return
    
    new_tf = context.args[0].lower()
    if new_tf not in ["1m", "5m"]:
        await update.message.reply_text("❌ الفريم غير صحيح")
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
    
    elif data == "balance":
        await query.edit_message_text(
            format_balance_message(),
            reply_markup=get_main_keyboard(),
            parse_mode="Markdown"
        )
    
    elif data == "trades":
        await query.edit_message_text(
            format_trades_message(),
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
    
    elif data == "reset":
        await query.edit_message_text(
            "⚠️ *هل تريد تصفير الرصيد والسجل؟*\n\n"
            "سيتم إعادة الرصيد إلى 1000 USDT\nوحذف جميع الصفقات المسجلة",
            reply_markup=get_confirm_keyboard(),
            parse_mode="Markdown"
        )
    
    elif data == "confirm_reset":
        paper_state.reset()
        reset_position_state()
        await query.edit_message_text(
            f"✅ تم تصفير الرصيد إلى {START_BALANCE:.0f} USDT\n\n" + format_status_message(),
            reply_markup=get_main_keyboard(),
            parse_mode="Markdown"
        )
    
    elif data == "cancel_reset":
        await query.edit_message_text(
            "❌ تم إلغاء التصفير\n\n" + format_status_message(),
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
    """حلقة فحص الإشارات - Paper Trading"""
    logger.info(f"بدء حلقة الإشارات (Paper Trading)")
    
    init_trades_file()
    init_paper_trades_file()
    
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
                # V3.1: Kill Switch logic moved below and modified for maturity
                pass
            
            analysis = analyze_market(candles)
            
            if "error" in analysis:
                await asyncio.sleep(POLL_INTERVAL)
                continue

            # ============================================================================
            # KILL SWITCH (V3.1)
            # ============================================================================
            stats = get_paper_stats()
            if stats['total'] >= 5:
                # أعد تفعيل منطق V3 الأصلي كما هو بعد نضج البيانات
                if stats['win_rate'] < MIN_WIN_RATE:
                    if state.signals_enabled:
                        state.signals_enabled = False
                        try:
                            await bot.send_message(
                                chat_id=chat_id,
                                text=f"⚠️ **Kill Switch Activated**\n\nWin Rate ({stats['win_rate']:.1f}%) < {MIN_WIN_RATE}%\nتم إيقاف الإشارات التلقائية لحماية الرصيد.",
                                parse_mode="Markdown"
                            )
                        except:
                            pass
                        continue
            
            if not state.signals_enabled:
                await asyncio.sleep(POLL_INTERVAL)
                continue
            
            if state.position_open:
                exit_reason = check_exit_signal(analysis)
                if exit_reason:
                    exit_price = analysis["close"]
                    duration = get_trade_duration_minutes()
                    
                    pnl_pct, pnl_usdt, balance = execute_paper_exit(
                        state.entry_price, exit_price, exit_reason,
                        state.last_signal_score, duration
                    )
                    
                    log_trade("EXIT", exit_reason.upper(), exit_price, pnl_pct)
                    
                    msg = format_exit_message(
                        state.entry_price, exit_price, pnl_pct, pnl_usdt,
                        exit_reason, duration, balance
                    )
                    sent = await send_signal_message(bot, chat_id, msg, "exit")
                    
                    if sent:
                        update_cooldown_after_exit(exit_reason)
                        reset_position_state()
                        logger.info(f"إغلاق المركز: {exit_reason} @ {exit_price:.4f} (PnL: {pnl_pct:.2f}%)")
            
            else:
                if check_buy_signal(analysis, candles):
                    entry_price = analysis["close"]
                    tp, sl = calculate_targets(entry_price)
                    
                    qty = execute_paper_buy(entry_price, state.last_signal_score, state.last_signal_reasons)
                    
                    log_trade("BUY", "SIGNAL", entry_price, None)
                    
                    msg = format_buy_message(
                        entry_price, tp, sl, state.timeframe,
                        state.last_signal_score, qty
                    )
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
    
    logger.info(f"بدء بوت إشارات {SYMBOL_DISPLAY} - Paper Trading")
    
    application = Application.builder().token(tg_token).build()
    
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("balance", cmd_balance))
    application.add_handler(CommandHandler("trades", cmd_trades))
    application.add_handler(CommandHandler("on", cmd_on))
    application.add_handler(CommandHandler("off", cmd_off))
    application.add_handler(CommandHandler("rules", cmd_rules))
    application.add_handler(CommandHandler("stats", cmd_stats))
    application.add_handler(CommandHandler("reset", cmd_reset))
    application.add_handler(CommandHandler("settf", cmd_timeframe))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    bot = application.bot
    
    await application.initialize()
    await application.start()
    await application.updater.start_polling(drop_pending_updates=True)
    
    print("=" * 50)
    print(f"🚀 بوت إشارات {SYMBOL_DISPLAY} - Paper Trading")
    print(f"💵 رأس المال: {START_BALANCE:.0f} USDT")
    print(f"📦 حجم الصفقة: {FIXED_TRADE_SIZE:.0f} USDT")
    print(f"💰 الرصيد الحالي: {paper_state.balance:.2f} USDT")
    print(f"📊 الفريم: {state.timeframe}")
    print(f"🎯 TP: +{TAKE_PROFIT_PCT}% | SL: -{STOP_LOSS_PCT}%")
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

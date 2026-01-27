#!/usr/bin/env python3
"""
XRP/USDT Telegram Signals Bot V3.2 + Paper Trading
بوت إشارات تداول يرسل إشارات دخول/خروج لزوج XRP/USDT
V3.2: Kill Switch متعدد الطبقات لحماية رأس المال
"""

import os
import csv
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict

import requests
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# --- Configuration ---
MODE = "PAPER"
TIMEFRAME = "1m"
SYMBOL = "XRPUSDT"
SYMBOL_DISPLAY = "XRP/USDT"

analysis_count = 0
last_analysis_time = None

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

DATA_MATURITY_TRADES = 5
LOSS_STREAK_LIMIT = 3
DRAWDOWN_LIMIT_PERCENT = 3.0
RECENT_WIN_RATE_MIN = 40.0
RECENT_TRADES_WINDOW = 10
AUTO_RESUME_MINUTES = 30
COOLDOWN_AFTER_LOSS_STREAK = 15

TRADES_FILE = "trades.csv"
PAPER_TRADES_FILE = "paper_trades.csv"

BINANCE_APIS = [
    "https://api.binance.us/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api2.binance.com/api/v3/klines",
    "https://api3.binance.com/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class KillSwitchState:
    def __init__(self):
        self.active: bool = False
        self.reason: str = ""
        self.triggered_at: Optional[datetime] = None
        self.resume_at: Optional[datetime] = None
        self.alert_sent: bool = False
    
    def activate(self, reason: str):
        self.active = True
        self.reason = reason
        self.triggered_at = datetime.now(timezone.utc)
        self.resume_at = self.triggered_at + timedelta(minutes=AUTO_RESUME_MINUTES)
        self.alert_sent = False
        logger.info(f"Kill Switch مفعّل: {reason}")
    
    def deactivate(self):
        self.active = False
        self.reason = ""
        self.triggered_at = None
        self.resume_at = None
        self.alert_sent = False
        logger.info("Kill Switch معطّل - تم استئناف التداول")
    
    def check_auto_resume(self) -> bool:
        if self.active and self.resume_at:
            if datetime.now(timezone.utc) >= self.resume_at:
                return True
        return False
    
    def get_remaining_minutes(self) -> int:
        if self.resume_at:
            remaining = self.resume_at - datetime.now(timezone.utc)
            return max(0, int(remaining.total_seconds() / 60))
        return 0

kill_switch = KillSwitchState()


class PaperTradingState:
    def __init__(self):
        self.balance: float = START_BALANCE
        self.peak_balance: float = START_BALANCE
        self.position_qty: float = 0.0
        self.entry_reason: str = ""
        self.loss_streak: int = 0
        self.load_balance()
    
    def load_balance(self):
        if os.path.exists(PAPER_TRADES_FILE):
            try:
                with open(PAPER_TRADES_FILE, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        last_row = rows[-1]
                        if last_row.get('balance_after'):
                            self.balance = float(last_row['balance_after'])
                        if last_row.get('balance_peak'):
                            self.peak_balance = float(last_row['balance_peak'])
                        if self.balance > self.peak_balance:
                            self.peak_balance = self.balance
            except:
                self.balance = START_BALANCE
                self.peak_balance = START_BALANCE
    
    def update_peak(self):
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
    
    def reset(self):
        self.balance = START_BALANCE
        self.peak_balance = START_BALANCE
        self.position_qty = 0.0
        self.entry_reason = ""
        self.loss_streak = 0
        if os.path.exists(PAPER_TRADES_FILE):
            os.remove(PAPER_TRADES_FILE)
        init_paper_trades_file()
        kill_switch.deactivate()

paper_state = PaperTradingState()


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


def init_paper_trades_file():
    if not os.path.exists(PAPER_TRADES_FILE):
        with open(PAPER_TRADES_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'action', 'entry_price', 'exit_price',
                'pnl_percent', 'pnl_usdt', 'balance_after', 'score',
                'entry_reason', 'exit_reason', 'duration_minutes',
                'kill_switch_triggered', 'kill_switch_reason', 'balance_peak'
            ])


def log_paper_trade(action: str, entry_price: float, exit_price: Optional[float],
                    pnl_pct: Optional[float], pnl_usdt: Optional[float],
                    balance_after: float, score: int, entry_reason: str,
                    exit_reason: str, duration_min: int,
                    ks_triggered: bool = False, ks_reason: str = ""):
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
            duration_min,
            str(ks_triggered),
            ks_reason,
            f"{paper_state.peak_balance:.2f}"
        ])


def get_closed_trades() -> List[Dict]:
    trades = []
    if not os.path.exists(PAPER_TRADES_FILE):
        return trades
    
    with open(PAPER_TRADES_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('action') == 'EXIT' and row.get('pnl_usdt'):
                try:
                    trades.append({
                        'timestamp': row.get('timestamp', ''),
                        'pnl_usdt': float(row['pnl_usdt']),
                        'pnl_percent': float(row.get('pnl_percent', 0)),
                        'result': 'WIN' if float(row['pnl_usdt']) >= 0 else 'LOSS'
                    })
                except:
                    pass
    return trades


def get_recent_trades(n: int = 10) -> List[Dict]:
    closed = get_closed_trades()
    return closed[-n:] if len(closed) >= n else closed


def get_paper_trades(limit: int = 5) -> List[Dict]:
    trades = []
    if not os.path.exists(PAPER_TRADES_FILE):
        return trades
    
    with open(PAPER_TRADES_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        exit_trades = [r for r in rows if r.get('action') == 'EXIT']
        
        for row in exit_trades[-limit:][::-1]:
            try:
                trades.append({
                    'timestamp': row.get('timestamp', ''),
                    'entry_price': row.get('entry_price', ''),
                    'exit_price': row.get('exit_price', ''),
                    'pnl_pct': float(row['pnl_percent']) if row.get('pnl_percent') else 0,
                    'pnl_usdt': float(row['pnl_usdt']) if row.get('pnl_usdt') else 0,
                    'balance': float(row['balance_after']) if row.get('balance_after') else 0,
                    'exit_reason': row.get('exit_reason', '')
                })
            except:
                pass
    
    return trades


def get_paper_stats() -> Dict:
    stats = {
        'total': 0,
        'wins': 0,
        'losses': 0,
        'win_rate': 0.0,
        'total_pnl': 0.0,
        'balance': paper_state.balance,
        'peak_balance': paper_state.peak_balance,
        'drawdown': 0.0,
        'loss_streak': paper_state.loss_streak
    }
    
    closed = get_closed_trades()
    for trade in closed:
        stats['total'] += 1
        stats['total_pnl'] += trade['pnl_usdt']
        if trade['result'] == 'WIN':
            stats['wins'] += 1
        else:
            stats['losses'] += 1
    
    if stats['total'] > 0:
        stats['win_rate'] = (stats['wins'] / stats['total']) * 100
    
    if paper_state.peak_balance > 0:
        stats['drawdown'] = ((paper_state.peak_balance - paper_state.balance) / paper_state.peak_balance) * 100
    
    return stats


def calculate_recent_win_rate() -> float:
    recent = get_recent_trades(RECENT_TRADES_WINDOW)
    if len(recent) < RECENT_TRADES_WINDOW:
        return 100.0
    wins = sum(1 for t in recent if t['result'] == 'WIN')
    return (wins / len(recent)) * 100


def check_data_maturity() -> bool:
    return len(get_closed_trades()) >= DATA_MATURITY_TRADES


def check_loss_streak() -> bool:
    return paper_state.loss_streak >= LOSS_STREAK_LIMIT


def check_drawdown() -> bool:
    if paper_state.peak_balance <= 0:
        return False
    drawdown_pct = ((paper_state.peak_balance - paper_state.balance) / paper_state.peak_balance) * 100
    return drawdown_pct >= DRAWDOWN_LIMIT_PERCENT


def check_recent_performance() -> bool:
    closed = get_closed_trades()
    if len(closed) < RECENT_TRADES_WINDOW:
        return False
    win_rate = calculate_recent_win_rate()
    return win_rate < RECENT_WIN_RATE_MIN


def evaluate_kill_switch() -> Optional[str]:
    if not check_data_maturity():
        return None
    
    if check_loss_streak():
        return f"{LOSS_STREAK_LIMIT} خسائر متتالية"
    
    if check_drawdown():
        return "تجاوز حد الخسارة الكلية (Drawdown)"
    
    if check_recent_performance():
        return f"انخفاض Win Rate في آخر {RECENT_TRADES_WINDOW} صفقات"
    
    return None


def resume_trading():
    kill_switch.deactivate()
    paper_state.loss_streak = 0
    logger.info("تم استئناف التداول")


def init_trades_file():
    if not os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['التاريخ', 'النوع', 'السبب', 'السعر', 'النتيجة%'])


def log_trade(trade_type: str, reason: str, price: float, result_pct: Optional[float] = None):
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


def get_klines(symbol: str, interval: str, limit: int = KLINE_LIMIT) -> Optional[List[dict]]:
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


def calculate_ema(prices: List[float], period: int) -> List[float]:
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


def analyze_market(candles: List[dict]) -> dict:
    global analysis_count, last_analysis_time
    analysis_count += 1
    last_analysis_time = datetime.now(timezone.utc)
    
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


def calculate_signal_score(analysis: dict, candles: List[dict]) -> tuple:
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


def is_low_liquidity_session() -> bool:
    now = datetime.now(timezone.utc)
    hour = now.hour
    if 21 <= hour or hour < 1:
        return True
    if 5 <= hour < 7:
        return True
    return False


def check_buy_signal(analysis: dict, candles: List[dict]) -> bool:
    if "error" in analysis:
        return False
    if kill_switch.active:
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
    
    return score >= MIN_SIGNAL_SCORE


def calculate_targets(entry_price: float) -> tuple:
    tp = entry_price * (1 + TAKE_PROFIT_PCT / 100)
    sl = entry_price * (1 - STOP_LOSS_PCT / 100)
    return tp, sl


def check_exit_signal(analysis: dict) -> Optional[str]:
    if not state.position_open or state.entry_price is None:
        return None
    
    current_price = analysis["close"]
    entry_price = state.entry_price
    pnl_pct = ((current_price - entry_price) / entry_price) * 100
    
    if pnl_pct >= TAKE_PROFIT_PCT:
        return "tp"
    
    if pnl_pct <= -STOP_LOSS_PCT:
        return "sl"
    
    if pnl_pct >= TRAILING_TRIGGER_PCT:
        state.trailing_activated = True
    
    if state.trailing_activated and current_price < analysis["ema_short"]:
        return "trailing_sl"
    
    if current_price < analysis["ema_short"]:
        state.candles_below_ema += 1
    else:
        state.candles_below_ema = 0
    
    if state.candles_below_ema >= 2:
        return "ema_confirmation"
    
    return None


def execute_paper_buy(price: float, score: int, reasons: List[str]) -> float:
    qty = FIXED_TRADE_SIZE / price
    paper_state.position_qty = qty
    paper_state.entry_reason = ", ".join(reasons)
    
    log_paper_trade(
        "BUY", price, None, None, None,
        paper_state.balance, score, paper_state.entry_reason,
        "", 0
    )
    return qty


def execute_paper_exit(entry_price: float, exit_price: float, reason: str,
                       score: int, duration_min: int) -> tuple:
    qty = paper_state.position_qty
    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
    pnl_usdt = (exit_price - entry_price) * qty
    
    paper_state.balance += pnl_usdt
    paper_state.update_peak()
    
    if pnl_usdt < 0:
        paper_state.loss_streak += 1
        state.consecutive_losses += 1
        state.consecutive_wins = 0
    else:
        paper_state.loss_streak = 0
        state.consecutive_wins += 1
        state.consecutive_losses = 0
    
    if state.consecutive_losses >= 2:
        state.pause_until = datetime.now(timezone.utc) + timedelta(minutes=COOLDOWN_PAUSE_MINUTES)
    
    log_paper_trade(
        "EXIT", entry_price, exit_price, pnl_pct, pnl_usdt,
        paper_state.balance, score, paper_state.entry_reason,
        reason, duration_min
    )
    
    paper_state.position_qty = 0.0
    paper_state.entry_reason = ""
    
    return pnl_pct, pnl_usdt, paper_state.balance


def reset_position_state():
    state.position_open = False
    state.entry_price = None
    state.entry_time = None
    state.entry_timeframe = None
    state.trailing_activated = False
    state.candles_below_ema = 0


def get_trade_duration_minutes() -> int:
    if state.entry_time:
        delta = datetime.now(timezone.utc) - state.entry_time
        return int(delta.total_seconds() / 60)
    return 0


def update_cooldown_after_exit(reason: str):
    if reason == "sl":
        state.current_cooldown = COOLDOWN_AFTER_SL
    elif state.consecutive_wins >= 2:
        state.current_cooldown = COOLDOWN_STREAK_WIN
    else:
        state.current_cooldown = COOLDOWN_NORMAL


def get_main_keyboard():
    keyboard = [
        [
            InlineKeyboardButton("🔄 تحديث الحالة", callback_data="status"),
            InlineKeyboardButton("🧪 تشخيص البوت", callback_data="diagnostic")
        ],
        [
            InlineKeyboardButton("💰 المحفظة", callback_data="balance"),
            InlineKeyboardButton("📊 الصفقات", callback_data="trades")
        ],
        [
            InlineKeyboardButton("⏱ فريم 1 دقيقة", callback_data="tf_1m"),
            InlineKeyboardButton("⏱ فريم 5 دقائق", callback_data="tf_5m")
        ],
        [
            InlineKeyboardButton("🟢 تشغيل", callback_data="on"),
            InlineKeyboardButton("⏸️ إيقاف", callback_data="off")
        ],
        [
            InlineKeyboardButton("🔴 تصفير البيانات", callback_data="reset")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_confirm_keyboard():
    keyboard = [[
        InlineKeyboardButton("✅ نعم، متأكد", callback_data="confirm_reset"),
        InlineKeyboardButton("❌ إلغاء", callback_data="cancel_reset")
    ]]
    return InlineKeyboardMarkup(keyboard)


def format_welcome_message() -> str:
    return (
        f"🚀 *مرحباً بك في بوت إشارات {SYMBOL_DISPLAY} V3.2*\n\n"
        "هذا البوت يقوم بتحليل السوق وإرسال إشارات شراء/بيع "
        "بناءً على استراتيجية الـ Breakout و EMA.\n\n"
        "⚠️ *نظام Paper Trading مفعل حالياً*\n"
        "يتم محاكاة الصفقات برصيد وهمي 1000 USDT.\n\n"
        "استخدم الأزرار أدناه للتحكم."
    )


def format_status_message() -> str:
    status = "🟢 نشط" if state.signals_enabled else "⏸️ متوقف"
    if kill_switch.active:
        status = f"🛑 متوقف (Kill Switch: {kill_switch.reason})"
    
    tf_display = "1 دقيقة" if state.timeframe == "1m" else "5 دقائق"
    
    pos_status = "📉 لا يوجد مركز مفتوح"
    if state.position_open:
        pnl = 0
        if state.last_close and state.entry_price:
            pnl = ((state.last_close - state.entry_price) / state.entry_price) * 100
        pos_status = (
            f"📈 مركز مفتوح @ {state.entry_price:.4f}\n"
            f"🕒 منذ: {get_trade_duration_minutes()} دقيقة\n"
            f"📊 PnL الحالي: {pnl:+.2f}%"
        )
    
    return (
        f"*📊 حالة البوت - V3.2*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 الحالة: {status}\n"
        f"🕒 الفريم الحالي: {tf_display}\n"
        f"🪙 الزوج: {SYMBOL_DISPLAY}\n"
        f"💵 السعر الحالي: {state.last_close if state.last_close else '---'}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"{pos_status}\n"
        f"━━━━━━━━━━━━━━━━━━━━━"
    )


def format_balance_message() -> str:
    stats = get_paper_stats()
    return (
        f"*💰 محفظة Paper Trading*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 الرصيد الحالي: {stats['balance']:.2f} USDT\n"
        f"📈 أعلى رصيد: {stats['peak_balance']:.2f} USDT\n"
        f"📉 Drawdown: {stats['drawdown']:.2f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 إجمالي الصفقات: {stats['total']}\n"
        f"✅ رابحة: {stats['wins']} | ❌ خاسرة: {stats['losses']}\n"
        f"⭐ Win Rate: {stats['win_rate']:.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━"
    )


def format_trades_message() -> str:
    trades = get_paper_trades(5)
    if not trades:
        return "📭 لا توجد صفقات مسجلة بعد."
    
    text = "*📊 آخر 5 صفقات منفذة:*\n\n"
    for t in trades:
        emoji = "✅" if t['pnl_usdt'] >= 0 else "❌"
        text += (
            f"{emoji} {t['timestamp']}\n"
            f"💰 PnL: {t['pnl_pct']:+.2f}% ({t['pnl_usdt']:+.2f} USDT)\n"
            f"📌 {t['exit_reason']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
        )
    return text


def format_stats_message() -> str:
    stats = get_paper_stats()
    return (
        f"*📈 إحصائيات الأداء الكاملة*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 صافي الربح: {stats['total_pnl']:+.2f} USDT\n"
        f"⭐ Win Rate: {stats['win_rate']:.1f}%\n"
        f"📊 إجمالي الصفقات: {stats['total']}\n"
        f"🔥 أطول سلسلة خسائر: {stats['loss_streak']}\n"
        f"━━━━━━━━━━━━━━━━━━━━━"
    )


def format_rules_message() -> str:
    return (
        f"*⚙️ قواعد التداول V3.2*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"• الرافعة المالية: 1x (Spot)\n"
        f"• حجم الصفقة: {FIXED_TRADE_SIZE} USDT\n"
        f"• Take Profit: +{TAKE_PROFIT_PCT}%\n"
        f"• Stop Loss: -{STOP_LOSS_PCT}%\n"
        f"• Trailing @ +{TRAILING_TRIGGER_PCT}%\n"
        f"• إغلاق تحت EMA{EMA_SHORT}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"*🛡️ Kill Switch V3.2:*\n"
        f"1️⃣ {LOSS_STREAK_LIMIT} خسائر متتالية\n"
        f"2️⃣ Drawdown > {DRAWDOWN_LIMIT_PERCENT}%\n"
        f"3️⃣ Win Rate < {RECENT_WIN_RATE_MIN}% (آخر 10)\n"
        f"• استئناف تلقائي: {AUTO_RESUME_MINUTES} دقيقة\n"
        f"━━━━━━━━━━━━━━━━━━━━━"
    )


def format_buy_message(entry: float, tp: float, sl: float, tf: str, score: int, qty: float) -> str:
    return (
        f"🟢 *إشارة شراء - Paper Trading*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 الزوج: {SYMBOL_DISPLAY}\n"
        f"⏱ الفريم: {tf}\n"
        f"💹 سعر الدخول: {entry:.4f}\n"
        f"🎯 Take Profit: {tp:.4f} (+{TAKE_PROFIT_PCT}%)\n"
        f"🛑 Stop Loss: {sl:.4f} (-{STOP_LOSS_PCT}%)\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"📦 الكمية: {qty:.2f} XRP\n"
        f"💵 القيمة: {FIXED_TRADE_SIZE:.0f} USDT\n"
        f"⭐ Score: {score}/10\n"
        f"━━━━━━━━━━━━━━━━━━━━━"
    )


def format_exit_message(entry: float, exit_price: float, pnl_pct: float,
                        pnl_usdt: float, reason: str, duration: int, balance: float) -> str:
    emoji = "🟢" if pnl_usdt >= 0 else "🔴"
    reason_text = {
        "tp": "Take Profit ✅",
        "sl": "Stop Loss ❌",
        "trailing_sl": "Trailing Stop 🔄",
        "ema_confirmation": "EMA Exit 📉"
    }.get(reason, reason.upper())
    
    return (
        f"{emoji} *إغلاق صفقة - Paper Trading*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 الزوج: {SYMBOL_DISPLAY}\n"
        f"📌 السبب: {reason_text}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"💹 سعر الدخول: {entry:.4f}\n"
        f"💹 سعر الخروج: {exit_price:.4f}\n"
        f"📈 النتيجة: {pnl_pct:+.2f}%\n"
        f"💵 الربح/الخسارة: {pnl_usdt:+.2f} USDT\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 الرصيد: {balance:.2f} USDT\n"
        f"⏱ المدة: {duration} دقيقة\n"
        f"━━━━━━━━━━━━━━━━━━━━━"
    )


async def send_signal_message(bot: Bot, chat_id: str, text: str, msg_type: str) -> bool:
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="Markdown"
        )
        state.last_message_time = time.time()
        return True
    except Exception as e:
        logger.error(f"فشل إرسال الرسالة: {e}")
        return False


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
        await update.message.reply_text("❌ استخدم: /settf 1m أو /settf 5m")
        return
    new_tf = context.args[0].lower()
    if new_tf not in ["1m", "5m"]:
        await update.message.reply_text("❌ الفريم غير صحيح")
        return
    state.timeframe = new_tf
    await update.message.reply_text(
        f"✅ تم تغيير الفريم إلى {new_tf}",
        reply_markup=get_main_keyboard()
    )


async def cmd_الفريم(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("❌ استخدم: /الفريم 1 أو /الفريم 5")
        return
    
    val = context.args[0]
    new_tf = ""
    if val == "1":
        new_tf = "1m"
    elif val == "5":
        new_tf = "5m"
    else:
        await update.message.reply_text("❌ الفريم غير صحيح (1 أو 5 فقط)")
        return
    
    state.timeframe = new_tf
    logger.info(f"تم تغيير الفريم إلى {new_tf} عبر الأمر العربي")
    
    # Update Job if exists
    application = context.application
    if application.job_queue:
        # Remove old jobs
        for job in application.job_queue.get_jobs_by_name("signal_loop"):
            job.schedule_removal()
        
        # Add new job
        chat_id = os.environ.get("TG_CHAT_ID")
        application.job_queue.run_repeating(
            lambda ctx: asyncio.create_task(signal_loop(application.bot, chat_id)),
            interval=POLL_INTERVAL,
            first=1,
            name="signal_loop"
        )
    
    await update.message.reply_text(
        f"✅ تم تغيير الفريم إلى {val} دقيقة\n\n" + format_status_message(),
        reply_markup=get_main_keyboard(),
        parse_mode="Markdown"
    )


async def cmd_diagnostic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    إجراء فحص تشغيلي شامل وعرض النتائج في تيليجرام
    """
    global analysis_count, last_analysis_time
    
    # 1. جلب البيانات وتحليلها
    candles = get_klines(SYMBOL, state.timeframe)
    if not candles:
        msg = "❌ فشل في جلب بيانات السوق."
        if update.message:
            await update.message.reply_text(msg)
        else:
            await update.callback_query.message.reply_text(msg)
        return
        
    analysis = analyze_market(candles)
    if "error" in analysis:
        msg = f"❌ خطأ في التحليل: {analysis['error']}"
        if update.message:
            await update.message.reply_text(msg)
        else:
            await update.callback_query.message.reply_text(msg)
        return

    score, reasons = calculate_signal_score(analysis, candles)
    ks_block = evaluate_kill_switch()
    
    # 2. بناء الرسالة
    msg = "🧪 *تشخيص البوت V3.2*\n\n"
    
    # حالة النظام
    job_status = "✅ يعمل" if analysis_count > 0 else "🛑 متوقف"
    last_time = last_analysis_time.strftime("%H:%M:%S") if last_analysis_time else "لا يوجد"
    msg += "🔧 *حالة النظام*\n"
    msg += f"• Job التحليل: {job_status}\n"
    msg += f"• مرات التنفيذ: {analysis_count}\n"
    msg += f"• آخر تنفيذ: {last_time}\n\n"
    
    # حالة التداول
    signals = "✅ مفعّلة" if state.signals_enabled else "🛑 موقوفة"
    ks_status = "⚠️ مفعل" if kill_switch.active else "✅ غير مفعل"
    cooldown = 0
    if state.pause_until:
        rem = (state.pause_until - datetime.now(timezone.utc)).total_seconds()
        cooldown = max(0, int(rem))
        
    msg += "⚙️ *حالة التداول*\n"
    msg += f"• الإشارات: {signals}\n"
    msg += f"• Paper Trading: ✅ مفعّل\n"
    msg += f"• Kill Switch: {ks_status}\n"
    if kill_switch.active:
        msg += f"  - السبب: {kill_switch.reason}\n"
    msg += f"• Cooldown: {cooldown} ثانية\n\n"
    
    # بيانات السوق
    last_candle_time = datetime.fromtimestamp(candles[-1]['open_time']/1000, tz=timezone.utc).strftime("%H:%M:%S")
    msg += "📊 *بيانات السوق (XRP/USDT)*\n"
    msg += f"• الفريم: {state.timeframe}\n"
    msg += f"• الشموع: {len(candles)}\n"
    msg += f"• آخر إغلاق: {analysis['close']:.4f}\n"
    msg += f"• وقت الشمعة: {last_candle_time}\n\n"
    
    # تحليل الدخول
    msg += "📈 *تحليل الدخول (آخر دورة)*\n"
    msg += f"{'✔️' if analysis['ema_bullish'] else '❌'} EMA20 > EMA50\n"
    msg += f"{'✔️' if analysis['breakout'] else '❌'} كسر قمة آخر 5 شموع\n"
    msg += f"{'✔️' if analysis['volume_confirmed'] else '❌'} فلتر الحجم (Volume)\n"
    msg += f"{'✔️' if analysis['range_confirmed'] else '❌'} فلتر التذبذب (Range)\n"
    msg += f"• Score الحالي: {score} / 10\n\n"
    
    # Paper Trading
    closed_trades = get_closed_trades()
    msg += "🧾 *Paper Trading*\n"
    msg += f"• الرصيد: {paper_state.balance:.2f} USDT\n"
    msg += f"• صفقة مفتوحة: {'نعم' if paper_state.position_qty > 0 else 'لا'}\n"
    if paper_state.position_qty > 0:
        msg += f"• سعر الدخول: {state.entry_price:.4f}\n"
    msg += f"• عدد الصفقات: {len(closed_trades)}\n\n"
    
    # الخلاصة الذكية
    summary = ""
    if kill_switch.active or not state.signals_enabled or ks_block:
        reason = kill_switch.reason if kill_switch.active else (ks_block if ks_block else "إيقاف يدوي")
        summary = f"🛑 التداول موقوف حاليًا بسبب: {reason}"
    elif score >= MIN_SIGNAL_SCORE:
        summary = "✅ البوت جاهز وسيدخل عند تحقق الشروط"
    else:
        summary = "⚠️ البوت يعمل لكن شروط الدخول غير مكتملة"
    
    msg += f"🧠 *الخلاصة الذكية*\n{summary}"
    
    if update.message:
        await update.message.reply_text(msg, parse_mode='Markdown')
    else:
        await update.callback_query.message.reply_text(msg, parse_mode='Markdown')

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    التعامل مع ضغطات الأزرار
    """
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data == "on":
        state.signals_enabled = True
        await query.edit_message_text("✅ تم تشغيل الإشارات\n\n" + format_status_message(), reply_markup=get_main_keyboard(), parse_mode="Markdown")
    elif data == "off":
        state.signals_enabled = False
        await query.edit_message_text("⏸️ تم إيقاف الإشارات\n\n" + format_status_message(), reply_markup=get_main_keyboard(), parse_mode="Markdown")
    elif data == "status":
        await query.edit_message_text(format_status_message(), reply_markup=get_main_keyboard(), parse_mode="Markdown")
    elif data == "balance":
        await query.edit_message_text(format_balance_message(), reply_markup=get_main_keyboard(), parse_mode="Markdown")
    elif data == "trades":
        await query.edit_message_text(format_trades_message(), reply_markup=get_main_keyboard(), parse_mode="Markdown")
    elif data == "stats":
        await query.edit_message_text(format_stats_message(), reply_markup=get_main_keyboard(), parse_mode="Markdown")
    elif data == "rules":
        await query.edit_message_text(format_rules_message(), reply_markup=get_main_keyboard(), parse_mode="Markdown")
    elif data == "diagnostic":
        await cmd_diagnostic(update, context)
    elif data == "reset":
        await query.edit_message_text("⚠️ *هل تريد تصفير الرصيد والسجل؟*\n\n", reply_markup=get_confirm_keyboard(), parse_mode="Markdown")
    elif data == "confirm_reset":
        paper_state.reset()
        reset_position_state()
        await query.edit_message_text(f"✅ تم تصفير الرصيد إلى {START_BALANCE:.0f} USDT\n\n" + format_status_message(), reply_markup=get_main_keyboard(), parse_mode="Markdown")
    elif data == "cancel_reset":
        await query.edit_message_text("❌ تم إلغاء التصفير\n\n" + format_status_message(), reply_markup=get_main_keyboard(), parse_mode="Markdown")
    elif data in ["tf_1m", "tf_5m"]:
        new_tf = "1m" if data == "tf_1m" else "5m"
        state.timeframe = new_tf
        logger.info(f"تم تغيير الفريم إلى {new_tf} عبر الأزرار")
        
        # Update Job
        application = context.application
        if application.job_queue:
            for job in application.job_queue.get_jobs_by_name("signal_loop"):
                job.schedule_removal()
            
            chat_id = os.environ.get("TG_CHAT_ID")
            application.job_queue.run_repeating(
                lambda ctx: asyncio.create_task(signal_loop(application.bot, chat_id)),
                interval=POLL_INTERVAL,
                first=1,
                name="signal_loop"
            )
            
        await query.edit_message_text(
            f"✅ تم تغيير الفريم إلى {'1 دقيقة' if new_tf == '1m' else '5 دقائق'}\n\n" + format_status_message(),
            reply_markup=get_main_keyboard(),
            parse_mode="Markdown"
        )


async def signal_loop(bot: Bot, chat_id: str) -> None:
    logger.info("حلقة الإشارات تعمل...")
    try:
        if kill_switch.check_auto_resume():
            resume_trading()
            await bot.send_message(chat_id=chat_id, text="✅ تم استئناف التداول تلقائياً", parse_mode="Markdown")
        
        if not state.signals_enabled or kill_switch.active:
            return
        
        if state.pause_until and datetime.now(timezone.utc) < state.pause_until:
            return
        
        candles = get_klines(SYMBOL, state.timeframe)
        if candles is None:
            return
        
        analysis = analyze_market(candles)
        if "error" in analysis:
            return
        
        ks_reason = evaluate_kill_switch()
        if ks_reason and not state.position_open:
            kill_switch.activate(ks_reason)
            return
        
        if state.position_open and state.entry_price is not None:
            exit_reason = check_exit_signal(analysis)
            if exit_reason:
                exit_price = analysis["close"]
                duration = get_trade_duration_minutes()
                pnl_pct, pnl_usdt, balance = execute_paper_exit(state.entry_price, exit_price, exit_reason, state.last_signal_score, duration)
                log_trade("EXIT", exit_reason.upper(), exit_price, pnl_pct)
                msg = format_exit_message(state.entry_price, exit_price, pnl_pct, pnl_usdt, exit_reason, duration, balance)
                await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
                update_cooldown_after_exit(exit_reason)
                reset_position_state()
        else:
            if check_buy_signal(analysis, candles):
                entry_price = analysis["close"]
                tp, sl = calculate_targets(entry_price)
                qty = execute_paper_buy(entry_price, state.last_signal_score, state.last_signal_reasons)
                log_trade("BUY", "SIGNAL", entry_price, None)
                msg = format_buy_message(entry_price, tp, sl, state.timeframe, state.last_signal_score, qty)
                await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
                state.position_open = True
                state.entry_price = entry_price
                state.entry_time = datetime.now(timezone.utc)
                state.trailing_activated = False
                state.candles_below_ema = 0

    except Exception as e:
        logger.error(f"Error in signal loop: {e}")


async def main() -> None:
    tg_token = os.environ.get("TG_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    
    if not tg_token or not chat_id:
        print("❌ الرجاء تعيين TG_TOKEN و TG_CHAT_ID")
        return
    
    # Initialize application
    application = Application.builder().token(tg_token).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("balance", cmd_balance))
    application.add_handler(CommandHandler("trades", cmd_trades))
    application.add_handler(CommandHandler("on", cmd_on))
    application.add_handler(CommandHandler("off", cmd_off))
    application.add_handler(CommandHandler("rules", cmd_rules))
    application.add_handler(CommandHandler("stats", cmd_stats))
    application.add_handler(CommandHandler("diagnostic", cmd_diagnostic))
    application.add_handler(CommandHandler("frame", cmd_الفريم))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Initialize the application
    await application.initialize()
    
    # Use JobQueue for signal loop if available, else use create_task
    if application.job_queue:
        application.job_queue.run_repeating(
            lambda context: asyncio.create_task(signal_loop(application.bot, chat_id)),
            interval=POLL_INTERVAL,
            first=1,
            name="signal_loop"
        )
        logger.info("Signal loop started via JobQueue")
    else:
        asyncio.create_task(signal_loop(application.bot, chat_id))
        logger.info("Signal loop started via create_task (JobQueue missing)")
    
    # Start the application
    await application.start()
    
    # Start polling
    logger.info("Starting polling...")
    await application.updater.start_polling(drop_pending_updates=True)
    
    print(f"🚀 بوت إشارات {SYMBOL_DISPLAY} V3.2 يعمل...")
    
    # Keep the loop running
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Stopping...")
    finally:
        if application.updater.running:
            await application.updater.stop()
        await application.stop()
        await application.shutdown()

if __name__ == "__main__":
    try:
        # Use simple run since we're in the main entry point
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass

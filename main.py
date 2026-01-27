#!/usr/bin/env python3
"""
XRP/USDT Telegram Signals Bot V3.3 + Paper Trading
بوت إشارات تداول يرسل إشارات دخول/خروج لزوج XRP/USDT
V3.3: TP Trigger & Risk-Free Management
"""

import os
import csv
import asyncio
import logging
import time
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
COOLDOWN_AFTER_LOSS_STREAK = 15
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

DOWNTREND_ALERT_COOLDOWN = 300  # 5 minutes in seconds

ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
VERSION = "3.4 – ATR SL + Loss Analysis Button"
LOSS_EVENTS_FILE = "loss_events.csv"

# --- In-memory counters for loss analysis ---
loss_counters = {
    "STOP_HUNT": 0,
    "NOISE": 0,
    "TREND_REVERSAL": 0,
    "WEAK_ENTRY": 0,
    "UNKNOWN": 0
}

def calculate_atr(candles: List[dict], period: int = ATR_PERIOD) -> Optional[float]:
    if len(candles) < period + 1:
        return None
    
    tr_values = []
    for i in range(1, len(candles)):
        high = candles[i]['high']
        low = candles[i]['low']
        prev_close = candles[i-1]['close']
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)
    
    if not tr_values:
        return None
        
    return sum(tr_values[-period:]) / period

def classify_loss(entry_price: float, exit_price: float, entry_candles: List[dict], exit_candles: List[dict]) -> str:
    """
    Classify the type of loss based on price action.
    """
    if not entry_candles or not exit_candles:
        return "UNKNOWN"
    
    current_close = exit_candles[-1]['close']
    
    # 1. STOP_HUNT: SL hit then price reclaims EMA20 within N candles
    # We check if the last few candles show a reclaim
    ema20_vals = calculate_ema([c['close'] for c in exit_candles], EMA_SHORT)
    if ema20_vals and current_close > ema20_vals[-1]:
        return "STOP_HUNT"
        
    # 2. NOISE: Loss < X% and trade duration < Y candles
    pnl_pct = abs((exit_price - entry_price) / entry_price) * 100
    duration = len(exit_candles) # Approximation if exit_candles are those during trade
    if pnl_pct < 0.15 and duration < 10:
        return "NOISE"
        
    # 3. TREND_REVERSAL: Close below EMA20 & EMA50 with continuation
    ema50_vals = calculate_ema([c['close'] for c in exit_candles], EMA_LONG)
    if ema20_vals and ema50_vals:
        if current_close < ema20_vals[-1] and current_close < ema50_vals[-1]:
            return "TREND_REVERSAL"
            
    # 4. WEAK_ENTRY: Entry followed by immediate volume drop
    if len(exit_candles) >= 2:
        entry_vol = entry_candles[-1]['volume']
        subsequent_vol = exit_candles[0]['volume'] if exit_candles else 0
        if subsequent_vol < entry_vol * 0.5:
            return "WEAK_ENTRY"
            
    return "UNKNOWN"

def log_loss_event(loss_type: str, pnl_pct: float, entry_price: float, exit_price: float):
    global loss_counters
    loss_counters[loss_type] = loss_counters.get(loss_type, 0) + 1
    
    file_exists = os.path.exists(LOSS_EVENTS_FILE)
    with open(LOSS_EVENTS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'loss_type', 'pnl_pct', 'entry_price', 'exit_price'])
        writer.writerow([
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            loss_type,
            f"{pnl_pct:.2f}",
            f"{entry_price:.4f}",
            f"{exit_price:.4f}"
        ])

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
        self.last_downtrend_alert_time: float = 0
        self.tp_triggered: bool = False
        self.risk_free_sl: Optional[float] = None
        self.current_sl: Optional[float] = None
        self.entry_candles_snapshot: List[dict] = []

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


def calculate_targets(entry_price: float, candles: List[dict]) -> tuple:
    tp = entry_price * (1 + TAKE_PROFIT_PCT / 100)
    
    # Smart Stop Loss (ATR)
    atr = calculate_atr(candles)
    fixed_sl = entry_price * (STOP_LOSS_PCT / 100)
    
    if atr:
        sl_dist = max(fixed_sl, atr * ATR_MULTIPLIER)
    else:
        sl_dist = fixed_sl
        
    sl = entry_price - sl_dist
    return tp, sl

def check_exit_signal(analysis: dict, candles: List[dict]) -> Optional[str]:
    if not state.position_open or state.entry_price is None:
        return None
    
    current_price = analysis["close"]
    entry_price = state.entry_price
    pnl_pct = ((current_price - entry_price) / entry_price) * 100
    
    # v3.3: TP Trigger Logic
    if not state.tp_triggered and pnl_pct >= TAKE_PROFIT_PCT:
        state.tp_triggered = True
        state.risk_free_sl = entry_price * 1.001  # +0.1% Small profit
        return "tp_trigger"

    # v3.3: Exit Conditions after TP Triggered or Smart SL
    if state.tp_triggered:
        if current_price <= state.risk_free_sl:
            return "risk_free_sl_hit"
        if current_price < analysis["ema_short"]:
            return "ema_exit_post_tp"
    else:
        # Check Smart SL
        if state.current_sl and current_price <= state.current_sl:
            return "sl"
        # Fallback to fixed if for some reason current_sl is missing
        if not state.current_sl and pnl_pct <= -STOP_LOSS_PCT:
            return "sl"
    
    # Trailing SL (Existing logic preserved but secondary to TP trigger)
    if not state.tp_triggered:
        if pnl_pct >= TRAILING_TRIGGER_PCT:
            state.trailing_activated = True
        
        if state.trailing_activated and current_price < analysis["ema_short"]:
            return "trailing_sl"
    
    # EMA Confirmation (Original logic)
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
        
        # Classify and log loss
        # We need current candles for classification
        candles = get_klines(SYMBOL, state.timeframe)
        if candles:
            ltype = classify_loss(entry_price, exit_price, state.entry_candles_snapshot, candles)
            log_loss_event(ltype, pnl_pct, entry_price, exit_price)
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
    state.tp_triggered = False
    state.risk_free_sl = None
    state.current_sl = None
    state.entry_candles_snapshot = []


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
            InlineKeyboardButton("📊 الإحصائيات", callback_data="stats")
        ],
        [
            InlineKeyboardButton("📉 تحليل الخسائر", callback_data="loss_analysis")
        ],
        [
            InlineKeyboardButton("⚙️ الإعدادات", callback_data="settings"),
            InlineKeyboardButton("📜 السجل", callback_data="history")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == "status":
        candles = get_klines(SYMBOL, state.timeframe)
        analysis = analyze_market(candles)
        status_text = format_status_message()
        await query.edit_message_text(text=status_text, reply_markup=get_main_keyboard(), parse_mode='Markdown')
    
    elif query.data == "loss_analysis":
        summary = "📉 <b>تحليل الخسائر الأخير</b>\n\n"
        total_losses = sum(loss_counters.values())
        if total_losses == 0:
            summary += "لا توجد بيانات خسائر كافية حالياً."
        else:
            for ltype, count in loss_counters.items():
                pct = (count / total_losses) * 100
                summary += f"• {ltype}: {count} ({pct:.1f}%)\n"
            
            # Find most frequent
            most_frequent = max(loss_counters, key=loss_counters.get)
            if loss_counters[most_frequent] > 0:
                summary += f"\n⚠️ نقطة الضعف الأكثر تكراراً: <b>{most_frequent}</b>"
        
        await query.edit_message_text(text=summary, reply_markup=get_main_keyboard(), parse_mode='HTML')

    elif query.data == "on":
        state.signals_enabled = True
        await query.edit_message_text("✅ تم تشغيل الإشارات\n\n" + format_status_message(), reply_markup=get_main_keyboard(), parse_mode="Markdown")
    elif query.data == "off":
        state.signals_enabled = False
        await query.edit_message_text("⏸️ تم إيقاف الإشارات\n\n" + format_status_message(), reply_markup=get_main_keyboard(), parse_mode="Markdown")
    elif query.data == "balance":
        await query.edit_message_text(format_balance_message(), reply_markup=get_main_keyboard(), parse_mode="Markdown")
    elif query.data == "trades":
        await query.edit_message_text(format_trades_message(), reply_markup=get_main_keyboard(), parse_mode="Markdown")
    elif query.data == "stats":
        await query.edit_message_text(format_stats_message(), reply_markup=get_main_keyboard(), parse_mode="Markdown")
    elif query.data == "rules":
        await query.edit_message_text(format_rules_message(), reply_markup=get_main_keyboard(), parse_mode="Markdown")
    elif query.data == "reset":
        await query.edit_message_text("⚠️ *هل تريد تصفير الرصيد والسجل؟*\n\n", reply_markup=get_confirm_keyboard(), parse_mode="Markdown")
    elif query.data == "confirm_reset":
        paper_state.reset()
        reset_position_state()
        await query.edit_message_text(f"✅ تم تصفير الرصيد إلى {START_BALANCE:.0f} USDT\n\n" + format_status_message(), reply_markup=get_main_keyboard(), parse_mode="Markdown")
    elif query.data == "cancel_reset":
        await query.edit_message_text("❌ تم إلغاء التصفير\n\n" + format_status_message(), reply_markup=get_main_keyboard(), parse_mode="Markdown")
    elif query.data in ["tf_1m", "tf_5m"]:
        new_tf = "1m" if query.data == "tf_1m" else "5m"
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


def get_confirm_keyboard():
    keyboard = [
        [
            InlineKeyboardButton("✅ نعم، متأكد", callback_data="confirm_reset"),
            InlineKeyboardButton("❌ إلغاء", callback_data="cancel_reset")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def format_welcome_message() -> str:
    return (
        f"🤖 *بوت إشارات {SYMBOL_DISPLAY} V3.2*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ تم تفعيل نظام Paper Trading\n"
        f"💰 الرصيد الابتدائي: {START_BALANCE} USDT\n"
        f"🛡️ نظام Kill Switch مفعّل للحماية\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"استخدم الأزرار أدناه للتحكم"
    )


def format_status_message() -> str:
    status = "🟢 يعمل" if state.signals_enabled else "⏸️ متوقف"
    ks_status = "✅ آمن"
    if kill_switch.active:
        ks_status = f"🛑 متوقف ({kill_switch.reason})"
    
    cooldown = 0
    if state.pause_until:
        rem = (state.pause_until - datetime.now(timezone.utc)).total_seconds()
        cooldown = max(0, int(rem))
    
    pos_status = "❌ لا توجد صفقة"
    if state.position_open:
        pnl = ((state.last_close - state.entry_price) / state.entry_price) * 100 if state.last_close and state.entry_price else 0
        pos_status = f"✅ صفقة مفتوحة ({pnl:+.2f}%)"
    
    return (
        f"📊 *حالة البوت الحالية*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 الحالة: {status}\n"
        f"🛡️ Kill Switch: {ks_status}\n"
        f"⏱️ الفريم: {state.timeframe}\n"
        f"💰 الرصيد: {paper_state.balance:.2f} USDT\n"
        f"📍 الصفقة: {pos_status}\n"
        f"⏳ Cooldown: {cooldown} ثانية\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"آخر سعر: {state.last_close if state.last_close else '---'}"
    )


def format_balance_message() -> str:
    stats = get_paper_stats()
    return (
        f"💰 *تفاصيل الرصيد - Paper Trading*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 الرصيد الحالي: {stats['balance']:.2f} USDT\n"
        f"📈 أعلى رصيد: {stats['peak_balance']:.2f} USDT\n"
        f"📉 أقصى تراجع: {stats['drawdown']:.2f}%\n"
        f"📊 إجمالي الربح/الخسارة: {stats['total_pnl']:+.2f} USDT\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"رأس المال الابتدائي: {START_BALANCE} USDT"
    )


def format_trades_message() -> str:
    trades = get_paper_trades(5)
    if not trades:
        return "📜 *لا توجد صفقات مغلقة بعد*"
    
    msg = "📜 *آخر 5 صفقات منفذة*\n━━━━━━━━━━━━━━━━━━━━━\n"
    for t in trades:
        emoji = "🟢" if t['pnl_usdt'] >= 0 else "🔴"
        msg += f"{emoji} {t['timestamp'].split(' ')[1]} | {t['pnl_pct']:+.2f}% | {t['pnl_usdt']:+.2f} $\n"
    
    return msg


def format_stats_message() -> str:
    stats = get_paper_stats()
    return (
        f"📊 *إحصائيات الأداء الكاملة*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔢 إجمالي الصفقات: {stats['total']}\n"
        f"✅ الصفقات الناجحة: {stats['wins']}\n"
        f"❌ الصفقات الخاسرة: {stats['losses']}\n"
        f"🎯 نسبة النجاح: {stats['win_rate']:.1f}%\n"
        f"🔥 سلسلة الخسائر: {stats['loss_streak']}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Win Rate (آخر 10): {calculate_recent_win_rate():.1f}%"
    )


def format_rules_message() -> str:
    return (
        f"⚖️ *قواعد التداول V3.3*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔹 حجم الصفقة: {FIXED_TRADE_SIZE} USDT\n"
        f"🔹 محفز الربح (Trigger): {TAKE_PROFIT_PCT}%\n"
        f"🔹 وقف الخسارة: {STOP_LOSS_PCT}%\n"
        f"🔹 تأمين الصفقة: رفع SL لـ +0.1%\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🛡️ *قواعد الحماية (Kill Switch)*\n"
        f"• 3 خسائر متتالية = إيقاف\n"
        f"• تراجع 3% من أعلى رصيد = إيقاف\n"
        f"• أقل من 40% نجاح (آخر 10) = إيقاف"
    )


def format_buy_message(price: float, tp: float, sl: float, tf: str, score: int, qty: float) -> str:
    return (
        f"🚀 *إشارة شراء جديدة - Paper Trading*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 الزوج: {SYMBOL_DISPLAY}\n"
        f"⏱ الفريم: {tf}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🟢 الدخول: {price:.4f}\n"
        f"🎯 الهدف (TP): {tp:.4f}\n"
        f"🛑 الوقف (SL): {sl:.4f}\n"
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
        "ema_confirmation": "EMA Exit 📉",
        "risk_free_sl_hit": "Trailing SL (Risk-Free) 🛡️",
        "ema_exit_post_tp": "EMA Exit (Post-TP) 📈"
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
    
    # Downtrend Alerts
    last_alert = "لا يوجد"
    if state.last_downtrend_alert_time > 0:
        last_alert = datetime.fromtimestamp(state.last_downtrend_alert_time, tz=timezone.utc).strftime("%H:%M:%S")
    msg += "📉 *تنبيهات الهبوط*\n"
    msg += f"• آخر تنبيه هبوط: {last_alert}\n\n"
    
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


async def check_downtrend_alerts(bot: Bot, chat_id: str, analysis: dict, candles: List[dict]):
    """
    إرسال تنبيهات الهبوط (للمراقبة فقط)
    """
    if state.position_open or kill_switch.active:
        return

    now = datetime.now(timezone.utc).timestamp()
    if now - state.last_downtrend_alert_time < DOWNTREND_ALERT_COOLDOWN:
        return

    reason = ""
    target = 0.0
    current_close = analysis["close"]
    
    # 1. Check conditions and set reason/target (Single primary reason as requested)
    if current_close < analysis["ema_short"]:
        reason = "كسر المتوسط المتحرك EMA20"
        if analysis["ema_long"] < current_close:
            target = analysis["ema_long"]
        else:
            # Lowest of last 10 candles
            last_10 = candles[-10:]
            target = min(c["low"] for c in last_10) if last_10 else current_close
            
    elif current_close < analysis["ema_long"]:
        reason = "كسر المتوسط المتحرك EMA50"
        # Nearest previous swing low that is < current_price
        # Search back up to 50 candles
        target = 0.0
        lookback = candles[-50:-1]
        for c in reversed(lookback):
            if c["low"] < current_close:
                target = c["low"]
                break
    else:
        # Lowest of previous 5 candles (excluding current)
        prev_lows = [c["low"] for c in candles[-6:-1]]
        lowest_low = min(prev_lows) if prev_lows else current_close
        
        if current_close < lowest_low:
            reason = "كسر قاع آخر 5 شموع"
            # Lowest LOW before those 5 candles
            before_5 = candles[-20:-6]
            target = min(c["low"] for c in before_5) if before_5 else 0.0

    if reason:
        # Final safety check
        if target >= current_close or target == 0.0:
            target_text = "الهدف غير واضح حاليًا"
        else:
            target_text = f"{target:.4f}"

        msg = (
            "⚠️ *تنبيه هبوط (مراقبة فقط)*\n\n"
            f"الزوج: {SYMBOL_DISPLAY}\n"
            f"الفريم: {state.timeframe}\n"
            f"السعر الحالي: {current_close:.4f}\n\n"
            "سبب التنبيه:\n"
            f"{reason}\n\n"
            "الهدف المحتمل للكسر:\n"
            f"{target_text}\n\n"
            f"⏱ الوقت: {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC\n\n"
            "❌ تنبيه فقط – لا يوجد أي تنفيذ تداول"
        )
        if await send_signal_message(bot, chat_id, msg, "downtrend_alert"):
            state.last_downtrend_alert_time = now


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
            
        # Downtrend Alerts (Monitoring Only)
        await check_downtrend_alerts(bot, chat_id, analysis, candles)
        
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
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        await application.stop()
        await application.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

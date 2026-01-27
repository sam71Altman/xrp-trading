#!/usr/bin/env python3
"""
XRP/USDT Telegram Signals Bot V3.2 + Paper Trading
بوت إشارات تداول يرسل إشارات دخول/خروج لزوج XRP/USDT
V3.2: Kill Switch متعدد الطبقات لحماية رأس المال
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


def get_historical_klines(symbol: str, interval: str, days: int = BACKTEST_DAYS) -> Optional[List[dict]]:
    if interval == "1m":
        limit = min(days * 24 * 60, 1000)
    elif interval == "5m":
        limit = min(days * 24 * 12, 1000)
    else:
        limit = 500
    
    return get_klines(symbol, interval, limit)


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


def run_backtest(candles: List[dict]) -> Dict:
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


def analyze_market(candles: List[dict]) -> dict:
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
    
    if score < MIN_SIGNAL_SCORE:
        return False
    
    return True


def check_exit_signal(analysis: dict) -> Optional[str]:
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
        paper_state.loss_streak += 1
        state.current_cooldown = COOLDOWN_AFTER_SL
        
        if state.consecutive_losses >= 2:
            state.pause_until = datetime.now(timezone.utc) + timedelta(minutes=COOLDOWN_PAUSE_MINUTES)
            state.pause_alerted = False
    
    elif exit_type == "tp":
        state.consecutive_wins += 1
        state.consecutive_losses = 0
        paper_state.loss_streak = 0
        
        if state.consecutive_wins >= 2:
            state.current_cooldown = COOLDOWN_STREAK_WIN
        else:
            state.current_cooldown = COOLDOWN_NORMAL
    
    else:
        state.current_cooldown = COOLDOWN_NORMAL


def execute_paper_buy(entry_price: float, score: int, reasons: List[str]) -> float:
    qty = FIXED_TRADE_SIZE / entry_price
    paper_state.position_qty = qty
    paper_state.entry_reason = "; ".join(reasons) if reasons else "Signal"
    
    log_paper_trade(
        action="BUY",
        entry_price=entry_price,
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
    qty = paper_state.position_qty
    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
    pnl_usdt = qty * (exit_price - entry_price)
    
    paper_state.balance += pnl_usdt
    paper_state.update_peak()
    
    ks_reason = evaluate_kill_switch()
    ks_triggered = ks_reason is not None
    
    log_paper_trade(
        action="EXIT",
        entry_price=entry_price,
        exit_price=exit_price,
        pnl_pct=pnl_pct,
        pnl_usdt=pnl_usdt,
        balance_after=paper_state.balance,
        score=score,
        entry_reason=paper_state.entry_reason,
        exit_reason=exit_reason.upper(),
        duration_min=duration_min,
        ks_triggered=ks_triggered,
        ks_reason=ks_reason if ks_reason else ""
    )
    
    return pnl_pct, pnl_usdt, paper_state.balance


def get_main_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("📊 الحالة", callback_data="status"),
            InlineKeyboardButton("💰 الرصيد", callback_data="balance"),
        ],
        [
            InlineKeyboardButton("📈 الصفقات", callback_data="trades"),
            InlineKeyboardButton("📋 الإحصائيات", callback_data="stats"),
        ],
        [
            InlineKeyboardButton("▶️ تشغيل", callback_data="on"),
            InlineKeyboardButton("⏸️ إيقاف", callback_data="off"),
        ],
        [
            InlineKeyboardButton("📖 القواعد", callback_data="rules"),
            InlineKeyboardButton("🔄 تصفير", callback_data="reset"),
        ],
        [
            InlineKeyboardButton("1m", callback_data="tf_1m"),
            InlineKeyboardButton("5m", callback_data="tf_5m"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_confirm_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("✅ تأكيد", callback_data="confirm_reset"),
            InlineKeyboardButton("❌ إلغاء", callback_data="cancel_reset"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def format_welcome_message() -> str:
    return (
        f"🤖 *بوت إشارات {SYMBOL_DISPLAY} V3.2*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 نمط: Paper Trading\n"
        f"💵 رأس المال: {START_BALANCE:.0f} USDT\n"
        f"📦 حجم الصفقة: {FIXED_TRADE_SIZE:.0f} USDT\n"
        f"🎯 TP: +{TAKE_PROFIT_PCT}% | SL: -{STOP_LOSS_PCT}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🛡️ *Kill Switch V3.2:*\n"
        f"• {LOSS_STREAK_LIMIT} خسائر متتالية = إيقاف\n"
        f"• Drawdown > {DRAWDOWN_LIMIT_PERCENT}% = إيقاف\n"
        f"• Win Rate < {RECENT_WIN_RATE_MIN}% (آخر 10) = إيقاف\n"
        f"• استئناف تلقائي بعد {AUTO_RESUME_MINUTES} دقيقة\n"
        f"━━━━━━━━━━━━━━━━━━━━━"
    )


def format_status_message() -> str:
    stats = get_paper_stats()
    
    status = "✅ نشط" if state.signals_enabled and not kill_switch.active else "⏸️ متوقف"
    position = "📈 مفتوحة" if state.position_open else "⚪ لا توجد"
    
    msg = (
        f"📊 *حالة البوت*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔔 الإشارات: {status}\n"
        f"📍 الصفقة: {position}\n"
        f"⏱ الفريم: {state.timeframe}\n"
        f"💰 الرصيد: {paper_state.balance:.2f} USDT\n"
        f"🏔 أعلى رصيد: {paper_state.peak_balance:.2f} USDT\n"
        f"📉 Drawdown: {stats['drawdown']:.2f}%\n"
        f"🔴 سلسلة الخسائر: {paper_state.loss_streak}\n"
    )
    
    if kill_switch.active:
        remaining = kill_switch.get_remaining_minutes()
        msg += (
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🛑 *Kill Switch مفعّل*\n"
            f"📌 السبب: {kill_switch.reason}\n"
            f"⏱ الاستئناف بعد: {remaining} دقيقة\n"
        )
    
    if state.position_open and state.entry_price:
        if state.last_close:
            pnl = calculate_pnl(state.entry_price, state.last_close)
            emoji = "🟢" if pnl >= 0 else "🔴"
            msg += (
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"💹 سعر الدخول: {state.entry_price:.4f}\n"
                f"📍 السعر الحالي: {state.last_close:.4f}\n"
                f"{emoji} PnL: {pnl:+.2f}%\n"
            )
    
    msg += f"━━━━━━━━━━━━━━━━━━━━━"
    return msg


def format_balance_message() -> str:
    stats = get_paper_stats()
    recent_win_rate = calculate_recent_win_rate()
    
    return (
        f"💰 *رصيد Paper Trading*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 الرصيد الحالي: {paper_state.balance:.2f} USDT\n"
        f"🏔 أعلى رصيد: {paper_state.peak_balance:.2f} USDT\n"
        f"📊 إجمالي الربح: {stats['total_pnl']:+.2f} USDT\n"
        f"📈 نسبة النجاح: {stats['win_rate']:.1f}%\n"
        f"📉 Drawdown: {stats['drawdown']:.2f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔴 سلسلة الخسائر: {paper_state.loss_streak}/{LOSS_STREAK_LIMIT}\n"
        f"📊 Win Rate (آخر 10): {recent_win_rate:.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━"
    )


def format_trades_message() -> str:
    trades = get_paper_trades(5)
    
    if not trades:
        return "📈 *آخر الصفقات*\n━━━━━━━━━━━━━━━━━━━━━\nلا توجد صفقات مسجلة"
    
    msg = "📈 *آخر الصفقات*\n━━━━━━━━━━━━━━━━━━━━━\n"
    
    for t in trades:
        emoji = "🟢" if t['pnl_usdt'] >= 0 else "🔴"
        msg += (
            f"{emoji} {t['exit_reason']}: {t['pnl_pct']:+.2f}% ({t['pnl_usdt']:+.2f}$)\n"
        )
    
    msg += f"━━━━━━━━━━━━━━━━━━━━━"
    return msg


def format_stats_message() -> str:
    stats = get_paper_stats()
    recent_win_rate = calculate_recent_win_rate()
    closed_count = len(get_closed_trades())
    
    maturity_status = "✅ نشط" if closed_count >= DATA_MATURITY_TRADES else f"⏳ {closed_count}/{DATA_MATURITY_TRADES}"
    
    return (
        f"📋 *إحصائيات التداول*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 إجمالي الصفقات: {stats['total']}\n"
        f"✅ رابحة: {stats['wins']}\n"
        f"❌ خاسرة: {stats['losses']}\n"
        f"📈 نسبة النجاح: {stats['win_rate']:.1f}%\n"
        f"💵 إجمالي الربح: {stats['total_pnl']:+.2f} USDT\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🛡️ *Kill Switch:*\n"
        f"• نضج البيانات: {maturity_status}\n"
        f"• سلسلة الخسائر: {paper_state.loss_streak}/{LOSS_STREAK_LIMIT}\n"
        f"• Drawdown: {stats['drawdown']:.2f}% / {DRAWDOWN_LIMIT_PERCENT}%\n"
        f"• Win Rate (آخر 10): {recent_win_rate:.1f}% / {RECENT_WIN_RATE_MIN}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━"
    )


def format_rules_message() -> str:
    return (
        f"📖 *قواعد الاستراتيجية V3.2*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"*شروط الدخول:*\n"
        f"• EMA{EMA_SHORT} > EMA{EMA_LONG}\n"
        f"• كسر قمة {BREAKOUT_CANDLES} شموع\n"
        f"• حجم أعلى من المتوسط\n"
        f"• Score >= {MIN_SIGNAL_SCORE}/10\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"*شروط الخروج:*\n"
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
        f"• /استئناف للاستئناف اليدوي\n"
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


async def cmd_الحالة(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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


async def cmd_احصائيات(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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


async def cmd_استئناف(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not kill_switch.active:
        await update.message.reply_text(
            "✅ التداول نشط بالفعل",
            reply_markup=get_main_keyboard()
        )
        return
    
    resume_trading()
    state.signals_enabled = True
    await update.message.reply_text(
        "✅ تم استئناف التداول يدوياً\n\n" + format_status_message(),
        reply_markup=get_main_keyboard(),
        parse_mode="Markdown"
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


async def signal_loop(bot: Bot, chat_id: str) -> None:
    logger.info(f"بدء حلقة الإشارات (Paper Trading V3.2)")
    
    init_trades_file()
    init_paper_trades_file()
    
    while True:
        try:
            if kill_switch.check_auto_resume():
                resume_trading()
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text="✅ تم استئناف التداول تلقائياً بعد انتهاء فترة الإيقاف",
                        parse_mode="Markdown"
                    )
                except:
                    pass
            
            if not state.signals_enabled:
                await asyncio.sleep(POLL_INTERVAL)
                continue
            
            if kill_switch.active:
                if not kill_switch.alert_sent:
                    try:
                        await bot.send_message(
                            chat_id=chat_id,
                            text=f"🛑 تم إيقاف التداول مؤقتًا بسبب {kill_switch.reason}",
                            parse_mode="Markdown"
                        )
                        kill_switch.alert_sent = True
                    except:
                        pass
                await asyncio.sleep(POLL_INTERVAL)
                continue
            
            if state.pause_until and datetime.now(timezone.utc) < state.pause_until:
                if not state.pause_alerted:
                    try:
                        await bot.send_message(
                            chat_id=chat_id,
                            text="⏸️ إيقاف مؤقت بعد خسارتين متتاليتين",
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
            
            analysis = analyze_market(candles)
            
            if "error" in analysis:
                await asyncio.sleep(POLL_INTERVAL)
                continue
            
            ks_reason = evaluate_kill_switch()
            if ks_reason and not state.position_open:
                kill_switch.activate(ks_reason)
                await asyncio.sleep(POLL_INTERVAL)
                continue
            
            if state.position_open and state.entry_price is not None:
                exit_reason = check_exit_signal(analysis)
                if exit_reason:
                    exit_price = analysis["close"]
                    duration = get_trade_duration_minutes()
                    entry = state.entry_price
                    
                    pnl_pct, pnl_usdt, balance = execute_paper_exit(
                        entry, exit_price, exit_reason,
                        state.last_signal_score, duration
                    )
                    
                    log_trade("EXIT", exit_reason.upper(), exit_price, pnl_pct)
                    
                    msg = format_exit_message(
                        entry, exit_price, pnl_pct, pnl_usdt,
                        exit_reason, duration, balance
                    )
                    sent = await send_signal_message(bot, chat_id, msg, "exit")
                    
                    if sent:
                        update_cooldown_after_exit(exit_reason)
                        reset_position_state()
                        logger.info(f"إغلاق المركز: {exit_reason} @ {exit_price:.4f} (PnL: {pnl_pct:.2f}%)")
                        
                        ks_reason = evaluate_kill_switch()
                        if ks_reason:
                            kill_switch.activate(ks_reason)
            
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
    
    logger.info(f"بدء بوت إشارات {SYMBOL_DISPLAY} V3.2 - Paper Trading")
    
    application = Application.builder().token(tg_token).build()
    
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("الحالة", cmd_الحالة))
    application.add_handler(CommandHandler("balance", cmd_balance))
    application.add_handler(CommandHandler("trades", cmd_trades))
    application.add_handler(CommandHandler("on", cmd_on))
    application.add_handler(CommandHandler("off", cmd_off))
    application.add_handler(CommandHandler("rules", cmd_rules))
    application.add_handler(CommandHandler("stats", cmd_stats))
    application.add_handler(CommandHandler("احصائيات", cmd_احصائيات))
    application.add_handler(CommandHandler("استئناف", cmd_استئناف))
    application.add_handler(CommandHandler("reset", cmd_reset))
    application.add_handler(CommandHandler("settf", cmd_timeframe))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    bot = application.bot
    
    await application.initialize()
    await application.start()
    await application.updater.start_polling(drop_pending_updates=True)
    
    print("=" * 50)
    print(f"🚀 بوت إشارات {SYMBOL_DISPLAY} V3.2 - Paper Trading")
    print(f"🛡️ Kill Switch: متعدد الطبقات")
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

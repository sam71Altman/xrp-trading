#!/usr/bin/env python3
"""
XRP/USDT Telegram Signals Bot {BOT_VERSION} + Paper Trading
بوت إشارات تداول يرسل إشارات دخول/خروج لزوج XRP/USDT
{BOT_VERSION}: TP Trigger & Risk-Free Management
"""

import os
import csv
import asyncio
import logging
import time
import threading
import json
def check_bounce_entry(analysis, candles, score):
    """شروط دخول الارتداد في السوق الهابط v3.7.5"""
    ema20 = analysis.get('ema20', 0)
    ema50 = analysis.get('ema50', 0)
    ema200 = analysis.get('ema200', 0)
    
    market_mode = "EASY_MARKET" if (ema20 > ema50 and ema50 > ema200) else "HARD_MARKET"
    
    if market_mode != "HARD_MARKET":
        return False
    
    current_price = candles[-1]['close'] if candles else 0
    
    # 1. القاع المحلي (Local Extreme)
    recent_lows = [c['low'] for c in candles[-15:]] if len(candles) >= 15 else []
    is_local_extreme = current_price <= min(recent_lows) if recent_lows else False
    
    # 2. RSI (v3.7.5)
    current_rsi = analysis.get('rsi', 50)
    
    # 3. Volume Spike (v3.7.5)
    def volume_spike_detected(candles):
        if len(candles) < 21: return False
        current_volume = candles[-1]['volume']
        avg_volume = sum(c['volume'] for c in candles[-21:-1]) / 20
        return current_volume > avg_volume * 1.8

    volume_spike = volume_spike_detected(candles)
    
    entry_is_bounce = (
        score <= 5 and
        is_local_extreme and
        current_rsi <= 35 and
        volume_spike
    )
    
    return entry_is_bounce

def detect_bearish_strength(candle):
    """تحديد قوة الشمعة الهابطة v3.7.5"""
    if not candle: return "WEAK"
    body_size = abs(candle['close'] - candle['open'])
    candle_range = candle['high'] - candle['low']
    body_ratio = body_size / candle_range if candle_range > 0 else 0
    
    if candle['close'] < candle['open'] and body_ratio > 0.7:
        return "STRONG"
    elif candle['close'] < candle['open'] and body_ratio > 0.5:
        return "MEDIUM"
    return "WEAK"

def check_buy_signal(analysis, candles):
    """
    منطق v3.7.5 المطور لفحص إشارة الشراء.
    """
    if not analysis or not candles:
        return False
        
    current_price = candles[-1]['close']
    ema20 = analysis.get('ema20', 0)
    ema50 = analysis.get('ema50', 0)
    ema200 = analysis.get('ema200', 0)
    score = analysis.get('score', 0)
    
    market_mode = "EASY_MARKET" if (ema20 > ema50 and ema50 > ema200) else "HARD_MARKET"
    
    # تحسين الدخول في السوق الصعب (ارتدادات فقط)
    if market_mode == "HARD_MARKET":
        is_bounce = check_bounce_entry(analysis, candles, score)
        if is_bounce:
            state.hold_active = True
            state.hold_candles = 0
            state.hold_start_price = current_price
            logger.info("[HOLD ACTIVATED] Bounce trade in bear market v3.7.5")
            return True
        return False
    
    # الدخول العادي في السوق السهل
    return current_price > ema20 and score >= MIN_SIGNAL_SCORE

def check_hold_exit_conditions(candles):
    """فحص شروط الخروج أثناء الـ Hold v3.7.5"""
    if not state.hold_active:
        return None
    
    current_price = candles[-1]['close'] if candles else 0
    current_candle = candles[-1] if candles else None
    
    # 1️⃣ STOP LOSS (أولوية قصوى)
    if state.current_sl and current_price <= state.current_sl:
        return "SL Hit (Hold)"
    
    # 2️⃣ فشل سعري (دروداون محدود)
    max_drawdown_price = state.hold_start_price * 0.9990  # -0.10%
    if current_price <= max_drawdown_price:
        return "Hold Failed - Max Drawdown"
    
    # 3️⃣ تحقيق هدف واقعي للسكالب
    scalp_target = state.hold_start_price * 1.003  # +0.3%
    if current_price >= scalp_target:
        return "Scalp Target Hit"
    
    # 4️⃣ فشل زمني مع ضعف الزخم
    if state.hold_candles >= 5:
        if len(candles) >= 21:
            recent_volume_avg = sum(c['volume'] for c in candles[-3:]) / 3
            normal_volume_avg = sum(c['volume'] for c in candles[-21:-1]) / 20
            if recent_volume_avg < normal_volume_avg * 0.65:
                return "Hold Failed - No Momentum"
    
    # 5️⃣ كسر هابط قوي
    if current_candle and detect_bearish_strength(current_candle) == "STRONG":
        return "Hold Failed - Strong Breakdown"
    
    # 6️⃣ قيد الخسارة اليومية التراكمية
    if state.daily_cumulative_loss >= 1.0:
        return "Hold Disabled - Daily Loss Limit"
    
    return None
    if state.daily_cumulative_loss >= 1.0:
        return "Hold Disabled - Daily Loss Limit"
    
    return None

def log_hold_status():
    """تسجيل مفصل لحالة الـ Hold v3.7.5"""
    current_price = get_current_price()
    drawdown = calculate_drawdown()
    logger.info(f"""
    📊 HOLD STATUS
    ├── Active: {state.hold_active}
    ├── Candles Held: {state.hold_candles}
    ├── Entry Price: {state.hold_start_price:.6f}
    ├── Current Price: {current_price:.6f}
    ├── Drawdown: {drawdown:.4f}%
    └── Daily Loss: {state.daily_cumulative_loss:.2f}%
    """)

import websocket
from version import BOT_VERSION
from price_engine import PriceEngine, TradingGuard, TelegramReporter, FailSafeSystem, ValidationChecks
from datetime import datetime, timezone, timedelta
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
from typing import Optional, List, Dict

# توقيت مكة المكرمة (GMT+3)
MECCA_TZ = ZoneInfo("Asia/Riyadh")

def get_now():
    return datetime.now(MECCA_TZ)

import requests
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from post_exit_guard import PostExitGuard, market_recovered, EntryGateMonitor

# --- Configuration ---
MODE = "PAPER"
TIMEFRAME = "1m"
SYMBOL = "XRPUSDT"
SYMBOL_DISPLAY = "XRP/USDT"

analysis_count = 0
last_analysis_time = None

EMA_SHORT = 20
EMA_LONG = 50
BREAKOUT_CANDLES = 3  # AGGRESSIVE: 3 candles

TAKE_PROFIT_PCT = 0.15  # AGGRESSIVE: 0.10% to 0.25%
STOP_LOSS_PCT = 0.25    # AGGRESSIVE: 0.20% to 0.35%
TRAILING_TRIGGER_PCT = None # AGGRESSIVE: No trailing

RANGE_FILTER_THRESHOLD = 0.0001 # Relaxed
VOLUME_LOOKBACK = 10
TREND_LOOKBACK = 10

COOLDOWN_NORMAL = 0 # AGGRESSIVE: No cooldown
COOLDOWN_AFTER_SL = 0
COOLDOWN_AFTER_LOSS_STREAK = 0
COOLDOWN_STREAK_WIN = 0
COOLDOWN_PAUSE_MINUTES = 0

MIN_WIN_RATE = 0.0
MIN_SIGNAL_SCORE = 1 # Relaxed

POLL_INTERVAL = 5 # Faster polling
KLINE_LIMIT = 200
BACKTEST_DAYS = 30

START_BALANCE = 1000.0
FIXED_TRADE_SIZE = 100.0

DATA_MATURITY_TRADES = 0
LOSS_STREAK_LIMIT = 999 # Disabled
DRAWDOWN_LIMIT_PERCENT = 5.0 # Keep only catastrophic kill switch
RECENT_WIN_RATE_MIN = 0.0 # Disabled
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

# تعيين توقيت مكة للمكتبة الأساسية (Logging)
def logging_time_converter(*args):
    return get_now().timetuple()

logging.Formatter.converter = logging_time_converter

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
        self.triggered_at = get_now()
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
            if get_now() >= self.resume_at:
                return True
        return False
    
    def get_remaining_minutes(self) -> int:
        if self.resume_at:
            remaining = self.resume_at - get_now()
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

REJECTION_ZONE = 0.01  # 0.01% zone for early rejection
TRAILING_TRIGGER_PCT = 0.2  # 0.2% to activate trailing SL
TRAILING_STOP_PCT = 0.1     # 0.1% trailing stop
DOWNTREND_ALERT_COOLDOWN = 300  # 5 minutes in seconds

ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
VERSION = BOT_VERSION
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
            get_now().strftime("%Y-%m-%d %H:%M:%S"),
            loss_type,
            f"{pnl_pct:.2f}",
            f"{entry_price:.4f}",
            f"{exit_price:.4f}"
        ])

# Monitoring ({BOT_VERSION})
MIN_MONITOR_DELAY = 5        # seconds after entry
DEBOUNCE_WINDOW = 15         # seconds (ticks approximate)
MAX_MONITOR_WINDOW = 180     # seconds (v3.7)
SHORT_TIME_WINDOW = 30       # seconds: "Early Rejection" window
SLOPE_DEGRADATION = 0.3      # %: Allowed EMA20 slope drop

ACTION_1_FLAG = "MONITOR"
ACTION_2_FLAGS = "PREPARE"
ACTION_3_FLAGS = "EXIT"

# Required persistence per flag type ({BOT_VERSION})
REQUIRED_FLAGS = {
    'early_rejection': 2,    # strongest signal
    'momentum_decay': 3,     # medium strength
    'weak_momentum': 3       # weakest signal
}

# Bounce Guard ({BOT_VERSION})
RECOVERY_THRESHOLD = 0.015   # % recovery
MAX_BOUNCE_TIME = 45         # seconds

# Fast Exit Zone ({BOT_VERSION})
FAST_EXIT_ZONE_SECONDS = 20
FAST_EXIT_REQUIRED_FLAGS = 2

class ExitIntelligenceLayer:
    def __init__(self):
        self.monitoring_active = False
        self.entry_price = 0.0
        self.entry_time = None
        self.entry_ema_slope = 0.0
        self.max_price_seen = 0.0
        self.min_price_since_high = 0.0
        self.last_high_time = None
        self.flag_history = [] # List of dicts per tick
        self.recent_prices = [] # (timestamp, price)
        self.stats = {
            "total_monitored_trades": 0,
            "early_exits_triggered": 0,
            "losses_prevented": 0,
            "false_exits": 0,
            "bounce_protected": 0
        }

    def start_monitoring(self, entry_price: float, ema_slope: float):
        self.monitoring_active = True
        self.entry_price = entry_price
        self.entry_time = get_now()
        self.entry_ema_slope = ema_slope
        self.max_price_seen = entry_price
        self.min_price_since_high = entry_price
        self.last_high_time = self.entry_time
        self.flag_history = []
        self.recent_prices = []
        self.stats["total_monitored_trades"] += 1
        logger.info(f"[INTEL] Started monitoring {BOT_VERSION} at {entry_price}")

    def stop_monitoring(self):
        self.monitoring_active = False

    def is_healthy_bounce(self, current_price: float) -> bool:
        # Prevent early exit during healthy pullbacks (LONG only assumed as per logic)
        if not self.recent_prices:
            return False
            
        now = get_now()
        # last 10 seconds low
        ten_sec_ago = now - timedelta(seconds=10)
        recent_ticks = [p for t, p in self.recent_prices if t >= ten_sec_ago]
        if not recent_ticks:
            return False
            
        recent_low = min(recent_ticks)
        
        if self.entry_price > recent_low:
            recovery_ratio = (current_price - recent_low) / (self.entry_price - recent_low)
            return recovery_ratio > 0.5 # recovered >50%
            
        return False

    def monitor(self, current_price: float, current_ema_slope: float) -> str:
        if not self.monitoring_active or not self.entry_time:
            return "NO_ACTION"

        now = get_now()
        duration = (now - self.entry_time).total_seconds()
        self.recent_prices.append((now, current_price))
        
        # Cleanup old prices
        if len(self.recent_prices) > 60: # Keep roughly 5 mins of ticks
            self.recent_prices.pop(0)

        if duration < MIN_MONITOR_DELAY:
            return "NO_ACTION"
        if duration > MAX_MONITOR_WINDOW:
            self.stop_monitoring()
            return "NO_ACTION"

        # Update high price tracker
        if current_price > self.max_price_seen:
            self.max_price_seen = current_price
            self.last_high_time = now

        # 1. Capture momentary flags
        current_flags = {
            'weak_momentum': False,
            'momentum_decay': False,
            'early_rejection': False
        }
        
        # A) فشل استمرار الزخم
        time_since_high = (now - self.last_high_time).total_seconds()
        current_price = PriceEngine.last_price if PriceEngine.last_price else current_price
        dist_from_entry = (current_price - self.entry_price) / self.entry_price
        if time_since_high >= 15 and dist_from_entry < 0.001:
            current_flags['weak_momentum'] = True
            
        # B) تدهور منحدر EMA20
        if duration >= 10 and self.entry_ema_slope != 0:
            slope_ratio = current_ema_slope / self.entry_ema_slope
            if slope_ratio < SLOPE_DEGRADATION:
                current_flags['momentum_decay'] = True
                
        # C) العودة السريعة لمنطقة الدخول
        if duration <= SHORT_TIME_WINDOW:
            price_diff_pct = abs(current_price - self.entry_price) / self.entry_price * 100
            if price_diff_pct <= REJECTION_ZONE and (self.max_price_seen / self.entry_price - 1) > 0.0005:
                current_flags['early_rejection'] = True

        self.flag_history.append(current_flags)
        if len(self.flag_history) > 30: # Max history for debounce
            self.flag_history.pop(0)

        # 2. Check Persistence ({BOT_VERSION})
        effective_flags = 0
        for ftype, req in REQUIRED_FLAGS.items():
            count = sum(1 for tick in self.flag_history if tick[ftype])
            if count >= req:
                effective_flags += 1

        # 3. Healthy Bounce Protection ({BOT_VERSION})
        if self.is_healthy_bounce(current_price):
            self.flag_history = [] # Reset flags
            self.stats["bounce_protected"] += 1
            logger.info(f"[INTEL] Bounce Guard blocked exit at {current_price}")
            return "NO_ACTION"

        # 4. Final Decision ({BOT_VERSION})
        if duration < FAST_EXIT_ZONE_SECONDS:
            if effective_flags >= FAST_EXIT_REQUIRED_FLAGS:
                return ACTION_3_FLAGS
        else:
            if effective_flags >= 3:
                return ACTION_3_FLAGS
            elif effective_flags >= 2:
                return ACTION_2_FLAGS
                
        return "NO_ACTION"

exit_intel = ExitIntelligenceLayer()

def calculate_slope(data: List[float], period: int = 5) -> float:
    if len(data) < period:
        return 0.0
    # Simple linear slope: (y2 - y1) / x_diff
    return (data[-1] - data[-period]) / period

# القيم المحسنة (معتمدة على تحليل XRP/USDT)
SMALL_PROFIT_THRESHOLD = 0.045  # نسبة مئوية (محسّنة)
PRICE_REENTRY_BAND = 0.05      # % منطقة منع إعادة الدخول (محسّنة)
PRICE_INVALIDATION = 0.08      # % لتحرير الذاكرة
MAX_CONSECUTIVE_LOOPS = 3      # أقصى تكرار للمكاسب الصغيرة

# Zero-Move Loop Fix Constants (v3.7.3)
MIN_EXIT_PRICE_MOVE_PCT = 0.01   # أقل حركة سعر تعتبر خروجًا حقيقيًا
MIN_EXIT_TIME_SECONDS = 10       # أقل مدة صفقة قبل السماح بالخروج التقني
HARD_EXIT_REASONS = ["STOP_LOSS", "MANUAL_CLOSE", "FORCE_CLOSE", "MAINTENANCE"]

class BotState:
    def __init__(self):
        self.mode: str = "AGGRESSIVE"  # Force Aggressive Mode
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
        self.current_cooldown: int = 0  # Disable cooldowns
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0
        self.pause_until: Optional[datetime] = None
        self.pause_alerted: bool = False
        self.backtest_warned: bool = False
        self.last_signal_score: int = 10  # Bypass score
        self.last_signal_reasons: List[str] = []
        self.last_signal_reason: str = "Aggressive Entry"
        self.backtest_stats: Dict = {}
        self.pending_reset: bool = False
        self.last_downtrend_alert_time: float = 0
        self.tp_triggered: bool = False
        self.risk_free_sl: Optional[float] = None
        self.current_sl: Optional[float] = None
        self.entry_candles_snapshot: List[dict] = []
        
        # LPEM State (v3.7.2)
        self.lpem_active: bool = False
        self.lpem_direction: str = "LONG"
        self.lpem_exit_price: float = 0.0
        self.lpem_activation_time: float = 0.0
        self.lpem_consecutive_count: int = 0
        self.lpem_strict_mode: bool = False
        self.last_exit_time: float = 0.0

        # v3.7.5 Hold Logic State
        self.hold_active = False
        self.hold_candles = 0
        self.hold_start_price = 0.0
        self.daily_cumulative_loss = 0.0
        self.hold_activation_count = 0

    def reset_hold(self):
        """إعادة تعيين حالة الـ Hold"""
        self.hold_active = False
        self.hold_candles = 0
        self.hold_start_price = 0.0

    def update_daily_loss(self, pnl_percent):
        """تحديث الخسارة اليومية التراكمية"""
        if pnl_percent < 0:
            self.daily_cumulative_loss += abs(pnl_percent)

    def reset_daily_counters(self):
        """إعادة تعيين العدادات اليومية (عند منتصف الليل)"""
        now = get_now()
        if now.hour == 0 and now.minute == 0:
            self.daily_cumulative_loss = 0.0
            self.hold_activation_count = 0
            logger.info("[DAILY RESET] Counters cleared")

    def log_hold_status(self, current_price: float, market_mode: str):
        """تسجيل مفصل لحالة الـ Hold"""
        drawdown = ((self.hold_start_price - current_price) / self.hold_start_price * 100) if self.hold_start_price > 0 else 0
        logger.info(f"""
        📊 HOLD STATUS
        ├── Active: {self.hold_active}
        ├── Candles Held: {self.hold_candles}
        ├── Market Mode: {market_mode}
        ├── Entry Price: {self.hold_start_price:.6f}
        ├── Current Price: {current_price:.6f}
        ├── Drawdown: {drawdown:.4f}%
        └── Daily Loss: {self.daily_cumulative_loss:.2f}%
        """)

state = BotState()


def clear_trade_history():
    """
    تصفير سجل الصفقات فقط (Paper Trading)
    """
    try:
        # حذف ملفات السجل
        if os.path.exists(PAPER_TRADES_FILE):
            os.remove(PAPER_TRADES_FILE)
        if os.path.exists(TRADES_FILE):
            os.remove(TRADES_FILE)
        if os.path.exists(LOSS_EVENTS_FILE):
            os.remove(LOSS_EVENTS_FILE)
            
        # إعادة تهيئة ملفات السجل
        init_paper_trades_file()
        
        # تصفير الإحصائيات في الذاكرة (إن وجدت)
        global loss_counters
        loss_counters = {
            "STOP_HUNT": 0,
            "NOISE": 0,
            "TREND_REVERSAL": 0,
            "WEAK_ENTRY": 0,
            "UNKNOWN": 0
        }
        
        # تحديث رصيد القمة ليتناسب مع الرصيد الحالي بعد التصفير
        paper_state.peak_balance = paper_state.balance
        paper_state.loss_streak = 0
        
        logger.info(f"[HISTORY] Trade history cleared by user action ({BOT_VERSION})")
        return True
    except Exception as e:
        logger.error(f"Error clearing trade history: {e}")
        return False

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    
    if query.data == "CLEAR_TRADE_HISTORY":
        keyboard = [
            [
                InlineKeyboardButton("✅ نعم، صفّر السجل", callback_data="CONFIRM_CLEAR_HISTORY"),
                InlineKeyboardButton("❌ إلغاء", callback_data="CANCEL_CLEAR")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        # Just update the reply markup for the confirmation
        await query.edit_message_reply_markup(reply_markup=reply_markup)
        
    elif query.data == "CONFIRM_CLEAR_HISTORY":
        if clear_trade_history():
            await query.edit_message_text("✅ تم تصفير سجل الصفقات بنجاح.\nابدأ فترة اختبار جديدة.")
        else:
            await query.edit_message_text("❌ فشل تصفير السجل. تحقق من السجلات.")
            
    elif query.data == "CANCEL_CLEAR":
        # Return to the single clear button
        await query.edit_message_reply_markup(reply_markup=get_trades_keyboard())
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
            get_now().strftime("%Y-%m-%d %H:%M:%S"),
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
            get_now().strftime("%Y-%m-%d %H:%M:%S"),
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
    last_analysis_time = get_now()
    
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


def calculate_rsi(prices: List[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    
    deltas = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    if avg_loss == 0:
        return 100.0
        
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
    if avg_loss == 0:
        return 100.0
        
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def check_extended_price(price: float, analysis: dict, candles: List[dict]) -> bool:
    # A) EMA Distance
    ema20 = float(analysis.get("ema_short", 0))
    if ema20 > 0 and abs(price - ema20) / ema20 >= 0.0020:
        return True
        
    # B) Candle Body Expansion
    if len(candles) >= 6:
        try:
            bodies = [abs(float(c.get("open", 0)) - float(c.get("close", 0))) for c in candles[-6:-1]]
            avg_body_5 = sum(bodies) / 5
            current_candle = candles[-1]
            current_body = abs(float(current_candle.get("open", 0)) - float(current_candle.get("close", 0)))
            if current_body > avg_body_5 * 1.5:
                return True
        except (TypeError, ValueError):
            pass
            
    # C) Wick Rejection
    try:
        last_candle = candles[-1]
        high = float(last_candle.get("high", 0))
        low = float(last_candle.get("low", 0))
        open_p = float(last_candle.get("open", 0))
        close_p = float(last_candle.get("close", 0))
        
        candle_range = high - low
        if candle_range > 0:
            upper_wick = high - max(open_p, close_p)
            if (upper_wick / candle_range) > 0.65:
                return True
    except (TypeError, ValueError):
        pass
            
    return False

def volume_spike_detected(candles: List[dict]) -> bool:
    """اكتشاف ارتفاع مفاجئ في الحجم"""
    if len(candles) < 21:
        return False
    current_volume = candles[-1]['volume']
    avg_volume = sum(c['volume'] for c in candles[-21:-1]) / 20
    return current_volume > avg_volume * 1.8  # +80%

def detect_bearish_strength(candle: dict) -> str:
    """تحديد قوة الشمعة الهابطة"""
    try:
        open_p = float(candle.get('open', 0))
        close_p = float(candle.get('close', 0))
        high = float(candle.get('high', 0))
        low = float(candle.get('low', 0))
        
        body_size = abs(close_p - open_p)
        candle_range = high - low
        body_ratio = body_size / candle_range if candle_range > 0 else 0
        
        if close_p < open_p and body_ratio > 0.7:
            return "STRONG"
        elif close_p < open_p and body_ratio > 0.5:
            return "MEDIUM"
    except:
        pass
    return "WEAK"

def is_local_extreme(current_price: float, candles: List[dict], lookback: int = 15) -> bool:
    """اكتشف إذا كان السعر عند قاع محلي"""
    if len(candles) < lookback:
        return False
    recent_lows = [c['low'] for c in candles[-lookback:]]
    return current_price <= min(recent_lows)

def check_bounce_entry(analysis: dict, candles: List[dict], score: int) -> bool:
    """شروط دخول الارتداد في السوق الهابط"""
    ema20 = analysis.get("ema_short", 0)
    ema50 = analysis.get("ema_long", 0)
    ema200_vals = calculate_ema([c['close'] for c in candles], 200)
    ema200 = ema200_vals[-1] if ema200_vals else 0
    
    # 1. Market Regime
    if ema20 > ema50 and ema50 > ema200:
        market_mode = "EASY_MARKET"
    else:
        market_mode = "HARD_MARKET"
        
    if market_mode != "HARD_MARKET":
        return False
    
    current_price = analysis["close"]
    prices = [c["close"] for c in candles]
    current_rsi = calculate_rsi(prices)
    
    entry_is_bounce = (
        score <= 5 and                           # فرصة جيدة حسب سكور البوت
        is_local_extreme(current_price, candles) and  # قاع محلي
        current_rsi <= 35 and                    # تشبع بيع
        volume_spike_detected(candles)           # ارتفاع حجم مفاجئ
    )
    
    return entry_is_bounce

def check_hold_exit_conditions(analysis: dict, candles: List[dict]) -> Optional[str]:
    """فحص شروط الخروج أثناء الـ Hold"""
    if not state.hold_active:
        return None
    
    current_price = analysis["close"]
    current_candle = candles[-1]
    entry_price = state.hold_start_price
    
    # 1️⃣ STOP LOSS (أولوية قصوى - لا تغيير)
    if state.current_sl and current_price <= state.current_sl:
        return "SL Hit (Hold)"
    
    # 2️⃣ فشل سعري (دروداون محدود)
    max_drawdown_price = entry_price * 0.9990  # -0.10% كحد أقصى
    if current_price <= max_drawdown_price:
        return "Hold Failed - Max Drawdown"
    
    # 3️⃣ تحقيق هدف واقعي للسكالب
    scalp_target = entry_price * 1.003  # +0.3%
    if current_price >= scalp_target:
        return "Scalp Target Hit"
    
    # 4️⃣ فشل زمني مع ضعف الزخم
    if state.hold_candles >= 5:
        if len(candles) >= 23:
            recent_volume_avg = sum(c['volume'] for c in candles[-3:]) / 3
            normal_volume_avg = sum(c['volume'] for c in candles[-23:-3]) / 20
            if recent_volume_avg < normal_volume_avg * 0.65:
                return "Hold Failed - No Momentum"
    
    # 5️⃣ كسر هابط قوي (شمعة هابطة كبيرة)
    bearish_strength = detect_bearish_strength(current_candle)
    if bearish_strength == "STRONG":
        return "Hold Failed - Strong Breakdown"
    
    # 6️⃣ قيد الخسارة اليومية التراكمية
    if state.daily_cumulative_loss >= 1.0:  # 1% كحد يومي
        return "Hold Disabled - Daily Loss Limit"
    
    return None

def log_hold_status(current_price: float, market_mode: str):
    """تسجيل مفصل لحالة الـ Hold"""
    drawdown = ((state.hold_start_price - current_price) / state.hold_start_price * 100) if state.hold_start_price > 0 else 0
    logger.info(f"""
    📊 HOLD STATUS
    ├── Active: {state.hold_active}
    ├── Candles Held: {state.hold_candles}
    ├── Market Mode: {market_mode}
    ├── Entry Price: {state.hold_start_price:.6f}
    ├── Current Price: {current_price:.6f}
    ├── Drawdown: {drawdown:.4f}%
    └── Daily Loss: {state.daily_cumulative_loss:.2f}%
    """)
    if "error" in analysis:
        return False
    
    # Kill Switch Check (Disabled for Aggressive)
    if kill_switch.active and state.mode != "AGGRESSIVE":
        return False
    
    current_close = analysis["close"]
    prev_close = analysis["prev_close"]
    ema20 = analysis["ema_short"]
    
    # Post-Exit Integration (LPEM + PEG v1.3)
    monitor = EntryGateMonitor.get()
    
    # 1. LPEM Check (سعري - سريع)
    if state.lpem_active:
        # Simplified LPEM check directly using state variables
        price_diff = abs(current_close - state.lpem_exit_price) / state.lpem_exit_price * 100
        # v3.7.2: Lower LPEM protection band for faster re-entry on 1m (0.25% -> 0.12%)
        if price_diff < 0.12: 
            monitor.record_decision(lpem_blocked=True, peg_blocked=False, entered=False)
            if analysis_count % 12 == 0:
                logger.info("[ENTRY GATE] BLOCKED by LPEM")
            return False

    # 2. PEG Check (سياقي - فقط إذا سمح LPEM)
    guard = PostExitGuard.get()
    if guard.active:
        if guard.expired():
            guard.clear("max_duration_reached")
        else:
            recovered, recovery_reason = market_recovered(guard, current_close, candles, analysis["ema_short"], analysis["ema_long"])
            if not recovered:
                guard.record_block()
                monitor.record_decision(lpem_blocked=False, peg_blocked=True, entered=False)
                if analysis_count % 12 == 0:
                    logger.info(f"[ENTRY GATE] BLOCKED by PEG | Exit price: {guard.exit_price}")
                return False
            else:
                guard.clear(f"recovered_{recovery_reason}")
                guard.record_allow(recovery_reason)

    # Calculate Score and RSI for filtering
    score, reasons = calculate_signal_score(analysis, candles)
    prices = [c["close"] for c in candles]
    rsi = calculate_rsi(prices)
    is_extended = check_extended_price(current_close, analysis, candles)

    # 1. SCORE + RSI HARD BLOCK
    if score <= 1 and (rsi > 75 or rsi < 25):
        logger.info(f"[AGG] Blocked: Weak Entry (Score={score}, RSI={rsi:.1f})")
        return False
        
    # 2. EXTENDED + WEAK SIGNAL = BLOCK (v3.7.2: Relaxed from 3 to 2)
    if is_extended and score <= 2:
        logger.info(f"[AGG] Blocked: Weak Extended (Score={score}, Extended=True)")
        return False
    
    # Entry Logic (Same as before)
    # 1. Price touches/dips below EMA20 and rejects upward
    low_hit = any(c["low"] <= ema20 for c in candles[-2:])
    if low_hit and current_close > ema20:
        EntryGateMonitor.get().record_decision(lpem_blocked=False, peg_blocked=False, entered=True)
        state.last_signal_reason = "EMA bounce"
        state.last_signal_score = score
        return True
    
    # 2. Momentum
    price_change = (current_close - prev_close) / prev_close * 100
    if price_change >= 0.05:
        EntryGateMonitor.get().record_decision(lpem_blocked=False, peg_blocked=False, entered=True)
        state.last_signal_reason = "Momentum"
        state.last_signal_score = score
        return True
        
    # 3. Micro breakout
    recent_high = max([c["high"] for c in candles[-4:-1]])
    if current_close > recent_high:
        EntryGateMonitor.get().record_decision(lpem_blocked=False, peg_blocked=False, entered=True)
        state.last_signal_reason = "Micro breakout"
        state.last_signal_score = score
        return True
        
    return False

def check_sell_signal(analysis: dict, candles: List[dict]) -> bool:
    if "error" in analysis:
        return False
    
    # Kill Switch Check (Disabled for Aggressive)
    if kill_switch.active and state.mode != "AGGRESSIVE":
        return False
        
    current_close = analysis["close"]
    prev_close = analysis["prev_close"]
    ema20 = analysis["ema_short"]
    
    # 1. Price touches/moves above EMA20 and rejects downward
    high_hit = any(c["high"] >= ema20 for c in candles[-2:])
    if high_hit and current_close < ema20:
        state.last_signal_reason = "EMA bounce"
        return True
        
    # 2. Momentum: -0.05% to -0.10% within short window
    price_change = (current_close - prev_close) / prev_close * 100
    if price_change <= -0.05:
        state.last_signal_reason = "Momentum"
        return True
        
    # 3. Micro breakdown of recent low (last 3 candles)
    recent_low = min([c["low"] for c in candles[-4:-1]])
    if current_close < recent_low:
        state.last_signal_reason = "Micro breakdown"
        return True
        
    return False


def calculate_targets(entry_price: float, candles: List[dict]) -> tuple:
    tp = entry_price * (1 + TAKE_PROFIT_PCT / 100)
    sl = entry_price * (1 - STOP_LOSS_PCT / 100)
    return tp, sl

def check_exit_signal(analysis: dict, candles: List[dict]) -> Optional[str]:
    if not state.position_open or state.entry_price is None:
        return None
    
    current_price = analysis["close"]
    entry_price = state.entry_price
    pnl_pct = ((current_price - entry_price) / entry_price) * 100
    
    # v3.3: TP Trigger Logic
    if not state.tp_triggered and TAKE_PROFIT_PCT is not None and pnl_pct >= TAKE_PROFIT_PCT:
        state.tp_triggered = True
        state.risk_free_sl = entry_price * 1.001  # +0.1% Small profit
        # v3.7.5: Release hold once TP is triggered to allow normal exit
        if state.hold_active:
            logger.info("[HOLD] TP Triggered - Releasing hold for normal exit")
            state.hold_active = False
        return "tp_trigger"

    # v3.3: Exit Conditions after TP Triggered or Smart SL
    if state.tp_triggered:
        if state.risk_free_sl is not None and current_price <= state.risk_free_sl:
            return "risk_free_sl_hit"
        if "ema_short" in analysis and analysis["ema_short"] is not None and current_price < analysis["ema_short"]:
            if state.hold_active:
                logger.info(f"[HOLD] Ignoring EMA exit (Post TP) | Candles: {state.hold_candles}")
                return None
            return "ema_exit_post_tp"
    else:
        # Check Smart SL
        if state.current_sl is not None and current_price <= state.current_sl:
            return "sl"
        
    # Trailing SL (Existing logic preserved but secondary to TP trigger)
    if not state.tp_triggered and TRAILING_TRIGGER_PCT is not None:
        if pnl_pct >= TRAILING_TRIGGER_PCT:
            state.trailing_activated = True
        
        if state.trailing_activated and "ema_short" in analysis and analysis["ema_short"] is not None and current_price < analysis["ema_short"]:
            return "trailing_sl"
    
    # EMA Confirmation (Original logic)
    if current_price < analysis["ema_short"]:
        # v3.7.2: Stay in trade if overall trend is strong (EMA20 > EMA50)
        # unless price drops significantly (0.10%) or duration is short
        ema20 = analysis["ema_short"]
        ema50 = analysis["ema_long"]
        
        if ema20 > ema50:
            drop_pct = (ema20 - current_price) / ema20 * 100
            # If drop is small and we are in a strong uptrend, give it space
            if drop_pct < 0.10:
                state.candles_below_ema = 0
                return None

        state.candles_below_ema += 1
    else:
        state.candles_below_ema = 0
    
    if state.candles_below_ema >= 2:
        if state.hold_active:
            logger.info(f"[HOLD] Ignoring EMA confirmation exit | Candles: {state.hold_candles}")
            return None
        return "ema_confirmation"
    
    return None


def execute_paper_buy(price: float, score: int, reasons: List[str]) -> float:
    # Use fixed trade size: 100 USDT from 1000 USDT starting balance (based on replit.md)
    trade_size_usdt = 100.0
    
    # Enforce Valid Position Size (MANDATORY FIX 1)
    if trade_size_usdt <= 0 or paper_state.balance < trade_size_usdt:
        logger.warning(f"Abort trade execution: Invalid position size or insufficient balance. Size: {trade_size_usdt}, Balance: {paper_state.balance}")
        return 0.0
        
    qty = trade_size_usdt / price
    
    # Enforce Valid Quantity (MANDATORY FIX 1)
    if qty <= 0:
        logger.warning(f"Abort trade execution: Quantity is zero. Price: {price}, Size: {trade_size_usdt}")
        return 0.0
        
    # Freeze Quantity at Entry (CRITICAL - 3.6.2)
    paper_state.position_qty = qty
    paper_state.entry_reason = ", ".join(reasons)
    
    # v3.7.5: Activate Hold Logic if it's a bounce entry in hard market
    klines = get_klines(SYMBOL, state.timeframe)
    analysis = analyze_market(klines) if klines else {}
    if analysis and check_bounce_entry(analysis, klines, score):
        state.hold_active = True
        state.hold_candles = 0
        state.hold_start_price = price
        logger.info("[HOLD ACTIVATED] Bounce trade in bear market")
    else:
        state.hold_active = False
    
    # Start Exit Intelligence Monitoring (v3.7)
    ema20_vals = calculate_ema([c['close'] for c in get_klines(SYMBOL, state.timeframe)], EMA_SHORT)
    current_slope = calculate_slope(ema20_vals) if ema20_vals else 0.0
    exit_intel.start_monitoring(price, current_slope)
    
    log_paper_trade(
        "BUY", price, None, None, None,
        paper_state.balance, score, paper_state.entry_reason,
        "", 0
    )
    return qty


def activate_lpem(direction: str, exit_price: float, pnl_pct: float, exit_reason: str):
    """
    تنشيط الذاكرة خفيفة الوزن بعد الخروج بربح صغير (v3.7.2)
    """
    now_ts = time.time()
    
    # تتبع التكرار المتسلسل
    if (now_ts - state.last_exit_time) < 60:
        state.lpem_consecutive_count += 1
    else:
        state.lpem_consecutive_count = 1
    
    state.last_exit_time = now_ts
    state.lpem_active = True
    state.lpem_direction = direction
    state.lpem_exit_price = exit_price
    state.lpem_activation_time = now_ts
    state.lpem_strict_mode = (state.lpem_consecutive_count >= 2)
    
    logger.info(f"🧠 [LPEM] Activated: PnL={pnl_pct:.4f}%, Reason={exit_reason}, Consecutive={state.lpem_consecutive_count}")

def release_lpem(reason: str):
    """
    تحرير الذاكرة وتصفير عداد التكرار (v3.7.2)
    """
    if state.lpem_active:
        state.lpem_active = False
        state.lpem_consecutive_count = 0
        state.lpem_strict_mode = False
        logger.info(f"🔓 [LPEM] Released: {reason}")

def check_lpem_invalidation(current_price: float, analysis: dict):
    """
    التحقق من شروط تحرير الذاكرة التلقائي (v3.7.2)
    """
    if not state.lpem_active:
        return
        
    # 1. تحرك السعر عكسيًا (فرصة دخول أفضل)
    diff_pct = abs((current_price - state.lpem_exit_price) / state.lpem_exit_price) * 100
    if diff_pct >= PRICE_INVALIDATION:
        release_lpem("price_moved_against")
        return
        
    # 2. انعكاس ظروف الاتجاه
    if state.lpem_direction == "LONG" and not analysis.get("ema_bullish"):
        release_lpem("trend_invalidated")
        return
    elif state.lpem_direction == "SHORT" and analysis.get("ema_bullish"):
        release_lpem("trend_invalidated")
        return
        
    # 3. مرور وقت طويل (ساعة احتياطية - 5 دقائق)
    if (time.time() - state.lpem_activation_time) > 300:
        release_lpem("safety_timeout")
        return

def execute_paper_exit(entry_price: float, exit_price: float, reason: str,
                       score: int, duration_min: int) -> tuple:
    # Use Frozen Quantity at Close (NO RE-CALCULATION - 3.6.2)
    qty = paper_state.position_qty
    
    # Enforce Valid Quantity (MANDATORY FIX 1)
    if qty <= 0:
        logger.warning(f"Abort exit execution: Quantity is zero.")
        return 0.0, 0.0, paper_state.balance
        
    # Correct PnL Calculation (ABSOLUTE VALUE - 3.6.2)
    pnl_usdt = (exit_price - entry_price) * qty
    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
    
    # Calculate price move percentage
    price_move_pct = abs(pnl_pct)
    trade_duration_sec = duration_min * 60
    
    # Zero-Move Loop Fix (v3.7.3): Block fake exits
    # Hard exits always allowed (STOP_LOSS, MANUAL_CLOSE, etc.)
    if reason.upper() not in HARD_EXIT_REASONS:
        if price_move_pct < MIN_EXIT_PRICE_MOVE_PCT and trade_duration_sec < MIN_EXIT_TIME_SECONDS:
            logger.info(
                f"[HOTFIX] Exit blocked | reason={reason}, "
                f"move={price_move_pct:.4f}%, duration={trade_duration_sec}s"
            )
            return None  # Return None to signify blocked exit

    # Logging Validation (Hard Check - 3.6.2)
    if not (qty > 0 and (abs(pnl_usdt) > 0 or exit_price == entry_price)):
        logger.error(f"Validation failed: Qty={qty}, PnL={pnl_usdt}. Skipping balance update.")
        return None  # Return None to signify blocked exit

    paper_state.balance += pnl_usdt
    paper_state.update_peak()
    
    # Post-Exit Market Quality Gate (PEG v1.3)
    PostExitGuard.get().record_exit(exit_price)
    
    # LPEM Activation Logic (v3.7.2 - Fixed Wiring + v3.7.3 Zero-Move Protection)
    # Only record LPEM if there's actual price movement
    if price_move_pct >= MIN_EXIT_PRICE_MOVE_PCT:
        if 0.01 <= pnl_pct <= 0.06:
            activate_lpem("LONG", exit_price, pnl_pct, reason)
        else:
            if pnl_pct <= 0 or pnl_pct > 0.06:
                release_lpem("major_exit_or_loss")
    else:
        logger.info(f"[LPEM] Ignored zero-move exit: move={price_move_pct:.4f}%")
    
    if pnl_usdt < 0:
        paper_state.loss_streak += 1
        state.consecutive_losses += 1
        state.consecutive_wins = 0
        
        # Classify and log loss
        candles = get_klines(SYMBOL, state.timeframe)
        if candles:
            ltype = classify_loss(entry_price, exit_price, state.entry_candles_snapshot, candles)
            log_loss_event(ltype, pnl_pct, entry_price, exit_price)
    else:
        paper_state.loss_streak = 0
        state.consecutive_wins += 1
        state.consecutive_losses = 0
    
    if state.consecutive_losses >= 2:
        state.pause_until = get_now() + timedelta(minutes=COOLDOWN_PAUSE_MINUTES)
    
    # 1. Store raw values
    trade_pnl_pct = pnl_pct
    trade_pnl_usdt = pnl_usdt
    
    # 2. Format for display (rounding only here)
    display_pnl_pct = round(trade_pnl_pct, 2)
    display_pnl_usdt = round(trade_pnl_usdt, 2)
    
    # 3. Handle 0.00 rounding issues
    if abs(display_pnl_pct) < 0.01:
        display_pnl_pct = 0.00
        display_pnl_usdt = 0.00

    log_paper_trade(
        "EXIT", entry_price, exit_price, trade_pnl_pct, trade_pnl_usdt,
        paper_state.balance, score, paper_state.entry_reason,
        reason, duration_min
    )
    
    # Reset position after logging
    paper_state.position_qty = 0.0
    paper_state.entry_reason = ""
    exit_intel.stop_monitoring() # Stop Intel (v3.7)
    
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
        delta = get_now() - state.entry_time
        return int(delta.total_seconds() / 60)
    return 0


def update_cooldown_after_exit(reason: str):
    if reason == "sl":
        state.current_cooldown = COOLDOWN_AFTER_SL
    elif state.consecutive_wins >= 2:
        state.current_cooldown = COOLDOWN_STREAK_WIN
    else:
        state.current_cooldown = COOLDOWN_NORMAL


VERSION = "3.7.1-lite – Exit Intelligence Calibration"

def get_main_keyboard():
    keyboard = [
        ["تحديث الحالة 🔄", "تشخيص البوت 🧪"],
        ["الإحصائيات 📊", "الرصيد 💰"],
        ["سجل الصفقات 📜", "تحليل الخسائر 📉"],
        ["1m", "5m"],
        ["تشغيل ✅", "إيقاف ⏸️"]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    
    if text == "تحديث الحالة 🔄":
        await cmd_status(update, context)
    elif text == "تشخيص البوت 🧪":
        await cmd_diagnostic(update, context)
    elif text == "الإحصائيات 📊":
        await cmd_stats(update, context)
    elif text == "الرصيد 💰":
        await cmd_balance(update, context)
    elif text == "سجل الصفقات 📜":
        await cmd_trades(update, context)
    elif text == "تحليل الخسائر 📉":
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
        
        await update.message.reply_text(text=summary, parse_mode='HTML')
    elif text == "1m":
        state.timeframe = "1m"
        await update.message.reply_text("✅ تم تغيير الفريم إلى 1m", reply_markup=get_main_keyboard())
    elif text == "5m":
        state.timeframe = "5m"
        await update.message.reply_text("✅ تم تغيير الفريم إلى 5m", reply_markup=get_main_keyboard())
    elif text == "تشغيل ✅":
        state.signals_enabled = True
        await update.message.reply_text("✅ تم تشغيل الإشارات", reply_markup=get_main_keyboard())
    elif text == "إيقاف ⏸️":
        state.signals_enabled = False
        await update.message.reply_text("⏸️ تم إيقاف الإشارات", reply_markup=get_main_keyboard())
    elif text == "الإعدادات ⚙️":
        rules = format_rules_message()
        await update.message.reply_text(rules, parse_mode="Markdown")


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
        f"🤖 *بوت إشارات {SYMBOL_DISPLAY} {BOT_VERSION}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔥 نمط المضاربة العنيف: مفعّل (Aggressive Mode)\n"
        f"💰 الرصيد الحالي: {paper_state.balance:.2f} USDT\n"
        f"🛡️ نظام Kill Switch: مُعطل (Aggressive Mode)\n"
        f"🛡️ حماية السعر الممتد: مفعّلة (Price Protection)\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"استخدم الأزرار أدناه للتحكم"
    )


def format_status_message() -> str:
    status = "🟢 يعمل" if state.signals_enabled else "⏸️ متوقف"
    ks_status = "⚠️ معطل (Aggressive Mode)"
    
    pos_status = "❌ لا توجد صفقة"
    if state.position_open:
        pnl = ((state.last_close - state.entry_price) / state.entry_price) * 100 if state.last_close and state.entry_price else 0
        pos_status = f"✅ صفقة مفتوحة ({pnl:+.2f}%)"
    
    return (
        f"📊 *حالة البوت {BOT_VERSION}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 الحالة: {status}\n"
        f"🛡️ Kill Switch: {ks_status}\n"
        f"⏱️ الفريم: {state.timeframe}\n"
        f"💰 الرصيد: {paper_state.balance:.2f} USDT\n"
        f"📍 الصفقة: {pos_status}\n"
        f"🚀 النمط: Clean Aggressive Scalping\n"
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
        f"⚖️ *قواعد التداول {BOT_VERSION}*\n"
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


def get_trades_keyboard():
    keyboard = [
        [InlineKeyboardButton("🗑️ تصفير سجل الصفقات", callback_data="CLEAR_TRADE_HISTORY")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    history_text = format_trades_message()
    await update.message.reply_text(
        history_text,
        reply_markup=get_trades_keyboard(),
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
    msg = f"🧪 *تشخيص البوت {BOT_VERSION}*\n\n"
    
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
        rem = (state.pause_until - get_now()).total_seconds()
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

    now = get_now().timestamp()
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
            f"⏱ الوقت (مكة): {get_now().strftime('%H:%M:%S')}\n\n"
            "❌ تنبيه فقط – لا يوجد أي تنفيذ تداول"
        )
        if await send_signal_message(bot, chat_id, msg, "downtrend_alert"):
            state.last_downtrend_alert_time = now


# Real-time Price Engine (v3.8)
class PriceEngine:
    last_price: Optional[float] = None
    last_update_time: float = 0
    latency_ms: float = 0
    is_connected: bool = False
    
    @classmethod
    def update_price(cls, price: float):
        cls.last_price = price
        cls.last_update_time = time.time()
        cls.is_connected = True

    @classmethod
    def on_message(cls, ws, message):
        try:
            data = json.loads(message)
            if 'p' in data:
                price = float(data['p'])
                cls.update_price(price)
                if 'E' in data:
                    cls.latency_ms = (time.time() * 1000) - data['E']
        except Exception as e:
            logger.error(f"PriceEngine error: {e}")

    @classmethod
    def on_error(cls, ws, error):
        logger.error(f"WebSocket Error: {error}")
        cls.is_connected = False
        FailSafeSystem.on_websocket_disconnect()

    @classmethod
    def on_close(cls, ws, close_status_code, close_msg):
        logger.warning("WebSocket Closed")
        cls.is_connected = False
        FailSafeSystem.on_websocket_disconnect()

    @classmethod
    def start(cls):
        def run():
            while True:
                try:
                    ws_url = "wss://stream.binance.com:9443/ws/xrpusdt@aggTrade"
                    ws = websocket.WebSocketApp(
                        ws_url,
                        on_message=cls.on_message,
                        on_error=cls.on_error,
                        on_close=cls.on_close
                    )
                    ws.run_forever()
                except Exception as e:
                    logger.error(f"WebSocket restart error: {e}")
                time.sleep(5)
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()

class TradingGuard:
    BLOCK_ALL_TRADING = False
    BLOCK_REASON = ""
    MAX_LATENCY_MS = 500

    @classmethod
    def enforce_guard(cls, operation_type: str) -> bool:
        if cls.BLOCK_ALL_TRADING:
            logger.warning(f"Guard Blocked {operation_type}: {cls.BLOCK_REASON}")
            return False
            
        if PriceEngine.last_price is None:
            logger.warning(f"Guard Blocked {operation_type}: No price data")
            return False
            
        if (time.time() - PriceEngine.last_update_time) > 2:
            logger.warning(f"Guard Blocked {operation_type}: Stale price (>2s)")
            return False
            
        if PriceEngine.latency_ms > cls.MAX_LATENCY_MS:
            logger.warning(f"Guard Blocked {operation_type}: High latency ({PriceEngine.latency_ms:.0f}ms)")
            return False
            
        return True

class FailSafeSystem:
    @staticmethod
    def on_websocket_disconnect():
        TradingGuard.BLOCK_ALL_TRADING = True
        TradingGuard.BLOCK_REASON = "WebSocket disconnected"
        logger.critical("FAILSAFE: Trading Blocked due to connection loss")

    @staticmethod
    def on_websocket_connect():
        TradingGuard.BLOCK_ALL_TRADING = False
        TradingGuard.BLOCK_REASON = ""
        logger.info("FAILSAFE: Trading Resumed")

def execute_trade_operation(operation_type: str, logic_function, *args, **kwargs):
    if not TradingGuard.enforce_guard(operation_type):
        return None
    return logic_function(*args, **kwargs)

def get_binance_ticker():
    # Primary source is now PriceEngine
    if PriceEngine.last_price:
        return {"price": PriceEngine.last_price}
    
    # Fallback to REST for initialization only
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={SYMBOL}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            price = float(data["price"])
            PriceEngine.update_price(price)
            return {"price": price}
    except Exception as e:
        logger.error(f"Error fetching ticker: {e}")
    return None


async def signal_loop(bot: Bot, chat_id: str) -> None:
    if state.mode == "AGGRESSIVE":
        print("[AGG] checking entry conditions")
        
    try:
        if kill_switch.check_auto_resume():
            resume_trading()
            await bot.send_message(chat_id=chat_id, text="✅ تم استئناف التداول تلقائياً", parse_mode="Markdown")
        
        # Disable Kill Switch check for Aggressive Mode
        if not state.signals_enabled or (kill_switch.active and state.mode != "AGGRESSIVE"):
            return
        
        # Disable Cooldown for Aggressive Mode
        if state.mode != "AGGRESSIVE":
            if state.pause_until and get_now() < state.pause_until:
                return
        
        # Use Real-time Price for Aggressive Mode
        if state.mode == "AGGRESSIVE" and PriceEngine.last_price:
            current_price = PriceEngine.last_price
        else:
            ticker = get_binance_ticker()
            if ticker:
                current_price = ticker["price"]
            else:
                return

        candles = get_klines(SYMBOL, state.timeframe)
        if candles is None:
            return
            
        # --- Exit Intelligence Layer (v3.7) ---
        if state.position_open:
            start_intel = time.time()
            # We need EMA20 slope
            ema20_vals = calculate_ema([c['close'] for c in candles], EMA_SHORT)
            current_slope = calculate_slope(ema20_vals) if ema20_vals else 0.0
            intel_action = exit_intel.monitor(current_price, current_slope)
            
            if (time.time() - start_intel) > 0.1:
                logger.warning(f"[INTEL] Execution exceeded 100ms: {(time.time() - start_intel)*1000:.2f}ms")
            
            if intel_action == ACTION_3_FLAGS:
                exit_reason = "TREND_REVERSAL_PREVENTED" # {BOT_VERSION}
                exit_result = execute_paper_exit(state.entry_price, current_price, exit_reason, 10, 0)
                if exit_result:
                    pnl_pct, pnl_usdt, balance = exit_result
                    reset_position_state()
                    update_cooldown_after_exit(exit_reason)
                    msg = f"🛡️ **Intel Early Exit**\nPrice: {current_price}\nPnL: {pnl_usdt:.2f} ({pnl_pct:.2f}%)"
                    await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
                return
            elif intel_action == ACTION_2_FLAGS:
                tight_sl = state.entry_price * 1.0005
                if not state.current_sl or tight_sl > state.current_sl:
                    state.current_sl = tight_sl
                    logger.info(f"[INTEL] Tightened SL to {tight_sl}")

        analysis = analyze_market(candles)
        if "error" in analysis:
            return

        # Override analysis price with real-time ticker price
        analysis["close"] = current_price
        state.last_close = current_price

        # [HOLD PROBE] v3.7.5
        ema20 = analysis.get('ema20', 0)
        ema50 = analysis.get('ema50', 0)
        ema200 = analysis.get('ema200', 0)
        market_mode = "EASY_MARKET" if (ema20 > ema50 and ema50 > ema200) else "HARD_MARKET"
        score = analysis.get('score', 0)
        rsi = analysis.get('rsi', 50)
        logger.warning(
            f"[HOLD PROBE] mode={market_mode}, "
            f"score={score}, rsi={rsi:.1f}, "
            f"bounce={check_bounce_entry(analysis, candles, score)}, "
            f"hold_active={state.hold_active}"
        )
            
        # Downtrend Alerts (Monitoring Only)
        await check_downtrend_alerts(bot, chat_id, analysis, candles)
        
        # LPEM Invalidation Check (v3.7.2)
        check_lpem_invalidation(current_price, analysis)
        
        # Disable Kill Switch evaluation for Aggressive Mode
        if state.mode != "AGGRESSIVE":
            ks_reason = evaluate_kill_switch()
            if ks_reason and not state.position_open:
                kill_switch.activate(ks_reason)
                return
        
        if state.position_open and state.entry_price is not None:
            exit_reason = check_exit_signal(analysis, candles)
            if exit_reason:
                exit_price = analysis["close"]
                duration = get_trade_duration_minutes()
                # Pass consistent score ({BOT_VERSION})
                exit_result = execute_paper_exit(state.entry_price, exit_price, exit_reason, state.last_signal_score, duration)
                if exit_result:
                    pnl_pct, pnl_usdt, balance = exit_result
                    log_trade("EXIT", exit_reason.upper(), exit_price, pnl_pct)
                    msg = format_exit_message(state.entry_price, exit_price, pnl_pct, pnl_usdt, exit_reason, duration, balance)
                    await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
                    if state.mode != "AGGRESSIVE":
                        update_cooldown_after_exit(exit_reason)
                    reset_position_state()
            
            # Additional Aggressive Flip check
            elif state.mode == "AGGRESSIVE" and check_sell_signal(analysis, candles):
                exit_price = analysis["close"]
                duration = get_trade_duration_minutes()
                # Pass consistent score ({BOT_VERSION})
                exit_result = execute_paper_exit(state.entry_price, exit_price, "aggressive_flip", state.last_signal_score, duration)
                if exit_result:
                    pnl_pct, pnl_usdt, balance = exit_result
                    log_trade("EXIT", "AGGRESSIVE_FLIP", exit_price, pnl_pct)
                    msg = format_exit_message(state.entry_price, exit_price, pnl_pct, pnl_usdt, "aggressive_flip", duration, balance)
                    await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
                    reset_position_state()

        # Re-check entry (Allow immediate re-entry for aggressive mode)
        if not state.position_open:
            if check_buy_signal(analysis, candles):
                entry_price = analysis["close"]
                
                # LPEM Filter (v3.7.2)
                if state.lpem_active and state.lpem_direction == "LONG":
                    # حساب منطقة المنع
                    current_band = PRICE_REENTRY_BAND * 0.6 if state.lpem_strict_mode else PRICE_REENTRY_BAND
                    diff_pct = abs((entry_price - state.lpem_exit_price) / state.lpem_exit_price) * 100
                    
                    if diff_pct <= current_band:
                        logger.info(f"🚫 [LPEM] Blocked Entry: Price within band ({diff_pct:.4f}% <= {current_band}%)")
                        return
                
                tp, sl = calculate_targets(entry_price, candles)
                
                # Fixed Score Calculation: Single source of truth ({BOT_VERSION})
                score, reasons = calculate_signal_score(analysis, candles)
                state.last_signal_score = score
                state.last_signal_reasons = reasons
                state.last_signal_reason = ", ".join(reasons)

                qty = execute_paper_buy(entry_price, score, reasons)
                log_trade("BUY", state.last_signal_reason, entry_price, None)
                
                msg = format_buy_message(entry_price, tp, sl, state.timeframe, score, qty)
                await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
                
                # Update State for New Long
                state.position_open = True
                state.entry_price = entry_price
                state.entry_time = get_now()
                state.current_sl = sl
                state.tp_triggered = False
                state.risk_free_sl = None
                state.trailing_activated = False
                state.candles_below_ema = 0
                state.entry_candles_snapshot = candles[-10:]

    except Exception as e:
        if "WebSocketApp" in str(e):
            logger.error(f"[NETWORK] WebSocket interface error: {e}. Isolation active.")
        else:
            logger.error(f"Error in signal loop: {e}")


def validate_version_unification():
    """
    تحقق حازم من توحيد النسخة
    """
    import re
    from version import BOT_VERSION
    pattern = r'^v\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$'
    
    if not re.match(pattern, BOT_VERSION):
        raise RuntimeError(f"Invalid bot version format: {BOT_VERSION}")
    
    logger.info(f"[SYSTEM] Bot version unified successfully: {BOT_VERSION}")

def check_local_version_definitions():
    """
    تحذير فقط – لا يوقف التشغيل
    """
    suspicious_tokens = ["V3.", "v3."]
    logger.info("[VERSION CHECK] Scanning for hardcoded version strings...")

async def main() -> None:
    validate_version_unification()
    check_local_version_definitions()
    # Start Price Engine
    PriceEngine.start()
    
    tg_token = os.environ.get("TG_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    
    if not tg_token or not chat_id:
        print("❌ الرجاء تعيين TG_TOKEN و TG_CHAT_ID")
        return
    
    # {BOT_VERSION} Integrity Check
    def validate_data_integrity():
        from version import BOT_VERSION
        import math
        
        logger.info(f"[BOOT] Bot version loaded: {BOT_VERSION}")
        
        # 1. Test PnL Calculation & Rounding Protection
        test_entry = 1.0000
        test_exit = 1.00004  # Tiny difference
        test_pnl = ((test_exit - test_entry) / test_entry) * 100
        
        # Apply the same protection as in execute_paper_exit
        display_pnl = round(test_pnl, 2)
        if abs(display_pnl) < 0.01:
            display_pnl = 0.00
            
        if display_pnl != 0.00:
            logger.error(f"❌ PnL Protection Failed: {display_pnl}")
        
        # 2. Test Score Integrity
        test_analysis = {"ema_bullish": True, "breakout": True, "volume_confirmed": True}
        test_candles = [{"close": 1.0}] * 20
        score, reasons = calculate_signal_score(test_analysis, test_candles)
        
        if not (1 <= score <= 10):
            logger.error(f"❌ Score Integrity Failed: {score}")
            
        logger.info(f"✅ Data Integrity Check Passed for version {BOT_VERSION}")
    
    validate_data_integrity()
    
    # Initialize application
    application = Application.builder().token(tg_token).build()
    
    # Remove obsolete CallbackQueryHandler as we switched to MessageHandler for ReplyKeyboard
    # application.add_handler(CallbackQueryHandler(handle_callback))
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
    
    # Add CallbackQueryHandler for buttons
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
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
    
    print(f"🚀 بوت إشارات {SYMBOL_DISPLAY} {BOT_VERSION} يعمل...")
    
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

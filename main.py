#!/usr/bin/env python3
"""
XRP/USDT Telegram Signals Bot V2
بوت إشارات تداول يرسل إشارات دخول/خروج لزوج XRP/USDT
باستخدام استراتيجية EMA20/EMA50 مع تأكيد الاختراق
V2: فلاتر متقدمة + Trailing Stop + تأكيد EMA + Cooldown تكيفي
"""

import os
import time
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, List

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

COOLDOWN_NORMAL = 60
COOLDOWN_AFTER_SL = 180

POLL_INTERVAL = 10
KLINE_LIMIT = 200

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

state = BotState()

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
# STRATEGY LOGIC V2
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

def check_buy_signal(analysis: dict) -> bool:
    """فحص شروط الدخول مع الفلاتر الجديدة"""
    if "error" in analysis:
        return False
    
    if not analysis["ema_bullish"]:
        return False
    if not analysis["breakout"]:
        return False
    
    if not analysis["range_confirmed"]:
        logger.debug("فشل فلتر التذبذب - EMA متقاربة جداً")
        return False
    
    if not analysis["volume_confirmed"]:
        logger.debug("فشل فلتر الحجم - الحجم أقل من المتوسط")
        return False
    
    return True

def check_exit_signal(analysis: dict) -> Optional[str]:
    """فحص شروط الخروج مع Smart Exit"""
    if "error" in analysis or not state.position_open or state.entry_price is None:
        return None
    
    current_close = analysis["close"]
    prev_close = analysis["prev_close"]
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
    """حساب مدة الصفقة بالدقائق"""
    if state.entry_time is None:
        return 0
    now = datetime.now(timezone.utc)
    duration = now - state.entry_time
    return int(duration.total_seconds() / 60)

def reset_position_state():
    """إعادة تعيين حالة المركز"""
    state.position_open = False
    state.entry_price = None
    state.entry_time = None
    state.entry_timeframe = None
    state.trailing_activated = False
    state.candles_below_ema = 0

def update_cooldown_after_exit(exit_type: str):
    """تحديث Cooldown بناءً على نوع الخروج"""
    state.last_exit_type = exit_type
    if exit_type == "sl":
        state.current_cooldown = COOLDOWN_AFTER_SL
        logger.info(f"Cooldown بعد SL: {COOLDOWN_AFTER_SL} ثانية")
    else:
        state.current_cooldown = COOLDOWN_NORMAL
        logger.info(f"Cooldown عادي: {COOLDOWN_NORMAL} ثانية")

# ============================================================================
# MESSAGE FORMATTING (Arabic) V2
# ============================================================================

def get_current_time_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def format_buy_message(entry: float, tp: float, sl: float, timeframe: str) -> str:
    return (
        f"🟢 *إشارة شراء*\n\n"
        f"📈 *الزوج:* {SYMBOL_DISPLAY}\n"
        f"📊 *الإطار الزمني:* {timeframe}\n"
        f"💰 *سعر الدخول:* {entry:.4f}\n"
        f"🎯 *جني الأرباح:* {tp:.4f} (+{TAKE_PROFIT_PCT}%)\n"
        f"🛑 *وقف الخسارة:* {sl:.4f} (-{STOP_LOSS_PCT}%)\n\n"
        f"📝 *السبب:* EMA{EMA_SHORT} > EMA{EMA_LONG} + اختراق أعلى قمة\n"
        f"🔥 *تقييم الإشارة:* قوية\n"
        f"📋 *ملاحظة:* تم تفعيل فلتر التذبذب والحجم\n"
        f"🕐 *الوقت:* {get_current_time_str()}"
    )

def format_exit_message(entry: float, exit_price: float, pnl: float, reason: str, duration_min: int) -> str:
    reason_text = {
        "tp": "وصول الهدف (TP)",
        "sl": "وصول وقف الخسارة (SL)",
        "trailing_sl": "Trailing Stop Loss",
        "ema_confirmation": f"تأكيد الإغلاق تحت EMA{EMA_SHORT} (شمعتين)",
    }.get(reason, "خروج يدوي")
    
    pnl_sign = "+" if pnl >= 0 else ""
    status_emoji = "✅" if pnl >= 0 else "❌"
    
    return (
        f"🔴 *إشارة خروج*\n\n"
        f"📈 *الزوج:* {SYMBOL_DISPLAY}\n"
        f"💰 *سعر الدخول:* {entry:.4f}\n"
        f"💵 *سعر الخروج:* {exit_price:.4f}\n"
        f"📊 *الربح/الخسارة:* {pnl_sign}{pnl:.2f}%\n\n"
        f"{status_emoji} *نوع الخروج:* {reason_text}\n"
        f"⏱️ *مدة الصفقة:* {duration_min} دقيقة\n"
        f"🕐 *الوقت:* {get_current_time_str()}"
    )

def format_status_message() -> str:
    status = "✅ نشط" if state.signals_enabled else "⏸️ متوقف"
    position = "📈 مفتوح" if state.position_open else "📉 مغلق"
    
    msg = (
        f"ℹ️ *حالة البوت*\n\n"
        f"🔔 *الإشارات:* {status}\n"
        f"📊 *الإطار الزمني:* {state.timeframe}\n"
        f"📈 *المركز:* {position}\n"
    )
    
    if state.position_open and state.entry_price:
        msg += f"💰 *سعر الدخول:* {state.entry_price:.4f}\n"
        if state.trailing_activated:
            msg += f"🔒 *Trailing Stop:* مفعّل (Break-even)\n"
        if state.last_close:
            pnl = calculate_pnl(state.entry_price, state.last_close)
            pnl_sign = "+" if pnl >= 0 else ""
            msg += f"📉 *الربح الحالي:* {pnl_sign}{pnl:.2f}%\n"
        duration = get_trade_duration_minutes()
        msg += f"⏱️ *مدة الصفقة:* {duration} دقيقة\n"
    
    if state.last_close:
        msg += f"🕯️ *آخر إغلاق:* {state.last_close:.4f}\n"
    
    msg += f"⏳ *Cooldown الحالي:* {state.current_cooldown} ثانية\n"
    msg += f"🕐 *التحديث:* {get_current_time_str()}"
    
    return msg

def format_welcome_message() -> str:
    return (
        f"🤖 *مرحباً بك في بوت إشارات {SYMBOL_DISPLAY} V2*\n\n"
        f"📊 *الاستراتيجية:* EMA{EMA_SHORT}/EMA{EMA_LONG} + Breakout\n"
        f"🎯 *الهدف:* +{TAKE_PROFIT_PCT}%\n"
        f"🛑 *وقف الخسارة:* -{STOP_LOSS_PCT}%\n"
        f"🔒 *Trailing:* +{TRAILING_TRIGGER_PCT}% → Break-even\n\n"
        f"✨ *ميزات V2:*\n"
        f"• فلتر التذبذب والحجم\n"
        f"• Trailing Stop ذكي\n"
        f"• تأكيد كسر EMA20 (شمعتين)\n"
        f"• Cooldown تكيفي\n\n"
        f"استخدم الأزرار أدناه للتحكم في البوت:\n"
    )

def format_rules_message() -> str:
    return (
        f"📜 *قواعد التداول V2*\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*شروط الدخول (BUY):*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"1️⃣ EMA{EMA_SHORT} > EMA{EMA_LONG}\n"
        f"2️⃣ اختراق أعلى قمة آخر {BREAKOUT_CANDLES} شموع\n"
        f"3️⃣ فرق EMA > {RANGE_FILTER_THRESHOLD * 100:.1f}% (فلتر التذبذب)\n"
        f"4️⃣ حجم الشمعة > متوسط آخر {VOLUME_LOOKBACK} شمعة\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*شروط الخروج (EXIT):*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ *TP:* +{TAKE_PROFIT_PCT}%\n"
        f"❌ *SL:* -{STOP_LOSS_PCT}%\n"
        f"🔒 *Trailing SL:* عند +{TRAILING_TRIGGER_PCT}% يتحرك SL إلى الدخول\n"
        f"📊 *EMA{EMA_SHORT}:* إغلاق شمعتين متتاليتين تحته\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*Cooldown:*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"• بعد TP/Trailing/EMA: {COOLDOWN_NORMAL} ثانية\n"
        f"• بعد SL: {COOLDOWN_AFTER_SL} ثانية\n"
    )

# ============================================================================
# INLINE KEYBOARD
# ============================================================================

def get_main_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("🔥 تشغيل الإشارات", callback_data="on"),
            InlineKeyboardButton("🛑 إيقاف الإشارات", callback_data="off"),
        ],
        [
            InlineKeyboardButton("📊 الحالة", callback_data="status"),
            InlineKeyboardButton("🔄 تحديث الآن", callback_data="force_check"),
        ],
        [
            InlineKeyboardButton("⏱ 1 دقيقة", callback_data="tf_1m"),
            InlineKeyboardButton("⏱ 5 دقائق", callback_data="tf_5m"),
        ],
        [
            InlineKeyboardButton("📜 القواعد", callback_data="rules"),
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
        logger.info(f"تخطي الرسالة بسبب cooldown ({signal_type})")
        return False
    
    if state.last_signal_type == signal_type and signal_type == "buy" and state.position_open:
        logger.info("تخطي رسالة شراء مكررة")
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

async def cmd_rules(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        format_rules_message(),
        reply_markup=get_main_keyboard(),
        parse_mode="Markdown"
    )

async def cmd_settf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("❌ استخدم: /settf 1m أو /settf 5m")
        return
    
    new_tf = context.args[0].lower()
    if new_tf not in ["1m", "5m"]:
        await update.message.reply_text("❌ الإطار الزمني غير صحيح. استخدم 1m أو 5m")
        return
    
    state.timeframe = new_tf
    await update.message.reply_text(
        f"✅ تم تغيير الإطار الزمني إلى {new_tf}",
        reply_markup=get_main_keyboard()
    )

async def cmd_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state.signals_enabled = True
    await update.message.reply_text(
        "✅ تم تفعيل إرسال الإشارات",
        reply_markup=get_main_keyboard()
    )

async def cmd_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state.signals_enabled = False
    await update.message.reply_text(
        "⏸️ تم إيقاف إرسال الإشارات مؤقتاً",
        reply_markup=get_main_keyboard()
    )

# ============================================================================
# CALLBACK QUERY HANDLER (Inline Buttons)
# ============================================================================

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data == "on":
        state.signals_enabled = True
        await query.edit_message_text(
            "✅ تم تفعيل إرسال الإشارات\n\n" + format_status_message(),
            reply_markup=get_main_keyboard(),
            parse_mode="Markdown"
        )
    
    elif data == "off":
        state.signals_enabled = False
        await query.edit_message_text(
            "⏸️ تم إيقاف إرسال الإشارات\n\n" + format_status_message(),
            reply_markup=get_main_keyboard(),
            parse_mode="Markdown"
        )
    
    elif data == "status":
        await query.edit_message_text(
            format_status_message(),
            reply_markup=get_main_keyboard(),
            parse_mode="Markdown"
        )
    
    elif data == "rules":
        await query.edit_message_text(
            format_rules_message(),
            reply_markup=get_main_keyboard(),
            parse_mode="Markdown"
        )
    
    elif data == "tf_1m":
        state.timeframe = "1m"
        await query.edit_message_text(
            f"✅ تم تغيير الإطار الزمني إلى 1m\n\n" + format_status_message(),
            reply_markup=get_main_keyboard(),
            parse_mode="Markdown"
        )
    
    elif data == "tf_5m":
        state.timeframe = "5m"
        await query.edit_message_text(
            f"✅ تم تغيير الإطار الزمني إلى 5m\n\n" + format_status_message(),
            reply_markup=get_main_keyboard(),
            parse_mode="Markdown"
        )
    
    elif data == "force_check":
        await query.edit_message_text(
            "🔄 جاري التحقق من السوق...",
            parse_mode="Markdown"
        )
        
        candles = get_klines(SYMBOL, state.timeframe)
        if candles:
            analysis = analyze_market(candles)
            
            result_msg = f"🔄 *نتيجة التحقق*\n\n"
            
            if "error" not in analysis:
                result_msg += (
                    f"💰 *السعر الحالي:* {analysis['close']:.4f}\n"
                    f"📊 *EMA{EMA_SHORT}:* {analysis['ema_short']:.4f}\n"
                    f"📊 *EMA{EMA_LONG}:* {analysis['ema_long']:.4f}\n"
                    f"📈 *أعلى قمة ({BREAKOUT_CANDLES} شموع):* {analysis['highest_high']:.4f}\n"
                    f"📊 *فرق EMA:* {analysis['ema_diff_pct']*100:.3f}%\n"
                    f"📊 *الحجم الحالي:* {analysis['current_volume']:.0f}\n"
                    f"📊 *متوسط الحجم:* {analysis['avg_volume']:.0f}\n\n"
                )
                
                if analysis["ema_bullish"]:
                    result_msg += f"✅ EMA{EMA_SHORT} > EMA{EMA_LONG}\n"
                else:
                    result_msg += f"❌ EMA{EMA_SHORT} < EMA{EMA_LONG}\n"
                
                if analysis["breakout"]:
                    result_msg += f"✅ اختراق صاعد\n"
                else:
                    result_msg += f"❌ لا يوجد اختراق\n"
                
                if analysis["range_confirmed"]:
                    result_msg += f"✅ فلتر التذبذب (>{RANGE_FILTER_THRESHOLD*100:.1f}%)\n"
                else:
                    result_msg += f"❌ فلتر التذبذب (<{RANGE_FILTER_THRESHOLD*100:.1f}%)\n"
                
                if analysis["volume_confirmed"]:
                    result_msg += f"✅ فلتر الحجم\n"
                else:
                    result_msg += f"❌ فلتر الحجم\n"
            else:
                result_msg += f"❌ {analysis['error']}\n"
            
            result_msg += f"\n" + format_status_message()
        else:
            result_msg = "❌ فشل في جلب البيانات\n\n" + format_status_message()
        
        await query.edit_message_text(
            result_msg,
            reply_markup=get_main_keyboard(),
            parse_mode="Markdown"
        )

# ============================================================================
# BACKGROUND SIGNAL LOOP (using asyncio)
# ============================================================================

async def signal_loop(bot: Bot, chat_id: str) -> None:
    """حلقة فحص الإشارات في الخلفية"""
    logger.info(f"بدء حلقة الإشارات V2 - التحديث كل {POLL_INTERVAL} ثانية")
    
    while True:
        try:
            if not state.signals_enabled:
                await asyncio.sleep(POLL_INTERVAL)
                continue
            
            candles = get_klines(SYMBOL, state.timeframe)
            
            if candles is None:
                state.consecutive_errors += 1
                logger.warning(f"فشل جلب البيانات (الأخطاء: {state.consecutive_errors})")
                
                if state.consecutive_errors >= 5 and not state.error_alerted:
                    try:
                        await bot.send_message(
                            chat_id=chat_id,
                            text="⚠️ مشكلة في الاتصال بـBinance API",
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
                logger.warning(f"خطأ في التحليل: {analysis['error']}")
                await asyncio.sleep(POLL_INTERVAL)
                continue
            
            if state.position_open:
                exit_reason = check_exit_signal(analysis)
                if exit_reason:
                    exit_price = analysis["close"]
                    pnl = calculate_pnl(state.entry_price, exit_price)
                    duration = get_trade_duration_minutes()
                    
                    msg = format_exit_message(state.entry_price, exit_price, pnl, exit_reason, duration)
                    sent = await send_signal_message(bot, chat_id, msg, "exit")
                    
                    if sent:
                        update_cooldown_after_exit(exit_reason)
                        reset_position_state()
                        logger.info(f"تم إغلاق المركز: {exit_reason} @ {exit_price:.4f} (PnL: {pnl:.2f}%)")
            
            else:
                if check_buy_signal(analysis):
                    entry_price = analysis["close"]
                    tp, sl = calculate_targets(entry_price)
                    
                    msg = format_buy_message(entry_price, tp, sl, state.timeframe)
                    sent = await send_signal_message(bot, chat_id, msg, "buy")
                    
                    if sent:
                        state.position_open = True
                        state.entry_price = entry_price
                        state.entry_time = datetime.now(timezone.utc)
                        state.entry_timeframe = state.timeframe
                        state.trailing_activated = False
                        state.candles_below_ema = 0
                        logger.info(f"تم فتح مركز @ {entry_price:.4f}")
        
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
        logger.error("TG_TOKEN environment variable not set!")
        print("❌ الرجاء تعيين TG_TOKEN في Replit Secrets")
        return
    
    if not chat_id:
        logger.error("TG_CHAT_ID environment variable not set!")
        print("❌ الرجاء تعيين TG_CHAT_ID في Replit Secrets")
        return
    
    logger.info(f"بدء بوت إشارات {SYMBOL_DISPLAY} V2 على الفريم {state.timeframe}")
    
    application = Application.builder().token(tg_token).build()
    
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("rules", cmd_rules))
    application.add_handler(CommandHandler("settf", cmd_settf))
    application.add_handler(CommandHandler("on", cmd_on))
    application.add_handler(CommandHandler("off", cmd_off))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    bot = application.bot
    
    await application.initialize()
    await application.start()
    await application.updater.start_polling(drop_pending_updates=True)
    
    print("=" * 50)
    print(f"🚀 بوت إشارات {SYMBOL_DISPLAY} V2")
    print(f"📊 الإطار الزمني: {state.timeframe}")
    print(f"📈 الاستراتيجية: EMA{EMA_SHORT}/EMA{EMA_LONG} + Breakout + Filters")
    print(f"🎯 TP: +{TAKE_PROFIT_PCT}% | SL: -{STOP_LOSS_PCT}%")
    print(f"🔒 Trailing: +{TRAILING_TRIGGER_PCT}% → Break-even")
    print(f"⏱️ Polling: كل {POLL_INTERVAL} ثواني")
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

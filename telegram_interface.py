"""
Telegram Interface - AI Control Commands
Provides buttons and handlers for mode, weight, and status.
"""
from typing import Callable, Dict, Any
import logging

from ai_state import AIMode, AIWeight

logger = logging.getLogger(__name__)


class TelegramAIInterface:
    
    def __init__(self, get_engine_fn: Callable):
        self.get_engine = get_engine_fn
    
    def handle_ai_mode(self, mode_str: str) -> str:
        try:
            mode_map = {
                "OFF": AIMode.OFF,
                "LEARN": AIMode.LEARN,
                "FULL": AIMode.FULL
            }
            
            mode = mode_map.get(mode_str.upper())
            if mode is None:
                return f"❌ وضع غير صالح: {mode_str}\nالأوضاع المتاحة: OFF, LEARN, FULL"
            
            engine = self.get_engine()
            engine.set_mode(mode)
            
            return f"✅ تم تغيير وضع الذكاء إلى: {mode.value}"
            
        except Exception as e:
            logger.error(f"[TG] Error setting mode: {e}")
            return f"❌ خطأ: {e}"
    
    def handle_ai_weight(self, weight_str: str) -> str:
        try:
            weight_map = {
                "OFF": AIWeight.OFF,
                "0": AIWeight.OFF,
                "0.0": AIWeight.OFF,
                "LOW": AIWeight.LOW,
                "0.3": AIWeight.LOW,
                "MEDIUM": AIWeight.MEDIUM,
                "0.6": AIWeight.MEDIUM,
                "HIGH": AIWeight.HIGH,
                "1": AIWeight.HIGH,
                "1.0": AIWeight.HIGH
            }
            
            weight = weight_map.get(weight_str.upper())
            if weight is None:
                return f"❌ وزن غير صالح: {weight_str}\nالأوزان المتاحة: OFF(0), LOW(0.3), MEDIUM(0.6), HIGH(1.0)"
            
            engine = self.get_engine()
            engine.set_weight(weight)
            
            return f"✅ تم تغيير وزن الذكاء إلى: {weight.name} ({weight.value})"
            
        except Exception as e:
            logger.error(f"[TG] Error setting weight: {e}")
            return f"❌ خطأ: {e}"
    
    def handle_ai_limit(self, limit_str: str) -> str:
        try:
            limit = int(limit_str)
            if limit < 1:
                return "❌ الحد يجب أن يكون 1 على الأقل"
            
            engine = self.get_engine()
            engine.set_daily_limit(limit)
            
            return f"✅ تم تغيير السقف اليومي إلى: {limit}"
            
        except ValueError:
            return f"❌ قيمة غير صالحة: {limit_str}"
        except Exception as e:
            logger.error(f"[TG] Error setting limit: {e}")
            return f"❌ خطأ: {e}"
    
    def handle_ai_status(self) -> str:
        try:
            engine = self.get_engine()
            status = engine.get_status()
            
            mode_emoji = {
                "OFF": "⚫",
                "LEARN": "🔵",
                "FULL": "🟢"
            }
            
            msg = [
                "📊 حالة نظام الذكاء الاصطناعي",
                "═" * 30,
                f"{mode_emoji.get(status['mode'], '⚪')} الوضع: {status['mode']}",
                f"⚖️ الوزن: {status['weight']}",
                f"📈 التدخلات اليومية: {status['daily_interventions']}/{status['daily_limit']}",
                f"🚫 الحد وصل: {'نعم' if status['limit_reached'] else 'لا'}",
                f"⏱️ الكولدوان: {status['cooldown_seconds']} ثانية",
            ]
            
            if status['active_cooldowns']:
                msg.append("⏳ كولدوان نشط:")
                for symbol, remaining in status['active_cooldowns'].items():
                    msg.append(f"   {symbol}: {remaining}s")
            
            return "\n".join(msg)
            
        except Exception as e:
            logger.error(f"[TG] Error getting status: {e}")
            return f"❌ خطأ في جلب الحالة: {e}"
    
    def get_mode_keyboard(self) -> list:
        return [
            [{"text": "⚫ OFF", "callback_data": "ai_mode_OFF"}],
            [{"text": "🔵 LEARN", "callback_data": "ai_mode_LEARN"}],
            [{"text": "🟢 FULL", "callback_data": "ai_mode_FULL"}]
        ]
    
    def get_weight_keyboard(self) -> list:
        return [
            [{"text": "⚫ OFF (0.0)", "callback_data": "ai_weight_OFF"}],
            [{"text": "🟡 LOW (0.3)", "callback_data": "ai_weight_LOW"}],
            [{"text": "🟠 MEDIUM (0.6)", "callback_data": "ai_weight_MEDIUM"}],
            [{"text": "🔴 HIGH (1.0)", "callback_data": "ai_weight_HIGH"}]
        ]
    
    def get_main_keyboard(self) -> list:
        return [
            [{"text": "📊 الحالة", "callback_data": "ai_status"}],
            [{"text": "🔄 تغيير الوضع", "callback_data": "ai_mode_menu"}],
            [{"text": "⚖️ تغيير الوزن", "callback_data": "ai_weight_menu"}],
            [{"text": "📈 تعيين السقف", "callback_data": "ai_limit_menu"}]
        ]

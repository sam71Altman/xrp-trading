#!/usr/bin/env python3
"""
Smart Adaptive Trading System - Trade Modes Module
نظام التداول الذكي التكيفي - وحدة أوضاع التداول
v4.2.PRO-AI | GOVERNED INTELLIGENCE
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json
import os

logger = logging.getLogger(__name__)

# Version Control (Unified)
AI_VERSION = "v4.2.PRO-AI"
BUILD_DATE = "2026"

MODE_PERFORMANCE_FILE = "mode_performance.json"
MODE_HISTORY_FILE = "mode_history.json"
AI_STATE_FILE = "ai_state.json"

# ⚡ HARD RULES (NON-NEGOTIABLE)
HARD_RULES = {
    "NO_DELETION": True,
    "NO_CORE_MODIFICATION": True,
    "AI_LAYER_ONLY": True,
    "NEXT_CANDLE_ONLY": True,
    "OPEN_TRADES_SAFE": True,
    "ONE_CLICK_DISABLE": True,
    "VERSION_UNIFIED": AI_VERSION
}

# 🔒 FINAL GUARANTEES
FINAL_GUARANTEES = {
    "NO_SYSTEM_RESTART": True,
    "NO_TRADE_INTERFERENCE": True,
    "NO_RISK_OVERRIDE": True,
    "AI_NO_MARTINGALE": True,
    "AI_NO_AVERAGING": True,
    "AI_LIMITS_ENFORCED": True,
    "INSTANT_REVERT": True,
    "FULL_TRANSPARENCY": True,
    "IMPACT_CAP_ACTIVE": True,
    "AUTO_SWITCH_ACTIVE": True,
    "WARNING_SYSTEM_ACTIVE": True,
    "DAILY_RESET_ACTIVE": True
}

# AI MODES
AI_MODES = {
    "OFF": "❌ معطل — منطق ثابت فقط",
    "LEARN": "📚 تعليمي — اقتراحات بدون تعديل",
    "FULL": "✅ مفعل — تعديلات ذكية ضمن حدود آمنة"
}

# AI IMPACT LEVELS
AI_IMPACT_LEVELS = {
    "LOW": {
        "label": "🟢 منخفض",
        "max_daily": 15,
        "max_change_pct": 0.15
    },
    "MEDIUM": {
        "label": "🟡 متوسط",
        "max_daily": 25,
        "max_change_pct": 0.25
    },
    "HIGH": {
        "label": "🔴 عالي",
        "max_daily": 40,
        "max_change_pct": 0.35,
        "confirmation_required": True
    }
}


class TradeMode:
    """الأنماط الرئيسية للتداول"""
    DEFAULT_CLEAN_AGGRESSIVE = "DEFAULT"
    FAST_SCALP_AGGRESSIVE = "FAST_SCALP"
    BOUNCE_FOCUS_MODE = "BOUNCE"
    
    ALL_MODES = [DEFAULT_CLEAN_AGGRESSIVE, FAST_SCALP_AGGRESSIVE, BOUNCE_FOCUS_MODE]
    
    DISPLAY_NAMES = {
        "DEFAULT": "🧠 الوضع الذكي (افتراضي)",
        "FAST_SCALP": "⚡ سكالب سريع",
        "BOUNCE": "🧲 اصطياد الارتدادات"
    }
    
    RISK_LEVELS = {
        "DEFAULT": "متوسطة",
        "FAST_SCALP": "عالية",
        "BOUNCE": "متوسطة إلى منخفضة"
    }
    
    DESCRIPTIONS = {
        "DEFAULT": "المنطق الحالي المحسّن - توازن بين الجودة والكمية",
        "FAST_SCALP": "صفقات سريعة متعددة - كمية على حساب الجودة",
        "BOUNCE": "اصطياد الارتدادات في الأسواق الهابطة - تركيز على القيعان"
    }
    
    EXPECTED_TRADES = {
        "DEFAULT": "5-10",
        "FAST_SCALP": "15-25",
        "BOUNCE": "3-6"
    }
    
    TARGET_PROFIT = {
        "DEFAULT": "0.15-0.25",
        "FAST_SCALP": "0.05-0.10",
        "BOUNCE": "0.20-0.40"
    }
    
    SUITABLE_MARKET = {
        "DEFAULT": "جميع الأسواق",
        "FAST_SCALP": "أسواق نشطة عالية التذبذب",
        "BOUNCE": "أسواق هابطة أو متراجعة"
    }
    
    MODE_TIPS = {
        "DEFAULT": "هذا الوضع مناسب لمعظم الظروف. راقب السوق وعدّل حسب الحاجة.",
        "FAST_SCALP": "توقع صفقات صغيرة متعددة. لا تقلق من الخسائر الصغيرة - المجموع هو الهدف.",
        "BOUNCE": "الصبر مطلوب. انتظر الإشارات القوية في القيعان فقط."
    }


class TradingLogicController:
    """التحكم في منطق التداول حسب الوضع"""
    
    def get_trading_params(self, trade_mode: str, market_data: Optional[Dict] = None) -> Dict:
        """إرجاع معاملات التداول الخاصة بكل وضع"""
        incoming_score = market_data.get("ai_score") if market_data else "N/A"
        logger.info(f"[MODE RECEIVED SCORE] mode={trade_mode} score={incoming_score}")
        
        params = {
            "price_protection": True,
            "volume_filter": True,
            "require_breakout": True,
            "hold_logic_enabled": True,
            "min_rsi": 30,
            "max_rsi": 70,
            "tp_target": 0.15,
            "sl_target": 0.25,
            "max_trade_duration": 3600,
            "cooldown_between_trades": 60,
            "min_signal_score": 0.4,
        }
        
        if trade_mode == TradeMode.FAST_SCALP_AGGRESSIVE:
            params.update({
                "price_protection": False,
                "volume_filter": False,
                "require_breakout": False,
                "hold_logic_enabled": False,
                "tp_target": 0.08,
                "sl_target": 0.05,
                "max_trade_duration": 900,
                "cooldown_between_trades": 30,
                "min_signal_score": 0.4,
            })
        
        elif trade_mode == TradeMode.BOUNCE_FOCUS_MODE:
            params.update({
                "price_protection": True,
                "volume_filter": True,
                "require_breakout": True,
                "hold_logic_enabled": True,
                "min_rsi": 20,
                "max_rsi": 40,
                "tp_target": 0.12,
                "sl_target": 0.06,
                "entry_conditions": self._get_bounce_conditions(market_data),
            })
        
        # 🧠 AI SCORE INTEGRATION (v4.2.PRO-AI)
        # AI score must remain unchanged across the entire decision pipeline.
        ai_score = market_data.get("ai_score") if market_data else 0.5
        real_score = ai_score  # CRITICAL: No modification, no clamping, no overrides.
        
        min_required = params.get("min_signal_score", 0.4)
        
        # [MODE VERIFY] - Debug verification log
        logger.info(
            f"[MODE VERIFY] mode={trade_mode} "
            f"ai_score={ai_score} "
            f"used_score={real_score} "
            f"min_required={min_required}"
        )
        
        # Log for backward compatibility with monitoring
        logger.info(f"[HOLD PROBE] mode={trade_mode} score={real_score} min_required={min_required}")
        
        return params
    
    def _get_bounce_conditions(self, market_data: Optional[Dict]) -> Dict:
        """شروط خاصة بالارتدادات"""
        return {
            "require_local_bottom": True,
            "require_oversold_rsi": True,
            "require_volume_spike": True,
            "require_support_level": True,
            "bounce_confirmation_candles": 2,
        }


class ModePerformanceTracker:
    """تتبع أداء كل وضع بشكل منفصل"""
    
    def __init__(self):
        self.performance = self._load_performance()
    
    def _load_performance(self) -> Dict:
        """تحميل بيانات الأداء من الملف"""
        if os.path.exists(MODE_PERFORMANCE_FILE):
            try:
                with open(MODE_PERFORMANCE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {
            "DEFAULT": {"trades": 0, "wins": 0, "losses": 0, "total_profit": 0.0},
            "FAST_SCALP": {"trades": 0, "wins": 0, "losses": 0, "total_profit": 0.0},
            "BOUNCE": {"trades": 0, "wins": 0, "losses": 0, "total_profit": 0.0}
        }
    
    def _save_performance(self):
        """حفظ بيانات الأداء"""
        try:
            with open(MODE_PERFORMANCE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.performance, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving mode performance: {e}")
    
    def record_trade(self, trade_mode: str, profit: float, is_win: bool):
        """تسجيل نتيجة صفقة"""
        if trade_mode not in self.performance:
            self.performance[trade_mode] = {"trades": 0, "wins": 0, "losses": 0, "total_profit": 0.0}
        
        stats = self.performance[trade_mode]
        stats["trades"] += 1
        if is_win:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        stats["total_profit"] += profit
        
        self._save_performance()
        logger.info(f"[MODE STATS] {trade_mode}: Recorded trade, profit={profit:.4f}, win={is_win}")
    
    def get_mode_stats(self, trade_mode: str) -> Dict:
        """جلب إحصائيات وضع معين"""
        stats = self.performance.get(trade_mode, {"trades": 0, "wins": 0, "losses": 0, "total_profit": 0.0})
        
        win_rate = 0.0
        avg_profit = 0.0
        
        if stats["trades"] > 0:
            win_rate = (stats["wins"] / stats["trades"]) * 100
            avg_profit = stats["total_profit"] / stats["trades"]
        
        return {
            "trades": stats["trades"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "win_rate": round(win_rate, 2),
            "avg_profit": round(avg_profit, 4),
            "total_profit": round(stats["total_profit"], 4)
        }
    
    def get_all_stats(self) -> Dict:
        """جلب إحصائيات جميع الأوضاع"""
        return {mode: self.get_mode_stats(mode) for mode in TradeMode.ALL_MODES}
    
    def get_best_mode(self) -> str:
        """تحديد أفضل وضع حسب الأداء"""
        best_mode = TradeMode.DEFAULT_CLEAN_AGGRESSIVE
        best_profit = -float('inf')
        
        for mode in TradeMode.ALL_MODES:
            stats = self.get_mode_stats(mode)
            if stats["trades"] >= 5 and stats["total_profit"] > best_profit:
                best_profit = stats["total_profit"]
                best_mode = mode
        
        return best_mode
    
    def reset_stats(self, mode: Optional[str] = None):
        """إعادة تعيين الإحصائيات"""
        if mode:
            self.performance[mode] = {"trades": 0, "wins": 0, "losses": 0, "total_profit": 0.0}
        else:
            self.performance = {m: {"trades": 0, "wins": 0, "losses": 0, "total_profit": 0.0} for m in TradeMode.ALL_MODES}
        self._save_performance()


class ModeStateManager:
    """إدارة حالة الأوضاع"""
    
    MIN_CHANGE_INTERVAL = 300
    MAX_CHANGES_PER_HOUR = 3
    
    def __init__(self):
        self.current_mode = TradeMode.DEFAULT_CLEAN_AGGRESSIVE
        self.previous_mode = None
        self.last_mode_change = None
        self.mode_change_history = self._load_history()
        self.mode_activated_at = datetime.now()
        self.hourly_changes = 0
        self.hourly_reset_time = datetime.now()
    
    def _load_history(self) -> List[Dict]:
        """تحميل سجل التغييرات"""
        if os.path.exists(MODE_HISTORY_FILE):
            try:
                with open(MODE_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def _save_history(self):
        """حفظ سجل التغييرات"""
        try:
            history_to_save = self.mode_change_history[-50:]
            with open(MODE_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(history_to_save, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving mode history: {e}")
    
    def can_change_mode(self) -> tuple:
        """التحقق من إمكانية تغيير الوضع"""
        now = datetime.now()
        
        if now - self.hourly_reset_time > timedelta(hours=1):
            self.hourly_changes = 0
            self.hourly_reset_time = now
        
        if self.hourly_changes >= self.MAX_CHANGES_PER_HOUR:
            return False, f"تجاوزت الحد الأقصى ({self.MAX_CHANGES_PER_HOUR} تغييرات/ساعة)"
        
        if self.last_mode_change:
            time_since_change = (now - self.last_mode_change).total_seconds()
            if time_since_change < self.MIN_CHANGE_INTERVAL:
                remaining = int(self.MIN_CHANGE_INTERVAL - time_since_change)
                return False, f"انتظر {remaining} ثانية قبل التغيير التالي"
        
        return True, "OK"
    
    def change_mode(self, new_mode: str) -> tuple:
        """تغيير الوضع"""
        if new_mode not in TradeMode.ALL_MODES:
            return False, "وضع غير صالح"
        
        if new_mode == self.current_mode:
            return False, "أنت بالفعل في هذا الوضع"
        
        can_change, reason = self.can_change_mode()
        if not can_change:
            return False, reason
        
        self.previous_mode = self.current_mode
        self.current_mode = new_mode
        self.last_mode_change = datetime.now()
        self.mode_activated_at = datetime.now()
        self.hourly_changes += 1
        
        self.mode_change_history.append({
            "timestamp": self.last_mode_change.isoformat(),
            "from": self.previous_mode,
            "to": new_mode
        })
        self._save_history()
        
        logger.info(f"[MODE CHANGE] {self.previous_mode} -> {new_mode}")
        return True, "تم تغيير الوضع بنجاح"
    
    def get_mode_duration(self) -> str:
        """الحصول على مدة الوضع الحالي"""
        if not self.mode_activated_at:
            return "غير معروف"
        
        duration = datetime.now() - self.mode_activated_at
        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)
        
        if hours > 0:
            return f"{hours} ساعة و {minutes} دقيقة"
        return f"{minutes} دقيقة"
    
    def get_recent_changes(self, limit: int = 5) -> List[Dict]:
        """جلب آخر التغييرات"""
        return self.mode_change_history[-limit:][::-1]


class ModeRecommender:
    """نظام اقتراح الأوضاع الذكي"""
    
    def analyze_and_suggest(self, market_data: Dict) -> Dict:
        """تحليل السوق واقتراح الوضع الأمثل"""
        
        analysis = {
            "volatility": self._calculate_volatility(market_data),
            "trend_strength": self._calculate_trend_strength(market_data),
            "market_phase": self._identify_market_phase(market_data),
            "rsi": market_data.get("rsi", 50),
            "ema_alignment": market_data.get("ema_bullish", True)
        }
        
        recommendation = self._get_recommendation(analysis)
        confidence = self._calculate_confidence(analysis)
        
        return {
            "recommended_mode": recommendation,
            "confidence": confidence,
            "analysis": analysis,
            "reason": self._get_recommendation_reason(analysis, recommendation)
        }
    
    def _calculate_volatility(self, market_data: Dict) -> float:
        """حساب التذبذب"""
        candles = market_data.get("candles", [])
        if len(candles) < 20:
            return 1.0
        
        ranges = [(c.get("high", 0) - c.get("low", 0)) / c.get("close", 1) * 100 for c in candles[-20:]]
        return sum(ranges) / len(ranges) if ranges else 1.0
    
    def _calculate_trend_strength(self, market_data: Dict) -> float:
        """حساب قوة الاتجاه"""
        ema20 = market_data.get("ema20", 0)
        ema50 = market_data.get("ema50", 0)
        if ema50 == 0:
            return 0
        return ((ema20 - ema50) / ema50) * 100
    
    def _identify_market_phase(self, market_data: Dict) -> str:
        """تحديد مرحلة السوق"""
        ema20 = market_data.get("ema20", 0)
        ema50 = market_data.get("ema50", 0)
        ema200 = market_data.get("ema200", 0)
        
        if ema20 > ema50 > ema200:
            return "BULLISH"
        elif ema20 < ema50 < ema200:
            return "BEARISH"
        else:
            return "RANGING"
    
    def _get_recommendation(self, analysis: Dict) -> str:
        """تحديد الوضع المناسب"""
        market_phase = analysis.get("market_phase", "RANGING")
        volatility = analysis.get("volatility", 1.0)
        rsi = analysis.get("rsi", 50)
        
        if market_phase == "BEARISH" and rsi < 35:
            return TradeMode.BOUNCE_FOCUS_MODE
        
        if volatility > 1.5 and market_phase in ["BULLISH", "RANGING"]:
            return TradeMode.FAST_SCALP_AGGRESSIVE
        
        return TradeMode.DEFAULT_CLEAN_AGGRESSIVE
    
    def _calculate_confidence(self, analysis: Dict) -> int:
        """حساب مستوى الثقة (0-100)"""
        confidence = 50
        
        volatility = analysis.get("volatility", 1.0)
        if volatility > 1.5:
            confidence += 15
        elif volatility < 0.5:
            confidence -= 10
        
        trend_strength = abs(analysis.get("trend_strength", 0))
        if trend_strength > 1.0:
            confidence += 20
        
        return min(100, max(0, confidence))
    
    def _get_recommendation_reason(self, analysis: Dict, recommendation: str) -> str:
        """سبب الاقتراح"""
        market_phase = analysis.get("market_phase", "RANGING")
        volatility = analysis.get("volatility", 1.0)
        rsi = analysis.get("rsi", 50)
        
        if recommendation == TradeMode.BOUNCE_FOCUS_MODE:
            return f"السوق هابط ({market_phase}) مع RSI منخفض ({rsi:.1f}) - فرص ارتداد محتملة"
        elif recommendation == TradeMode.FAST_SCALP_AGGRESSIVE:
            return f"تذبذب عالي ({volatility:.2f}%) - مناسب للسكالبينج السريع"
        else:
            return "ظروف السوق عادية - الوضع الذكي الافتراضي مناسب"


class ModeValidator:
    """التحقق من صحة تطبيق الأوضاع"""
    
    def __init__(self, logic_controller: TradingLogicController):
        self.logic_controller = logic_controller
    
    def validate_mode_application(self, current_mode: str, runtime_params: Dict) -> Dict:
        """التحقق من أن الوضع مطبق بشكل صحيح"""
        expected_params = self.logic_controller.get_trading_params(current_mode)
        
        logs = []
        mismatches = 0
        
        for param, expected_value in expected_params.items():
            if param == "entry_conditions":
                continue
            
            actual_value = runtime_params.get(param)
            if actual_value is not None and actual_value != expected_value:
                logs.append(f"⚠️ {param}: متوقع={expected_value}, فعلي={actual_value}")
                mismatches += 1
            else:
                logs.append(f"✅ {param}: {expected_value}")
        
        return {
            "mode": current_mode,
            "applied_correctly": mismatches == 0,
            "mismatches": mismatches,
            "details": logs
        }


class AIImpactGuard:
    """نظام سقف تأثير الذكاء - AI Impact Cap"""
    
    def __init__(self):
        self.impact_level = "LOW"
        self.daily_adjustments = 0
        self.daily_reset_time = datetime.now()
        self.warning_sent_70 = False
        self.warning_sent_90 = False
        self._load_state()
    
    def _load_state(self):
        """تحميل حالة الحارس"""
        if os.path.exists(AI_STATE_FILE):
            try:
                with open(AI_STATE_FILE, 'r') as f:
                    data = json.load(f)
                    self.daily_adjustments = data.get("daily_adjustments", 0)
                    self.impact_level = data.get("impact_level", "LOW")
                    reset_str = data.get("daily_reset_time")
                    if reset_str:
                        self.daily_reset_time = datetime.fromisoformat(reset_str)
            except:
                pass
    
    def _save_state(self):
        """حفظ حالة الحارس"""
        try:
            with open(AI_STATE_FILE, 'w') as f:
                json.dump({
                    "daily_adjustments": self.daily_adjustments,
                    "impact_level": self.impact_level,
                    "daily_reset_time": self.daily_reset_time.isoformat()
                }, f)
        except Exception as e:
            logger.error(f"Error saving AI state: {e}")
    
    def check_daily_reset(self):
        """إعادة تعيين العداد اليومي"""
        now = datetime.now()
        if now.date() > self.daily_reset_time.date():
            self.daily_adjustments = 0
            self.daily_reset_time = now
            self.warning_sent_70 = False
            self.warning_sent_90 = False
            self._save_state()
            logger.info("[AI GUARD] Daily reset performed")
    
    def can_adjust(self) -> bool:
        """التحقق من إمكانية إجراء تعديل"""
        self.check_daily_reset()
        max_daily = AI_IMPACT_LEVELS[self.impact_level]["max_daily"]
        return self.daily_adjustments < max_daily
    
    def record_adjustment(self):
        """تسجيل تعديل"""
        self.daily_adjustments += 1
        self._save_state()
        logger.info(f"[AI GUARD] Adjustment recorded: {self.daily_adjustments}/{AI_IMPACT_LEVELS[self.impact_level]['max_daily']}")
    
    def get_usage_percentage(self) -> float:
        """نسبة الاستخدام"""
        max_daily = AI_IMPACT_LEVELS[self.impact_level]["max_daily"]
        return (self.daily_adjustments / max_daily) * 100 if max_daily > 0 else 0
    
    def get_warning_status(self) -> Optional[str]:
        """حالة التحذيرات"""
        usage = self.get_usage_percentage()
        
        if usage >= 100:
            return "CRITICAL_100"
        elif usage >= 90 and not self.warning_sent_90:
            self.warning_sent_90 = True
            return "WARNING_90"
        elif usage >= 70 and not self.warning_sent_70:
            self.warning_sent_70 = True
            return "WARNING_70"
        return None
    
    def set_impact_level(self, level: str) -> bool:
        """تغيير مستوى التأثير"""
        if level not in AI_IMPACT_LEVELS:
            return False
        self.impact_level = level
        self._save_state()
        return True
    
    def get_status(self) -> Dict:
        """حالة الحارس الكاملة"""
        self.check_daily_reset()
        return {
            "level": self.impact_level,
            "level_label": AI_IMPACT_LEVELS[self.impact_level]["label"],
            "daily_used": self.daily_adjustments,
            "daily_max": AI_IMPACT_LEVELS[self.impact_level]["max_daily"],
            "usage_pct": round(self.get_usage_percentage(), 1),
            "can_adjust": self.can_adjust(),
            "time_to_reset": self._get_time_to_reset()
        }
    
    def _get_time_to_reset(self) -> str:
        """الوقت المتبقي للإعادة"""
        now = datetime.now()
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        remaining = tomorrow - now
        hours = int(remaining.total_seconds() // 3600)
        minutes = int((remaining.total_seconds() % 3600) // 60)
        return f"{hours}س {minutes}د"


class AISystem:
    """نظام الذكاء التكيفي - AI System v4.2.PRO-AI"""
    
    def __init__(self):
        self.enabled = True
        self.mode = "FULL"  # OFF / LEARN / FULL
        self.toggle_cooldown = 60
        self.guard_active = True
        self.impact_stats = {
            "adjustments_made": 0,
            "suggestions_given": 0,
            "user_accepted": 0,
            "performance_boost": 0.0
        }
        self.toggle_history = []
        self.last_toggle_time = None
        self.consecutive_losses = 0
        self.silent_pause_active = False
        self._load_state()
    
    def _load_state(self):
        """تحميل حالة الذكاء"""
        if os.path.exists(AI_STATE_FILE):
            try:
                with open(AI_STATE_FILE, 'r') as f:
                    data = json.load(f)
                    self.enabled = data.get("ai_enabled", True)
                    self.mode = data.get("ai_mode", "FULL")
            except:
                pass
    
    def _save_state(self):
        """حفظ حالة الذكاء"""
        try:
            if os.path.exists(AI_STATE_FILE):
                with open(AI_STATE_FILE, 'r') as f:
                    data = json.load(f)
            else:
                data = {}
            data["ai_enabled"] = self.enabled
            data["ai_mode"] = self.mode
            with open(AI_STATE_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving AI state: {e}")
    
    def toggle(self) -> tuple:
        """تبديل حالة الذكاء"""
        now = datetime.now()
        if self.last_toggle_time:
            elapsed = (now - self.last_toggle_time).total_seconds()
            if elapsed < self.toggle_cooldown:
                remaining = int(self.toggle_cooldown - elapsed)
                return False, f"انتظر {remaining} ثانية"
        
        self.enabled = not self.enabled
        self.last_toggle_time = now
        self.toggle_history.append({
            "time": now.isoformat(),
            "action": "enabled" if self.enabled else "disabled"
        })
        self._save_state()
        
        status = "مفعل" if self.enabled else "معطل"
        logger.info(f"[AI SYSTEM] Toggled to: {status}")
        return True, f"الذكاء الآن {status}"
    
    def set_mode(self, new_mode: str) -> tuple:
        """تغيير وضع الذكاء"""
        if new_mode not in AI_MODES:
            return False, "وضع غير صالح"
        
        if new_mode == self.mode:
            return False, "أنت بالفعل في هذا الوضع"
        
        old_mode = self.mode
        self.mode = new_mode
        self._save_state()
        
        logger.info(f"[AI SYSTEM] Mode changed: {old_mode} -> {new_mode}")
        return True, f"تم تغيير وضع الذكاء إلى {AI_MODES[new_mode]}"
    
    def emergency_shutdown(self, reason: str):
        """إيقاف طارئ للذكاء"""
        self.enabled = False
        self.mode = "OFF"
        self._save_state()
        logger.warning(f"[AI EMERGENCY] Shutdown triggered: {reason}")
        return f"🚨 إيقاف طارئ للذكاء: {reason}"
    
    def record_loss(self):
        """تسجيل خسارة للحوكمة الصامتة"""
        self.consecutive_losses += 1
        if self.consecutive_losses >= 3:
            self.silent_pause_active = True
            logger.info("[AI SYSTEM] Silent pause activated after 3 losses")
    
    def record_win(self):
        """تسجيل ربح"""
        self.consecutive_losses = 0
        self.silent_pause_active = False
    
    def can_make_adjustment(self, has_open_trade: bool, impact_guard: 'AIImpactGuard') -> bool:
        """التحقق من إمكانية إجراء تعديل ذكي"""
        if not self.enabled:
            return False
        if self.mode == "OFF":
            return False
        if has_open_trade:  # OPEN_TRADES_SAFE
            return False
        if not impact_guard.can_adjust():
            return False
        if self.silent_pause_active:
            return False
        return True
    
    def get_status(self) -> Dict:
        """حالة النظام"""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "mode_label": AI_MODES.get(self.mode, "غير معروف"),
            "guard_active": self.guard_active,
            "silent_pause": self.silent_pause_active,
            "consecutive_losses": self.consecutive_losses,
            "stats": self.impact_stats
        }


# Global instances
performance_tracker = ModePerformanceTracker()
mode_state = ModeStateManager()
logic_controller = TradingLogicController()
mode_recommender = ModeRecommender()
mode_validator = ModeValidator(logic_controller)
ai_system = AISystem()
ai_impact_guard = AIImpactGuard()


def get_current_mode() -> str:
    """الحصول على الوضع الحالي"""
    return mode_state.current_mode


def get_mode_params() -> Dict:
    """الحصول على معاملات الوضع الحالي"""
    return logic_controller.get_trading_params(mode_state.current_mode)


def change_trade_mode(new_mode: str) -> tuple:
    """تغيير وضع التداول"""
    return mode_state.change_mode(new_mode)


def record_mode_trade(profit: float, is_win: bool):
    """تسجيل صفقة للوضع الحالي"""
    performance_tracker.record_trade(mode_state.current_mode, profit, is_win)


def get_mode_recommendation(market_data: Dict) -> Dict:
    """الحصول على اقتراح الوضع الأمثل"""
    return mode_recommender.analyze_and_suggest(market_data)


def format_mode_stats_message() -> str:
    """تنسيق رسالة إحصائيات الأوضاع"""
    all_stats = performance_tracker.get_all_stats()
    
    message = "📊 *إحصائيات أوضاع التداول*\n"
    message += "═══════════════════════\n\n"
    
    for mode, stats in all_stats.items():
        display_name = TradeMode.DISPLAY_NAMES.get(mode, mode)
        win_rate_emoji = "🟢" if stats["win_rate"] >= 50 else "🔴"
        profit_emoji = "📈" if stats["total_profit"] >= 0 else "📉"
        
        message += f"*{display_name}*\n"
        message += f"├ الصفقات: {stats['trades']}\n"
        message += f"├ الفوز/الخسارة: {stats['wins']}/{stats['losses']}\n"
        message += f"├ {win_rate_emoji} نسبة الفوز: {stats['win_rate']}%\n"
        message += f"└ {profit_emoji} إجمالي الربح: {stats['total_profit']:.4f}$\n\n"
    
    best_mode = performance_tracker.get_best_mode()
    message += f"🏆 *أفضل وضع:* {TradeMode.DISPLAY_NAMES.get(best_mode, best_mode)}\n"
    
    return message


def format_mode_confirmation_message(new_mode: str) -> str:
    """تنسيق رسالة تأكيد تغيير الوضع"""
    display_name = TradeMode.DISPLAY_NAMES.get(new_mode, new_mode)
    risk_level = TradeMode.RISK_LEVELS.get(new_mode, "غير محدد")
    description = TradeMode.DESCRIPTIONS.get(new_mode, "")
    expected_trades = TradeMode.EXPECTED_TRADES.get(new_mode, "غير محدد")
    target_profit = TradeMode.TARGET_PROFIT.get(new_mode, "غير محدد")
    suitable_market = TradeMode.SUITABLE_MARKET.get(new_mode, "غير محدد")
    tip = TradeMode.MODE_TIPS.get(new_mode, "")
    
    message = f"""
🎯 *تم تغيير وضع التداول بنجاح*

✅ *الوضع الجديد:* {display_name}
📊 *مستوى المخاطرة:* {risk_level}
📝 *الوصف:* {description}
🕒 *وقت التفعيل:* الشمعة القادمة

🔍 *التوقعات تحت هذا الوضع:*
• عدد الصفقات/يوم: {expected_trades}
• متوسط الربح المستهدف: {target_profit}%
• نوع السوق المناسب: {suitable_market}

💡 *نصيحة:* {tip}

📈 لمشاهدة الإحصائيات: /modestats
🔄 للتغيير مجدداً: /mode
    """
    return message


def format_dashboard_message(market_data: Optional[Dict] = None) -> str:
    """تنسيق لوحة التحكم الشاملة"""
    current_mode = mode_state.current_mode
    display_name = TradeMode.DISPLAY_NAMES.get(current_mode, current_mode)
    risk_level = TradeMode.RISK_LEVELS.get(current_mode, "غير محدد")
    mode_duration = mode_state.get_mode_duration()
    
    params = logic_controller.get_trading_params(current_mode)
    
    all_stats = performance_tracker.get_all_stats()
    
    message = f"""
📊 *لوحة تحكم التداول الذكي*
════════════════════════════

🧠 *الوضع الحالي:* {display_name}
⚡ *المخاطرة:* {risk_level}
🕒 *مفعل منذ:* {mode_duration}

📈 *أداء الأوضاع:*
"""
    
    for mode, stats in all_stats.items():
        mode_emoji = "✅" if mode == current_mode else "➡️"
        mode_name = TradeMode.DISPLAY_NAMES.get(mode, mode).split()[0]
        profit_sign = "+" if stats["total_profit"] >= 0 else ""
        message += f"{mode_emoji} {mode_name}: {stats['trades']} صفقة | {stats['win_rate']}% | {profit_sign}{stats['total_profit']:.2f}$\n"
    
    message += f"""
⚙️ *المعاملات النشطة:*
├ فلتر السعر: {'✅' if params.get('price_protection') else '❌'}
├ فلتر الحجم: {'✅' if params.get('volume_filter') else '❌'}
├ تأكيد الاختراق: {'✅' if params.get('require_breakout') else '❌'}
├ نظام الهولد: {'✅' if params.get('hold_logic_enabled') else '❌'}
├ TP الهدف: {params.get('tp_target', 0)*100:.1f}%
└ SL الهدف: {params.get('sl_target', 0)*100:.1f}%
"""
    
    if market_data:
        recommendation = mode_recommender.analyze_and_suggest(market_data)
        rec_mode = TradeMode.DISPLAY_NAMES.get(recommendation["recommended_mode"], "")
        message += f"""
🎯 *اقتراح الذكاء الاصطناعي:*
├ الوضع المقترح: {rec_mode}
├ الثقة: {recommendation['confidence']}%
└ السبب: {recommendation['reason']}
"""
    
    recent_changes = mode_state.get_recent_changes(3)
    if recent_changes:
        message += "\n🔄 *آخر التغييرات:*\n"
        for change in recent_changes:
            from_mode = TradeMode.DISPLAY_NAMES.get(change['from'], change['from']).split()[0]
            to_mode = TradeMode.DISPLAY_NAMES.get(change['to'], change['to']).split()[0]
            message += f"├ {from_mode} ← {to_mode}\n"
    
    message += """
════════════════════════════
🔧 /mode - تغيير الوضع
📈 /modestats - إحصائيات مفصلة
🎯 /recommend - اقتراح ذكي
⚠️ /validate - التحقق من التطبيق
"""
    
    return message


def format_recommendation_message(market_data: Dict) -> str:
    """تنسيق رسالة الاقتراح"""
    recommendation = mode_recommender.analyze_and_suggest(market_data)
    rec_mode = recommendation["recommended_mode"]
    rec_display = TradeMode.DISPLAY_NAMES.get(rec_mode, rec_mode)
    confidence = recommendation["confidence"]
    reason = recommendation["reason"]
    analysis = recommendation["analysis"]
    
    current_mode = mode_state.current_mode
    current_display = TradeMode.DISPLAY_NAMES.get(current_mode, current_mode)
    
    same_mode = rec_mode == current_mode
    
    message = f"""
🎯 *اقتراح الوضع الذكي*
════════════════════════════

📊 *تحليل السوق:*
├ مرحلة السوق: {analysis.get('market_phase', 'غير محدد')}
├ التذبذب: {analysis.get('volatility', 0):.2f}%
├ قوة الاتجاه: {analysis.get('trend_strength', 0):.2f}%
└ RSI: {analysis.get('rsi', 50):.1f}

🧠 *الوضع الحالي:* {current_display}
{'✅' if same_mode else '🔄'} *الوضع المقترح:* {rec_display}
📊 *مستوى الثقة:* {confidence}%

💡 *السبب:* {reason}
"""
    
    if not same_mode:
        message += f"\n⚡ لتغيير الوضع، استخدم الأزرار أدناه أو /mode"
    else:
        message += "\n✅ أنت بالفعل في الوضع المقترح!"
    
    return message

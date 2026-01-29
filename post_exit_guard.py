import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PostExitGuard:
    """
    حارس خفيف يمنع الدخول بعد الخروج في سوق غير متعافٍ (v1.3)
    """
    _instance = None
    MAX_BLOCK_DURATION = 3600  # 1 ساعة (حماية فقط)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.active = False
        self.exit_price = None
        self.exit_time = None
        self._stats = {
            "entries_blocked": 0,
            "entries_allowed": 0,
            "total_block_duration": 0.0,
            "clear_reasons": {},
            "recovery_reasons": {}
        }

    def record_exit(self, exit_price):
        """تسجيل خروج جديد"""
        self.active = True
        self.exit_price = exit_price
        self.exit_time = time.time()
        logger.info(f"[PEG] Exit recorded @ {exit_price}")

    def clear(self, reason):
        """مسح حالة الحارس"""
        if self.active:
            if self.exit_time:
                block_duration = time.time() - self.exit_time
                self._stats["total_block_duration"] += block_duration
            self._stats["clear_reasons"][reason] = self._stats["clear_reasons"].get(reason, 0) + 1
            logger.info(f"[PEG] Cleared | reason={reason}")
        self.active = False
        self.exit_price = None
        self.exit_time = None

    def expired(self):
        """هل انتهت صلاحية الحارس؟"""
        if not self.active or self.exit_time is None:
            return False
        return (time.time() - self.exit_time) > self.MAX_BLOCK_DURATION

    def record_block(self):
        self._stats["entries_blocked"] += 1

    def record_allow(self, recovery_reason=None):
        self._stats["entries_allowed"] += 1
        if recovery_reason:
            self._stats["recovery_reasons"][recovery_reason] = \
                self._stats["recovery_reasons"].get(recovery_reason, 0) + 1

    def get_stats_summary(self):
        total = self._stats["entries_blocked"] + self._stats["entries_allowed"]
        return {
            "entries_blocked": self._stats["entries_blocked"],
            "entries_allowed": self._stats["entries_allowed"],
            "block_percentage": (self._stats["entries_blocked"] / total * 100) if total > 0 else 0,
            "clear_reasons": dict(self._stats["clear_reasons"]),
            "recovery_reasons": dict(self._stats["recovery_reasons"])
        }

def market_recovered(guard, current_price, candles, ema20, ema50):
    """تقييم تعافي السوق (v1.3)"""
    if not guard.active or guard.exit_time is None:
        return True, "no_guard"

    # 1. اختراق قمة الارتداد
    last_high = get_pullback_high_since_exit(guard, candles)
    if last_high is not None and current_price > last_high:
        return True, f"breakout_above_{last_high:.4f}"

    # 2. Higher Low (Simplified version for 1m timeframe)
    if higher_low_confirmed(guard, candles):
        return True, "higher_low_confirmed"

    # 3. عودة EMA20 فوق EMA50
    if ema20 > ema50:
        return True, "ema20_above_ema50"

    # 4. ارتداد قوي فوري
    lowest = get_lowest_price_since_exit(guard, candles)
    if lowest is not None:
        recovery_pct = ((current_price - lowest) / lowest) * 100
        if recovery_pct >= 0.15: # Standard for scalping (instead of 1.5% in doc which is huge for 1m)
            return True, f"strong_recovery_{recovery_pct:.2f}%"

    return False, None

def get_pullback_high_since_exit(guard, candles):
    relevant = [c for c in candles if c[0]/1000 > guard.exit_time]
    if len(relevant) < 3: return None
    recent = relevant[-min(5, len(relevant)):]
    return max(c[2] for c in recent) # High is index 2

def get_lowest_price_since_exit(guard, candles):
    relevant = [c for c in candles if c[0]/1000 > guard.exit_time]
    if not relevant: return None
    return min(c[3] for c in relevant) # Low is index 3

def higher_low_confirmed(guard, candles):
    # Simplified Logic: Check if the lowest point in the last 10 candles is higher than the lowest point since exit
    relevant_since_exit = [c for c in candles if c[0]/1000 > guard.exit_time]
    if len(relevant_since_exit) < 6: return False
    
    lowest_since_exit = min(c[3] for c in relevant_since_exit)
    current_lowest_window = min(c[3] for c in relevant_since_exit[-3:])
    
    if current_lowest_window > lowest_since_exit * 1.0005: # 0.05% higher
        return True
    return False

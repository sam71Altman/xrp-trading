import time
import logging

logger = logging.getLogger(__name__)

class PostExitGuard:
    _instance = None
    MAX_BLOCK_DURATION = 3600  # 1 hour

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.active = False
        self.exit_price = None
        self.exit_time = None

    def record_exit(self, exit_price):
        self.active = True
        self.exit_price = exit_price
        self.exit_time = time.time()
        logger.info(f"[PEG] Exit recorded @ {exit_price}")

    def clear(self, reason):
        if self.active:
            logger.info(f"[PEG] Cleared | reason={reason}")
        self.active = False
        self.exit_price = None
        self.exit_time = None

    def expired(self):
        if not self.active or not self.exit_time:
            return False
        return (time.time() - self.exit_time) > self.MAX_BLOCK_DURATION

class EntryGateMonitor:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.stats = {
            "clean_entries": 0,
            "lpem_only_blocks": 0,
            "peg_only_blocks": 0
        }

    def record_decision(self, lpem_blocked, peg_blocked, entered):
        if entered:
            self.stats["clean_entries"] += 1
        elif lpem_blocked:
            self.stats["lpem_only_blocks"] += 1
        elif peg_blocked:
            self.stats["peg_only_blocks"] += 1

def market_recovered(guard, current_price, candles, ema20, ema50):
    # 1. Breakout after exit
    last_high = get_pullback_high_since_exit(guard, candles)
    if last_high and current_price > last_high:
        return True, "breakout"

    # 2. Higher Low confirmed
    if higher_low_confirmed(guard, candles):
        return True, "higher_low"

    # 3. EMA recovery
    if ema20 > ema50:
        return True, "ema_recovery"

    # 4. Strong immediate recovery
    lowest = get_lowest_price_since_exit(guard, candles)
    if lowest:
        recovery_pct = (current_price - lowest) / lowest * 100
        if recovery_pct >= 0.15: # Standard for scalping
            return True, "strong_recovery"

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
    relevant_since_exit = [c for c in candles if c[0]/1000 > guard.exit_time]
    if len(relevant_since_exit) < 6: return False
    
    lowest_since_exit = min(c[3] for c in relevant_since_exit)
    current_lowest_window = min(c[3] for c in relevant_since_exit[-3:])
    
    if current_lowest_window > lowest_since_exit * 1.0005: # 0.05% higher
        return True
    return False

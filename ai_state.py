"""
AI State Management - Production Ready
Handles mode, weight, counters, cooldowns, and limits per engine instance.
"""
from typing import Dict, Optional
from enum import Enum
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


class AIMode(Enum):
    OFF = "OFF"
    LEARN = "LEARN"
    FULL = "FULL"


class AIWeight(Enum):
    OFF = 0.0
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 1.0


@dataclass
class AIState:
    mode: AIMode = AIMode.OFF
    weight: AIWeight = AIWeight.MEDIUM
    
    daily_interventions: int = 0
    daily_limit: int = 50
    last_reset_day: int = field(default_factory=lambda: time.gmtime().tm_yday)
    
    cooldown_seconds: int = 30
    last_trade_time: Dict[str, float] = field(default_factory=dict)
    
    def set_mode(self, mode: AIMode) -> None:
        self.mode = mode
        logger.info(f"[AI] Mode changed to: {mode.value}")
    
    def set_weight(self, weight: AIWeight) -> None:
        self.weight = weight
        logger.info(f"[AI] Weight changed to: {weight.value}")
    
    def set_daily_limit(self, limit: int) -> None:
        if limit < 1:
            limit = 1
        self.daily_limit = limit
        logger.info(f"[AI] Daily limit changed to: {limit}")
    
    def record_intervention(self) -> None:
        self._check_daily_reset()
        self.daily_interventions += 1
        logger.info(f"[AI] Intervention recorded: {self.daily_interventions}/{self.daily_limit}")
    
    def is_limit_reached(self) -> bool:
        self._check_daily_reset()
        return self.daily_interventions >= self.daily_limit
    
    def _check_daily_reset(self) -> None:
        current_day = time.gmtime().tm_yday
        if current_day != self.last_reset_day:
            self.daily_interventions = 0
            self.last_reset_day = current_day
            logger.info("[AI] Daily counters reset")
    
    def is_cooldown_active(self, symbol: str) -> bool:
        last_time = self.last_trade_time.get(symbol)
        if last_time is None:
            return False
        elapsed = time.time() - last_time
        return elapsed < self.cooldown_seconds
    
    def get_cooldown_remaining(self, symbol: str) -> int:
        last_time = self.last_trade_time.get(symbol)
        if last_time is None:
            return 0
        elapsed = time.time() - last_time
        remaining = self.cooldown_seconds - elapsed
        return max(0, int(remaining))
    
    def record_trade(self, symbol: str) -> None:
        self.last_trade_time[symbol] = time.time()
    
    def get_status(self) -> Dict:
        self._check_daily_reset()
        return {
            "mode": self.mode.value,
            "weight": self.weight.value,
            "daily_interventions": self.daily_interventions,
            "daily_limit": self.daily_limit,
            "limit_reached": self.is_limit_reached(),
            "cooldown_seconds": self.cooldown_seconds,
            "active_cooldowns": {
                k: self.get_cooldown_remaining(k) 
                for k, v in self.last_trade_time.items() 
                if self.is_cooldown_active(k)
            }
        }

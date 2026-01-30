"""
Trading Engine - Single Entry Path
All trades go through check_and_execute_trade() only.
Uses Dependency Injection - no monkey patching.
"""
from typing import Callable, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass
import logging

from ai_state import AIState, AIMode, AIWeight
from ai_filter import SimpleAIFilter, MarketData

logger = logging.getLogger(__name__)


class TradeDecision(Enum):
    ALLOWED = "ALLOWED"
    BLOCKED_LOW_SCORE = "BLOCKED_LOW_SCORE"
    BLOCKED_COOLDOWN = "BLOCKED_COOLDOWN"
    BLOCKED_SYSTEM_ERROR = "BLOCKED_SYSTEM_ERROR"
    ALLOWED_LIMIT_FALLBACK = "ALLOWED_LIMIT_FALLBACK"
    ALLOWED_LEARN_MODE = "ALLOWED_LEARN_MODE"
    ALLOWED_OFF_MODE = "ALLOWED_OFF_MODE"


@dataclass
class TradeResult:
    decision: TradeDecision
    score: Optional[float]
    weight: float
    executed: bool
    details: str


class TradingEngine:
    
    def __init__(
        self,
        execute_trade_fn: Callable[[str, str, float], bool],
        get_market_data_fn: Callable[[str], Optional[MarketData]]
    ):
        self.execute_trade_fn = execute_trade_fn
        self.get_market_data_fn = get_market_data_fn
        self.ai_state = AIState()
        self.ai_filter = SimpleAIFilter()
    
    def check_and_execute_trade(
        self,
        symbol: str,
        direction: str,
        amount: float,
        original_conditions_met: bool
    ) -> TradeResult:
        
        if not original_conditions_met:
            return TradeResult(
                decision=TradeDecision.BLOCKED_LOW_SCORE,
                score=None,
                weight=self.ai_state.weight.value,
                executed=False,
                details="Original conditions not met"
            )
        
        if self.ai_state.is_cooldown_active(symbol):
            remaining = self.ai_state.get_cooldown_remaining(symbol)
            self._log_decision(symbol, None, TradeDecision.BLOCKED_COOLDOWN)
            return TradeResult(
                decision=TradeDecision.BLOCKED_COOLDOWN,
                score=None,
                weight=self.ai_state.weight.value,
                executed=False,
                details=f"Cooldown active: {remaining}s remaining"
            )
        
        if self.ai_state.mode == AIMode.OFF:
            return self._execute_with_result(
                symbol, direction, amount,
                TradeDecision.ALLOWED_OFF_MODE, None, "AI OFF"
            )
        
        market_data = self.get_market_data_fn(symbol)
        if market_data is None:
            self._log_decision(symbol, None, TradeDecision.BLOCKED_SYSTEM_ERROR)
            return TradeResult(
                decision=TradeDecision.BLOCKED_SYSTEM_ERROR,
                score=None,
                weight=self.ai_state.weight.value,
                executed=False,
                details="Failed to get market data"
            )
        
        score = self.ai_filter.calculate_score(market_data)
        
        if score is None:
            self._log_decision(symbol, None, TradeDecision.BLOCKED_SYSTEM_ERROR)
            return TradeResult(
                decision=TradeDecision.BLOCKED_SYSTEM_ERROR,
                score=None,
                weight=self.ai_state.weight.value,
                executed=False,
                details="Score calculation error"
            )
        
        if self.ai_state.mode == AIMode.LEARN:
            self._log_decision(symbol, score, TradeDecision.ALLOWED_LEARN_MODE)
            return self._execute_with_result(
                symbol, direction, amount,
                TradeDecision.ALLOWED_LEARN_MODE, score, "LEARN mode - analysis only"
            )
        
        weight = self.ai_state.weight.value
        
        if self.ai_state.is_limit_reached():
            self._log_decision(symbol, score, TradeDecision.ALLOWED_LIMIT_FALLBACK)
            return self._execute_with_result(
                symbol, direction, amount,
                TradeDecision.ALLOWED_LIMIT_FALLBACK, score, "Daily limit reached - fallback"
            )
        
        if score >= weight:
            self._log_decision(symbol, score, TradeDecision.ALLOWED)
            return self._execute_with_result(
                symbol, direction, amount,
                TradeDecision.ALLOWED, score, f"Score {score} >= Weight {weight}"
            )
        else:
            self.ai_state.record_intervention()
            self._log_decision(symbol, score, TradeDecision.BLOCKED_LOW_SCORE)
            return TradeResult(
                decision=TradeDecision.BLOCKED_LOW_SCORE,
                score=score,
                weight=weight,
                executed=False,
                details=f"Score {score} < Weight {weight}"
            )
    
    def _execute_with_result(
        self,
        symbol: str,
        direction: str,
        amount: float,
        decision: TradeDecision,
        score: Optional[float],
        details: str
    ) -> TradeResult:
        try:
            executed = self.execute_trade_fn(symbol, direction, amount)
            if executed:
                self.ai_state.record_trade(symbol)
            return TradeResult(
                decision=decision,
                score=score,
                weight=self.ai_state.weight.value,
                executed=executed,
                details=details
            )
        except Exception as e:
            logger.error(f"[TRADE] Execution error: {e}")
            return TradeResult(
                decision=TradeDecision.BLOCKED_SYSTEM_ERROR,
                score=score,
                weight=self.ai_state.weight.value,
                executed=False,
                details=f"Execution error: {e}"
            )
    
    def _log_decision(
        self,
        symbol: str,
        score: Optional[float],
        decision: TradeDecision
    ) -> None:
        weight = self.ai_state.weight.value
        score_str = f"{score:.3f}" if score is not None else "N/A"
        logger.info(f"[AI] {symbol} | score={score_str} | weight={weight} | {decision.value}")
    
    def set_mode(self, mode: AIMode) -> None:
        self.ai_state.set_mode(mode)
    
    def set_weight(self, weight: AIWeight) -> None:
        self.ai_state.set_weight(weight)
    
    def set_daily_limit(self, limit: int) -> None:
        self.ai_state.set_daily_limit(limit)
    
    def get_status(self) -> Dict[str, Any]:
        return self.ai_state.get_status()

"""
Trading Engine - Single Entry Path
All trades go through check_and_execute_trade() only.
Uses Dependency Injection - no monkey patching.
"""
from typing import Callable, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass
import logging
import asyncio

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

    QUICK_DOWN_TP = 0.10
    QUICK_DOWN_SL = 0.12
    
    def __init__(
        self,
        execute_trade_fn: Callable[[str, str, float], bool],
        get_market_data_fn: Callable[[str], Optional[MarketData]],
        broker=None,
        telegram=None
    ):
        self.execute_trade_fn = execute_trade_fn
        self.get_market_data_fn = get_market_data_fn
        self.ai_state = AIState()
        self.ai_filter = SimpleAIFilter()
        self.broker = broker
        self.telegram = telegram

        self._trade_lock = asyncio.Lock()
        self._trade_queue = asyncio.Queue(maxsize=1)

        self._position_open = False
        self._position_symbol = None
        self._entry_price = 0.0
        self._position_version = 0

        self._last_trade_id = None
        self._pipeline_task = None

        self.fast_submode = None
        self._closing = False
    
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
            self._log_decision(symbol, None, TradeDecision.ALLOWED_OFF_MODE)
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

    async def request_trade(self, signal):
        """
        Wait briefly instead of dropping immediately.
        Prevents silent signal loss.
        """
        try:
            async with asyncio.timeout(1.0):
                await self._trade_queue.put(signal)
                return True
        except asyncio.TimeoutError:
            return False

    async def _trade_pipeline(self):
        while True:
            try:
                try:
                    async with asyncio.timeout(1.0):
                        signal = await self._trade_queue.get()
                except asyncio.TimeoutError:
                    continue

                await self._execute_trade_atomically(signal)

                self._trade_queue.task_done()

            except Exception:
                logger.exception("[Pipeline crash prevented]")

    async def _execute_trade_atomically(self, signal):
        try:
            async with asyncio.timeout(0.5):
                async with self._trade_lock:
                    return await self._execute_under_lock(signal)
        except asyncio.TimeoutError:
            return False

    async def _execute_under_lock(self, signal):
        signal_type = signal.type if hasattr(signal, 'type') else signal.get("type")
        
        if signal_type == "OPEN":
            if self._position_open:
                return False

            symbol = signal.symbol if hasattr(signal, 'symbol') else signal.get("symbol")
            amount = signal.amount if hasattr(signal, 'amount') else signal.get("amount", 0)

            try:
                async with asyncio.timeout(3.0):
                    order = await self.broker.order(
                        symbol,
                        "BUY",
                        amount
                    )
            except asyncio.TimeoutError:
                return False

            self._position_open = True
            self._position_symbol = symbol
            self._entry_price = order.price
            self._position_version += 1

            await self._notify_trade_once(order)

            return True

        elif signal_type == "CLOSE":
            if self._closing or not self._position_open:
                return False

            self._closing = True

            try:
                async with asyncio.timeout(3.0):
                    order = await self.broker.order(
                        self._position_symbol,
                        "SELL",
                        signal.amount if hasattr(signal, 'amount') else signal.get("amount", 0)
                    )

                self._position_open = False
                self._position_symbol = None
                self._entry_price = 0.0
                self._position_version += 1

                await self._notify_trade_once(order)

                return True
            except asyncio.TimeoutError:
                return False
            finally:
                self._closing = False

        return False

    async def _notify_trade_once(self, order):
        if order.id == self._last_trade_id:
            return

        message = f"{order.side} {order.symbol} @ {order.price}"

        try:
            await self.telegram.send(message)
            self._last_trade_id = order.id
        except Exception:
            logger.exception("Telegram send failed")

    async def start_trading_core(self):
        if not self._pipeline_task:
            self._pipeline_task = asyncio.create_task(self._trade_pipeline())

    async def _check_quick_down_exit(self, price: float):
        """
        STRICT TP/SL exit ONLY for FAST_SCALP DOWN submode.
        TP = +0.10%, SL = -0.12%
        Hard exit only - no trailing, no continuation.
        """
        if self.fast_submode != "DOWN":
            return

        if not self._position_open:
            return

        if self._entry_price <= 0:
            return

        pnl = (price - self._entry_price) / self._entry_price * 100

        if pnl >= self.QUICK_DOWN_TP:
            logger.info(f"[DOWN EXIT] TP hit: {pnl:.4f}% >= {self.QUICK_DOWN_TP}%")
            await self.request_trade({"type": "CLOSE"})
            return

        if pnl <= -self.QUICK_DOWN_SL:
            logger.info(f"[DOWN EXIT] SL hit: {pnl:.4f}% <= -{self.QUICK_DOWN_SL}%")
            await self.request_trade({"type": "CLOSE"})
            return

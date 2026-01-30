"""
AI Integration - Connects new AI filter system with existing bot
Production-ready adapter layer
"""
from typing import Optional, Dict, Callable
import logging
import time

from ai_state import AIState, AIMode, AIWeight
from ai_filter import SimpleAIFilter, MarketData
from trading_engine import TradingEngine, TradeDecision, TradeResult

logger = logging.getLogger(__name__)

_global_engine: Optional[TradingEngine] = None


def create_market_data_from_analysis(analysis: Dict, candles: list) -> Optional[MarketData]:
    try:
        if not analysis or not candles or len(candles) < 20:
            return None
        
        closes = [c['close'] for c in candles]
        volumes = [c['volume'] for c in candles]
        
        current_volume = volumes[-1] if volumes else 0
        avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
        
        highs = [c['high'] for c in candles[-14:]]
        lows = [c['low'] for c in candles[-14:]]
        tr_values = [h - l for h, l in zip(highs, lows)]
        current_atr = sum(tr_values) / len(tr_values) if tr_values else 0.001
        
        all_highs = [c['high'] for c in candles[-28:]]
        all_lows = [c['low'] for c in candles[-28:]]
        all_tr = [h - l for h, l in zip(all_highs, all_lows)]
        avg_atr = sum(all_tr) / len(all_tr) if all_tr else current_atr
        
        ema20 = analysis.get('ema20', closes[-1])
        ema50 = analysis.get('ema50', closes[-1])
        
        if ema50 != 0:
            trend_diff = abs(ema20 - ema50) / ema50
            trend_strength = min(1.0, trend_diff * 100)
        else:
            trend_strength = 0.5
        
        rsi = analysis.get('rsi', 50)
        if rsi is None or rsi < 0 or rsi > 100:
            rsi = 50
        
        high = candles[-1]['high']
        low = candles[-1]['low']
        mid = (high + low) / 2 if (high + low) > 0 else 1
        spread = (high - low) / mid if mid > 0 else 0.001
        
        spreads = []
        for c in candles[-20:]:
            h, l = c['high'], c['low']
            m = (h + l) / 2 if (h + l) > 0 else 1
            spreads.append((h - l) / m if m > 0 else 0)
        avg_spread = sum(spreads) / len(spreads) if spreads else spread
        
        if avg_volume <= 0:
            avg_volume = 1
        if avg_atr <= 0:
            avg_atr = 0.001
        if avg_spread <= 0:
            avg_spread = 0.001
        
        return MarketData(
            volume=current_volume,
            avg_volume=avg_volume,
            atr=current_atr,
            avg_atr=avg_atr,
            trend_strength=trend_strength,
            rsi=rsi,
            spread=spread,
            avg_spread=avg_spread
        )
    except Exception as e:
        logger.error(f"[AI INTEGRATION] Error creating market data: {e}")
        return None


def init_ai_engine(execute_trade_fn: Callable) -> TradingEngine:
    global _global_engine
    
    def get_market_data_placeholder(symbol: str) -> Optional[MarketData]:
        return None
    
    _global_engine = TradingEngine(
        execute_trade_fn=execute_trade_fn,
        get_market_data_fn=get_market_data_placeholder
    )
    
    logger.info("[AI ENGINE] Initialized successfully")
    return _global_engine


def get_ai_engine() -> Optional[TradingEngine]:
    return _global_engine


def check_ai_filter(
    symbol: str,
    analysis: Dict,
    candles: list,
    original_conditions_met: bool = True
) -> TradeResult:
    global _global_engine
    
    if _global_engine is None:
        logger.warning("[AI FILTER] Engine not initialized, allowing trade")
        return TradeResult(
            decision=TradeDecision.ALLOWED_OFF_MODE,
            score=None,
            weight=0.0,
            executed=False,
            details="Engine not initialized"
        )
    
    if _global_engine.ai_state.mode == AIMode.OFF:
        return TradeResult(
            decision=TradeDecision.ALLOWED_OFF_MODE,
            score=None,
            weight=_global_engine.ai_state.weight.value,
            executed=False,
            details="AI OFF"
        )
    
    if _global_engine.ai_state.is_cooldown_active(symbol):
        remaining = _global_engine.ai_state.get_cooldown_remaining(symbol)
        logger.info(f"[AI] {symbol} | BLOCKED_COOLDOWN | remaining={remaining}s")
        return TradeResult(
            decision=TradeDecision.BLOCKED_COOLDOWN,
            score=None,
            weight=_global_engine.ai_state.weight.value,
            executed=False,
            details=f"Cooldown: {remaining}s"
        )
    
    market_data = create_market_data_from_analysis(analysis, candles)
    if market_data is None:
        logger.warning(f"[AI] {symbol} | BLOCKED_SYSTEM_ERROR | Failed to create market data")
        return TradeResult(
            decision=TradeDecision.BLOCKED_SYSTEM_ERROR,
            score=None,
            weight=_global_engine.ai_state.weight.value,
            executed=False,
            details="Failed to create market data"
        )
    
    score = _global_engine.ai_filter.calculate_score(market_data)
    
    if score is None:
        logger.warning(f"[AI] {symbol} | BLOCKED_SYSTEM_ERROR | Score calculation failed")
        return TradeResult(
            decision=TradeDecision.BLOCKED_SYSTEM_ERROR,
            score=None,
            weight=_global_engine.ai_state.weight.value,
            executed=False,
            details="Score calculation error"
        )
    
    weight = _global_engine.ai_state.weight.value
    mode = _global_engine.ai_state.mode
    
    if mode == AIMode.LEARN:
        logger.info(f"[AI] {symbol} | score={score:.3f} | weight={weight} | ALLOWED_LEARN_MODE")
        return TradeResult(
            decision=TradeDecision.ALLOWED_LEARN_MODE,
            score=score,
            weight=weight,
            executed=False,
            details="LEARN mode"
        )
    
    if _global_engine.ai_state.is_limit_reached():
        logger.info(f"[AI] {symbol} | score={score:.3f} | weight={weight} | ALLOWED_LIMIT_FALLBACK")
        return TradeResult(
            decision=TradeDecision.ALLOWED_LIMIT_FALLBACK,
            score=score,
            weight=weight,
            executed=False,
            details="Daily limit fallback"
        )
    
    if score >= weight:
        logger.info(f"[AI] {symbol} | score={score:.3f} | weight={weight} | ALLOWED")
        return TradeResult(
            decision=TradeDecision.ALLOWED,
            score=score,
            weight=weight,
            executed=False,
            details=f"Score {score} >= Weight {weight}"
        )
    else:
        _global_engine.ai_state.record_intervention()
        logger.info(f"[AI] {symbol} | score={score:.3f} | weight={weight} | BLOCKED_LOW_SCORE")
        return TradeResult(
            decision=TradeDecision.BLOCKED_LOW_SCORE,
            score=score,
            weight=weight,
            executed=False,
            details=f"Score {score} < Weight {weight}"
        )


def record_trade_executed(symbol: str):
    global _global_engine
    if _global_engine:
        _global_engine.ai_state.record_trade(symbol)


def set_ai_mode(mode_str: str) -> str:
    global _global_engine
    if _global_engine is None:
        return "Engine not initialized"
    
    mode_map = {"OFF": AIMode.OFF, "LEARN": AIMode.LEARN, "FULL": AIMode.FULL}
    mode = mode_map.get(mode_str.upper())
    if mode is None:
        return f"Invalid mode: {mode_str}"
    
    _global_engine.set_mode(mode)
    return f"Mode set to: {mode.value}"


def set_ai_weight(weight_str: str) -> str:
    global _global_engine
    if _global_engine is None:
        return "Engine not initialized"
    
    weight_map = {
        "OFF": AIWeight.OFF, "0": AIWeight.OFF, "0.0": AIWeight.OFF,
        "LOW": AIWeight.LOW, "0.3": AIWeight.LOW,
        "MEDIUM": AIWeight.MEDIUM, "0.6": AIWeight.MEDIUM,
        "HIGH": AIWeight.HIGH, "1": AIWeight.HIGH, "1.0": AIWeight.HIGH
    }
    weight = weight_map.get(weight_str.upper())
    if weight is None:
        return f"Invalid weight: {weight_str}"
    
    _global_engine.set_weight(weight)
    return f"Weight set to: {weight.name} ({weight.value})"


def set_ai_limit(limit: int) -> str:
    global _global_engine
    if _global_engine is None:
        return "Engine not initialized"
    
    _global_engine.set_daily_limit(limit)
    return f"Daily limit set to: {limit}"


def get_ai_status() -> Dict:
    global _global_engine
    if _global_engine is None:
        return {"error": "Engine not initialized"}
    
    return _global_engine.get_status()


def is_trade_allowed(result: TradeResult) -> bool:
    allowed_decisions = [
        TradeDecision.ALLOWED,
        TradeDecision.ALLOWED_OFF_MODE,
        TradeDecision.ALLOWED_LEARN_MODE,
        TradeDecision.ALLOWED_LIMIT_FALLBACK
    ]
    return result.decision in allowed_decisions

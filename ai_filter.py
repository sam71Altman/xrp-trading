"""
AI Filter - Simple Scalping Filter
Calculates score based on: Volume, ATR, Trend, RSI, Spread
Returns None on any error for safety.
"""
from typing import Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    volume: float
    avg_volume: float
    atr: float
    avg_atr: float
    trend_strength: float
    rsi: float
    spread: float
    avg_spread: float


class SimpleAIFilter:
    
    WEIGHTS = {
        "volume": 0.25,
        "atr": 0.20,
        "trend": 0.20,
        "rsi": 0.20,
        "spread": 0.15
    }
    
    def calculate_score(self, data: MarketData) -> Optional[float]:
        try:
            if not self._validate_data(data):
                logger.error("[AI FILTER] Invalid market data")
                return None
            
            volume_score = self._score_volume(data.volume, data.avg_volume)
            atr_score = self._score_atr(data.atr, data.avg_atr)
            trend_score = self._score_trend(data.trend_strength)
            rsi_score = self._score_rsi(data.rsi)
            spread_score = self._score_spread(data.spread, data.avg_spread)
            
            if None in [volume_score, atr_score, trend_score, rsi_score, spread_score]:
                logger.error("[AI FILTER] Score calculation failed")
                return None
            
            total = (
                volume_score * self.WEIGHTS["volume"] +
                atr_score * self.WEIGHTS["atr"] +
                trend_score * self.WEIGHTS["trend"] +
                rsi_score * self.WEIGHTS["rsi"] +
                spread_score * self.WEIGHTS["spread"]
            )
            
            final_score = max(0.0, min(1.0, total))
            return round(final_score, 3)
            
        except Exception as e:
            logger.error(f"[AI FILTER] Error calculating score: {e}")
            return None
    
    def _validate_data(self, data: MarketData) -> bool:
        try:
            if data.avg_volume <= 0 or data.avg_atr <= 0 or data.avg_spread <= 0:
                return False
            if not (0 <= data.rsi <= 100):
                return False
            if not (0 <= data.trend_strength <= 1):
                return False
            return True
        except Exception:
            return False
    
    def _score_volume(self, volume: float, avg_volume: float) -> Optional[float]:
        try:
            ratio = volume / avg_volume
            if ratio >= 1.5:
                return 1.0
            elif ratio >= 1.0:
                return 0.7
            elif ratio >= 0.5:
                return 0.4
            else:
                return 0.2
        except Exception:
            return None
    
    def _score_atr(self, atr: float, avg_atr: float) -> Optional[float]:
        try:
            ratio = atr / avg_atr
            if 0.8 <= ratio <= 1.5:
                return 1.0
            elif 0.5 <= ratio <= 2.0:
                return 0.6
            else:
                return 0.3
        except Exception:
            return None
    
    def _score_trend(self, strength: float) -> Optional[float]:
        try:
            if strength >= 0.7:
                return 1.0
            elif strength >= 0.4:
                return 0.7
            elif strength >= 0.2:
                return 0.4
            else:
                return 0.2
        except Exception:
            return None
    
    def _score_rsi(self, rsi: float) -> Optional[float]:
        try:
            if 40 <= rsi <= 60:
                return 1.0
            elif 30 <= rsi <= 70:
                return 0.7
            elif 20 <= rsi <= 80:
                return 0.4
            else:
                return 0.2
        except Exception:
            return None
    
    def _score_spread(self, spread: float, avg_spread: float) -> Optional[float]:
        try:
            ratio = spread / avg_spread
            if ratio <= 1.0:
                return 1.0
            elif ratio <= 1.5:
                return 0.7
            elif ratio <= 2.0:
                return 0.4
            else:
                return 0.2
        except Exception:
            return None

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
    volume: float = 1
    avg_volume: float = 1
    atr: float = 1
    avg_atr: float = 1
    trend_strength: float = 0.5
    rsi: float = 50
    spread: float = 1
    avg_spread: float = 1


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

            logger.info(f"[AI INPUT] vol={data.volume} avg_vol={data.avg_volume} atr={data.atr} avg_atr={data.avg_atr} trend={data.trend_strength} rsi={data.rsi} spread={data.spread} avg_spread={data.avg_spread}")

            if not self._validate_data(data):
                logger.warning("[AI FILTER] validation failed — using fallback score")
                return 0.5

            volume_score = self._score_volume(data.volume, data.avg_volume)
            atr_score = self._score_atr(data.atr, data.avg_atr)
            trend_score = self._score_trend(max(0.01, data.trend_strength))
            rsi_score = self._score_rsi(data.rsi)
            spread_score = self._score_spread(data.spread, data.avg_spread)

            logger.info(f"[AI PARTS] vol={volume_score} atr={atr_score} trend={trend_score} rsi={rsi_score} spread={spread_score}")

            scores = [volume_score, atr_score, trend_score, rsi_score, spread_score]

            if any(s is None for s in scores):

                logger.warning("[AI FILTER] partial None score — using fallback")

                return 0.5

            total = (
                volume_score * self.WEIGHTS["volume"] +
                atr_score * self.WEIGHTS["atr"] +
                trend_score * self.WEIGHTS["trend"] +
                rsi_score * self.WEIGHTS["rsi"] +
                spread_score * self.WEIGHTS["spread"]
            )

            final_score = max(0.05, min(1.0, total))

            logger.info(f"[AI FINAL SCORE] {final_score}")

            return round(final_score, 3)

        except Exception as e:

            logger.error(f"[AI FILTER ERROR] {e}")

            return 0.5
    
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

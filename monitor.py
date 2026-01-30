"""
System Monitor - Health Check and Validation
"""
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

from ai_state import AIState, AIMode, AIWeight
from ai_filter import SimpleAIFilter, MarketData
from trading_engine import TradingEngine

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    checks: List[Tuple[str, bool, str]]


class SystemMonitor:
    
    def __init__(self, engine: TradingEngine):
        self.engine = engine
    
    def get_status(self) -> Dict[str, Any]:
        ai_status = self.engine.get_status()
        validation = self.validate_system()
        
        return {
            "ai": ai_status,
            "system_healthy": validation.passed,
            "validation_checks": [
                {"name": name, "passed": passed, "message": msg}
                for name, passed, msg in validation.checks
            ]
        }
    
    def validate_system(self) -> ValidationResult:
        checks = []
        
        checks.append(self._check_mode_valid())
        checks.append(self._check_weight_valid())
        checks.append(self._check_filter_functional())
        checks.append(self._check_no_global_state())
        checks.append(self._check_single_entry_path())
        
        all_passed = all(passed for _, passed, _ in checks)
        return ValidationResult(passed=all_passed, checks=checks)
    
    def _check_mode_valid(self) -> Tuple[str, bool, str]:
        try:
            mode = self.engine.ai_state.mode
            valid = isinstance(mode, AIMode)
            return ("mode_valid", valid, f"Mode: {mode.value}" if valid else "Invalid mode type")
        except Exception as e:
            return ("mode_valid", False, str(e))
    
    def _check_weight_valid(self) -> Tuple[str, bool, str]:
        try:
            weight = self.engine.ai_state.weight
            valid = isinstance(weight, AIWeight)
            valid_values = [0.0, 0.3, 0.6, 1.0]
            value_valid = weight.value in valid_values
            return ("weight_valid", valid and value_valid, f"Weight: {weight.value}")
        except Exception as e:
            return ("weight_valid", False, str(e))
    
    def _check_filter_functional(self) -> Tuple[str, bool, str]:
        try:
            test_data = MarketData(
                volume=100.0,
                avg_volume=100.0,
                atr=0.01,
                avg_atr=0.01,
                trend_strength=0.5,
                rsi=50.0,
                spread=0.001,
                avg_spread=0.001
            )
            score = self.engine.ai_filter.calculate_score(test_data)
            valid = score is not None and 0.0 <= score <= 1.0
            return ("filter_functional", valid, f"Test score: {score}")
        except Exception as e:
            return ("filter_functional", False, str(e))
    
    def _check_no_global_state(self) -> Tuple[str, bool, str]:
        try:
            state1 = self.engine.ai_state
            state2 = AIState()
            
            state1.set_mode(AIMode.FULL)
            state2.set_mode(AIMode.OFF)
            
            independent = state1.mode != state2.mode
            return ("no_global_state", independent, "States are independent" if independent else "DANGER: Shared state detected")
        except Exception as e:
            return ("no_global_state", False, str(e))
    
    def _check_single_entry_path(self) -> Tuple[str, bool, str]:
        try:
            has_execute = callable(getattr(self.engine, 'execute_trade_fn', None))
            has_check = callable(getattr(self.engine, 'check_and_execute_trade', None))
            
            execute_private = not hasattr(TradingEngine, 'execute_trade')
            
            valid = has_execute and has_check and execute_private
            return ("single_entry_path", valid, "check_and_execute_trade is the only entry point")
        except Exception as e:
            return ("single_entry_path", False, str(e))
    
    def run_diagnostics(self) -> str:
        validation = self.validate_system()
        
        lines = [
            "🔍 تشخيص النظام",
            "═" * 40
        ]
        
        for name, passed, msg in validation.checks:
            emoji = "✅" if passed else "❌"
            lines.append(f"{emoji} {name}: {msg}")
        
        lines.append("═" * 40)
        
        if validation.passed:
            lines.append("✅ النظام يعمل بشكل صحيح")
        else:
            lines.append("❌ يوجد مشاكل تحتاج إصلاح")
        
        return "\n".join(lines)

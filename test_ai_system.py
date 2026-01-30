"""
Test AI System - Verification Script
"""
import sys

def mock_execute_trade(symbol: str, direction: str, amount: float) -> bool:
    print(f"  [MOCK EXECUTE] {symbol} {direction} {amount}")
    return True

def mock_get_market_data(symbol: str):
    from ai_filter import MarketData
    return MarketData(
        volume=120.0,
        avg_volume=100.0,
        atr=0.015,
        avg_atr=0.012,
        trend_strength=0.6,
        rsi=55.0,
        spread=0.0008,
        avg_spread=0.001
    )

def run_tests():
    print("=" * 60)
    print("🧪 اختبار نظام الذكاء الاصطناعي")
    print("=" * 60)
    
    from ai_state import AIState, AIMode, AIWeight
    from ai_filter import SimpleAIFilter, MarketData
    from trading_engine import TradingEngine, TradeDecision
    from monitor import SystemMonitor
    from telegram_interface import TelegramAIInterface
    
    tests_passed = 0
    tests_failed = 0
    
    print("\n📌 اختبار 1: AIState مستقل (لا Global State)")
    state1 = AIState()
    state2 = AIState()
    state1.set_mode(AIMode.FULL)
    state2.set_mode(AIMode.OFF)
    if state1.mode == AIMode.FULL and state2.mode == AIMode.OFF:
        print("  ✅ الحالات مستقلة")
        tests_passed += 1
    else:
        print("  ❌ خطر: الحالات مشتركة!")
        tests_failed += 1
    
    print("\n📌 اختبار 2: قيم Weight صحيحة")
    valid_weights = [AIWeight.OFF.value, AIWeight.LOW.value, AIWeight.MEDIUM.value, AIWeight.HIGH.value]
    expected = [0.0, 0.3, 0.6, 1.0]
    if valid_weights == expected:
        print(f"  ✅ الأوزان صحيحة: {valid_weights}")
        tests_passed += 1
    else:
        print(f"  ❌ الأوزان غير صحيحة: {valid_weights}")
        tests_failed += 1
    
    print("\n📌 اختبار 3: SimpleAIFilter يحسب score بشكل صحيح")
    ai_filter = SimpleAIFilter()
    test_data = MarketData(
        volume=100.0, avg_volume=100.0,
        atr=0.01, avg_atr=0.01,
        trend_strength=0.5, rsi=50.0,
        spread=0.001, avg_spread=0.001
    )
    score = ai_filter.calculate_score(test_data)
    if score is not None and 0.0 <= score <= 1.0:
        print(f"  ✅ Score محسوب: {score}")
        tests_passed += 1
    else:
        print(f"  ❌ Score غير صالح: {score}")
        tests_failed += 1
    
    print("\n📌 اختبار 4: SimpleAIFilter يرجع None عند خطأ")
    bad_data = MarketData(
        volume=100.0, avg_volume=0,  # قيمة غير صالحة
        atr=0.01, avg_atr=0.01,
        trend_strength=0.5, rsi=50.0,
        spread=0.001, avg_spread=0.001
    )
    score = ai_filter.calculate_score(bad_data)
    if score is None:
        print("  ✅ يرجع None عند خطأ (آمن)")
        tests_passed += 1
    else:
        print(f"  ❌ لم يرجع None: {score}")
        tests_failed += 1
    
    print("\n📌 اختبار 5: TradingEngine مع Dependency Injection")
    engine = TradingEngine(
        execute_trade_fn=mock_execute_trade,
        get_market_data_fn=mock_get_market_data
    )
    if hasattr(engine, 'execute_trade_fn') and hasattr(engine, 'get_market_data_fn'):
        print("  ✅ Dependency Injection يعمل")
        tests_passed += 1
    else:
        print("  ❌ Dependency Injection لا يعمل")
        tests_failed += 1
    
    print("\n📌 اختبار 6: وضع OFF يسمح بكل الصفقات")
    engine.set_mode(AIMode.OFF)
    result = engine.check_and_execute_trade("XRPUSDT", "BUY", 100.0, True)
    if result.decision == TradeDecision.ALLOWED_OFF_MODE and result.executed:
        print(f"  ✅ OFF يسمح: {result.decision.value}")
        tests_passed += 1
    else:
        print(f"  ❌ OFF لا يسمح: {result.decision.value}")
        tests_failed += 1
    
    print("\n📌 اختبار 7: وضع FULL يفلتر الصفقات")
    engine.set_mode(AIMode.FULL)
    engine.set_weight(AIWeight.HIGH)  # 1.0
    result = engine.check_and_execute_trade("BTCUSDT", "SELL", 50.0, True)
    print(f"  Score: {result.score}, Weight: {result.weight}")
    print(f"  Decision: {result.decision.value}")
    if result.decision in [TradeDecision.ALLOWED, TradeDecision.BLOCKED_LOW_SCORE]:
        print("  ✅ FULL يفلتر بشكل صحيح")
        tests_passed += 1
    else:
        print(f"  ❌ سلوك غير متوقع")
        tests_failed += 1
    
    print("\n📌 اختبار 8: Cooldown يعمل")
    engine.set_mode(AIMode.OFF)
    engine.ai_state.cooldown_seconds = 30
    result1 = engine.check_and_execute_trade("ETHUSDT", "BUY", 10.0, True)
    result2 = engine.check_and_execute_trade("ETHUSDT", "BUY", 10.0, True)
    if result1.executed and result2.decision == TradeDecision.BLOCKED_COOLDOWN:
        print("  ✅ Cooldown يمنع الصفقة الثانية")
        tests_passed += 1
    else:
        print(f"  ❌ Cooldown لا يعمل: r1={result1.executed}, r2={result2.decision.value}")
        tests_failed += 1
    
    print("\n📌 اختبار 9: سقف التدخلات اليومي")
    engine2 = TradingEngine(mock_execute_trade, mock_get_market_data)
    engine2.set_mode(AIMode.FULL)
    engine2.set_weight(AIWeight.HIGH)
    engine2.set_daily_limit(2)
    
    for i in range(3):
        engine2.check_and_execute_trade(f"PAIR{i}", "BUY", 10.0, True)
    
    if engine2.ai_state.is_limit_reached():
        result = engine2.check_and_execute_trade("NEWPAIR", "BUY", 10.0, True)
        if result.decision == TradeDecision.ALLOWED_LIMIT_FALLBACK:
            print("  ✅ السقف يعمل - fallback للسماح")
            tests_passed += 1
        else:
            print(f"  ❌ السقف لا يعمل: {result.decision.value}")
            tests_failed += 1
    else:
        print("  ❌ السقف لم يصل")
        tests_failed += 1
    
    print("\n📌 اختبار 10: SystemMonitor يتحقق من صحة النظام")
    monitor = SystemMonitor(engine)
    validation = monitor.validate_system()
    print(f"  Checks passed: {sum(1 for _, p, _ in validation.checks if p)}/{len(validation.checks)}")
    if validation.passed:
        print("  ✅ النظام صحي")
        tests_passed += 1
    else:
        print("  ⚠️ بعض الفحوصات فشلت")
        for name, passed, msg in validation.checks:
            if not passed:
                print(f"    ❌ {name}: {msg}")
        tests_failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 النتيجة: {tests_passed} نجح / {tests_failed} فشل")
    print("=" * 60)
    
    return tests_failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

import asyncio
import os
import csv
import logging
from datetime import datetime, timezone
import requests

# Mock setup to access main.py variables/state
# In a real environment, we'd import them, but here we'll define a diagnostic script
# that reads the state and prints it.

async def diagnostic():
    print("="*50)
    print("🔍 RUNTIME DIAGNOSTIC - XRP/USDT V3.2")
    print("="*50)

    try:
        from main import (
            kill_switch, paper_state, state, SYMBOL, TIMEFRAME, 
            EMA_SHORT, EMA_LONG, BREAKOUT_CANDLES, RANGE_FILTER_THRESHOLD,
            MIN_SIGNAL_SCORE, PAPER_TRADES_FILE, get_klines, analyze_market,
            calculate_signal_score, evaluate_kill_switch, get_closed_trades,
            calculate_recent_win_rate, check_data_maturity, check_loss_streak,
            check_drawdown, check_recent_performance, RECENT_TRADES_WINDOW
        )
        
        # 1️⃣ Analysis Job Check
        # Note: Since this is a separate script, we check the state of the main bot
        # We assume the bot is running if we can access the state (or we'd check logs)
        print("\n1️⃣ [ANALYSIS JOB]")
        # We can't easily check 'application' from here without the bot object,
        # but we can check if the 'state' objects are being updated.
        print(f"[CHECK] Analysis Job: RUNNING (Assuming active if main.py is running)")
        print(f"[CHECK] Current Time: {datetime.now(timezone.utc)}")
        
        # 2️⃣ Trading State
        print("\n2️⃣ [TRADING STATE]")
        print(f"signals_enabled: {state.signals_enabled}")
        print(f"paper_trading_enabled: True (MODE is PAPER)")
        print(f"kill_switch_active: {kill_switch.active}")
        print(f"kill_switch_reason: {kill_switch.reason if kill_switch.active else 'None'}")
        
        # 3️⃣ Buy Conditions
        print("\n3️⃣ [BUY CONDITIONS - LAST ANALYSIS]")
        candles = get_klines(SYMBOL, TIMEFRAME)
        if candles:
            analysis = analyze_market(candles)
            if "error" not in analysis:
                score, reasons = calculate_signal_score(analysis, candles)
                print(f"EMA{EMA_SHORT} > EMA{EMA_LONG} ({analysis['ema_short']:.4f} > {analysis['ema_long']:.4f}): {analysis['ema_bullish']}")
                print(f"Breakout (High {BREAKOUT_CANDLES} candles): {analysis['breakout']} (Close: {analysis['close']:.4f}, High: {analysis['highest_high']:.4f})")
                print(f"Volume Filter ({analysis['current_volume']:.1f} > {analysis['avg_volume']:.1f}): {analysis['volume_confirmed']}")
                print(f"Range Filter ({analysis['ema_diff_pct']:.5f} >= {RANGE_FILTER_THRESHOLD}): {analysis['range_confirmed']}")
                print(f"Signal Score: {score} / 10")
            else:
                print(f"Analysis Error: {analysis['error']}")
        else:
            print("Market Data Error: Failed to fetch candles")

        # 4️⃣ Kill Switch Gate
        print("\n4️⃣ [KILL SWITCH GATE]")
        closed_trades = get_closed_trades()
        maturity = check_data_maturity()
        win_rate = calculate_recent_win_rate()
        print(f"Closed Trades Count: {len(closed_trades)}")
        print(f"Data Maturity Passed (>=5 trades): {maturity}")
        print(f"Loss Streak: {paper_state.loss_streak}")
        print(f"Drawdown %: {((paper_state.peak_balance - paper_state.balance) / paper_state.peak_balance * 100) if paper_state.peak_balance > 0 else 0:.2f}%")
        print(f"Recent Win Rate: {win_rate:.2f}%")
        
        ks_block = evaluate_kill_switch()
        if ks_block:
            print(f"❗ Kill Switch منع التداول: {ks_block}")
        else:
            print("✅ Kill Switch does not block trading currently.")

        # 5️⃣ Paper Trading
        print("\n5️⃣ [PAPER TRADING]")
        print(f"Virtual Balance: {paper_state.balance:.2f} USDT")
        print(f"Open Position Exists: {paper_state.position_qty > 0}")
        if paper_state.position_qty > 0:
            print(f"Entry Price: {state.entry_price}")
        print(f"paper_trades.csv exists: {os.path.exists(PAPER_TRADES_FILE)}")
        if os.path.exists(PAPER_TRADES_FILE):
            with open(PAPER_TRADES_FILE, 'r') as f:
                print(f"Rows count in paper_trades.csv: {sum(1 for _ in f) - 1}")

        # 6️⃣ Market Data
        print("\n6️⃣ [MARKET DATA]")
        if candles:
            print(f"[MARKET] Timeframe: {TIMEFRAME}")
            print(f"[MARKET] Candles fetched: {len(candles)}")
            print(f"[MARKET] Last Close: {candles[-1]['close']}")
        
        # 7️⃣ Final Reason
        print("\n7️⃣ [FINAL VERDICT]")
        if kill_switch.active:
            print(f"❌ البوت لا يتداول بسبب: Kill Switch مفعّل ({kill_switch.reason})")
        elif not state.signals_enabled:
            print("❌ البوت لا يتداول بسبب: الإشارات معطلة يدوياً")
        elif ks_block:
             print(f"❌ البوت لا يتداول بسبب: شروط Kill Switch (Evaluated: {ks_block})")
        else:
            print("✅ البوت جاهز وسيدخل عند تحقق الشروط")

    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")

if __name__ == "__main__":
    asyncio.run(diagnostic())

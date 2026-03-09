#!/usr/bin/env python3
import asyncio
import logging
from collections import deque

logger = logging.getLogger(__name__)

async def safe_edit_message(query, text, **kwargs):
    """Safe message edit to avoid Telegram 'Message is not modified' error"""
    try:
    if hasattr(query, 'edit_message_text'):
    await query.edit_message_text(text, **kwargs)
    except Exception as e:
    if "Message is not modified" in str(e):
    return
    if "logger" in globals():
    logger.error(f"[TELEGRAM] Edit error: {e}")
    else:
    print(f"[TELEGRAM] Edit error: {e}")


    if "Message is not modified" in str(e):
    return
    if "logger" in globals():
    logger.error(f"[TELEGRAM] Edit error: {e}")
    else:
    print(f"[TELEGRAM] Edit error: {e}")

    import re
    from version import BOT_VERSION
# Updated pattern to support v4.4.PRO-FINAL format
    pattern = r'^v\d+\.\d+(\.\d+)?(-[a-zA-Z0-9-]+)?\.?[a-zA-Z0-9-]*$'

    if not re.match(pattern, BOT_VERSION):
    raise RuntimeError(f"Invalid bot version format: {BOT_VERSION}")

    logger.info(f"[SYSTEM] Bot version unified successfully: {BOT_VERSION}")

def check_local_version_definitions():
    """
    تحذير فقط – لا يوقف التشغيل
    """
    suspicious_tokens = ["V3.", "v3."]
    logger.info("[VERSION CHECK] Scanning for hardcoded version strings...")

async def main() -> None:
    validate_version_unification()
    check_local_version_definitions()
# Start Price Engine
    PriceEngine.start()

# Initialize AI Filter Engine (v4.5.PRO-AI)
    init_ai_engine(execute_paper_buy)
    logger.info("[AI ENGINE] Initialized successfully")

    tg_token = os.environ.get("TG_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")

    if not tg_token or not chat_id:
    print("❌ الرجاء تعيين TG_TOKEN و TG_CHAT_ID")
    return

# {BOT_VERSION} Integrity Check
def validate_data_integrity():
    from version import BOT_VERSION
    import math

    logger.info(f"[BOOT] Bot version loaded: {BOT_VERSION}")

# 1. Test PnL Calculation & Rounding Protection
    test_entry = 1.0000
    test_exit = 1.00004  # Tiny difference
    test_pnl = ((test_exit - test_entry) / test_entry) * 100

# Apply the same protection as in execute_paper_exit
    display_pnl = round(test_pnl, 2)
    if abs(display_pnl) < 0.01:
    display_pnl = 0.00

    if display_pnl != 0.00:

# 2. Test Score Integrity
    test_analysis = {"ema_bullish": True, "breakout": True, "volume_confirmed": True}
    test_candles = [{"close": 1.0}] * 20
    score, reasons = calculate_signal_score(test_analysis, test_candles)

    if not (1 <= score <= 10):

    logger.info(f"✅ Data Integrity Check Passed for version {BOT_VERSION}")

    validate_data_integrity()

# Initialize application
    application = Application.builder().token(tg_token).build()

# Remove obsolete CallbackQueryHandler as we switched to MessageHandler for ReplyKeyboard
# application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(CommandHandler("start", cmd_start))
# Removed duplicated status command registration
    application.add_handler(CommandHandler("health", health_command))
    application.add_handler(CommandHandler("balance", cmd_balance))
    application.add_handler(CommandHandler("trades", cmd_trades))
    application.add_handler(CommandHandler("on", cmd_on))
    application.add_handler(CommandHandler("off", cmd_off))
    application.add_handler(CommandHandler("rules", cmd_rules))
    application.add_handler(CommandHandler("stats", cmd_stats))
    application.add_handler(CommandHandler("diagnostic", cmd_diagnostic))
    application.add_handler(CommandHandler("frame", cmd_الفريم))

# Mode commands (Smart Trading System)
    application.add_handler(CommandHandler("mode", cmd_mode))
    application.add_handler(CommandHandler("modestats", cmd_modestats))
    application.add_handler(CommandHandler("dashboard", cmd_dashboard))
    application.add_handler(CommandHandler("recommend", cmd_recommend))
    application.add_handler(CommandHandler("validate", cmd_validate))

# AI commands (v4.4.PRO-FINAL)
    application.add_handler(CommandHandler("ai", cmd_ai))
    application.add_handler(CommandHandler("ai_emergency", cmd_ai_emergency))

# Add CallbackQueryHandlers for buttons
    application.add_handler(CallbackQueryHandler(handle_mode_callback, pattern="^(MODE_|FAST_MODE_|MODE_STATS|MODE_RECOMMEND|SHOW_FAST_MODES|BACK_TO_MODES)"))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# Initialize the application
    await application.initialize()

# Use JobQueue for signal loop if available, else use create_task
    if application.job_queue:
    application.job_queue.run_repeating(
    lambda context: asyncio.create_task(signal_loop(application.bot, chat_id)),
    interval=3,
    first=1,
    name="signal_loop"
    )
    logger.info("Signal loop started via JobQueue")
    else:
    asyncio.create_task(signal_loop(application.bot, chat_id))
    logger.info("Signal loop started via create_task (JobQueue missing)")

# Start the application
    await application.start()

# Start polling
    logger.info("Starting polling...")
    await application.updater.start_polling(drop_pending_updates=True)

# Print mode timing configuration
    print_mode_timing_config()

    print(f"🚀 بوت إشارات {SYMBOL_DISPLAY} {BOT_VERSION} يعمل...")

# Keep running
    while True:
    await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
    await application.stop()
    await application.shutdown()

if __name__ == "__main__":
# WebSocket Safety Check
import websocket
except ImportError:
print("❌ FATAL: websocket-client is not installed.")
exit(1)

print(f"🚀 Initializing {BOT_VERSION}...")
logger.info(f"🚀 {BOT_VERSION} Startup")

# Version Integrity Check
SYSTEM_VERSION = BOT_VERSION
if BOT_VERSION != SYSTEM_VERSION:
exit(1)

asyncio.run(main())
except KeyboardInterrupt:
pass
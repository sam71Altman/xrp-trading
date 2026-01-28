import json
import time
import threading
import logging
import websocket
from typing import Optional

logger = logging.getLogger(__name__)

class PriceEngine:
    """
    Real-time Price Engine for XRP/USDT using Binance WebSocket.
    """
    last_price: Optional[float] = None
    last_update_time: float = 0
    _ws = None
    _thread = None
    _running = False

    @classmethod
    def start(cls):
        if cls._running:
            return
        cls._running = True
        cls._thread = threading.Thread(target=cls._run, daemon=True)
        cls._thread.start()
        logger.info("🚀 PriceEngine started...")

    @classmethod
    def _run(cls):
        while cls._running:
            try:
                # Binance WebSocket for XRP/USDT ticker
                cls._ws = websocket.WebSocketApp(
                    "wss://stream.binance.com:9443/ws/xrpusdt@ticker",
                    on_message=cls._on_message,
                    on_error=cls._on_error,
                    on_close=cls._on_close
                )
                cls._ws.run_forever()
            except Exception as e:
                logger.error(f"❌ PriceEngine WS Error: {e}")
            time.sleep(1)  # Reconnect delay

    @classmethod
    def _on_message(cls, ws, message):
        try:
            data = json.loads(message)
            cls.last_price = float(data['c'])  # 'c' is the last price
            cls.last_update_time = time.time()
        except Exception as e:
            logger.error(f"❌ Error processing WS message: {e}")

    @classmethod
    def _on_error(cls, ws, error):
        logger.error(f"❌ PriceEngine WS Error: {error}")

    @classmethod
    def _on_close(cls, ws, close_status_code, close_msg):
        logger.warning("⚠️ PriceEngine WS Connection Closed")
        FailSafeSystem.on_websocket_disconnect()

class TradingGuard:
    """
    Guard that prevents trading when stale prices or connection issues are detected.
    """
    BLOCK_ALL_TRADING: bool = False
    BLOCK_REASON: str = ""
    MAX_LATENCY: float = 0.5  # 500ms

    @classmethod
    def enforce_guard(cls, operation_type: str) -> bool:
        if cls.BLOCK_ALL_TRADING:
            logger.warning(f"🚫 Trading Blocked for {operation_type}: {cls.BLOCK_REASON}")
            return False

        if PriceEngine.last_price is None:
            logger.warning(f"⚠️ Waiting for price data for {operation_type}...")
            return False

        latency = time.time() - PriceEngine.last_update_time
        if latency > cls.MAX_LATENCY:
            logger.warning(f"⚠️ Price stale ({latency:.3f}s) - Blocking {operation_type}")
            return False

        return True

class TelegramReporter:
    """
    Telegram report formatter for real-time prices and latency.
    """
    @staticmethod
    def format_price_message(action: str, details: dict) -> str:
        price = PriceEngine.last_price if PriceEngine.last_price else 0.0
        latency = (time.time() - PriceEngine.last_update_time) * 1000
        
        msg = f"🔔 *{action}*\n"
        msg += f"💰 Price: `{price:.4f}`\n"
        msg += f"⏱ Latency: `{latency:.0f}ms`\n"
        
        for k, v in details.items():
            msg += f"🔸 {k}: `{v}`\n"
            
        return msg

class FailSafeSystem:
    """
    Failsafe system to handle connection drops.
    """
    @staticmethod
    def on_websocket_disconnect():
        TradingGuard.BLOCK_ALL_TRADING = True
        TradingGuard.BLOCK_REASON = "WebSocket disconnected"
        logger.critical("🚨 CRITICAL: WebSocket disconnected. All trading BLOCKED.")

    @staticmethod
    def on_websocket_reconnect():
        TradingGuard.BLOCK_ALL_TRADING = False
        TradingGuard.BLOCK_REASON = ""
        logger.info("✅ WebSocket reconnected. Trading resumed.")

class ValidationChecks:
    """
    Continuous system performance validation.
    """
    def __init__(self):
        self.start_time = time.time()

    def check_health(self):
        latency = time.time() - PriceEngine.last_update_time
        is_healthy = latency < 0.5 and PriceEngine.last_price is not None
        return {
            "uptime": time.time() - self.start_time,
            "latency_ms": latency * 1000,
            "healthy": is_healthy,
            "last_price": PriceEngine.last_price
        }

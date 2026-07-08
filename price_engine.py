import json
import time
import threading
import logging
import websocket
from typing import Optional

logger = logging.getLogger(__name__)


class PriceEngine:
    """
    Real-time Price Engine for XRP/USDT using Binance WebSocket (aggTrade stream).

    SINGLE SOURCE OF TRUTH for the live price. Both main.py and
    trading_engine.py (via TradingGuard) read this class. Do NOT define
    another PriceEngine anywhere else — a duplicate class caused the guard
    to silently block all trades ("Waiting for price data for TRADE").
    """
    last_price: Optional[float] = None
    last_update_time: float = 0
    latency_ms: float = 0
    is_connected: bool = False
    _running = False

    @classmethod
    def update_price(cls, price: float):
        cls.last_price = price
        cls.last_update_time = time.time()
        cls.is_connected = True
        # Auto-resume trading if we were blocked only because the
        # websocket dropped: fresh price data means the feed is back.
        if TradingGuard.BLOCK_ALL_TRADING and TradingGuard.BLOCK_REASON == "WebSocket disconnected":
            FailSafeSystem.on_websocket_reconnect()

    @classmethod
    def on_message(cls, ws, message):
        try:
            data = json.loads(message)
            if 'p' in data:
                price = float(data['p'])
                cls.update_price(price)
                if 'E' in data:
                    cls.latency_ms = (time.time() * 1000) - data['E']
        except Exception as e:
            logger.error(f"PriceEngine error: {e}")

    @classmethod
    def on_error(cls, ws, error):
        logger.error(f"WebSocket Error: {error}")
        cls.is_connected = False
        FailSafeSystem.on_websocket_disconnect()

    @classmethod
    def on_close(cls, ws, close_status_code, close_msg):
        logger.warning("WebSocket Closed")
        cls.is_connected = False
        FailSafeSystem.on_websocket_disconnect()

    @classmethod
    def start(cls):
        if cls._running:
            return
        cls._running = True

        def run():
            while True:
                try:
                    ws_url = "wss://stream.binance.com:9443/ws/xrpusdt@aggTrade"
                    ws = websocket.WebSocketApp(
                        ws_url,
                        on_message=cls.on_message,
                        on_error=cls.on_error,
                        on_close=cls.on_close
                    )
                    ws.run_forever()
                except Exception as e:
                    logger.error(f"WebSocket restart error: {e}")
                time.sleep(5)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        logger.info("🚀 PriceEngine started...")


class TradingGuard:
    """
    Guard that prevents trading when stale prices or connection issues are detected.
    Reads PriceEngine above — the same class the websocket updates.
    """
    BLOCK_ALL_TRADING: bool = False
    BLOCK_REASON: str = ""
    # Binance @ticker stream pushes ~1 update/second, so a 0.5s staleness
    # ceiling spuriously blocked ~half of all real trade attempts
    # ("Price stale (0.8s) - Blocking TRADE"). 2.0s matches the staleness
    # window used by main.py's own TradingGuard.
    MAX_LATENCY_MS: float = 2000

    @classmethod
    def enforce_guard(cls, operation_type: str) -> bool:
        if cls.BLOCK_ALL_TRADING:
            logger.warning(f"Guard Blocked {operation_type}: {cls.BLOCK_REASON}")
            return False

        if PriceEngine.last_price is None:
            logger.warning(f"Guard Blocked {operation_type}: No price data")
            return False

        if (time.time() - PriceEngine.last_update_time) > 2:
            logger.warning(f"Guard Blocked {operation_type}: Stale price (>2s)")
            return False

        if PriceEngine.latency_ms > cls.MAX_LATENCY_MS:
            logger.warning(f"Guard Blocked {operation_type}: High latency ({PriceEngine.latency_ms:.0f}ms)")
            return False

        return True


class FailSafeSystem:
    """
    Failsafe system to handle connection drops.

    A notifier callback (registered by main.py at startup via set_notifier)
    sends Telegram alerts on block/resume. price_engine.py must NEVER import
    anything from main.py — the callback is injected, not imported.
    """
    ALERT_COOLDOWN_SECONDS = 300  # max one disconnect alert per 5 minutes

    _notifier = None            # callable(text: str), thread-safe, injected by main.py
    _last_disconnect_alert_time: float = 0
    _disconnect_alerted: bool = False

    @classmethod
    def set_notifier(cls, notifier):
        cls._notifier = notifier
        # If the WebSocket disconnected before the notifier was registered
        # (which happens when Binance blocks the connection within the first
        # second of startup, before main.py reaches set_notifier), send the
        # disconnect alert now so that the reconnect notification can fire later.
        if TradingGuard.BLOCK_ALL_TRADING and TradingGuard.BLOCK_REASON == "WebSocket disconnected":
            now = time.time()
            if (now - cls._last_disconnect_alert_time) >= cls.ALERT_COOLDOWN_SECONDS:
                cls._last_disconnect_alert_time = now
                cls._disconnect_alerted = True
                cls._notify(
                    "🚨 *تنبيه: انقطاع مصدر الأسعار*\n"
                    "تم إيقاف فتح صفقات جديدة مؤقتاً حتى عودة بيانات الأسعار.\n"
                    "الصفقات المفتوحة تُدار كالمعتاد."
                )

    @classmethod
    def _notify(cls, text: str):
        if cls._notifier is None:
            return
        try:
            cls._notifier(text)
        except Exception as e:
            logger.error(f"FAILSAFE notifier error: {e}")

    @classmethod
    def on_websocket_disconnect(cls):
        TradingGuard.BLOCK_ALL_TRADING = True
        TradingGuard.BLOCK_REASON = "WebSocket disconnected"
        logger.critical("FAILSAFE: Trading Blocked due to connection loss")

        # Only consume the cooldown window when a notifier is registered,
        # so an early disconnect (before main.py registers the callback)
        # doesn't silently swallow the first real alert.
        now = time.time()
        if cls._notifier is not None and (now - cls._last_disconnect_alert_time) >= cls.ALERT_COOLDOWN_SECONDS:
            cls._last_disconnect_alert_time = now
            cls._disconnect_alerted = True
            cls._notify(
                "🚨 *تنبيه: انقطاع مصدر الأسعار*\n"
                "تم إيقاف فتح صفقات جديدة مؤقتاً حتى عودة بيانات الأسعار.\n"
                "الصفقات المفتوحة تُدار كالمعتاد."
            )

    @classmethod
    def on_websocket_reconnect(cls):
        TradingGuard.BLOCK_ALL_TRADING = False
        TradingGuard.BLOCK_REASON = ""
        logger.info("FAILSAFE: Trading Resumed")

        if cls._disconnect_alerted:
            cls._disconnect_alerted = False
            cls._notify(
                "✅ *عاد مصدر الأسعار*\n"
                "تم استئناف التداول تلقائياً."
            )

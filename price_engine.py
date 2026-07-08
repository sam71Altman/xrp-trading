import json
import time
import threading
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

# REST endpoints tried in order on each poll cycle.
# api.binance.us is the only endpoint reachable from Replit servers.
_TICKER_URLS = [
    "https://api.binance.us/api/v3/ticker/price?symbol=XRPUSDT",
    "https://api.binance.com/api/v3/ticker/price?symbol=XRPUSDT",
    "https://api1.binance.com/api/v3/ticker/price?symbol=XRPUSDT",
    "https://api2.binance.com/api/v3/ticker/price?symbol=XRPUSDT",
    "https://api3.binance.com/api/v3/ticker/price?symbol=XRPUSDT",
]

POLL_INTERVAL_SECONDS = 2


class PriceEngine:
    """
    Real-time Price Engine for XRP/USDT using Binance REST API polling.

    Polls every 2 seconds across multiple fallback endpoints.
    SINGLE SOURCE OF TRUTH for the live price. Both main.py and
    trading_engine.py (via TradingGuard) read this class. Do NOT define
    another PriceEngine anywhere else.
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
        # Auto-resume trading if we were blocked only because the feed dropped
        if TradingGuard.BLOCK_ALL_TRADING and TradingGuard.BLOCK_REASON == "Price feed disconnected":
            FailSafeSystem.on_feed_reconnect()

    @classmethod
    def _fetch_price(cls) -> Optional[float]:
        """Try each endpoint in order; return price on first success."""
        for url in _TICKER_URLS:
            try:
                t0 = time.time()
                resp = requests.get(url, timeout=4)
                if resp.status_code == 200:
                    price = float(resp.json()["price"])
                    cls.latency_ms = (time.time() - t0) * 1000
                    return price
            except Exception:
                continue
        return None

    @classmethod
    def start(cls):
        if cls._running:
            return
        cls._running = True

        def run():
            consecutive_failures = 0
            while True:
                try:
                    price = cls._fetch_price()
                    if price is not None:
                        consecutive_failures = 0
                        cls.update_price(price)
                    else:
                        consecutive_failures += 1
                        cls.is_connected = False
                        if consecutive_failures >= 3:
                            FailSafeSystem.on_feed_disconnect()
                except Exception as e:
                    logger.error(f"PriceEngine poll error: {e}")
                time.sleep(POLL_INTERVAL_SECONDS)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        logger.info("🚀 PriceEngine started (REST polling)...")


class TradingGuard:
    """
    Guard that prevents trading when stale prices or connection issues are detected.
    Reads PriceEngine above — the same class the REST poller updates.
    """
    BLOCK_ALL_TRADING: bool = False
    BLOCK_REASON: str = ""
    MAX_LATENCY_MS: float = 5000

    @classmethod
    def enforce_guard(cls, operation_type: str) -> bool:
        if cls.BLOCK_ALL_TRADING:
            logger.warning(f"Guard Blocked {operation_type}: {cls.BLOCK_REASON}")
            return False

        if PriceEngine.last_price is None:
            logger.warning(f"Guard Blocked {operation_type}: No price data")
            return False

        if (time.time() - PriceEngine.last_update_time) > 8:
            logger.warning(f"Guard Blocked {operation_type}: Stale price (>8s)")
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

    _notifier = None
    _last_disconnect_alert_time: float = 0
    _disconnect_alerted: bool = False

    @classmethod
    def set_notifier(cls, notifier):
        cls._notifier = notifier
        # If the feed was already down before the notifier was registered
        # (race between PriceEngine startup and main.py finishing init),
        # send the alert now so the resume notification can fire later.
        if TradingGuard.BLOCK_ALL_TRADING and TradingGuard.BLOCK_REASON == "Price feed disconnected":
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
    def on_feed_disconnect(cls):
        TradingGuard.BLOCK_ALL_TRADING = True
        TradingGuard.BLOCK_REASON = "Price feed disconnected"
        logger.critical("FAILSAFE: Trading Blocked due to connection loss")

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
    def on_feed_reconnect(cls):
        TradingGuard.BLOCK_ALL_TRADING = False
        TradingGuard.BLOCK_REASON = ""
        logger.info("FAILSAFE: Trading Resumed")

        if cls._disconnect_alerted:
            cls._disconnect_alerted = False
            cls._notify(
                "✅ *عاد مصدر الأسعار*\n"
                "تم استئناف التداول تلقائياً."
            )

    # ── backward-compat aliases (called from main.py in some places) ──────────
    @classmethod
    def on_websocket_disconnect(cls):
        cls.on_feed_disconnect()

    @classmethod
    def on_websocket_reconnect(cls):
        cls.on_feed_reconnect()

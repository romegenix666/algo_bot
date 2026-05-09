"""Telegram bot — daily summary + critical alerts.

We use **simple HTTP POST** to Telegram's Bot API rather than the
``python-telegram-bot`` async framework. We never need to *receive*
messages — we only push. One requests dependency, one method.

Setup (one-time):
    1. Talk to @BotFather on Telegram → /newbot → get a token
    2. Send a message to your bot from your Telegram account
    3. Visit  https://api.telegram.org/bot<TOKEN>/getUpdates
       Find your chat_id in the JSON response.
    4. Put both in ``.env``:
           TELEGRAM_BOT_TOKEN=...
           TELEGRAM_CHAT_ID=...

Failure mode:
    If credentials are missing OR the API call fails, we LOG and CARRY ON.
    The bot must never crash because Telegram is down.

Alerts we send:
    - End-of-day summary (always)
    - Order fills (configurable; can be silenced for high-frequency runs)
    - Stop-loss triggers
    - Circuit-breaker trips
    - Manual kill-switch acknowledgements
    - Errors / data outages
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import httpx
except ImportError:  # pragma: no cover - dev guard
    httpx = None  # type: ignore[assignment]

from src.utils.logging import logger
from src.utils.settings import settings


@dataclass
class TelegramNotifier:
    """Thin Telegram client. ``enabled`` controls whether anything is sent."""

    bot_token: str | None = None
    chat_id: str | None = None
    timeout_s: float = 8.0
    parse_mode: str = "HTML"

    @classmethod
    def from_settings(cls) -> TelegramNotifier:
        creds = settings.notifications
        return cls(
            bot_token=creds.telegram_bot_token,
            chat_id=creds.telegram_chat_id,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id and httpx is not None)

    # ----------------------------------------------------------------
    def send(self, text: str, *, silent: bool = False) -> bool:
        """Send a single message. Returns True on success, False otherwise."""
        if not self.enabled:
            return False
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text[:4000],  # Telegram cap is 4096 chars
            "parse_mode": self.parse_mode,
            "disable_notification": silent,
        }
        try:
            r = httpx.post(url, json=payload, timeout=self.timeout_s)
            if r.status_code != 200:
                logger.warning("Telegram send failed (HTTP {}): {}", r.status_code, r.text)
                return False
            return True
        except Exception as exc:  # pragma: no cover - network
            logger.warning("Telegram exception: {}", exc)
            return False

    # ----------------------------------------------------------------
    # High-level alert helpers — opinionated formatting so callers stay tidy.
    # ----------------------------------------------------------------
    def fill(
        self,
        ticker: str,
        side: str,
        quantity: int,
        price: float,
        reason: str | None = None,
    ) -> bool:
        emoji = "🟢" if side.lower() == "buy" else "🔴"
        msg = f"{emoji} <b>FILL {side.upper()}</b>\n  {ticker}  qty={quantity}  @ ₹{price:,.2f}"
        if reason:
            msg += f"\n  reason: <i>{reason}</i>"
        return self.send(msg)

    def stop_hit(self, ticker: str, price: float, strategy: str) -> bool:
        msg = (
            f"🛑 <b>STOP HIT</b>\n  {ticker} closed @ ₹{price:,.2f}\n  strategy: <i>{strategy}</i>"
        )
        return self.send(msg)

    def circuit_breaker_tripped(self, state: str, reason: str) -> bool:
        msg = (
            f"⚠️ <b>CIRCUIT BREAKER {state.upper()}</b>\n"
            f"  reason: {reason}\n"
            f"  no new entries until reset"
        )
        return self.send(msg)

    def kill_switch(self, n_closed: int) -> bool:
        msg = f"🚨 <b>KILL SWITCH</b>\nClosed {n_closed} positions. Bot is now in HALTED state."
        return self.send(msg)

    def daily_summary(
        self,
        as_of: str,
        equity: float,
        daily_pct: float,
        drawdown_pct: float,
        regime: str,
        n_open: int,
        n_filled_today: int,
        breaker_state: str,
    ) -> bool:
        sign = "📈" if daily_pct >= 0 else "📉"
        dd_sign = "🔻" if drawdown_pct < -0.005 else "🟢"
        msg = (
            f"<b>📊 Daily Summary — {as_of}</b>\n"
            f"  Equity   : ₹{equity:>12,.0f}\n"
            f"  Day      : {sign} {daily_pct:+.2%}\n"
            f"  Drawdown : {dd_sign} {drawdown_pct:+.2%}\n"
            f"  Regime   : {regime}\n"
            f"  Open pos : {n_open}\n"
            f"  Filled   : {n_filled_today}\n"
            f"  Breaker  : {breaker_state}"
        )
        return self.send(msg, silent=True)


__all__ = ["TelegramNotifier"]

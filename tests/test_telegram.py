"""Telegram notifier tests.

We don't hit the real API. We verify:
    - When credentials are missing, ``enabled`` is False and no exceptions.
    - HTTP calls are made to the right URL with the right payload.
    - Failures are logged and don't crash.
"""

from __future__ import annotations

from unittest.mock import patch

from src.monitor.telegram import TelegramNotifier


def test_notifier_disabled_without_creds() -> None:
    n = TelegramNotifier(bot_token=None, chat_id=None)
    assert n.enabled is False
    # send() must NOT raise even when disabled.
    assert n.send("hello") is False


def test_notifier_sends_when_enabled() -> None:
    n = TelegramNotifier(bot_token="dummy_token", chat_id="123")
    fake_resp = type("R", (), {"status_code": 200, "text": "ok"})()
    with patch("src.monitor.telegram.httpx") as mock_httpx:
        mock_httpx.post.return_value = fake_resp
        ok = n.send("hello")
    assert ok is True
    mock_httpx.post.assert_called_once()
    call_args = mock_httpx.post.call_args
    assert "https://api.telegram.org/botdummy_token/sendMessage" in call_args.args[0]
    payload = call_args.kwargs["json"]
    assert payload["chat_id"] == "123"
    assert payload["text"] == "hello"


def test_notifier_handles_http_error() -> None:
    n = TelegramNotifier(bot_token="dummy_token", chat_id="123")
    fake_resp = type("R", (), {"status_code": 401, "text": "unauthorised"})()
    with patch("src.monitor.telegram.httpx") as mock_httpx:
        mock_httpx.post.return_value = fake_resp
        ok = n.send("hello")
    assert ok is False


def test_notifier_handles_exception() -> None:
    n = TelegramNotifier(bot_token="dummy_token", chat_id="123")
    with patch("src.monitor.telegram.httpx") as mock_httpx:
        mock_httpx.post.side_effect = RuntimeError("network down")
        ok = n.send("hello")
    assert ok is False


def test_fill_helper_format() -> None:
    n = TelegramNotifier(bot_token="dummy_token", chat_id="123")
    with patch("src.monitor.telegram.httpx") as mock_httpx:
        fake_resp = type("R", (), {"status_code": 200, "text": "ok"})()
        mock_httpx.post.return_value = fake_resp
        n.fill(ticker="RELIANCE.NS", side="buy", quantity=100, price=1500.0, reason="momentum")
    payload = mock_httpx.post.call_args.kwargs["json"]
    assert "RELIANCE.NS" in payload["text"]
    assert "100" in payload["text"]
    assert "1,500" in payload["text"]


def test_daily_summary_format() -> None:
    n = TelegramNotifier(bot_token="dummy_token", chat_id="123")
    with patch("src.monitor.telegram.httpx") as mock_httpx:
        fake_resp = type("R", (), {"status_code": 200, "text": "ok"})()
        mock_httpx.post.return_value = fake_resp
        n.daily_summary(
            as_of="2026-05-08",
            equity=1_000_000.0,
            daily_pct=0.005,
            drawdown_pct=-0.02,
            regime="trending_up_low_vol",
            n_open=5,
            n_filled_today=2,
            breaker_state="healthy",
        )
    payload = mock_httpx.post.call_args.kwargs["json"]
    assert "Daily Summary" in payload["text"]
    assert "1,000,000" in payload["text"]
    assert payload["disable_notification"] is True  # daily summary is silent

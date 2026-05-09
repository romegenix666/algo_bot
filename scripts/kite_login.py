"""Daily Kite Connect access-token refresh.

Kite access tokens expire every day at ~8 AM IST. Run this script each
morning *before* the scheduler kicks off the trading jobs.

Flow:
    1. Print the Kite login URL.
    2. User opens it, logs in, gets redirected to your registered redirect
       URI with ``?request_token=...`` in the query string.
    3. User pastes the request_token here.
    4. We exchange it for an access_token and write it to ``.env``.

Usage::

    python -m scripts.kite_login

You'll need your KITE_API_KEY and KITE_API_SECRET in .env (one-time setup
in the Kite developer portal: https://developers.kite.trade/).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    from kiteconnect import KiteConnect
except ImportError:  # pragma: no cover - optional
    KiteConnect = None

from src.utils.settings import PROJECT_ROOT, settings

ENV_FILE = PROJECT_ROOT / ".env"


def main() -> int:
    if KiteConnect is None:
        print("ERROR: pip install kiteconnect")
        return 2

    creds = settings.broker
    if not creds.kite_api_key or not creds.kite_api_secret:
        print("ERROR: set KITE_API_KEY and KITE_API_SECRET in .env first.")
        return 2

    kite = KiteConnect(api_key=creds.kite_api_key)
    print("=" * 70)
    print("Step 1 — Open this URL in your browser, log in, and copy the")
    print("request_token from the redirect URL (e.g. ?request_token=XXXX):")
    print()
    print(f"  {kite.login_url()}")
    print()
    print("Step 2 — Paste the request_token below:")
    print("=" * 70)
    request_token = input("request_token: ").strip()
    if not request_token:
        print("Aborted.")
        return 1

    try:
        data = kite.generate_session(request_token, api_secret=creds.kite_api_secret)
    except Exception as exc:
        print(f"ERROR: token exchange failed — {exc}")
        return 2

    access_token = data["access_token"]
    print(f"\n✓ Access token obtained: {access_token[:8]}...{access_token[-4:]}")
    _write_env(ENV_FILE, "KITE_ACCESS_TOKEN", access_token)
    print(f"✓ Updated {ENV_FILE}")
    return 0


def _write_env(path: Path, key: str, value: str) -> None:
    """Update or append a single KEY=VALUE in .env."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text() if path.exists() else ""
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    new_line = f"{key}={value}"
    if pattern.search(text):
        text = pattern.sub(new_line, text)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += new_line + "\n"
    path.write_text(text)


_ = sys


if __name__ == "__main__":
    raise SystemExit(main())

"""Long-running daily scheduler.

Runs the bot's three daily jobs at IST times:
    - 08:30 — refresh sentiment
    - 16:30 — fetch closing prices
    - 16:35 — paper-trade

Usage::

    python -m scripts.run_scheduler

Stop with Ctrl-C. Designed for local-laptop / cloud-VM deployment.
For production use systemd or supervisord to auto-restart on crashes.
"""

from __future__ import annotations

from src.monitor.scheduler import run_blocking


def main() -> int:
    run_blocking()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""APScheduler-based daily runner.

Provides a long-running process that fires three jobs per day on weekdays:

    08:30 IST   — refresh sentiment (RSS scrape + scoring) — pre-market
    16:30 IST   — daily fetch_history (closing prices)
    16:35 IST   — paper_trade (run strategies, route signals, alert)

Why a separate scheduler instead of system cron:
    - State is in-process: re-loading APScheduler is fast.
    - Failures fire Telegram alerts (we wire that in).
    - Dev-friendly: local laptop testing without crontab.

For production deployment, a system-wide cron OR systemd timer is fine
(running each command at the chosen IST time). This module is the
stand-in until then.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
except ImportError:  # pragma: no cover - dev guard
    BlockingScheduler = None
    CronTrigger = None

from src.monitor.telegram import TelegramNotifier
from src.utils.logging import logger
from src.utils.settings import PROJECT_ROOT

IST = ZoneInfo("Asia/Kolkata")


def _run_module(module: str, *args: str) -> int:
    """Run a project script as a subprocess. Returns exit code."""
    cmd = [sys.executable, "-m", module, *args]
    logger.info("Scheduler running: {}", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    return proc.returncode


def _job_refresh_sentiment() -> None:
    code = _run_module("scripts.refresh_sentiment")
    if code != 0:
        TelegramNotifier.from_settings().send(f"⚠️ refresh_sentiment exited with code {code}")


def _job_fetch_history() -> None:
    code = _run_module("scripts.fetch_history", "--mode", "daily")
    if code != 0:
        TelegramNotifier.from_settings().send(f"⚠️ fetch_history exited with code {code}")


def _job_paper_trade() -> None:
    code = _run_module("scripts.paper_trade")
    if code != 0:
        TelegramNotifier.from_settings().send(f"⚠️ paper_trade exited with code {code}")


def build_scheduler() -> object:
    """Construct the BlockingScheduler with the four jobs registered."""
    if BlockingScheduler is None or CronTrigger is None:
        raise ImportError("Install apscheduler: `pip install apscheduler`")
    sched = BlockingScheduler(timezone=IST)
    sched.add_job(
        _job_refresh_sentiment,
        CronTrigger(day_of_week="mon-fri", hour=8, minute=30, timezone=IST),
        id="refresh_sentiment",
        name="Refresh sentiment (8:30 IST)",
    )
    sched.add_job(
        _job_fetch_history,
        CronTrigger(day_of_week="mon-fri", hour=16, minute=30, timezone=IST),
        id="fetch_history",
        name="Fetch closing prices (16:30 IST)",
    )
    sched.add_job(
        _job_paper_trade,
        CronTrigger(day_of_week="mon-fri", hour=16, minute=35, timezone=IST),
        id="paper_trade",
        name="Paper trade (16:35 IST)",
    )
    # Optional intraday sentiment refresh at 12:30 IST (lunch).
    sched.add_job(
        _job_refresh_sentiment,
        CronTrigger(day_of_week="mon-fri", hour=12, minute=30, timezone=IST),
        id="refresh_sentiment_lunch",
        name="Refresh sentiment (lunch)",
    )
    logger.info("Scheduler initialised. Jobs: %s", [j.name for j in sched.get_jobs()])
    return sched


def run_blocking() -> None:  # pragma: no cover - long-running
    """Start the scheduler. Blocks until SIGTERM / Ctrl-C."""
    sched = build_scheduler()
    notifier = TelegramNotifier.from_settings()
    notifier.send(
        f"🤖 Algo Bot scheduler started @ {datetime.now(IST).isoformat(timespec='seconds')}"
    )
    try:
        sched.start()  # type: ignore[attr-defined]
    except (KeyboardInterrupt, SystemExit):
        notifier.send("🤖 Algo Bot scheduler stopped (manual)")


# Stub for static-analysis tools — `Path` is referenced in CLI use later.
_ = Path

__all__ = ["IST", "build_scheduler", "run_blocking"]

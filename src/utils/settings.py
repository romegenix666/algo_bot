"""Typed configuration loader.

Reads ``config/{ALGO_MODE}.yaml`` (falls back to ``config/default.yaml``) and
overlays environment variables from ``.env``. Every other module imports
``settings`` from here so we only have one source of truth.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class BrokerCreds(BaseSettings):
    """Broker credentials pulled exclusively from environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    kite_api_key: str | None = Field(default=None, alias="KITE_API_KEY")
    kite_api_secret: str | None = Field(default=None, alias="KITE_API_SECRET")
    kite_access_token: str | None = Field(default=None, alias="KITE_ACCESS_TOKEN")

    upstox_api_key: str | None = Field(default=None, alias="UPSTOX_API_KEY")
    upstox_api_secret: str | None = Field(default=None, alias="UPSTOX_API_SECRET")
    upstox_redirect_uri: str | None = Field(default=None, alias="UPSTOX_REDIRECT_URI")


class NotificationCreds(BaseSettings):
    """Telegram + news API credentials."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    telegram_bot_token: str | None = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str | None = Field(default=None, alias="TELEGRAM_CHAT_ID")
    newsapi_key: str | None = Field(default=None, alias="NEWSAPI_KEY")
    huggingface_token: str | None = Field(default=None, alias="HUGGINGFACE_TOKEN")


class AppEnv(BaseSettings):
    """High-level environment toggles."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mode: str = Field(default="paper", alias="ALGO_MODE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    timezone: str = Field(default="Asia/Kolkata", alias="TZ")
    database_url: str = Field(default="sqlite:///data/algobot.db", alias="DATABASE_URL")


class Settings(BaseModel):
    """All loaded settings: yaml config + env-derived credentials."""

    config: dict[str, Any]
    env: AppEnv
    broker: BrokerCreds
    notifications: NotificationCreds

    @property
    def mode(self) -> str:
        return self.env.mode

    def get(self, *keys: str, default: Any = None) -> Any:
        """Dotted-path getter into the YAML config: ``settings.get("risk", "per_trade_pct")``."""
        node: Any = self.config
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file {path} must be a mapping at the top level")
    return loaded


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out: dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached project settings. Call this from anywhere."""
    env = AppEnv()
    base_cfg = _load_yaml("default")
    overlay_cfg = _load_yaml(env.mode) if env.mode != "default" else {}
    merged = _deep_merge(base_cfg, overlay_cfg)

    return Settings(
        config=merged,
        env=env,
        broker=BrokerCreds(),
        notifications=NotificationCreds(),
    )


settings = get_settings()


__all__ = [
    "PROJECT_ROOT",
    "AppEnv",
    "BrokerCreds",
    "NotificationCreds",
    "Settings",
    "get_settings",
    "settings",
]

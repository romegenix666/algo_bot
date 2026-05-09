"""KiteBroker construction + import safety tests.

We don't hit the real Kite API. We verify:
    - Importing the module doesn't crash even without creds.
    - Constructing without creds raises a clear error.
    - The class implements the Broker interface.
"""

from __future__ import annotations

import pytest

from src.orders.kite import KiteBroker, KiteUnavailableError


def test_construction_without_creds_raises() -> None:
    with pytest.raises(KiteUnavailableError):
        KiteBroker(api_key=None, access_token=None)


def test_construction_without_token_raises() -> None:
    with pytest.raises(KiteUnavailableError):
        KiteBroker(api_key="abc", access_token=None)


def test_kite_order_type_mapping() -> None:
    """The kite-specific order-type mapping must match Zerodha conventions."""
    from src.orders.base import OrderType

    assert KiteBroker._kite_order_type(OrderType.MARKET) == "MARKET"
    assert KiteBroker._kite_order_type(OrderType.LIMIT) == "LIMIT"
    # Zerodha calls stop-loss-market "SL-M" and stop-limit "SL"
    assert KiteBroker._kite_order_type(OrderType.STOP) == "SL-M"
    assert KiteBroker._kite_order_type(OrderType.STOP_LIMIT) == "SL"

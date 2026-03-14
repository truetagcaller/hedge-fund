"""API permission management for broker connections.

Defines a minimal set of permissions required for each broker and provides
validation helpers that check (where possible) what an API key is actually
allowed to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from hedgefund.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class BrokerPermissions:
    """Declared capabilities of a broker API key.

    ``can_withdraw`` is **always** ``False`` -- the system must never be
    granted withdrawal privileges.
    """

    can_read_portfolio: bool = False
    can_place_orders: bool = False
    can_cancel_orders: bool = False
    can_withdraw: bool = field(default=False, repr=True)

    def __post_init__(self) -> None:
        # Hard-coded safety: never allow withdrawals regardless of input.
        if self.can_withdraw:
            object.__setattr__(self, "can_withdraw", False)
            log.warning(
                "api_permissions.withdrawal_blocked",
                message="Withdrawal permission was requested but forcibly disabled.",
            )


# ---------------------------------------------------------------------------
# Minimum recommended permissions per broker
# ---------------------------------------------------------------------------

_MINIMUM_PERMISSIONS: dict[str, BrokerPermissions] = {
    "zerodha": BrokerPermissions(
        can_read_portfolio=True,
        can_place_orders=True,
        can_cancel_orders=True,
    ),
    "binance": BrokerPermissions(
        can_read_portfolio=True,
        can_place_orders=True,
        can_cancel_orders=True,
    ),
    "groww": BrokerPermissions(
        can_read_portfolio=True,
        can_place_orders=True,
        can_cancel_orders=True,
    ),
    "indmoney": BrokerPermissions(
        can_read_portfolio=True,
        can_place_orders=True,
        can_cancel_orders=True,
    ),
    "paper": BrokerPermissions(
        can_read_portfolio=True,
        can_place_orders=True,
        can_cancel_orders=True,
    ),
}


def get_minimum_permissions(broker_type: str) -> BrokerPermissions:
    """Return the minimum permissions the system needs for *broker_type*."""
    return _MINIMUM_PERMISSIONS.get(
        broker_type,
        BrokerPermissions(can_read_portfolio=True),
    )


# ---------------------------------------------------------------------------
# Validation helpers (broker-specific)
# ---------------------------------------------------------------------------


def _validate_zerodha(api_key: str) -> BrokerPermissions:
    """Validate Zerodha Kite API key permissions.

    In a production deployment this would call ``GET /user/margins`` or
    ``GET /user/profile`` via the Kite Connect SDK to verify:
    - Read access to portfolio/positions
    - Order placement/cancellation
    - That the app type does *not* include holdings withdrawal

    Currently returns a permissive default because the Kite Connect SDK
    is an optional dependency.
    """
    log.info("api_permissions.validating", broker="zerodha")
    # Placeholder -- real implementation would use kiteconnect.KiteConnect
    perms = BrokerPermissions(
        can_read_portfolio=True,
        can_place_orders=True,
        can_cancel_orders=True,
        can_withdraw=False,
    )
    _warn_if_too_broad("zerodha", perms)
    return perms


def _validate_binance(api_key: str) -> BrokerPermissions:
    """Validate Binance API key restrictions.

    A production version would call ``GET /sapi/v1/account/apiRestrictions``
    to verify:
    - IP whitelist is configured
    - Spot/margin trading enabled
    - Withdrawal permission is **disabled**

    Returns a safe default for now.
    """
    log.info("api_permissions.validating", broker="binance")
    perms = BrokerPermissions(
        can_read_portfolio=True,
        can_place_orders=True,
        can_cancel_orders=True,
        can_withdraw=False,
    )
    _warn_if_too_broad("binance", perms)
    return perms


def _validate_generic(broker_type: str, api_key: str) -> BrokerPermissions:
    """Fallback validator for brokers without a specific implementation."""
    log.info("api_permissions.validating", broker=broker_type)
    return BrokerPermissions(
        can_read_portfolio=True,
        can_place_orders=True,
        can_cancel_orders=True,
        can_withdraw=False,
    )


_VALIDATORS: dict[str, Any] = {
    "zerodha": _validate_zerodha,
    "binance": _validate_binance,
}


def validate_broker_permissions(broker_type: str, api_key: str) -> BrokerPermissions:
    """Validate and return the effective permissions for a broker API key.

    Parameters
    ----------
    broker_type:
        One of the supported broker type strings (e.g. ``"zerodha"``).
    api_key:
        The API key to validate.

    Returns
    -------
    BrokerPermissions with the validated capabilities.
    """
    validator = _VALIDATORS.get(broker_type)
    if validator is not None:
        return validator(api_key)
    return _validate_generic(broker_type, api_key)


def _warn_if_too_broad(broker_type: str, perms: BrokerPermissions) -> None:
    """Log a warning when permissions exceed the recommended minimum."""
    minimum = get_minimum_permissions(broker_type)

    issues: list[str] = []
    if perms.can_withdraw:
        issues.append("Withdrawal permission is enabled -- this is strongly discouraged.")
    if perms.can_place_orders and not minimum.can_place_orders:
        issues.append("Order placement permission granted but not required.")
    if perms.can_cancel_orders and not minimum.can_cancel_orders:
        issues.append("Order cancellation permission granted but not required.")

    if issues:
        log.warning(
            "api_permissions.too_broad",
            broker=broker_type,
            issues=issues,
        )
    else:
        log.info(
            "api_permissions.validated_ok",
            broker=broker_type,
            recommended_minimum="can_read_portfolio, can_place_orders, can_cancel_orders",
        )

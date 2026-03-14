"""Order construction for single-leg and multi-leg option strategies.

The :class:`OrderBuilder` produces fully-validated :class:`Order` objects
(or lists thereof for multi-leg structures) ready for submission.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Sequence

import structlog

from hedgefund.types import (
    OptionContract,
    OptionType,
    Order,
    OrderType,
    Side,
)

logger = structlog.get_logger(__name__)


class OrderValidationError(ValueError):
    """Raised when order parameters fail validation."""


class OrderBuilder:
    """Constructs single- and multi-leg option orders.

    Parameters
    ----------
    default_order_type:
        Default order type when not explicitly specified (default ``LIMIT``).
    """

    def __init__(self, default_order_type: OrderType = OrderType.LIMIT) -> None:
        self._default_type = default_order_type

    # ── Single leg ────────────────────────────────────────────────────────

    def single_leg(
        self,
        contract: OptionContract,
        side: Side,
        quantity: int,
        signal_id: str,
        order_type: OrderType | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        """Build a single-leg option order."""
        otype = order_type or self._default_type
        self._validate_base(quantity, otype, limit_price, stop_price)

        order = Order(
            order_id=Order.generate_id(),
            signal_id=signal_id,
            contract=contract,
            side=side,
            order_type=otype,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
        )
        logger.info(
            "order_builder.single_leg",
            order_id=order.order_id,
            symbol=contract.symbol,
            side=side.value,
            qty=quantity,
        )
        return order

    # ── Vertical spread ───────────────────────────────────────────────────

    def vertical_spread(
        self,
        underlying: str,
        option_type: OptionType,
        long_strike: float,
        short_strike: float,
        expiration: date,
        quantity: int,
        signal_id: str,
        limit_price: float | None = None,
        multiplier: int = 100,
    ) -> list[Order]:
        """Build a vertical (bull/bear) spread.

        For a bull call spread: buy the lower strike, sell the higher strike.
        For a bear put spread: buy the higher strike, sell the lower strike.
        """
        if long_strike == short_strike:
            raise OrderValidationError("Long and short strikes must differ")
        if quantity <= 0:
            raise OrderValidationError("Quantity must be positive")

        long_contract = OptionContract(
            symbol=f"{underlying}_{expiration}_{option_type.value}_{long_strike}",
            underlying=underlying,
            option_type=option_type,
            strike=long_strike,
            expiration=expiration,
            multiplier=multiplier,
        )
        short_contract = OptionContract(
            symbol=f"{underlying}_{expiration}_{option_type.value}_{short_strike}",
            underlying=underlying,
            option_type=option_type,
            strike=short_strike,
            expiration=expiration,
            multiplier=multiplier,
        )

        orders = [
            self.single_leg(long_contract, Side.BUY, quantity, signal_id, limit_price=limit_price),
            self.single_leg(short_contract, Side.SELL, quantity, signal_id, limit_price=limit_price),
        ]
        logger.info(
            "order_builder.vertical_spread",
            underlying=underlying,
            type=option_type.value,
            long_strike=long_strike,
            short_strike=short_strike,
            qty=quantity,
        )
        return orders

    # ── Iron condor ───────────────────────────────────────────────────────

    def iron_condor(
        self,
        underlying: str,
        put_long_strike: float,
        put_short_strike: float,
        call_short_strike: float,
        call_long_strike: float,
        expiration: date,
        quantity: int,
        signal_id: str,
        limit_price: float | None = None,
        multiplier: int = 100,
    ) -> list[Order]:
        """Build an iron condor (short strangle + long wings).

        Strikes must satisfy:
        put_long < put_short < call_short < call_long
        """
        strikes = [put_long_strike, put_short_strike, call_short_strike, call_long_strike]
        if strikes != sorted(strikes) or len(set(strikes)) != 4:
            raise OrderValidationError(
                "Iron condor strikes must be strictly increasing: "
                f"put_long < put_short < call_short < call_long, got {strikes}"
            )
        if quantity <= 0:
            raise OrderValidationError("Quantity must be positive")

        def _contract(strike: float, otype: OptionType) -> OptionContract:
            return OptionContract(
                symbol=f"{underlying}_{expiration}_{otype.value}_{strike}",
                underlying=underlying,
                option_type=otype,
                strike=strike,
                expiration=expiration,
                multiplier=multiplier,
            )

        orders = [
            self.single_leg(_contract(put_long_strike, OptionType.PUT), Side.BUY, quantity, signal_id, limit_price=limit_price),
            self.single_leg(_contract(put_short_strike, OptionType.PUT), Side.SELL, quantity, signal_id, limit_price=limit_price),
            self.single_leg(_contract(call_short_strike, OptionType.CALL), Side.SELL, quantity, signal_id, limit_price=limit_price),
            self.single_leg(_contract(call_long_strike, OptionType.CALL), Side.BUY, quantity, signal_id, limit_price=limit_price),
        ]
        logger.info(
            "order_builder.iron_condor",
            underlying=underlying,
            strikes=strikes,
            qty=quantity,
        )
        return orders

    # ── Straddle / Strangle ───────────────────────────────────────────────

    def straddle(
        self,
        underlying: str,
        strike: float,
        expiration: date,
        quantity: int,
        signal_id: str,
        side: Side = Side.BUY,
        limit_price: float | None = None,
        multiplier: int = 100,
    ) -> list[Order]:
        """Build a straddle (same strike, both call and put)."""
        if quantity <= 0:
            raise OrderValidationError("Quantity must be positive")

        def _contract(otype: OptionType) -> OptionContract:
            return OptionContract(
                symbol=f"{underlying}_{expiration}_{otype.value}_{strike}",
                underlying=underlying,
                option_type=otype,
                strike=strike,
                expiration=expiration,
                multiplier=multiplier,
            )

        orders = [
            self.single_leg(_contract(OptionType.CALL), side, quantity, signal_id, limit_price=limit_price),
            self.single_leg(_contract(OptionType.PUT), side, quantity, signal_id, limit_price=limit_price),
        ]
        logger.info(
            "order_builder.straddle",
            underlying=underlying,
            strike=strike,
            side=side.value,
            qty=quantity,
        )
        return orders

    def strangle(
        self,
        underlying: str,
        put_strike: float,
        call_strike: float,
        expiration: date,
        quantity: int,
        signal_id: str,
        side: Side = Side.BUY,
        limit_price: float | None = None,
        multiplier: int = 100,
    ) -> list[Order]:
        """Build a strangle (different strikes, both call and put)."""
        if put_strike >= call_strike:
            raise OrderValidationError(
                f"Put strike ({put_strike}) must be below call strike ({call_strike})"
            )
        if quantity <= 0:
            raise OrderValidationError("Quantity must be positive")

        put_contract = OptionContract(
            symbol=f"{underlying}_{expiration}_PUT_{put_strike}",
            underlying=underlying,
            option_type=OptionType.PUT,
            strike=put_strike,
            expiration=expiration,
            multiplier=multiplier,
        )
        call_contract = OptionContract(
            symbol=f"{underlying}_{expiration}_CALL_{call_strike}",
            underlying=underlying,
            option_type=OptionType.CALL,
            strike=call_strike,
            expiration=expiration,
            multiplier=multiplier,
        )

        orders = [
            self.single_leg(put_contract, side, quantity, signal_id, limit_price=limit_price),
            self.single_leg(call_contract, side, quantity, signal_id, limit_price=limit_price),
        ]
        logger.info(
            "order_builder.strangle",
            underlying=underlying,
            put_strike=put_strike,
            call_strike=call_strike,
            side=side.value,
            qty=quantity,
        )
        return orders

    # ── Calendar spread ───────────────────────────────────────────────────

    def calendar_spread(
        self,
        underlying: str,
        option_type: OptionType,
        strike: float,
        near_expiration: date,
        far_expiration: date,
        quantity: int,
        signal_id: str,
        limit_price: float | None = None,
        multiplier: int = 100,
    ) -> list[Order]:
        """Build a calendar (time) spread — sell near, buy far.

        Parameters
        ----------
        near_expiration:
            The earlier expiration (short leg).
        far_expiration:
            The later expiration (long leg).
        """
        if near_expiration >= far_expiration:
            raise OrderValidationError(
                f"Near expiration ({near_expiration}) must precede far ({far_expiration})"
            )
        if quantity <= 0:
            raise OrderValidationError("Quantity must be positive")

        near_contract = OptionContract(
            symbol=f"{underlying}_{near_expiration}_{option_type.value}_{strike}",
            underlying=underlying,
            option_type=option_type,
            strike=strike,
            expiration=near_expiration,
            multiplier=multiplier,
        )
        far_contract = OptionContract(
            symbol=f"{underlying}_{far_expiration}_{option_type.value}_{strike}",
            underlying=underlying,
            option_type=option_type,
            strike=strike,
            expiration=far_expiration,
            multiplier=multiplier,
        )

        orders = [
            self.single_leg(near_contract, Side.SELL, quantity, signal_id, limit_price=limit_price),
            self.single_leg(far_contract, Side.BUY, quantity, signal_id, limit_price=limit_price),
        ]
        logger.info(
            "order_builder.calendar_spread",
            underlying=underlying,
            strike=strike,
            near=str(near_expiration),
            far=str(far_expiration),
            qty=quantity,
        )
        return orders

    # ── Validation helpers ────────────────────────────────────────────────

    @staticmethod
    def _validate_base(
        quantity: int,
        order_type: OrderType,
        limit_price: float | None,
        stop_price: float | None,
    ) -> None:
        if quantity <= 0:
            raise OrderValidationError(f"Quantity must be positive, got {quantity}")

        if order_type == OrderType.LIMIT and limit_price is None:
            raise OrderValidationError("Limit orders require a limit_price")

        if order_type == OrderType.STOP and stop_price is None:
            raise OrderValidationError("Stop orders require a stop_price")

        if order_type == OrderType.STOP_LIMIT:
            if limit_price is None or stop_price is None:
                raise OrderValidationError(
                    "Stop-limit orders require both limit_price and stop_price"
                )

        if limit_price is not None and limit_price <= 0:
            raise OrderValidationError(f"Limit price must be positive, got {limit_price}")

        if stop_price is not None and stop_price <= 0:
            raise OrderValidationError(f"Stop price must be positive, got {stop_price}")

"""Custom exception hierarchy for the trading system."""


class HedgeFundError(Exception):
    """Base exception for all trading system errors."""


# ── Data Errors ──────────────────────────────────────────

class DataError(HedgeFundError):
    """Error in data ingestion or processing."""


class DataConnectionError(DataError):
    """Failed to connect to data source."""


class DataValidationError(DataError):
    """Received data failed validation checks."""


class StaleDataError(DataError):
    """Data is older than acceptable threshold."""


# ── Execution Errors ─────────────────────────────────────

class ExecutionError(HedgeFundError):
    """Error during trade execution."""


class OrderRejectedError(ExecutionError):
    """Order was rejected by broker."""

    def __init__(self, order_id: str, reason: str):
        self.order_id = order_id
        self.reason = reason
        super().__init__(f"Order {order_id} rejected: {reason}")


class InsufficientFundsError(ExecutionError):
    """Not enough capital to execute order."""


class BrokerConnectionError(ExecutionError):
    """Lost connection to broker."""


# ── Risk Errors ──────────────────────────────────────────

class RiskError(HedgeFundError):
    """Risk management violation."""


class RiskLimitBreachedError(RiskError):
    """A risk limit has been exceeded."""

    def __init__(self, limit_name: str, current: float, maximum: float):
        self.limit_name = limit_name
        self.current = current
        self.maximum = maximum
        super().__init__(
            f"Risk limit '{limit_name}' breached: {current:.4f} > {maximum:.4f}"
        )


class CircuitBreakerTrippedError(RiskError):
    """Circuit breaker has been activated - trading halted."""


class DrawdownExceededError(RiskError):
    """Portfolio drawdown exceeds maximum threshold."""


# ── Signal Errors ────────────────────────────────────────

class SignalError(HedgeFundError):
    """Error in signal generation."""


class ModelNotLoadedError(SignalError):
    """ML model has not been loaded or trained."""


class InsufficientDataError(SignalError):
    """Not enough data to generate a reliable signal."""


# ── Configuration Errors ─────────────────────────────────

class ConfigError(HedgeFundError):
    """Configuration error."""


class InvalidConfigError(ConfigError):
    """Configuration values are invalid."""


# ── Data Source Errors ─────────────────────────────────────

class DataSourceError(HedgeFundError):
    """Error related to data source validation."""


class DataSourceNotConnectedError(DataSourceError):
    """No valid data source is connected."""


class SyntheticDataError(DataSourceError):
    """Attempted to use synthetic/mock data."""


# ── Agent Errors ───────────────────────────────────────────

class AgentError(HedgeFundError):
    """Error in AI agent system."""


class AgentNotActiveError(AgentError):
    """Agent is not active (data source not verified)."""


# ── Write Guard Errors ─────────────────────────────────────

class ForbiddenWriteError(HedgeFundError):
    """Attempted to write forbidden data to database."""

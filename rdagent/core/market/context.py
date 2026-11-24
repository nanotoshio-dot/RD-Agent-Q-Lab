from dataclasses import dataclass
from datetime import date
from pathlib import Path

from rdagent.core.market.trading_rules import TradingRules


@dataclass(frozen=True)
class MarketContext:
    """Immutable context describing a market and its research environment."""

    market_id: str
    provider_uri: Path
    region: str
    currency: str
    qlib_market: str
    benchmark: str
    universe: tuple[str, ...]

    workspace_path: Path
    cache_path: Path
    memory_path: Path
    calendar: Path

    train_start: date
    train_end: date
    valid_start: date
    valid_end: date
    test_start: date
    test_end: date

    # None keeps the China rules written as defaults in the Qlib templates.
    trading_rules: TradingRules | None = None

    def __post_init__(self) -> None:
        self._validate_market_strings()
        self._validate_paths()
        self._validate_universe()
        self._validate_dates()
        self._validate_period_order()
        self._validate_trading_rules()

    def _validate_market_strings(self) -> None:
        required_strings = {
            "market_id": self.market_id,
            "region": self.region,
            "currency": self.currency,
            "qlib_market": self.qlib_market,
            "benchmark": self.benchmark,
        }
        for field_name, value in required_strings.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")  # noqa: EM102, TRY003

    def _validate_paths(self) -> None:
        path_fields = (
            "provider_uri",
            "workspace_path",
            "cache_path",
            "memory_path",
            "calendar",
        )
        for field_name in path_fields:
            if not isinstance(getattr(self, field_name), Path):
                raise TypeError(f"{field_name} must be a pathlib.Path")  # noqa: EM102, TRY003

    def _validate_universe(self) -> None:
        if (
            not isinstance(self.universe, tuple)
            or not self.universe
            or any(not isinstance(item, str) or not item.strip() for item in self.universe)
        ):
            raise ValueError("universe must contain non-empty instrument groups")  # noqa: EM101, TRY003

    def _validate_dates(self) -> None:
        date_fields = (
            self.train_start,
            self.train_end,
            self.valid_start,
            self.valid_end,
            self.test_start,
            self.test_end,
        )
        if any(type(value) is not date for value in date_fields):
            raise TypeError("Research period fields must be datetime.date objects")  # noqa: EM101, TRY003

        if not self.train_start <= self.train_end:
            raise ValueError("Invalid training period")  # noqa: EM101, TRY003
        if not self.valid_start <= self.valid_end:
            raise ValueError("Invalid validation period")  # noqa: EM101, TRY003
        if not self.test_start <= self.test_end:
            raise ValueError("Invalid test period")  # noqa: EM101, TRY003

    def _validate_period_order(self) -> None:
        if not (self.train_end < self.valid_start and self.valid_end < self.test_start):
            message = "Training, validation and test periods must be chronological and non-overlapping"
            raise ValueError(message)

    def _validate_trading_rules(self) -> None:
        if self.trading_rules is None:
            return
        if not isinstance(self.trading_rules, TradingRules):
            raise TypeError("trading_rules must be a TradingRules")  # noqa: EM101, TRY003
        # Qlib replaces a null limit with the region default (0.095 for cn), so null means no limit only in us.
        if self.trading_rules.limit_threshold is None and self.region != "us":
            raise ValueError("A null price limit is only supported in region us")  # noqa: EM101, TRY003

import math
from dataclasses import dataclass
from decimal import Decimal

TRADING_RULE_ENV_KEYS = (
    "RD_MARKET_LIMIT_THRESHOLD",
    "RD_MARKET_OPEN_COST",
    "RD_MARKET_CLOSE_COST",
    "RD_MARKET_MIN_COST",
    "RD_MARKET_TOPK",
    "RD_MARKET_N_DROP",
)


@dataclass(frozen=True)
class TradingRules:
    """Backtest rules that the Qlib templates render for one market.

    limit_threshold is the daily price limit; None renders as YAML null, which Qlib reads as
    "no limit" only in region us (other regions fall back to their own default limit).
    """

    limit_threshold: float | None
    open_cost: float
    close_cost: float
    min_cost: float
    topk: int
    n_drop: int

    def __post_init__(self) -> None:
        if self.limit_threshold is not None and not (
            type(self.limit_threshold) is float and 0 < self.limit_threshold < 1
        ):
            raise ValueError("limit_threshold must be None or a float between 0 and 1")  # noqa: EM101, TRY003
        for field_name in ("open_cost", "close_cost"):
            value = getattr(self, field_name)
            if not _is_number(value) or not 0 <= value < 1:
                raise ValueError(f"{field_name} must be a number in [0, 1)")  # noqa: EM102, TRY003
        if not _is_number(self.min_cost) or self.min_cost < 0:
            raise ValueError("min_cost must be a non-negative number")  # noqa: EM101, TRY003
        if type(self.topk) is not int or self.topk < 1:
            raise ValueError("topk must be a positive integer")  # noqa: EM101, TRY003
        if type(self.n_drop) is not int or not 1 <= self.n_drop <= self.topk:
            raise ValueError("n_drop must be an integer between 1 and topk")  # noqa: EM101, TRY003

    def to_env(self) -> dict[str, str]:
        """Return the template variables, written so every YAML loader reads them the same way."""
        values = (
            "null" if self.limit_threshold is None else _yaml_number(self.limit_threshold),
            _yaml_number(self.open_cost),
            _yaml_number(self.close_cost),
            _yaml_number(self.min_cost),
            str(self.topk),
            str(self.n_drop),
        )
        return dict(zip(TRADING_RULE_ENV_KEYS, values, strict=True))


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _yaml_number(value: float) -> str:
    # Plain decimal text: YAML loaders disagree on exponent forms such as 5e-05.
    return format(Decimal(str(value)).normalize(), "f")

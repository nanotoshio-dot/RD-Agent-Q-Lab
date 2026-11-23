from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class MarketContext:
    """Immutable context describing a market and its research environment."""

    market_id: str
    provider_uri: Path
    region: str
    currency: str
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

    def __post_init__(self) -> None:
        required_strings = {
            "market_id": self.market_id,
            "region": self.region,
            "currency": self.currency,
        }

        for field_name, value in required_strings.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        path_fields = (
            "provider_uri",
            "workspace_path",
            "cache_path",
            "memory_path",
            "calendar",
        )

        for field_name in path_fields:
            if not isinstance(getattr(self, field_name), Path):
                raise TypeError(f"{field_name} must be a pathlib.Path")

        if not isinstance(self.universe, tuple) or not self.universe or any(
            not isinstance(item, str) or not item.strip()
            for item in self.universe
        ):
            raise ValueError("universe must contain non-empty instrument groups")

        date_fields = (
            self.train_start,
            self.train_end,
            self.valid_start,
            self.valid_end,
            self.test_start,
            self.test_end,
        )

        if any(type(value) is not date for value in date_fields):
            raise TypeError("Research period fields must be datetime.date objects")

        if not self.train_start <= self.train_end:
            raise ValueError("Invalid training period")

        if not self.valid_start <= self.valid_end:
            raise ValueError("Invalid validation period")

        if not self.test_start <= self.test_end:
            raise ValueError("Invalid test period")

        if not (
            self.train_end < self.valid_start
            and self.valid_end < self.test_start
        ):
            raise ValueError(
                "Training, validation and test periods must be chronological "
                "and non-overlapping"
            )

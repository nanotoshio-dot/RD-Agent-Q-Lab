import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from rdagent.core.market.context import MarketContext
from rdagent.core.market.trading_rules import TRADING_RULE_ENV_KEYS


@dataclass(frozen=True)
class MarketProcessConfig:
    context: MarketContext
    model_env_type: str = "local"
    factor_python_bin: str = "python"

    def __post_init__(self) -> None:
        if not isinstance(self.context, MarketContext):
            raise TypeError("context must be a MarketContext")  # noqa: EM101, TRY003
        if not self.model_env_type.strip() or not self.factor_python_bin.strip():
            raise ValueError("Environment settings cannot be empty")  # noqa: EM101, TRY003

    def build_env(self, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
        ctx = self.context
        env = dict(base_env) if base_env is not None else {}
        # Stale rule values from the parent must not reach a context that keeps the template defaults.
        for key in TRADING_RULE_ENV_KEYS:
            env.pop(key, None)

        # RD-Agent cache keys ignore data, dates and rules, so each configuration gets its own cache
        # folder. Factor data depends on the provider only, so dates and rules can share it.
        # The calendar's last day changes whenever the provider is updated in place.
        data_identity = {
            "provider_uri": str(ctx.provider_uri),
            "region": ctx.region,
            "calendar_last_day": _last_line(ctx.calendar),
        }
        run_identity = {
            **data_identity,
            "qlib_market": ctx.qlib_market,
            "benchmark": ctx.benchmark,
            "periods": [
                period.isoformat()
                for period in (
                    ctx.train_start,
                    ctx.train_end,
                    ctx.valid_start,
                    ctx.valid_end,
                    ctx.test_start,
                    ctx.test_end,
                )
            ],
            "trading_rules": None if ctx.trading_rules is None else ctx.trading_rules.to_env(),
        }
        data_key = _short_hash(data_identity)

        env.update(
            {
                "WORKSPACE_PATH": str(ctx.workspace_path),
                "PICKLE_CACHE_FOLDER_PATH_STR": str(ctx.cache_path / _short_hash(run_identity)),
                "QLIB_FACTOR_TRAIN_START": ctx.train_start.isoformat(),
                "QLIB_FACTOR_TRAIN_END": ctx.train_end.isoformat(),
                "QLIB_FACTOR_VALID_START": ctx.valid_start.isoformat(),
                "QLIB_FACTOR_VALID_END": ctx.valid_end.isoformat(),
                "QLIB_FACTOR_TEST_START": ctx.test_start.isoformat(),
                "QLIB_FACTOR_TEST_END": ctx.test_end.isoformat(),
                "QLIB_MODEL_TRAIN_START": ctx.train_start.isoformat(),
                "QLIB_MODEL_TRAIN_END": ctx.train_end.isoformat(),
                "QLIB_MODEL_VALID_START": ctx.valid_start.isoformat(),
                "QLIB_MODEL_VALID_END": ctx.valid_end.isoformat(),
                "QLIB_MODEL_TEST_START": ctx.test_start.isoformat(),
                "QLIB_MODEL_TEST_END": ctx.test_end.isoformat(),
                "QLIB_QUANT_TRAIN_START": ctx.train_start.isoformat(),
                "QLIB_QUANT_TRAIN_END": ctx.train_end.isoformat(),
                "QLIB_QUANT_VALID_START": ctx.valid_start.isoformat(),
                "QLIB_QUANT_VALID_END": ctx.valid_end.isoformat(),
                "QLIB_QUANT_TEST_START": ctx.test_start.isoformat(),
                "QLIB_QUANT_TEST_END": ctx.test_end.isoformat(),
                "FACTOR_CoSTEER_DATA_FOLDER": str(ctx.workspace_path / f"factor_data-{data_key}"),
                "FACTOR_CoSTEER_DATA_FOLDER_DEBUG": str(ctx.workspace_path / f"factor_data_debug-{data_key}"),
                "FACTOR_CoSTEER_PYTHON_BIN": self.factor_python_bin,
                "MODEL_CoSTEER_ENV_TYPE": self.model_env_type,
                "RD_MARKET_ID": ctx.market_id,
                "RD_MARKET_PROVIDER_URI": str(ctx.provider_uri),
                "RD_MARKET_REGION": ctx.region,
                "RD_MARKET_QLIB_MARKET": ctx.qlib_market,
                "RD_MARKET_BENCHMARK": ctx.benchmark,
                "RD_MARKET_CURRENCY": ctx.currency,
                "RD_MARKET_UNIVERSE": json.dumps(ctx.universe),
                "RD_MARKET_CALENDAR": str(ctx.calendar),
                "RD_MARKET_MEMORY_PATH": str(ctx.memory_path),
                "RD_MARKET_CACHE_PATH": str(ctx.cache_path),
            },
        )
        if ctx.trading_rules is not None:
            env.update(ctx.trading_rules.to_env())
        return env


def _short_hash(values: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:12]


def _last_line(path: Path) -> str | None:
    if not path.is_file():
        return None
    lines = path.read_text().split()
    return lines[-1] if lines else None

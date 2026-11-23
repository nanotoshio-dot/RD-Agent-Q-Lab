import json
from collections.abc import Mapping
from dataclasses import dataclass

from rdagent.core.market.context import MarketContext


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

        env.update(
            {
                "WORKSPACE_PATH": str(ctx.workspace_path),
                "PICKLE_CACHE_FOLDER_PATH_STR": str(ctx.cache_path),
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
                "FACTOR_CoSTEER_DATA_FOLDER": str(ctx.workspace_path / "factor_data"),
                "FACTOR_CoSTEER_DATA_FOLDER_DEBUG": str(ctx.workspace_path / "factor_data_debug"),
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
            }
        )
        return env

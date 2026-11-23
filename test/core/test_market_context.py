import json
from datetime import date
from pathlib import Path

import pytest

from rdagent.core.market import MarketContext, MarketProcessConfig


def make_context(**overrides):
    values = {
        "market_id": "us-sp500",
        "provider_uri": Path("/data/us"),
        "region": "us",
        "currency": "USD",
        "qlib_market": "us",
        "benchmark": "^GSPC",
        "universe": ("sp500",),
        "workspace_path": Path("/work/us-sp500"),
        "cache_path": Path("/cache/us-sp500"),
        "memory_path": Path("/memory/us-sp500"),
        "calendar": Path("/data/us/calendars/day.txt"),
        "train_start": date(2010, 1, 1),
        "train_end": date(2018, 12, 31),
        "valid_start": date(2019, 1, 1),
        "valid_end": date(2020, 12, 31),
        "test_start": date(2021, 1, 1),
        "test_end": date(2022, 12, 31),
    }
    values.update(overrides)
    return MarketContext(**values)


def test_market_context_accepts_valid_configuration():
    context = make_context()

    assert context.market_id == "us-sp500"
    assert context.universe == ("sp500",)
    assert context.provider_uri == Path("/data/us")


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("market_id", "  ", ValueError),
        ("benchmark", "", ValueError),
        ("provider_uri", "/data/us", TypeError),
        ("universe", (), ValueError),
        ("universe", ("sp500", ""), ValueError),
        ("train_start", "2010-01-01", TypeError),
    ],
)
def test_market_context_rejects_invalid_fields(field, value, error):
    with pytest.raises(error):
        make_context(**{field: value})


def test_market_context_rejects_overlapping_periods():
    with pytest.raises(ValueError, match="chronological and non-overlapping"):
        make_context(valid_start=date(2018, 12, 1))


def test_market_process_config_builds_isolated_environment():
    context = make_context()
    config = MarketProcessConfig(
        context=context,
        model_env_type="local",
        factor_python_bin="/opt/rdagent/bin/python",
    )

    env = config.build_env({"EXISTING_SETTING": "keep-me"})

    assert env["EXISTING_SETTING"] == "keep-me"
    assert env["WORKSPACE_PATH"] == "/work/us-sp500"
    assert env["PICKLE_CACHE_FOLDER_PATH_STR"] == "/cache/us-sp500"
    assert env["RD_MARKET_ID"] == "us-sp500"
    assert env["RD_MARKET_PROVIDER_URI"] == "/data/us"
    assert env["RD_MARKET_BENCHMARK"] == "^GSPC"
    assert json.loads(env["RD_MARKET_UNIVERSE"]) == ["sp500"]
    assert env["FACTOR_CoSTEER_PYTHON_BIN"] == "/opt/rdagent/bin/python"
    assert env["MODEL_CoSTEER_ENV_TYPE"] == "local"


def test_market_process_config_does_not_mutate_base_environment():
    base_env = {"EXISTING_SETTING": "original"}
    config = MarketProcessConfig(context=make_context())

    env = config.build_env(base_env)

    assert base_env == {"EXISTING_SETTING": "original"}
    assert env is not base_env


def test_market_process_config_rejects_wrong_context_type():
    with pytest.raises(TypeError, match="context must be a MarketContext"):
        MarketProcessConfig(context=object())

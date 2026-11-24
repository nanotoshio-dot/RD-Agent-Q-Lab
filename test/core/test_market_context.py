import json
from datetime import date
from pathlib import Path

import pytest

from rdagent.core.market import MarketContext, MarketProcessConfig, TradingRules
from rdagent.core.market.trading_rules import TRADING_RULE_ENV_KEYS

pytestmark = pytest.mark.offline

US_RULES = TradingRules(limit_threshold=None, open_cost=0.001, close_cost=0.001, min_cost=1, topk=50, n_drop=5)


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
    assert Path(env["PICKLE_CACHE_FOLDER_PATH_STR"]).parent == Path("/cache/us-sp500")
    assert Path(env["FACTOR_CoSTEER_DATA_FOLDER"]).parent == Path("/work/us-sp500")
    assert Path(env["FACTOR_CoSTEER_DATA_FOLDER_DEBUG"]).parent == Path("/work/us-sp500")
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


def test_market_context_rejects_rules_of_the_wrong_type():
    with pytest.raises(TypeError, match="trading_rules"):
        make_context(trading_rules={"topk": 50})


def test_no_price_limit_is_only_accepted_in_region_us():
    # Qlib replaces a null limit with the region default (0.095 for cn), so null means "no limit" only in us.
    with pytest.raises(ValueError, match="region us"):
        make_context(region="cn", trading_rules=US_RULES)


def test_trading_rules_reach_the_environment():
    env = MarketProcessConfig(context=make_context(trading_rules=US_RULES)).build_env({})

    assert {key: env[key] for key in TRADING_RULE_ENV_KEYS} == US_RULES.to_env()


def test_context_without_rules_drops_stale_rule_variables():
    stale = {"RD_MARKET_TOPK": "20", "RD_MARKET_OPEN_COST": "0.001"}

    env = MarketProcessConfig(context=make_context()).build_env(stale)

    assert not set(TRADING_RULE_ENV_KEYS) & set(env)


def test_cache_folder_changes_with_the_research_configuration():
    def cache_folder(context):
        return MarketProcessConfig(context=context).build_env({})["PICKLE_CACHE_FOLDER_PATH_STR"]

    base = cache_folder(make_context())
    variants = {
        cache_folder(make_context(test_end=date(2022, 6, 30))),
        cache_folder(make_context(trading_rules=US_RULES)),
        cache_folder(make_context(benchmark="^NDX")),
        cache_folder(make_context(provider_uri=Path("/data/us-2026"))),
    }

    assert cache_folder(make_context()) == base
    assert base not in variants
    assert len(variants) == 4


def test_factor_data_folder_changes_when_the_provider_is_updated_in_place(tmp_path):
    calendar = tmp_path / "us" / "calendars" / "day.txt"
    calendar.parent.mkdir(parents=True)

    def data_folder():
        context = make_context(provider_uri=tmp_path / "us", calendar=calendar)
        return MarketProcessConfig(context=context).build_env({})["FACTOR_CoSTEER_DATA_FOLDER"]

    calendar.write_text("2020-11-09\n2020-11-10\n")
    before = data_folder()
    calendar.write_text("2020-11-09\n2020-11-10\n2020-11-11\n")

    assert data_folder() != before


def test_factor_data_folder_follows_the_provider_but_not_the_dates():
    def data_folders(context):
        env = MarketProcessConfig(context=context).build_env({})
        return env["FACTOR_CoSTEER_DATA_FOLDER"], env["FACTOR_CoSTEER_DATA_FOLDER_DEBUG"]

    base = data_folders(make_context())

    assert data_folders(make_context(test_end=date(2022, 6, 30), trading_rules=US_RULES)) == base
    assert data_folders(make_context(provider_uri=Path("/data/us-2026"))) != base
    assert base[0] != base[1]

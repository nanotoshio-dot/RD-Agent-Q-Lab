from datetime import date
from pathlib import Path

import pytest

from rdagent.app.market_launcher import (
    MARKET_PROFILES,
    build_child_environment,
    build_market_context,
    main,
)


@pytest.mark.parametrize("market_id", sorted(MARKET_PROFILES))
def test_market_profiles_build_independent_contexts(market_id, tmp_path):
    context = build_market_context(
        market_id=market_id,
        provider_uri=tmp_path / "provider",
        train_start=date(2010, 1, 1),
        train_end=date(2017, 12, 31),
        valid_start=date(2018, 1, 1),
        valid_end=date(2019, 12, 31),
        test_start=date(2020, 1, 1),
        test_end=date(2021, 12, 31),
        root=tmp_path / "markets",
    )

    profile = MARKET_PROFILES[market_id]
    assert context.market_id == market_id
    assert context.region == profile.region
    assert context.currency == profile.currency
    assert context.qlib_market == profile.qlib_market
    assert context.benchmark == profile.benchmark
    assert context.universe == profile.universe
    assert context.calendar == tmp_path / "provider" / "calendars" / "day.txt"
    assert context.workspace_path == tmp_path / "markets" / market_id / "workspace"
    assert context.cache_path == tmp_path / "markets" / market_id / "cache"
    assert context.memory_path == tmp_path / "markets" / market_id / "memory"


def test_us_markets_share_provider_location_but_not_research_directories(tmp_path):
    common = {
        "provider_uri": tmp_path / "us_data",
        "train_start": date(2010, 1, 1),
        "train_end": date(2017, 12, 31),
        "valid_start": date(2018, 1, 1),
        "valid_end": date(2019, 12, 31),
        "test_start": date(2020, 1, 1),
        "test_end": date(2021, 12, 31),
        "root": tmp_path / "markets",
    }
    sp500 = build_market_context(market_id="us-sp500", **common)
    nasdaq = build_market_context(market_id="us-nasdaq100", **common)

    assert sp500.provider_uri == nasdaq.provider_uri
    assert sp500.universe != nasdaq.universe
    assert sp500.benchmark != nasdaq.benchmark
    assert sp500.workspace_path != nasdaq.workspace_path
    assert sp500.cache_path != nasdaq.cache_path
    assert sp500.memory_path != nasdaq.memory_path


def test_unknown_market_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown market"):
        build_market_context(
            market_id="unknown",
            provider_uri=tmp_path,
            train_start=date(2010, 1, 1),
            train_end=date(2017, 12, 31),
            valid_start=date(2018, 1, 1),
            valid_end=date(2019, 12, 31),
            test_start=date(2020, 1, 1),
            test_end=date(2021, 12, 31),
        )


def test_child_environment_isolated_from_parent_environment(tmp_path):
    context = build_market_context(
        market_id="us-sp500",
        provider_uri=tmp_path / "provider",
        train_start=date(2010, 1, 1),
        train_end=date(2017, 12, 31),
        valid_start=date(2018, 1, 1),
        valid_end=date(2019, 12, 31),
        test_start=date(2020, 1, 1),
        test_end=date(2021, 12, 31),
        root=tmp_path / "markets",
    )
    parent = {"KEEP_ME": "yes", "WORKSPACE_PATH": "/parent/workspace"}

    child = build_child_environment(context, base_env=parent)

    assert parent == {"KEEP_ME": "yes", "WORKSPACE_PATH": "/parent/workspace"}
    assert child["KEEP_ME"] == "yes"
    assert child["WORKSPACE_PATH"] == str(context.workspace_path)
    assert child["RD_MARKET_ID"] == "us-sp500"
    assert child["RD_MARKET_BENCHMARK"] == "^GSPC"
    assert child["LOG_TRACE_PATH"] == str(tmp_path / "markets" / "us-sp500" / "log")
    assert child["STEP_SEMAPHORE"] == "1"


def test_launcher_defaults_to_validation_only(tmp_path, capsys, monkeypatch):
    def unexpected_run(*args, **kwargs):
        pytest.fail("Research loop must not start without --execute")

    monkeypatch.setattr("rdagent.app.market_launcher.subprocess.run", unexpected_run)
    result = main([
        "--market", "china",
        "--provider-uri", str(tmp_path / "cn_data"),
        "--train-start", "2010-01-01",
        "--train-end", "2017-12-31",
        "--valid-start", "2018-01-01",
        "--valid-end", "2019-12-31",
        "--test-start", "2020-01-01",
        "--test-end", "2021-12-31",
    ])

    output = capsys.readouterr().out
    assert result == 0
    assert "Validation only" in output
    assert "Benchmark: SH000300" in output
    assert "Add --execute" in output

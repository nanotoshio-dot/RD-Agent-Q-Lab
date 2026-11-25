import os
from datetime import date
from pathlib import Path

import pytest

from rdagent.app.market_launcher import (
    FACTOR_DATA_START,
    MARKET_PROFILES,
    build_child_environment,
    build_market_context,
    load_base_environment,
    main,
)
from rdagent.core.market.trading_rules import TRADING_RULE_ENV_KEYS

pytestmark = pytest.mark.offline

GENERATE_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "rdagent"
    / "scenarios"
    / "qlib"
    / "experiment"
    / "factor_data_template"
    / "generate.py"
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
    assert child["MLFLOW_ALLOW_FILE_STORE"] == "true"
    assert build_child_environment(context, base_env={"MLFLOW_ALLOW_FILE_STORE": "false"})[
        "MLFLOW_ALLOW_FILE_STORE"
    ] == ("false")


def test_launcher_defaults_to_validation_only(tmp_path, capsys, monkeypatch):
    def unexpected_run(*args, **kwargs):
        pytest.fail("Research loop must not start without --execute")

    monkeypatch.setattr("rdagent.app.market_launcher.subprocess.run", unexpected_run)
    result = main(
        [
            "--market",
            "china",
            "--provider-uri",
            str(tmp_path / "cn_data"),
            "--train-start",
            "2010-01-01",
            "--train-end",
            "2017-12-31",
            "--valid-start",
            "2018-01-01",
            "--valid-end",
            "2019-12-31",
            "--test-start",
            "2020-01-01",
            "--test-end",
            "2021-12-31",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Validation only" in output
    assert "Benchmark: SH000300" in output
    assert "Add --execute" in output


def launcher_args(market, provider, **dates):
    values = {
        "train-start": "2010-01-01",
        "train-end": "2014-12-31",
        "valid-start": "2015-01-01",
        "valid-end": "2016-12-31",
        "test-start": "2017-01-01",
        "test-end": "2020-06-30",
    }
    values.update({key.replace("_", "-"): value for key, value in dates.items()})
    args = ["--market", market, "--provider-uri", str(provider)]
    for key, value in values.items():
        args += [f"--{key}", value]
    return args


def write_calendar(provider, first, last, future=None):
    calendars = provider / "calendars"
    calendars.mkdir(parents=True)
    (calendars / "day.txt").write_text(f"{first}\n2015-06-01\n{last}\n")
    if future is not None:
        days = sorted({first, last, future})
        (calendars / "day_future.txt").write_text("\n".join(days) + "\n")


def test_china_profile_keeps_the_template_rules():
    assert MARKET_PROFILES["china"].trading_rules.to_env() == {
        "RD_MARKET_LIMIT_THRESHOLD": "0.095",
        "RD_MARKET_OPEN_COST": "0.0005",
        "RD_MARKET_CLOSE_COST": "0.0015",
        "RD_MARKET_MIN_COST": "5",
        "RD_MARKET_TOPK": "50",
        "RD_MARKET_N_DROP": "5",
    }


@pytest.mark.parametrize(("market_id", "topk", "n_drop"), [("us-sp500", 50, 5), ("us-nasdaq100", 20, 2)])
def test_us_profiles_have_no_price_limit_and_their_own_portfolio_size(market_id, topk, n_drop):
    rules = MARKET_PROFILES[market_id].trading_rules

    assert rules.limit_threshold is None
    assert (rules.topk, rules.n_drop) == (topk, n_drop)


def test_child_environment_carries_the_market_rules(tmp_path):
    context = build_market_context(
        market_id="us-nasdaq100",
        provider_uri=tmp_path / "us_data",
        train_start=date(2010, 1, 1),
        train_end=date(2014, 12, 31),
        valid_start=date(2015, 1, 1),
        valid_end=date(2016, 12, 31),
        test_start=date(2017, 1, 1),
        test_end=date(2020, 6, 30),
        root=tmp_path / "markets",
    )

    child = build_child_environment(context, base_env={"RD_MARKET_TOPK": "999"})

    assert {key: child[key] for key in TRADING_RULE_ENV_KEYS} == MARKET_PROFILES["us-nasdaq100"].trading_rules.to_env()


def test_docker_is_refused_for_us_markets_even_without_execute(tmp_path):
    with pytest.raises(SystemExit, match="cannot run in Docker"):
        main([*launcher_args("us-sp500", tmp_path / "us_data"), "--model-env", "docker"])


def test_training_before_the_factor_data_start_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="--train-start"):
        main(launcher_args("us-sp500", tmp_path / "us_data", train_start="2008-01-01"))


@pytest.mark.parametrize("test_end", ["2020-11-10", "2021-01-01"])
def test_test_period_must_end_before_the_last_trading_day(tmp_path, test_end):
    provider = tmp_path / "us_data"
    write_calendar(provider, "1999-12-31", "2020-11-10")

    with pytest.raises(SystemExit, match="--test-end"):
        main(launcher_args("us-sp500", provider, test_end=test_end))


def test_a_future_calendar_allows_testing_up_to_the_last_trading_day(tmp_path, capsys):
    provider = tmp_path / "cn_data"
    write_calendar(provider, "2005-01-04", "2020-11-10", future="2099-12-31")

    assert main(launcher_args("china", provider, test_end="2020-11-10")) == 0
    assert "Validation only" in capsys.readouterr().out


def test_a_future_calendar_must_extend_past_the_test_period(tmp_path):
    provider = tmp_path / "cn_data"
    write_calendar(provider, "2005-01-04", "2020-11-10", future="2020-11-10")

    with pytest.raises(SystemExit, match="day_future.txt"):
        main(launcher_args("china", provider, test_end="2020-11-10"))


def test_relative_provider_paths_are_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    context = build_market_context(
        market_id="us-sp500",
        provider_uri=Path("us_data"),
        train_start=date(2010, 1, 1),
        train_end=date(2014, 12, 31),
        valid_start=date(2015, 1, 1),
        valid_end=date(2016, 12, 31),
        test_start=date(2017, 1, 1),
        test_end=date(2020, 6, 30),
        root=tmp_path / "markets",
    )

    assert context.provider_uri == tmp_path / "us_data"
    assert context.calendar == tmp_path / "us_data" / "calendars" / "day.txt"


def test_training_before_the_provider_calendar_is_refused(tmp_path):
    provider = tmp_path / "us_data"
    write_calendar(provider, "2012-01-03", "2020-11-10")

    with pytest.raises(SystemExit, match="--train-start"):
        main(launcher_args("us-sp500", provider))


def test_factor_data_start_matches_the_generation_script():
    assert f'"{FACTOR_DATA_START.isoformat()}"' in GENERATE_SCRIPT.read_text()


def test_dotenv_values_are_overridden_by_the_process_environment(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("CHAT_MODEL=from-file\nEMBEDDING_MODEL=from-file\nEMPTY_VALUE\n")
    monkeypatch.setenv("CHAT_MODEL", "from-shell")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    env = load_base_environment(dotenv)

    assert env["CHAT_MODEL"] == "from-shell"
    assert env["EMBEDDING_MODEL"] == "from-file"
    assert "EMPTY_VALUE" not in env
    assert "EMBEDDING_MODEL" not in os.environ


def extended_context(tmp_path, test_start, test_end):
    provider = tmp_path / "us_data_2026"
    (provider / "_sources").mkdir(parents=True)
    (provider / "_sources" / "cutover.txt").write_text("2020-11-10\n")
    return build_market_context(
        market_id="us-sp500",
        provider_uri=provider,
        train_start=date(2010, 1, 1),
        train_end=date(2014, 12, 31),
        valid_start=date(2015, 1, 1),
        valid_end=date(2016, 12, 31),
        test_start=test_start,
        test_end=test_end,
        root=tmp_path / "markets",
    )


def test_a_test_period_crossing_the_extension_day_is_warned_about(tmp_path):
    from rdagent.app.market_launcher import extension_warning

    crossing = extended_context(tmp_path, date(2020, 1, 1), date(2026, 11, 20))

    assert "crosses 2020-11-10" in extension_warning(crossing)


def test_a_provider_extended_twice_still_warns_about_the_first_extension_day(tmp_path):
    from rdagent.app.market_launcher import extension_warning

    crossing = extended_context(tmp_path, date(2019, 1, 2), date(2026, 11, 20))
    sources = crossing.provider_uri / "_sources"
    (sources / "cutover.txt").write_text("2026-11-23\n")
    (sources / "earlier_cutovers.txt").write_text("2020-11-10\n")

    assert "crosses 2020-11-10" in extension_warning(crossing)


def test_a_test_period_after_the_extension_day_needs_no_warning(tmp_path):
    from rdagent.app.market_launcher import extension_warning

    after = extended_context(tmp_path, date(2021, 1, 4), date(2026, 11, 20))

    assert extension_warning(after) is None


def test_a_run_id_gives_the_run_its_own_log_folder(tmp_path):
    context = extended_context(tmp_path, date(2021, 1, 4), date(2026, 11, 20))

    child = build_child_environment(context, base_env={}, run_id="2026-11-25_14-00-00")

    assert child["LOG_TRACE_PATH"] == str(tmp_path / "markets" / "us-sp500" / "log" / "2026-11-25_14-00-00")


def test_a_cut_off_knowledge_file_is_set_aside_and_a_sound_one_is_kept(tmp_path, monkeypatch):
    from rdagent.app.market_launcher import set_aside_damaged_knowledge
    from rdagent.core import serialization
    from rdagent.core.conf import RD_AGENT_SETTINGS

    monkeypatch.setattr(RD_AGENT_SETTINGS, "artifact_signing_key", "a key for the launcher tests, 32+ bytes long")
    sound = serialization.dumps({"learned": "use $close"})
    (tmp_path / "factor_coder_knowledge.pkl").write_bytes(sound)
    (tmp_path / "model_coder_knowledge.pkl").write_bytes(sound[: len(sound) // 2])  # stopped while writing

    moved = set_aside_damaged_knowledge(tmp_path, "run-7")

    assert moved == [tmp_path / "model_coder_knowledge.pkl.damaged-run-7"]
    assert (tmp_path / "factor_coder_knowledge.pkl").read_bytes() == sound
    assert not (tmp_path / "model_coder_knowledge.pkl").exists()


def launch(tmp_path, *extra):
    provider = tmp_path / "cn_data"
    (provider / "calendars").mkdir(parents=True, exist_ok=True)
    (provider / "calendars" / "day.txt").write_text("2010-01-04\n2024-12-31\n2025-01-02\n")
    (provider / "instruments").mkdir(exist_ok=True)
    (provider / "instruments" / "csi300.txt").write_text("SH600000\t2005-04-08\t2099-12-31\n")
    periods = "--train-start 2010-01-04 --train-end 2017-12-31 --valid-start 2018-01-01 --valid-end 2019-12-31"
    periods += " --test-start 2020-01-01 --test-end 2024-12-30"
    root = ["--root", str(tmp_path / "markets")]
    return main(["--market", "china", "--provider-uri", str(provider), *periods.split(), *root, "--execute", *extra])


def test_an_executed_run_records_its_setup_and_ends_with_a_report(tmp_path, monkeypatch, capsys):
    import json
    import types

    import rdagent.app.market_report as market_report

    started = {}

    def fake_run(command, env, check):
        started["log"] = Path(env["LOG_TRACE_PATH"])
        started["record"] = json.loads((started["log"] / "run.json").read_text())
        return types.SimpleNamespace(returncode=3)

    reported = []
    monkeypatch.setattr("rdagent.app.market_launcher.subprocess.run", fake_run)
    monkeypatch.setattr(market_report, "write_report", lambda *args: reported.append(args) or "report.md")

    assert launch(tmp_path) == 3  # the run's own exit code, whatever the report did

    market_root = tmp_path / "markets" / "china"
    assert started["log"].parent == market_root / "log"
    assert started["record"]["Market"] == "china"
    assert started["record"]["Test period"] == "2020-01-01 to 2024-12-30"
    assert reported == [("china", market_root, started["log"].name)]
    assert "Report: report.md" in capsys.readouterr().out


def test_a_failing_report_does_not_change_the_exit_code_and_can_be_switched_off(tmp_path, monkeypatch, capsys):
    import types

    import rdagent.app.market_report as market_report

    def no_loops(*args):
        raise SystemExit("No research loops were logged")

    monkeypatch.setattr(
        "rdagent.app.market_launcher.subprocess.run", lambda *a, **k: types.SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(market_report, "write_report", no_loops)

    assert launch(tmp_path) == 0
    assert "No report was written: No research loops were logged" in capsys.readouterr().out

    assert launch(tmp_path, "--no-report") == 0
    output = capsys.readouterr().out
    assert "Report:" not in output
    assert "No report was written" not in output


def test_a_provider_without_the_market_members_is_refused_before_the_run_starts(tmp_path, monkeypatch):
    def unexpected_run(*args, **kwargs):
        pytest.fail("The research loop must not start without the market's member file")

    monkeypatch.setattr("rdagent.app.market_launcher.subprocess.run", unexpected_run)
    provider = tmp_path / "cn_data"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text("2010-01-04\n2024-12-31\n2025-01-02\n")
    arguments = "--market china --train-start 2010-01-04 --train-end 2017-12-31 --valid-start 2018-01-01"
    arguments += " --valid-end 2019-12-31 --test-start 2020-01-01 --test-end 2024-12-30 --execute"

    with pytest.raises(SystemExit, match="member file does not exist"):
        main([*arguments.split(), "--provider-uri", str(provider), "--root", str(tmp_path / "markets")])


def test_knowledge_learned_with_another_model_is_set_aside(tmp_path):
    from rdagent.app.market_launcher import knowledge_stamp, retire_stale_knowledge

    first = knowledge_stamp(
        {"CHAT_MODEL": "ollama_chat/qwen2.5-coder:7b-8k", "EMBEDDING_MODEL": "ollama/nomic-embed-text"}
    )
    assert retire_stale_knowledge(tmp_path, first, "run-1") == []  # nothing learned yet
    (tmp_path / "factor_coder_knowledge.pkl").write_bytes(b"learned in run 1")

    assert retire_stale_knowledge(tmp_path, first, "run-2") == []  # same models: kept
    assert (tmp_path / "factor_coder_knowledge.pkl").exists()

    other = knowledge_stamp(
        {"CHAT_MODEL": "ollama_chat/qwen2.5-coder:7b-8k", "EMBEDDING_MODEL": "ollama/mxbai-embed-large"}
    )
    assert retire_stale_knowledge(tmp_path, other, "run-3") == [tmp_path / "factor_coder_knowledge.pkl.retired-run-3"]
    assert not (tmp_path / "factor_coder_knowledge.pkl").exists()
    assert retire_stale_knowledge(tmp_path, other, "run-4") == []


def test_a_second_run_of_a_market_is_refused_while_the_first_one_lives(tmp_path):
    from rdagent.app.market_launcher import claim_market

    marker = claim_market(tmp_path)
    assert marker.read_text() == str(os.getpid())

    with pytest.raises(SystemExit, match="already going"):
        claim_market(tmp_path)

    marker.write_text("999999999")  # a run that was killed left its marker
    assert claim_market(tmp_path).read_text() == str(os.getpid())


def test_the_market_is_released_when_the_run_ends(tmp_path, monkeypatch):
    import types

    seen = {}

    def fake_run(command, env, check):
        seen["marker"] = (tmp_path / "markets" / "china" / "run.pid").read_text()
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("rdagent.app.market_launcher.subprocess.run", fake_run)

    assert launch(tmp_path, "--no-report") == 0

    assert seen["marker"] == str(os.getpid())
    assert not (tmp_path / "markets" / "china" / "run.pid").exists()
    assert (tmp_path / "markets" / "china" / "memory" / "knowledge_stamp.json").is_file()


def test_signatures_are_checked_with_the_key_the_child_uses(monkeypatch):
    from rdagent.app.market_launcher import adopt_signing_settings
    from rdagent.core.conf import RD_AGENT_SETTINGS

    monkeypatch.setattr(RD_AGENT_SETTINGS, "artifact_signing_key", "")
    monkeypatch.setattr(
        RD_AGENT_SETTINGS, "artifact_signing_key_path", Path("/home/user/.rdagent/artifact_signing.key")
    )

    adopt_signing_settings({"ARTIFACT_SIGNING_KEY": "a key that only the .env file holds, 32+ bytes"})
    assert RD_AGENT_SETTINGS.artifact_signing_key == "a key that only the .env file holds, 32+ bytes"

    adopt_signing_settings({"ARTIFACT_SIGNING_KEY_PATH": "/keys/market.key"})
    assert RD_AGENT_SETTINGS.artifact_signing_key_path == Path("/keys/market.key")

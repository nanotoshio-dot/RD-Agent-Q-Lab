import numpy as np
import pandas as pd
import pytest

import rdagent.scenarios.qlib.developer.factor_pruning as pruning
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
from rdagent.components.coder.factor_coder.evaluators import FactorSingleFeedback
from rdagent.core.exception import FactorEmptyError
from rdagent.scenarios.qlib.developer.factor_pruning import (
    FactorPruningSettings,
    PruningFactorRunner,
    Universe,
    build_universe,
    daily_rank_corr,
    judge_factor,
    t_statistic,
)
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment

pytestmark = pytest.mark.offline

DAYS = pd.date_range("2015-01-01", periods=120, freq="B", name="datetime")
STOCKS = [f"S{number:02d}" for number in range(30)]
INDEX = pd.MultiIndex.from_product([DAYS, STOCKS], names=["datetime", "instrument"])


def series(values, name="factor"):
    return pd.Series(np.asarray(values, dtype=float), index=INDEX, name=name)


@pytest.fixture
def market():
    """A label, a factor that predicts it and a factor that is pure noise."""
    rng = np.random.default_rng(7)
    signal = series(rng.normal(size=len(INDEX)), "signal")
    label = (0.2 * signal + series(rng.normal(size=len(INDEX)))).rename("label")
    noise = series(rng.normal(size=len(INDEX)), "noise")
    return Universe(priced=INDEX, label=label), signal, noise


def test_rank_correlation_ignores_monotone_changes_and_sees_the_sign(market):
    _, signal, noise = market

    assert daily_rank_corr(signal, signal**3).mean() == pytest.approx(1.0)
    assert daily_rank_corr(signal, -signal).mean() == pytest.approx(-1.0)
    assert abs(daily_rank_corr(signal, noise).mean()) < 0.05
    assert len(daily_rank_corr(signal, noise)) == len(DAYS)


def test_rank_correlation_stays_fast_on_a_market_sized_history():
    # A correlation computed group by group took minutes on the S&P 500 (2,500 days of 500 stocks).
    import time

    days = pd.date_range("2010-01-01", periods=600, freq="B", name="datetime")
    index = pd.MultiIndex.from_product([days, range(300)], names=["datetime", "instrument"])
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(size=len(index)), index=index)
    y = pd.Series(rng.normal(size=len(index)), index=index)

    started = time.perf_counter()
    corr = daily_rank_corr(x, y)

    assert len(corr) == 600
    assert time.perf_counter() - started < 10


def test_days_with_too_few_stocks_are_left_out(market):
    _, signal, noise = market
    few = signal.where(signal.index.get_level_values("instrument").isin(STOCKS[:5]))

    assert daily_rank_corr(few, noise).empty


def test_t_statistic_needs_enough_days():
    assert t_statistic(pd.Series([0.05] * 10)) == 0.0
    assert t_statistic(pd.Series([0.05, 0.03] * 50)) > 30
    assert t_statistic(pd.Series([0.0] * 100)) == 0.0
    assert t_statistic(pd.Series([-0.05] * 100)) == -np.inf  # the same value every day, without rounding noise


def test_a_predictive_factor_is_kept(market):
    universe, signal, _ = market

    verdict = judge_factor("task", signal, universe, {}, FactorPruningSettings())

    assert verdict.reason is None
    assert verdict.coverage == 1.0
    assert verdict.rank_ic > 0.1
    assert verdict.rank_ic_t > 10


def test_a_factor_without_predictive_power_is_weak(market):
    universe, _, noise = market

    verdict = judge_factor("task", noise, universe, {}, FactorPruningSettings())

    assert "no measurable predictive power" in verdict.reason
    assert abs(verdict.rank_ic_t) < 2


def test_weakness_is_not_judged_when_the_threshold_is_zero(market):
    universe, _, noise = market

    verdict = judge_factor("task", noise, universe, {}, FactorPruningSettings(min_abs_rank_ic_t=0))

    assert verdict.reason is None


def test_a_negated_or_reshaped_copy_of_a_kept_factor_is_redundant(market):
    universe, signal, _ = market
    library = {"SOTA_momentum": signal}

    for copy in (-signal, signal**3, signal * 2 + 1):
        verdict = judge_factor("task", copy.rename("copy"), universe, library, FactorPruningSettings())

        assert "ranks stocks like the factor 'SOTA_momentum'" in verdict.reason
        assert verdict.abs_rank_corr == pytest.approx(1.0)


def test_a_factor_with_few_values_is_pruned_before_anything_else(market):
    universe, signal, _ = market
    sparse = signal.where(signal.index.get_level_values("datetime") >= DAYS[90])  # 25% of the days

    verdict = judge_factor("task", sparse, universe, {}, FactorPruningSettings())

    assert "on only 25% of the days" in verdict.reason
    assert np.isnan(verdict.rank_ic)


def test_a_factor_that_is_the_same_for_every_stock_cannot_rank(market):
    universe, _, _ = market
    market_wide = pd.Series(INDEX.get_level_values("datetime").dayofyear.to_numpy(float), index=INDEX, name="index")

    verdict = judge_factor("task", market_wide, universe, {}, FactorPruningSettings())

    assert "the same value for every stock on 100% of the days" in verdict.reason


def test_infinite_values_count_as_missing(market):
    universe, signal, _ = market
    broken = signal.copy()
    broken[broken.index.get_level_values("instrument").isin(STOCKS[:10])] = np.inf

    verdict = judge_factor("task", broken, universe, {}, FactorPruningSettings())

    assert "on only 67% of the days" in verdict.reason


def test_the_universe_holds_member_days_of_the_period_and_no_label_from_after_it():
    days = pd.date_range("2016-12-26", periods=10, freq="B", name="datetime")
    index = pd.MultiIndex.from_product([days, ["IN", "LATE", "OUT"]], names=["datetime", "instrument"])
    close = pd.Series(np.arange(1.0, len(index) + 1), index=index)
    close[(days[2], "IN")] = np.nan  # a suspended day has no price
    spans = [("IN", days[0], days[-1]), ("LATE", days[4], days[-1]), ("GONE", days[0], days[-1])]

    universe = build_universe(close, spans, start=str(days[1].date()), end=str(days[7].date()))

    member_days = universe.priced
    assert set(member_days.get_level_values("instrument")) == {"IN", "LATE"}
    assert member_days.get_level_values("datetime").min() == days[1]
    # The label reads two days ahead, so the last two days of the period are left out.
    assert member_days.get_level_values("datetime").max() == days[5]
    assert (days[2], "IN") not in member_days
    assert (days[3], "LATE") not in member_days
    assert (days[4], "LATE") in member_days
    expected = close[(days[7], "LATE")] / close[(days[6], "LATE")] - 1
    assert universe.label[(days[5], "LATE")] == pytest.approx(expected)


class FakeTask:
    def __init__(self, name):
        self.name = name
        self.factor_implementation = True

    def get_task_information(self):
        return f"factor_name: {self.name}\nfactor_description: test"


class FakeImplementation:
    def __init__(self, frame):
        self.frame = frame

    def execute(self, data_type):
        assert data_type == "All"
        return "executed", self.frame


def coder_feedback(decision=True):
    """The class the factor coder really produces: it has a constructor of its own."""
    return FactorSingleFeedback(
        execution_feedback="ok", value_feedback="ok", code_feedback="ok", final_feedback="ok", final_decision=decision
    )


def experiment(frames):
    exp = QlibFactorExperiment(sub_tasks=[FakeTask(name) for name in frames])
    exp.sub_workspace_list = [FakeImplementation(frame) for frame in frames.values()]
    exp.prop_dev_feedback = CoSTEERMultiFeedback([coder_feedback() for _ in frames])
    return exp


@pytest.fixture
def runner(market, monkeypatch):
    universe, _, _ = market
    logged = []
    monkeypatch.setattr(pruning.logger, "log_object", lambda obj, tag: logged.append((tag, obj)))
    runner = PruningFactorRunner(scen=None)
    monkeypatch.setattr(runner, "load_universe", lambda: universe)
    runner.logged = logged
    return runner


def test_pruned_factors_are_marked_and_the_others_reach_the_backtest(market, runner):
    _, signal, noise = market
    exp = experiment({"Momentum": signal.to_frame("Momentum"), "Noise": noise.to_frame("Noise")})

    verdicts = runner.prune(exp)

    assert [verdict.reason is None for verdict in verdicts] == [True, False]
    kept, pruned = exp.prop_dev_feedback.feedback_list
    assert kept.final_decision is True
    assert pruned.final_decision is False
    assert "Pruned before the backtest. 'Noise': it has no measurable predictive power" in pruned.final_feedback
    assert "Pruned before the backtest" in str(pruned)
    assert [task.factor_implementation for task in exp.sub_tasks] == [True, False]
    ((tag, record),) = runner.logged
    assert tag == "factor pruning"
    assert [item["column"] for item in record] == ["Momentum", "Noise"]


def test_a_second_new_factor_that_repeats_the_first_is_pruned(market, runner):
    _, signal, _ = market
    exp = experiment({"Momentum": signal.to_frame("Momentum"), "Reversal": (-signal).to_frame("Reversal")})

    verdicts = runner.prune(exp)

    assert verdicts[0].reason is None
    assert "ranks stocks like the factor 'Momentum'" in verdicts[1].reason


def test_when_nothing_is_left_the_loop_is_told_why(market, runner):
    _, _, noise = market
    exp = experiment({"Noise": noise.to_frame("Noise")})

    with pytest.raises(FactorEmptyError, match="Factor 'Noise' was pruned because it has no measurable predictive"):
        runner.prune(exp)


def test_accepted_factors_of_earlier_loops_make_a_copy_redundant(market, runner, monkeypatch):
    _, signal, _ = market
    monkeypatch.setattr(runner, "accepted_factors", lambda exp, universe: {"Accepted": signal})
    exp = experiment({"Copy": (signal * 3).to_frame("Copy")})

    with pytest.raises(FactorEmptyError, match="ranks stocks like the factor 'Accepted'"):
        runner.prune(exp)


def test_factors_the_coder_did_not_finish_are_left_to_the_upstream_runner(market, runner):
    _, signal, _ = market
    exp = experiment({"Momentum": signal.to_frame("Momentum"), "Failed": None})
    exp.sub_workspace_list[1] = None

    verdicts = runner.prune(exp)

    assert [verdict.column for verdict in verdicts] == ["Momentum"]


def test_a_factor_the_coder_failed_does_not_switch_pruning_off_for_the_others(market, runner, monkeypatch):
    # The feedback of several factors is falsy as soon as one of them failed.
    calls = []
    monkeypatch.setattr(runner, "prune", lambda exp: calls.append("prune"))
    monkeypatch.setattr(pruning.QlibFactorRunner, "develop", lambda self, exp: calls.append("upstream") or exp)
    _, signal, noise = market
    exp = experiment({"Momentum": signal.to_frame("Momentum"), "Broken": noise.to_frame("Broken")})
    exp.prop_dev_feedback = CoSTEERMultiFeedback([coder_feedback(True), coder_feedback(False)])
    assert not exp.prop_dev_feedback

    runner.develop(exp)

    assert calls == ["prune", "upstream"]


def test_a_series_and_repeated_column_names_are_judged_like_a_frame(market, runner):
    _, signal, _ = market
    twice = pd.concat([signal.rename("Same"), (-signal).rename("Same")], axis=1)
    exp = experiment({"AsSeries": signal.rename("Momentum"), "Twice": twice})

    verdicts = runner.prune(exp)

    assert [(verdict.column, verdict.reason is None) for verdict in verdicts] == [
        ("Momentum", True),
        ("Same", False),
        ("Same", False),
    ]


def test_a_factor_that_cannot_be_measured_is_pruned_instead_of_ending_the_run(market, runner):
    _, signal, _ = market
    repeated = pd.concat([signal, signal]).to_frame("Repeated")  # every (datetime, instrument) pair twice
    exp = experiment({"Repeated": repeated})

    with pytest.raises(FactorEmptyError, match="'Repeated' was pruned because its values could not be measured"):
        runner.prune(exp)


@pytest.fixture
def market_on_disk(market, tmp_path, monkeypatch):
    """A data folder and a provider as the launcher sets them up, for the runner's own loading."""
    universe, signal, _ = market
    close = (1 + signal.abs()).rename("$close")
    data_folder = tmp_path / "factor_data"
    data_folder.mkdir()
    close.to_frame().to_hdf(data_folder / "daily_pv.h5", key="data")
    members = tmp_path / "provider" / "instruments"
    members.mkdir(parents=True)
    lines = [f"{stock}\t1999-01-01\t2099-12-31" for stock in STOCKS[:20]] + ["NODATA\t1999-01-01\t2099-12-31"]
    (members / "sp500.txt").write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("RD_MARKET_PROVIDER_URI", str(tmp_path / "provider"))
    monkeypatch.setenv("RD_MARKET_QLIB_MARKET", "sp500")
    monkeypatch.setenv("QLIB_FACTOR_TRAIN_START", str(DAYS[10].date()))
    monkeypatch.setenv("QLIB_FACTOR_VALID_END", str(DAYS[99].date()))
    monkeypatch.setattr(pruning.FACTOR_COSTEER_SETTINGS, "data_folder", str(data_folder))
    monkeypatch.setattr(pruning, "_UNIVERSES", {})
    return close


def test_the_runner_reads_the_market_the_launcher_configured(market_on_disk):
    runner = PruningFactorRunner(scen=None)

    universe = runner.load_universe()

    days = universe.priced.get_level_values("datetime")
    assert (days.min(), days.max()) == (DAYS[10], DAYS[97])  # the label of the last two days is not known yet
    assert set(universe.priced.get_level_values("instrument")) == set(STOCKS[:20])
    assert runner.load_universe() is universe  # read once
    # The loop pickles its runner at every step; the universe must not travel with it.
    assert not any(isinstance(value, (Universe, tuple)) for value in vars(runner).values())


def test_develop_prunes_only_experiments_with_coded_factors(market, runner, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "prune", lambda exp: calls.append("prune"))
    monkeypatch.setattr(pruning.QlibFactorRunner, "develop", lambda self, exp: calls.append("upstream") or exp)
    baseline = QlibFactorExperiment(sub_tasks=[])
    _, signal, _ = market

    runner.develop(baseline)
    runner.develop(experiment({"Momentum": signal.to_frame("Momentum")}))

    assert calls == ["upstream", "prune", "upstream"]


def test_the_launcher_names_this_runner():
    from rdagent.core.market.process_config import PRUNING_FACTOR_RUNNER
    from rdagent.core.utils import import_class

    assert import_class(PRUNING_FACTOR_RUNNER) is PruningFactorRunner

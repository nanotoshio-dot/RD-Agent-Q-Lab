"""A barely trained model is measured, shown to the feedback step and written in the report."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import rdagent.scenarios.qlib.developer.feedback as feedback
from rdagent.app.market_report import series_metrics, show
from rdagent.scenarios.qlib.developer.training_health import training_warning

pytestmark = pytest.mark.offline

EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "rdagent" / "scenarios" / "qlib" / "experiment"


def template_module():
    spec = importlib.util.spec_from_file_location("model_health", EXPERIMENT_DIR / "factor_template/model_health.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["model_health.py", "read_exp_res.py"])
def test_both_template_folders_measure_alike(name):
    assert (EXPERIMENT_DIR / "factor_template" / name).read_text() == (
        EXPERIMENT_DIR / "model_template" / name
    ).read_text()


def test_the_best_round_is_the_lowest_validation_loss():
    history = [SimpleNamespace(step=step, value=value) for step, value in enumerate([0.9, 0.5, 0.7, float("nan")])]

    assert template_module().best_iteration(history) == 2
    assert template_module().best_iteration([]) is None


def test_ties_at_the_cut_are_counted():
    day1 = pd.Series([3.0, 2.0, 2.0, 2.0, 1.0], index=list("ABCDE"))  # two of three places go to three names
    day2 = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=list("ABCDE"))  # no tie
    pred = pd.concat({pd.Timestamp("2020-01-02"): day1, pd.Timestamp("2020-01-03"): day2})
    pred.index.names = ["datetime", "instrument"]

    stats = template_module().ranking_stats(pred.to_frame("score"), topk=3)

    assert stats == {
        "signal.median_distinct_scores": 4.0,
        "signal.median_names": 5.0,
        "signal.topk_share_in_ties": pytest.approx(1 / 3),
    }
    assert template_module().ranking_stats(pred.to_frame("score"), topk=5)["signal.topk_share_in_ties"] == 0


def figures(**values):
    names = {"rounds": "model.best_iteration", "distinct": "signal.median_distinct_scores"}
    names.update({"names": "signal.median_names", "ties": "signal.topk_share_in_ties"})
    return pd.Series({names[key]: value for key, value in values.items()}, dtype=float)


def test_a_model_that_stops_at_once_or_ties_most_names_is_flagged():
    warning = training_warning(figures(rounds=1, distinct=21, names=481, ties=0.337))

    assert "best boosting round was 1" in warning
    assert "21 different scores a day for 481 names" in warning
    assert "34% of the top places" in warning
    assert training_warning(figures(rounds=60, distinct=100, names=481)) is not None
    assert training_warning(figures(rounds=2)) is not None


@pytest.mark.parametrize(
    "result",
    [None, pd.Series({"IC": 0.01}), figures(rounds=np.nan), figures(rounds=40, distinct=470, names=481, ties=0)],
)
def test_a_model_that_ranks_or_says_nothing_is_not_flagged(result):
    assert training_warning(result) is None


def backtest(rounds):
    base = pd.Series(
        {
            "IC": 0.01,
            "1day.excess_return_with_cost.annualized_return": 0.02,
            "1day.excess_return_with_cost.max_drawdown": -0.1,
        }
    )
    return pd.concat([base, figures(rounds=rounds)]).to_frame("0")["0"]


def test_the_feedback_prompt_says_which_result_barely_trained(monkeypatch):
    prompts = []

    def answer(user_prompt, system_prompt, **kwargs):
        prompts.append(user_prompt)
        return json.dumps({"Replace Best Result": "no"})

    backend = SimpleNamespace(build_messages_and_create_chat_completion=answer)
    monkeypatch.setattr(feedback, "APIBackend", lambda *args, **kwargs: backend)
    exp = SimpleNamespace(
        hypothesis=SimpleNamespace(hypothesis="momentum"),
        result=backtest(40),
        sub_tasks=[],
        based_experiments=[SimpleNamespace(result=backtest(1))],
    )
    scen = SimpleNamespace(get_scenario_all_desc=lambda **kwargs: "scenario")

    feedback.QlibFactorExperiment2Feedback(scen).generate_feedback(exp, trace=None)

    assert "SOTA Result: Warning: the model barely ranks" in prompts[0]
    assert "Current Result: Warning" not in prompts[0]


def test_the_report_shows_the_figures():
    metrics = series_metrics(figures(rounds=1, distinct=21, names=481, ties=0.337))

    assert show("model.best_iteration", metrics["model.best_iteration"]) == "1"
    assert show("signal.median_distinct_scores", metrics["signal.median_distinct_scores"]) == "21"
    assert show("signal.topk_share_in_ties", metrics["signal.topk_share_in_ties"]) == "34%"

import pandas as pd
import pytest

import rdagent.scenarios.qlib.developer.feedback_guard as feedback_guard
from rdagent.core.proposal import HypothesisFeedback
from rdagent.scenarios.qlib.developer.feedback_guard import (
    ReturnGuardedFactorFeedback,
    annualized_return,
)

pytestmark = pytest.mark.offline


def result(with_cost):
    return pd.Series({"IC": 0.0148, "1day.excess_return_with_cost.annualized_return": with_cost})


class Experiment:
    def __init__(self, current, best):
        self.result = current
        self.based_experiments = [Experiment(best, None)] if best is not None else []


def answer(decision):
    return HypothesisFeedback(reason="", decision=decision, observations="An improvement in the annualized return.")


@pytest.fixture
def guard(monkeypatch):
    """The guard, with the LLM's answer given by the test instead of a model."""
    answers = []
    upstream = feedback_guard.QlibFactorExperiment2Feedback
    monkeypatch.setattr(upstream, "generate_feedback", lambda self, exp, trace: answers.pop())
    monkeypatch.setattr(feedback_guard.logger, "warning", lambda message: None)
    guard = ReturnGuardedFactorFeedback(scen=None)
    guard.answers = answers
    return guard


def test_a_replace_that_the_return_contradicts_keeps_the_best_result(guard):
    # The 7B model called 4.5% against 9.2% "an improvement in the annualized return".
    guard.answers.append(answer(True))

    feedback = guard.generate_feedback(Experiment(result(0.0445), result(0.0917)), trace=None)

    assert feedback.decision is False
    assert "fell from 9.17% to 4.45%" in feedback.observations
    assert feedback.observations.startswith("An improvement in the annualized return.")


def test_a_replace_with_a_better_return_stands(guard):
    guard.answers.append(answer(True))

    feedback = guard.generate_feedback(Experiment(result(0.0917), result(0.0847)), trace=None)

    assert feedback.decision is True
    assert feedback.observations == "An improvement in the annualized return."


def test_a_keep_is_never_turned_into_a_replace(guard):
    guard.answers.append(answer(False))

    assert guard.generate_feedback(Experiment(result(0.20), result(0.05)), trace=None).decision is False


@pytest.mark.parametrize(
    ("current", "best"),
    [
        (result(0.05), None),
        (result(float("nan")), result(0.08)),
        (None, result(0.08)),
        (result(0.05), pd.Series({"IC": 0.01})),
    ],
)
def test_without_two_returns_to_compare_the_answer_stands(guard, current, best):
    guard.answers.append(answer(True))

    assert guard.generate_feedback(Experiment(current, best), trace=None).decision is True


def test_the_return_is_read_from_a_result_or_missing():
    assert annualized_return(result(0.0917)) == pytest.approx(0.0917)
    assert annualized_return(None) is None
    assert annualized_return(pd.Series({"IC": 0.01})) is None


def test_the_launcher_names_this_feedback():
    from rdagent.core.market.process_config import GUARDED_FACTOR_SUMMARIZER
    from rdagent.core.utils import import_class

    assert import_class(GUARDED_FACTOR_SUMMARIZER) is ReturnGuardedFactorFeedback

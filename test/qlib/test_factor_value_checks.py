import numpy as np
import pandas as pd
import pytest

import rdagent.components.coder.factor_coder.eva_utils as eva_utils
from rdagent.components.coder.factor_coder.eva_utils import (
    FactorEmptyValueEvaluator,
    FactorOutputFormatEvaluator,
)

pytestmark = pytest.mark.offline


class FakeImplementation:
    def __init__(self, result):
        self.result = result

    def execute(self):
        return "execution log", self.result


def factor_frame(values):
    days = pd.date_range("2020-01-01", periods=len(values), name="datetime")
    index = pd.MultiIndex.from_product([days, ["SH600000"]], names=["datetime", "instrument"])
    return pd.DataFrame({"Factor": values}, index=index)


def check(result):
    return FactorEmptyValueEvaluator().evaluate(FakeImplementation(result), None)


def test_a_result_with_values_passes():
    assert check(factor_frame([1.0, np.nan, 2.0])) == ("The source dataframe has factor values.", True)


def test_an_empty_result_is_named_with_its_usual_cause():
    # What pd.DataFrame(series, columns=[name]) returns for a series called "$close".
    series = factor_frame([1.0, 2.0])["Factor"].rename("$close")
    empty = pd.DataFrame(series, columns=["20_day_MA"])

    feedback, passed = check(empty)

    assert not passed
    assert "has no rows" in feedback
    assert "to_frame" in feedback


def test_a_result_with_only_nan_values_gets_feedback_but_no_verdict():
    # A correct three-year window has no values on the two-year evaluation sample.
    feedback, passed = check(factor_frame([np.nan, np.nan]))

    assert passed is None
    assert "NaN" in feedback


def test_a_missing_result_fails():
    assert check(None)[1] is False


def format_check(result):
    return FactorOutputFormatEvaluator().evaluate(FakeImplementation(result), None, check_qlib_structure=True)


def test_a_correct_qlib_result_passes_without_asking_the_llm(monkeypatch):
    monkeypatch.setattr(eva_utils, "APIBackend", None)  # any LLM call would fail

    assert format_check(factor_frame([1.0, 2.0]))[1] is True


@pytest.mark.parametrize(
    ("result", "problem"),
    [
        (factor_frame([1.0, 2.0]).reset_index(), "index levels are [None]"),
        (factor_frame([1.0, 2.0]).assign(Other=1.0), "2 columns"),
        (factor_frame([1.0, 2.0]).astype(str), "must be numeric"),
    ],
)
def test_a_wrong_qlib_structure_is_named_exactly(result, problem):
    feedback, passed = format_check(result)

    assert not passed
    assert problem in feedback

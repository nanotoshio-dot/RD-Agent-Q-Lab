"""Factor feedback whose "replace the best result" must agree with the numbers.

The feedback prompt tells the LLM to replace the best result when the annualized return improves. A small
model can misread the two numbers: in a run with a 7B model it wrote "an improvement in the annualized
return" for a result of 4.5% against 9.2%, and the worse result became the best one. Every later loop is
then compared with it, and its factor stays in the library.

``ReturnGuardedFactorFeedback`` keeps the best result in that case and says so in the feedback, which the
next hypothesis prompt shows. It never turns a "keep" into a "replace". Select it with
``QLIB_QUANT_FACTOR_SUMMARIZER`` (quant loop) or ``QLIB_FACTOR_SUMMARIZER`` (factor loop); the market
launcher does.
"""

from __future__ import annotations

import math

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.developer.feedback import QlibFactorExperiment2Feedback

# The return the feedback prompt shows the LLM for both results.
ANNUALIZED_RETURN = "1day.excess_return_with_cost.annualized_return"


def annualized_return(result: object) -> float | None:
    """The annualized excess return with cost of a backtest result, or None when it has none."""
    if result is None or not hasattr(result, "get"):
        return None
    try:
        value = float(result.get(ANNUALIZED_RETURN))
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


class ReturnGuardedFactorFeedback(QlibFactorExperiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        feedback = super().generate_feedback(exp, trace)
        if not feedback.decision or not exp.based_experiments:
            return feedback
        current = annualized_return(exp.result)
        best = annualized_return(exp.based_experiments[-1].result)
        if current is None or best is None or current > best:
            return feedback
        note = (
            f"The answer was to replace the best result, but the annualized excess return with cost fell from "
            f"{best:.2%} to {current:.2%}. The best result is kept and this loop's factors are not added to "
            "the library."
        )
        logger.warning(note)
        feedback.decision = False
        feedback.observations = f"{feedback.observations or ''} {note}".strip()
        return feedback

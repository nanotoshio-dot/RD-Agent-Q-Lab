"""A warning for a backtest whose model barely ranks, so its return is not read as evidence.

The thresholds only warn. They change no decision.
"""

from __future__ import annotations

import math

# Added to the result by read_exp_res.py of the Qlib templates (see model_health.py there).
BEST_ITERATION = "model.best_iteration"
DISTINCT_SCORES = "signal.median_distinct_scores"
NAMES = "signal.median_names"
TIE_SHARE = "signal.topk_share_in_ties"
# At this many boosting rounds or fewer, or with fewer distinct scores a day than this share of the names,
# the model barely ranks.
FEW_ITERATIONS = 3
FEW_DISTINCT_SHARE = 0.5


def _figure(result: object, name: str) -> float | None:
    if result is None or not hasattr(result, "get"):
        return None
    try:
        value = float(result.get(name))
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def training_warning(result: object) -> str | None:
    """A sentence about a barely trained model, or None when it ranks or the result does not say."""
    rounds = _figure(result, BEST_ITERATION)
    distinct, names = _figure(result, DISTINCT_SCORES), _figure(result, NAMES)
    ties = _figure(result, TIE_SHARE)
    few_rounds = rounds is not None and rounds <= FEW_ITERATIONS
    few_scores = distinct is not None and bool(names) and distinct / names < FEW_DISTINCT_SHARE
    if not (few_rounds or few_scores):
        return None
    facts = []
    if rounds is not None:
        facts.append(f"its best boosting round was {rounds:.0f}")
    if distinct is not None and names:
        facts.append(f"it gave about {distinct:.0f} different scores a day for {names:.0f} names")
    if ties is not None:
        facts.append(f"ties decided {ties:.0%} of the top places")
    return (
        f"Warning: the model barely ranks these names ({'; '.join(facts)}). "
        "A difference in return from this run is weak evidence about what the run changed."
    )

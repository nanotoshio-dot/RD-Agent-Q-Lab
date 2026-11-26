"""How much a backtest's model ranks, from what its run recorded.

A model that stops after a few trees, or that gives most names the same score, leaves part of the top-k
choice to the order of ties. Its return then says little about the features. read_exp_res.py adds these
figures to qlib_res.csv.
"""

import pandas as pd


def best_iteration(history: list) -> int | None:
    """The boosting round with the lowest validation loss, from MLflow metric history (step 0 is round 1)."""
    losses = [(float(entry.value), int(entry.step)) for entry in history if entry.value == entry.value]
    return min(losses)[1] + 1 if losses else None


def ranking_stats(pred: pd.DataFrame, topk: int) -> dict[str, float]:
    """Medians per day of distinct scores and of names, and the mean share of top-k places decided by ties."""
    scores = pred.iloc[:, 0].dropna()
    if scores.empty:
        return {}
    days = scores.groupby(level="datetime")
    names = days.size()
    # The k-th highest score of each day, on every row of that day.
    place = days.rank(method="first", ascending=False)
    kth = scores[place == topk].droplevel([level for level in scores.index.names if level != "datetime"])
    cut = kth.reindex(scores.index.get_level_values("datetime")).to_numpy()
    above = (scores > cut).groupby(level="datetime").sum()
    tied = (scores == cut).groupby(level="datetime").sum()
    places = topk - above
    # The places at the cut are decided by tie order only when more names share that score than fit.
    tied_share = (places / topk).where((names > topk) & (tied > places), 0.0)
    return {
        "signal.median_distinct_scores": float(days.nunique().median()),
        "signal.median_names": float(names.median()),
        "signal.topk_share_in_ties": float(tied_share.mean()),
    }

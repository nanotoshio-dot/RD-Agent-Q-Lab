"""Factor pruning: drop weak or redundant new factors before the backtest.

Upstream only drops a new factor whose values correlate at 0.99 or more with an accepted one, and only from
the second loop on. Nothing measures one factor alone, so a factor without values, without variation
across stocks or without predictive power reaches the backtest, where the feedback step can accept it on
test-period noise.

``PruningFactorRunner`` judges each new factor on the training and validation periods only, on the days its
stocks are members of the market, and then runs the upstream runner with the factors that are left:

- coverage: the share of member days with a value;
- variation: the share of days on which the factor differs across stocks;
- redundancy: the absolute rank correlation with every accepted factor and every new factor kept before it;
- strength: the t-statistic of the daily rank IC against the label the models are trained on.

Select it with ``QLIB_QUANT_FACTOR_RUNNER`` (quant loop) or ``QLIB_FACTOR_RUNNER`` (factor loop); the market
launcher does. The thresholds are ``FACTOR_PRUNING_*`` settings.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic_settings import SettingsConfigDict

from rdagent.app.qlib_rd_loop.conf import FactorBasePropSetting
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.core.conf import ExtendedBaseSettings
from rdagent.core.exception import FactorEmptyError
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
from rdagent.scenarios.qlib.developer.utils import (
    _normalize_factor_index,
    process_factor_data,
)
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment

# The label of the Qlib templates is Ref($close, -2)/Ref($close, -1) - 1, so it reads two days ahead.
LABEL_HORIZON = 2
MIN_NAMES_PER_DAY = 10
MIN_DAYS = 30
NO_SPREAD = 1e-12  # below this a standard deviation is rounding noise


class FactorPruningSettings(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="FACTOR_PRUNING_", protected_namespaces=())

    min_coverage: float = 0.8
    """A factor needs a value on at least this share of member days."""

    min_varying_days: float = 0.5
    """A factor must differ across stocks on at least this share of days."""

    max_abs_rank_corr: float = 0.99
    """A factor at or above this absolute rank correlation with a kept factor is redundant."""

    min_abs_rank_ic_t: float = 1.5
    """A factor below this absolute t-statistic of the daily rank IC is weak. 0 keeps every factor.

    Pure noise stays below 1.5 about 87% of the time. A stricter 2.0 also rejects most textbook
    price-volume factors on a large-cap market (a 20-day return on the S&P 500 has about -1.8), which the
    backtest should get to judge.
    """


@dataclass
class FactorVerdict:
    """What was measured for one new factor, and why it was pruned (``reason`` is None when kept)."""

    task: str
    column: str
    coverage: float = math.nan
    varying_days: float = math.nan
    rank_ic: float = math.nan
    rank_ic_t: float = math.nan
    most_similar: str | None = None
    abs_rank_corr: float = math.nan
    reason: str | None = None


@dataclass
class Universe:
    """The member days of the market in the training and validation periods."""

    priced: pd.Index  # (datetime, instrument) of member days with a close
    label: pd.Series  # forward return on those days; NaN where it is not known

    def select(self, values: pd.Series) -> pd.Series:
        """``values`` on the member days, NaN where it has none, with infinities as NaN.

        The stocks are filtered first: a factor covers the whole market (15 million rows for China), and
        looking up member days in all of it is slow.
        """
        members = self.priced.get_level_values("instrument").unique()
        of_members = values[values.index.get_level_values("instrument").isin(members)]
        return of_members.reindex(self.priced).replace([np.inf, -np.inf], np.nan)


def daily_rank_corr(x: pd.Series, y: pd.Series) -> pd.Series:
    """Rank correlation of ``x`` and ``y`` across stocks, one value per day with enough stocks."""
    pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if pair.empty:
        return pd.Series(dtype=float)
    # Sums per day instead of a correlation per group: groupby(...).corr(other) aligns ``other`` once per
    # day, which takes minutes on a market's history.
    ranks = pair.groupby(level="datetime").rank()
    centered = ranks - ranks.groupby(level="datetime").transform("mean")
    products = pd.DataFrame(
        {"xy": centered["x"] * centered["y"], "xx": centered["x"] ** 2, "yy": centered["y"] ** 2, "n": 1}
    )
    sums = products.groupby(level="datetime").sum()
    corr = sums["xy"] / np.sqrt(sums["xx"] * sums["yy"])
    return corr[sums["n"] >= MIN_NAMES_PER_DAY].replace([np.inf, -np.inf], np.nan).dropna()


def t_statistic(values: pd.Series) -> float:
    """Mean over its standard error; 0 when there are too few days to tell."""
    if len(values) < MIN_DAYS:
        return 0.0
    mean, spread = float(values.mean()), float(values.std(ddof=1))
    if not spread > NO_SPREAD:
        return 0.0 if abs(mean) <= NO_SPREAD else math.copysign(math.inf, mean)
    return mean / (spread / math.sqrt(len(values)))


def judge_factor(
    task: str, values: pd.Series, universe: Universe, library: dict[str, pd.Series], settings: FactorPruningSettings
) -> FactorVerdict:
    """Measure one factor column on the universe; ``library`` holds the factors it must differ from."""
    verdict = FactorVerdict(task=task, column=str(values.name))
    on_member_days = universe.select(values)
    verdict.coverage = float(on_member_days.notna().mean()) if len(on_member_days) else 0.0
    if verdict.coverage < settings.min_coverage:
        verdict.reason = (
            f"it has a value on only {verdict.coverage:.0%} of the days its stocks are in the market "
            f"(at least {settings.min_coverage:.0%} is needed)"
        )
        return verdict

    distinct = on_member_days.groupby(level="datetime").nunique()
    verdict.varying_days = float((distinct > 1).mean()) if len(distinct) else 0.0
    if verdict.varying_days < settings.min_varying_days:
        verdict.reason = (
            f"it has the same value for every stock on {1 - verdict.varying_days:.0%} of the days, "
            "so it cannot rank stocks"
        )
        return verdict

    for name, other in library.items():
        corr = daily_rank_corr(on_member_days, other)
        similarity = abs(float(corr.mean())) if len(corr) >= MIN_DAYS else 0.0
        if not similarity <= verdict.abs_rank_corr:  # also true for the first one (NaN)
            verdict.most_similar, verdict.abs_rank_corr = name, similarity
    if verdict.most_similar is not None and verdict.abs_rank_corr >= settings.max_abs_rank_corr:
        verdict.reason = (
            f"it ranks stocks like the factor '{verdict.most_similar}' "
            f"(absolute rank correlation {verdict.abs_rank_corr:.3f})"
        )
        return verdict

    rank_ic = daily_rank_corr(on_member_days, universe.label)
    verdict.rank_ic = float(rank_ic.mean()) if len(rank_ic) else 0.0
    verdict.rank_ic_t = t_statistic(rank_ic)
    if abs(verdict.rank_ic_t) < settings.min_abs_rank_ic_t:
        verdict.reason = (
            f"it has no measurable predictive power on the training and validation periods "
            f"(rank IC {verdict.rank_ic:+.4f}, t-statistic {verdict.rank_ic_t:+.1f}, "
            f"at least {settings.min_abs_rank_ic_t:.1f} in absolute value is needed)"
        )
    return verdict


def read_member_spans(provider_uri: Path, market: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    path = provider_uri / "instruments" / f"{market}.txt"
    spans = []
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:  # noqa: PLR2004 - code, first day, last day
            spans.append((parts[0], pd.Timestamp(parts[1]), pd.Timestamp(parts[2])))
    return spans


def build_universe(
    close: pd.Series, spans: list[tuple[str, pd.Timestamp, pd.Timestamp]], start: str, end: str
) -> Universe:
    """Member days between ``start`` and ``end``, with the label read from ``close``.

    ``close`` is indexed by (datetime, instrument) with a row for every day of each stock, as in
    daily_pv.h5. The last ``LABEL_HORIZON`` days are left out: their label needs prices after ``end``.
    """
    close = close[close.index.get_level_values("instrument").isin({stock for stock, _, _ in spans})]
    by_stock = close.groupby(level="instrument")
    label = by_stock.shift(-LABEL_HORIZON) / by_stock.shift(-1) - 1
    days = close.index.get_level_values("datetime").unique().sort_values()
    days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))][:-LABEL_HORIZON]
    stocks = set(close.index.get_level_values("instrument").unique())
    day_parts, stock_parts = [], []
    for stock, first, last in spans:
        if stock not in stocks:
            continue
        member = days[(days >= first) & (days <= last)]
        day_parts.append(member.to_numpy())
        stock_parts.append(np.full(len(member), stock, dtype=object))
    if not day_parts:
        empty = pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"])
        return Universe(priced=empty, label=pd.Series(dtype=float, index=empty))
    member_days = pd.MultiIndex.from_arrays(
        [np.concatenate(day_parts), np.concatenate(stock_parts)], names=["datetime", "instrument"]
    ).drop_duplicates()
    priced = close.reindex(member_days).dropna().index
    return Universe(priced=priced, label=label.reindex(priced))


# Kept outside the runner: the loop pickles its runner into every session snapshot, and a universe is
# megabytes.
_UNIVERSES: dict[tuple, Universe] = {}


def mark_pruned(feedback: object, note: str) -> object:
    """A copy of a coder's feedback that says the factor is not to be used, and why.

    The factor coder's feedback class has a constructor of its own, so dataclasses.replace cannot copy it.
    """
    pruned = copy.copy(feedback)
    pruned.final_decision = False  # type: ignore[attr-defined]
    text_field = "final_feedback" if hasattr(pruned, "final_feedback") else "return_checking"
    setattr(pruned, text_field, f"{getattr(pruned, text_field) or ''}\n{note}".strip())
    return pruned


def factor_columns(result: object) -> list[pd.Series] | None:
    """The columns a factor's code wrote, or None for a result the upstream runner will report itself.

    The coder accepts a Series as well as a frame, and the names of a frame's columns may repeat.
    """
    if isinstance(result, pd.Series):
        result = result.to_frame(result.name if result.name is not None else "factor")
    if not isinstance(result, pd.DataFrame):
        return None
    frame = _normalize_factor_index(result)
    if frame is None:
        return None
    return [frame.iloc[:, position] for position in range(frame.shape[1])]


class PruningFactorRunner(QlibFactorRunner):
    """The upstream factor runner, run on the new factors that survive pruning."""

    def develop(self, exp: QlibFactorExperiment) -> QlibFactorExperiment:
        # "is not None": the feedback of several factors is falsy as soon as the coder failed one of them.
        coded = isinstance(exp, QlibFactorExperiment) and exp.sub_tasks
        if coded and getattr(exp, "prop_dev_feedback", None) is not None:
            self.prune(exp)
        return super().develop(exp)

    def load_universe(self) -> Universe:
        """The market's member days for the configured periods; read once per data folder and period."""
        periods = FactorBasePropSetting()
        provider = Path(os.environ.get("RD_MARKET_PROVIDER_URI") or "~/.qlib/qlib_data/cn_data").expanduser()
        market = os.environ.get("RD_MARKET_QLIB_MARKET") or "csi300"
        data_file = Path(FACTOR_COSTEER_SETTINGS.data_folder) / "daily_pv.h5"
        key = (str(data_file), str(provider), market, periods.train_start, periods.valid_end)
        if key not in _UNIVERSES:
            close = pd.read_hdf(data_file, key="data")["$close"]
            spans = read_member_spans(provider, market)
            _UNIVERSES.clear()  # one run uses one universe
            _UNIVERSES[key] = build_universe(close, spans, periods.train_start, periods.valid_end)
        return _UNIVERSES[key]

    def accepted_factors(self, exp: QlibFactorExperiment, universe: Universe) -> dict[str, pd.Series]:
        """The accepted factors of earlier loops on the universe, as the upstream runner combines them."""
        accepted = [base for base in exp.based_experiments if isinstance(base, QlibFactorExperiment)]
        if len(accepted) <= 1:  # the first one is the baseline without learned factors
            return {}
        frame = process_factor_data(accepted)
        frame = frame.loc[:, ~frame.columns.duplicated(keep="last")]
        return {str(name): universe.select(frame[name]) for name in frame.columns}

    def judge(self, task: str, values: pd.Series, universe: Universe, library: dict[str, pd.Series]) -> FactorVerdict:
        """``judge_factor``, with a factor that cannot be measured pruned for that reason.

        Any other exception would end a run of many loops; a pruned factor only ends its own loop.
        """
        try:
            return judge_factor(task, values, universe, library, FactorPruningSettings())
        except Exception as error:  # noqa: BLE001 - whatever the factor's values are, the run goes on
            logger.warning(f"Factor pruning could not measure '{values.name}': {type(error).__name__}: {error}")
            reason = f"its values could not be measured ({type(error).__name__}: {error})"
            return FactorVerdict(task=task, column=str(values.name), reason=reason)

    def prune(self, exp: QlibFactorExperiment) -> list[FactorVerdict]:
        """Judge every new factor and mark the pruned ones as not implemented, with the reason."""
        universe = self.load_universe()
        library = self.accepted_factors(exp, universe)
        verdicts: list[FactorVerdict] = []
        kept = 0
        feedback_list = exp.prop_dev_feedback.feedback_list
        for position, implementation in enumerate(exp.sub_workspace_list):
            feedback = feedback_list[position]
            if not (implementation and feedback):
                continue
            task = exp.sub_tasks[position].get_task_information().splitlines()[0]
            columns = factor_columns(implementation.execute("All")[1])
            if columns is None:
                continue  # the upstream runner reports factors that fail on the full data
            task_verdicts = [self.judge(task, column, universe, library) for column in columns]
            verdicts.extend(task_verdicts)
            reasons = [f"'{verdict.column}': {verdict.reason}" for verdict in task_verdicts if verdict.reason]
            if reasons:
                feedback_list[position] = mark_pruned(feedback, "Pruned before the backtest. " + "; ".join(reasons))
                if hasattr(exp.sub_tasks[position], "factor_implementation"):
                    # The feedback step lists each factor with this flag, as tested or not.
                    exp.sub_tasks[position].factor_implementation = False
                continue
            kept += 1
            for verdict, column in zip(task_verdicts, columns, strict=True):
                library[verdict.column] = universe.select(column)

        for verdict in verdicts:
            logger.info(f"Factor pruning: {verdict}")
        logger.log_object([asdict(verdict) for verdict in verdicts], tag="factor pruning")
        pruned = [verdict for verdict in verdicts if verdict.reason]
        if pruned and kept == 0:
            reasons = " ".join(f"Factor '{verdict.column}' was pruned because {verdict.reason}." for verdict in pruned)
            raise FactorEmptyError(
                f"No new factor reached the backtest. {reasons} "
                "Propose a factor built on a different idea from the earlier ones."
            )
        return verdicts

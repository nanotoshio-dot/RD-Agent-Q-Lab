"""Report what one market run did, loop by loop, from the objects the research loop logged.

    python -m rdagent.app.market_report --market china
    python -m rdagent.app.market_report --market us-sp500 --run 2026-11-25_14-09-35

It only reads the log folder. It writes ``reports/<run>/report.md`` and ``loops.csv`` under the market's
folder, and keeps ``memory/journal.csv``: one line per loop of every reported run, the market's record of
what was tried and what came of it. The market launcher calls it when a run ends.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rdagent.core import serialization

METRICS = {
    "IC": "IC",
    "Rank IC": "Rank IC",
    "1day.excess_return_with_cost.annualized_return": "Excess return with cost",
    "1day.excess_return_with_cost.information_ratio": "Information ratio with cost",
    "1day.excess_return_with_cost.max_drawdown": "Max drawdown with cost",
    "1day.excess_return_without_cost.annualized_return": "Excess return without cost",
}
RETURN_WITH_COST = "1day.excess_return_with_cost.annualized_return"
PERCENT = {name for name in METRICS if name.endswith(("annualized_return", "max_drawdown"))}
JOURNAL_FIELDS = (
    "run",
    "loop",
    "action",
    "hypothesis",
    "tasks",
    "coder_rounds",
    "coder_accepted",
    "pruned",
    "ic",
    "excess_return_with_cost",
    "best_before_ic",
    "best_before_excess_return_with_cost",
    "decision",
    "outcome",
    "minutes",
)


@dataclass
class LoopRecord:
    """What the log holds about one loop; a field stays empty when its step did not run."""

    loop: int
    action: str = ""
    hypothesis: str = ""
    reason: str = ""
    tasks: list[str] = field(default_factory=list)
    coder_rounds: int = 0
    coder_accepted: list[bool] = field(default_factory=list)
    pruning: list[dict] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    best_before: dict[str, float] = field(default_factory=dict)
    code: dict[str, str] = field(default_factory=dict)
    decision: bool | None = None
    observations: str = ""
    evaluation: str = ""
    seconds: dict[str, float] = field(default_factory=dict)
    last_step: str = ""

    @property
    def outcome(self) -> str:
        """One line saying how the loop ended."""
        if self.decision is None:
            # No feedback was logged, so the run stopped inside this loop. What the loop did until then is
            # not a research result, and the time of the step it stopped in was never logged.
            return f"unfinished (stopped during {self.last_step or 'its first step'})"
        if self.decision:
            return "accepted as the new best"
        if self.metrics:
            return "backtested, not better than the best before"
        if self.pruning and all(verdict.get("reason") for verdict in self.pruning):
            return "pruned before the backtest"
        if self.coder_rounds and not any(self.coder_accepted):
            return "the coder did not produce working code"
        return "skipped: " + " ".join(self.observations.split())[:120]


def series_metrics(result: object) -> dict[str, float]:
    """The report's metrics from an experiment's result (a Series of Qlib figures), where present."""
    if result is None or not hasattr(result, "get"):
        return {}
    found = {}
    for name in METRICS:
        value = result.get(name)
        try:
            found[name] = float(value)
        except (TypeError, ValueError):
            continue
    return {name: value for name, value in found.items() if value == value}  # noqa: PLR0124 - drops NaN


def latest_session(log_dir: Path) -> tuple[str, int]:
    """The process id of the newest session in a log folder, and how many sessions the folder holds.

    A log folder that several runs wrote to keeps them apart only by process id.
    """
    newest: dict[str, str] = {}
    for path in log_dir.glob("Loop_*/**/*.pkl"):
        pid = path.parent.name
        newest[pid] = max(newest.get(pid, ""), path.name)
    if not newest:
        return "", 0
    return max(newest, key=lambda pid: newest[pid]), len(newest)


def read_run(log_dir: Path) -> tuple[list[LoopRecord], int]:
    """The loops of the newest session in ``log_dir``, and the number of files that could not be read."""
    pid, _ = latest_session(log_dir)
    loops: dict[int, LoopRecord] = {}
    unreadable = 0
    files = sorted(log_dir.glob(f"Loop_*/**/{pid}/*.pkl"), key=lambda path: path.name) if pid else []
    for path in files:
        parts = path.relative_to(log_dir).parts  # Loop_N / step / [evo_loop_K /] tag / pid / time.pkl
        tag, step = parts[-3], parts[1]
        index = int(parts[0].split("_")[1])
        record = loops.setdefault(index, LoopRecord(loop=index))
        record.last_step = step  # files are read in the order they were written, whatever they hold
        if tag not in READERS:
            continue
        try:
            logged = serialization.loads(path.read_bytes())
        except Exception:  # noqa: BLE001 - a file from other code or a killed write must not stop the report
            unreadable += 1
            continue
        try:
            READERS[tag](record, logged, step)
        except Exception:  # noqa: BLE001 - an object of an unexpected shape is skipped like an unreadable file
            unreadable += 1
    return [loops[index] for index in sorted(loops)], unreadable


def _read_hypothesis(record: LoopRecord, logged: object, _: str) -> None:
    record.action = str(getattr(logged, "action", "") or "")
    record.hypothesis = " ".join(str(getattr(logged, "hypothesis", "") or "").split())
    record.reason = " ".join(str(getattr(logged, "reason", "") or "").split())


def _read_tasks(record: LoopRecord, logged: object, _: str) -> None:
    record.tasks = [str(getattr(task, "factor_name", None) or getattr(task, "name", "?")) for task in logged]


def _read_coder_feedback(record: LoopRecord, logged: object, _: str) -> None:
    record.coder_rounds += 1
    record.coder_accepted = [bool(item and item.final_decision) for item in logged.feedback_list]


def _read_pruning(record: LoopRecord, logged: object, _: str) -> None:
    record.pruning = [dict(verdict) for verdict in logged]


def _read_result(record: LoopRecord, logged: object, _: str) -> None:
    record.metrics = series_metrics(getattr(logged, "result", None))
    earlier = getattr(logged, "based_experiments", None) or []
    record.best_before = series_metrics(earlier[-1].result) if earlier else {}
    feedback = getattr(logged, "prop_dev_feedback", None)
    for position, workspace in enumerate(getattr(logged, "sub_workspace_list", None) or []):
        accepted = feedback is None or bool(feedback[position])
        files = getattr(workspace, "file_dict", None) or {}
        name = record.tasks[position] if position < len(record.tasks) else f"task {position}"
        source = files.get("factor.py") or files.get("model.py")
        if accepted and source:
            record.code[name] = source


def _read_feedback(record: LoopRecord, logged: object, _: str) -> None:
    record.decision = bool(logged.decision)
    record.observations = str(getattr(logged, "observations", "") or "")
    record.evaluation = str(getattr(logged, "hypothesis_evaluation", "") or "")


def _read_time(record: LoopRecord, logged: object, step: str) -> None:
    record.seconds[step] = (logged["end_time"] - logged["start_time"]).total_seconds()


READERS = {
    "hypothesis generation": _read_hypothesis,
    "experiment generation": _read_tasks,
    "evolving feedback": _read_coder_feedback,
    "factor pruning": _read_pruning,
    "runner result": _read_result,
    "feedback": _read_feedback,
    "time_info": _read_time,
}


def show(name: str, value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.2%}" if name in PERCENT else f"{value:.4f}"


def journal_row(run: str, record: LoopRecord) -> dict[str, object]:
    pruned = [str(verdict.get("column")) for verdict in record.pruning if verdict.get("reason")]
    return {
        "run": run,
        "loop": record.loop,
        "action": record.action,
        "hypothesis": record.hypothesis,
        "tasks": "; ".join(record.tasks),
        "coder_rounds": record.coder_rounds,
        "coder_accepted": sum(record.coder_accepted),
        "pruned": "; ".join(pruned),
        "ic": record.metrics.get("IC", ""),
        "excess_return_with_cost": record.metrics.get(RETURN_WITH_COST, ""),
        "best_before_ic": record.best_before.get("IC", ""),
        "best_before_excess_return_with_cost": record.best_before.get(RETURN_WITH_COST, ""),
        "decision": "" if record.decision is None else record.decision,
        "outcome": record.outcome,
        "minutes": round(sum(record.seconds.values()) / 60, 1),
    }


def render(market: str, run: str, loops: list[LoopRecord], details: dict, notes: list[str]) -> str:
    lines = [f"# Research report: {market}, run {run}", ""]
    lines += [f"- {key}: {value}" for key, value in details.items()]
    lines += [f"- {note}" for note in notes]
    accepted = sum(bool(record.decision) for record in loops)
    lines += [f"- Loops: {len(loops)}, of which {accepted} accepted as a new best", ""]
    lines += ["| Loop | Action | Tasks | Coder rounds | IC | Excess return with cost | Outcome | Minutes |"]
    lines += ["| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for record in loops:
        row = journal_row(run, record)
        cells = [
            str(record.loop),
            record.action,
            row["tasks"],
            str(record.coder_rounds),
            show("IC", record.metrics.get("IC")),
            show(RETURN_WITH_COST, record.metrics.get(RETURN_WITH_COST)),
            record.outcome,
            str(row["minutes"]),
        ]
        lines.append("| " + " | ".join(str(cell).replace("|", "/") for cell in cells) + " |")
    for record in loops:
        lines += ["", f"## Loop {record.loop}: {record.outcome}", ""]
        lines += [f"**Hypothesis** ({record.action or 'no action logged'}): {record.hypothesis or 'not logged'}", ""]
        if record.tasks:
            passed = f"{sum(record.coder_accepted)} of {len(record.coder_accepted)} accepted"
            lines += [f"**Coder**: {', '.join(record.tasks)}; {record.coder_rounds} round(s), {passed}.", ""]
        for verdict in record.pruning:
            state = f"pruned because {verdict['reason']}" if verdict.get("reason") else "kept"
            measured = f"coverage {verdict.get('coverage', float('nan')):.0%}, rank IC {verdict.get('rank_ic', float('nan')):+.4f}"
            lines += [
                f"- Pruning, `{verdict.get('column')}`: {state} ({measured}, t {verdict.get('rank_ic_t', float('nan')):+.1f})."
            ]
        if record.metrics:
            lines += ["", "| Metric | This loop | Best before |", "| --- | --- | --- |"]
            lines += [
                f"| {label} | {show(name, record.metrics.get(name))} | {show(name, record.best_before.get(name))} |"
                for name, label in METRICS.items()
            ]
        if record.decision is not None:
            verdict_text = "replace the best result" if record.decision else "keep the best result"
            lines += ["", f"**Decision**: {verdict_text}. {' '.join(record.observations.split())[:600]}"]
        for name, source in record.code.items():
            lines += ["", f"Code of `{name}`:", "", "```python", source.strip(), "```"]
    return "\n".join(lines) + "\n"


def update_journal(path: Path, run: str, rows: list[dict[str, object]]) -> None:
    """Replace this run's lines in the market's journal, so reporting a run twice changes nothing."""
    kept: list[dict[str, object]] = []
    if path.is_file():
        with path.open(newline="") as handle:
            kept = [row for row in csv.DictReader(handle) if row.get("run") != run]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=JOURNAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept + rows)


def resolve_log_dir(market_root: Path, run: str) -> tuple[Path, str]:
    """The log folder of ``run`` ("latest" for the newest one) and the run's name.

    Each run has a folder under ``log``. Runs from before that wrote into ``log`` itself; that folder is
    the run called "log".
    """
    log_root = market_root / "log"
    runs = sorted(path for path in log_root.glob("*") if path.is_dir() and any(path.glob("Loop_*")))
    if run != "latest":
        chosen = log_root if run == "log" else log_root / run
    elif runs:
        chosen = runs[-1]
    else:
        chosen = log_root
    if not any(chosen.glob("Loop_*")):
        raise SystemExit(f"No research loops were logged in {chosen}")
    return chosen, ("log" if chosen == log_root else chosen.name)


def write_report(market: str, market_root: Path, run: str = "latest") -> Path:
    """Write the report of ``run`` and update the journal; return the report's path."""
    log_dir, run_name = resolve_log_dir(market_root, run)
    loops, unreadable = read_run(log_dir)
    _, sessions = latest_session(log_dir)
    details: dict[str, object] = {"Log folder": log_dir}
    run_file = log_dir / "run.json"
    if run_file.is_file():
        details.update(json.loads(run_file.read_text()))
    notes = []
    if sessions > 1:
        notes.append(f"The folder holds {sessions} sessions; this report covers the newest one")
    if unreadable:
        notes.append(f"{unreadable} logged objects could not be read and were left out")
    rows = [journal_row(run_name, record) for record in loops]
    out_dir = market_root / "reports" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.md"
    report.write_text(render(market, run_name, loops, details, notes))
    with (out_dir / "loops.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=JOURNAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "loops.json").write_text(json.dumps([asdict(record) for record in loops], indent=1, default=str))
    update_journal(market_root / "memory" / "journal.csv", run_name, rows)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--market", required=True)
    parser.add_argument("--root", type=Path, default=Path.home() / ".rdagent" / "markets")
    parser.add_argument("--run", default="latest", help='a folder under the market\'s log folder, or "latest"')
    args = parser.parse_args(argv)
    from rdagent.app.market_launcher import (  # noqa: PLC0415 - only the command line needs it
        adopt_signing_settings,
        load_base_environment,
    )

    adopt_signing_settings(load_base_environment())  # the runs signed their logs with the key .env may set
    report = write_report(args.market, args.root.expanduser() / args.market, args.run)
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

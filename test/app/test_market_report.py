import csv
from datetime import datetime, timedelta

import pandas as pd
import pytest

from rdagent.app.market_report import (
    LoopRecord,
    latest_session,
    main,
    read_run,
    resolve_log_dir,
    update_journal,
    write_report,
)
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core import serialization
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.proposal import HypothesisFeedback
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.proposal.quant_proposal import QlibQuantHypothesis

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def signing_key(monkeypatch):
    monkeypatch.setattr(RD_AGENT_SETTINGS, "artifact_signing_key", "a key for the report tests, 32+ bytes long")


class Workspace:
    def __init__(self, code):
        self.file_dict = {"factor.py": code}


def result(ic, with_cost):
    return pd.Series(
        {
            "IC": ic,
            "Rank IC": ic / 2,
            "1day.excess_return_with_cost.annualized_return": with_cost,
            "1day.excess_return_with_cost.information_ratio": 0.8,
            "1day.excess_return_with_cost.max_drawdown": -0.16,
            "1day.excess_return_without_cost.annualized_return": with_cost + 0.05,
        }
    )


def coder_feedback(decision):
    return CoSTEERMultiFeedback(
        [CoSTEERSingleFeedback(execution="ok", return_checking="ok", code="ok", final_decision=decision)]
    )


class LogWriter:
    """Writes objects the way the research loop's logger lays them out on disk."""

    def __init__(self, folder, pid="4242"):
        self.folder, self.pid, self.clock = folder, pid, datetime(2026, 11, 25, 14, 0, 0)

    def log(self, loop, step, tag, logged, evo=None):
        self.clock += timedelta(seconds=1)
        parts = [f"Loop_{loop}", step] + ([f"evo_loop_{evo}"] if evo is not None else []) + [tag, self.pid]
        path = self.folder.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{self.clock:%Y-%m-%d_%H-%M-%S-%f}.pkl").write_bytes(serialization.dumps(logged))

    def hypothesis(self, loop, text):
        self.log(
            loop, "direct_exp_gen", "hypothesis generation", QlibQuantHypothesis(text, "why", "", "", "", "", "factor")
        )

    def tasks(self, loop, name):
        self.log(loop, "direct_exp_gen", "experiment generation", [FactorTask(name, "description", "formulation")])

    def step_time(self, loop, step, seconds):
        start = datetime(2026, 11, 25, 14, 0, 0)
        self.log(loop, step, "time_info", {"start_time": start, "end_time": start + timedelta(seconds=seconds)})


def write_three_loops(folder):
    """An accepted loop, a pruned one and one the coder could not finish."""
    writer = LogWriter(folder)
    baseline = QlibFactorExperiment(sub_tasks=[])
    baseline.result = result(0.0144, 0.0847)

    writer.hypothesis(0, "20-day momentum\npredicts returns")
    writer.tasks(0, "Momentum20")
    writer.log(0, "coding", "evolving feedback", coder_feedback(False), evo=0)
    writer.log(0, "coding", "evolving feedback", coder_feedback(True), evo=1)
    writer.log(
        0,
        "running",
        "factor pruning",
        [{"column": "Momentum20", "coverage": 0.99, "rank_ic": -0.03, "rank_ic_t": -6.5, "reason": None}],
    )
    experiment = QlibFactorExperiment(sub_tasks=[FactorTask("Momentum20", "d", "f")])
    experiment.based_experiments = [baseline]
    experiment.sub_workspace_list = [Workspace("momentum = close / close.shift(20) - 1")]
    experiment.prop_dev_feedback = coder_feedback(True)
    experiment.result = result(0.0180, 0.0950)
    writer.log(0, "running", "runner result", experiment)
    writer.log(0, "feedback", "feedback", HypothesisFeedback(reason="better", decision=True, observations="IC rose"))
    writer.step_time(0, "coding", 600)
    writer.step_time(0, "running", 240)

    writer.hypothesis(1, "20-day average price")
    writer.tasks(1, "SMA20")
    writer.log(1, "coding", "evolving feedback", coder_feedback(True), evo=0)
    pruned = {
        "column": "SMA20",
        "coverage": 0.99,
        "rank_ic": 0.002,
        "rank_ic_t": 1.0,
        "reason": "it has no measurable predictive power",
    }
    writer.log(1, "running", "factor pruning", [pruned])
    skipped = HypothesisFeedback(reason="", decision=False, observations="No new factor reached the backtest.")
    writer.log(1, "feedback", "feedback", skipped)

    writer.hypothesis(2, "something complicated")
    writer.tasks(2, "Complicated")
    for step in range(3):
        writer.log(2, "coding", "evolving feedback", coder_feedback(False), evo=step)
    writer.log(
        2, "feedback", "feedback", HypothesisFeedback(reason="", decision=False, observations="All tasks are failed")
    )
    return writer


def test_each_loop_is_read_with_its_outcome(tmp_path):
    write_three_loops(tmp_path)

    loops, unreadable = read_run(tmp_path)

    assert unreadable == 0
    assert [record.outcome for record in loops] == [
        "accepted as the new best",
        "pruned before the backtest",
        "the coder did not produce working code",
    ]
    first = loops[0]
    assert first.hypothesis == "20-day momentum predicts returns"
    assert first.tasks == ["Momentum20"]
    assert (first.coder_rounds, first.coder_accepted) == (2, [True])
    assert first.metrics["IC"] == pytest.approx(0.0180)
    assert first.best_before["1day.excess_return_with_cost.annualized_return"] == pytest.approx(0.0847)
    assert first.code == {"Momentum20": "momentum = close / close.shift(20) - 1"}
    assert first.seconds == {"coding": 600.0, "running": 240.0}
    assert loops[1].metrics == {}


def test_an_unreadable_file_is_counted_and_the_rest_is_reported(tmp_path):
    write_three_loops(tmp_path)
    broken = tmp_path / "Loop_1" / "feedback" / "feedback" / "4242"
    next(broken.glob("*.pkl")).write_bytes(b"cut off while being written")

    loops, unreadable = read_run(tmp_path)

    assert unreadable == 1
    assert loops[1].decision is None
    assert loops[1].outcome == "unfinished (stopped during feedback)"


def test_only_the_newest_session_of_a_shared_folder_is_reported(tmp_path):
    old = LogWriter(tmp_path, pid="111")
    old.hypothesis(0, "from an earlier run")
    new = LogWriter(tmp_path, pid="222")
    new.clock += timedelta(hours=1)
    new.hypothesis(0, "from the latest run")

    assert latest_session(tmp_path) == ("222", 2)
    assert [record.hypothesis for record in read_run(tmp_path)[0]] == ["from the latest run"]


def test_report_files_are_written_and_the_journal_keeps_one_copy_of_a_run(tmp_path):
    market_root = tmp_path / "china"
    run_dir = market_root / "log" / "2026-11-25_14-00-00"
    write_three_loops(run_dir)
    (run_dir / "run.json").write_text('{"Test period": "2020-01-01 to 2024-12-31"}')

    report = write_report("china", market_root)
    write_report("china", market_root)  # a second report of the same run

    text = report.read_text()
    assert report == market_root / "reports" / "2026-11-25_14-00-00" / "report.md"
    assert "# Research report: china, run 2026-11-25_14-00-00" in text
    assert "- Test period: 2020-01-01 to 2024-12-31" in text
    assert "- Loops: 3, of which 1 accepted as a new best" in text
    assert "| 0 | factor | Momentum20 | 2 | 0.0180 | +9.50% | accepted as the new best | 14.0 |" in text
    assert "| Information ratio with cost | 0.8000 | 0.8000 |" in text
    assert "`SMA20`: pruned because it has no measurable predictive power" in text
    assert "momentum = close / close.shift(20) - 1" in text
    with (market_root / "memory" / "journal.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["run"], row["loop"], row["outcome"]) for row in rows] == [
        ("2026-11-25_14-00-00", "0", "accepted as the new best"),
        ("2026-11-25_14-00-00", "1", "pruned before the backtest"),
        ("2026-11-25_14-00-00", "2", "the coder did not produce working code"),
    ]
    assert rows[1]["pruned"] == "SMA20"


def test_the_journal_keeps_the_lines_of_other_runs(tmp_path):
    journal = tmp_path / "memory" / "journal.csv"
    update_journal(journal, "run-a", [{"run": "run-a", "loop": 0, "outcome": "unfinished"}])
    update_journal(journal, "run-b", [{"run": "run-b", "loop": 0, "outcome": "accepted as the new best"}])
    update_journal(journal, "run-a", [{"run": "run-a", "loop": 0, "outcome": "pruned before the backtest"}])

    with journal.open(newline="") as handle:
        rows = [(row["run"], row["outcome"]) for row in csv.DictReader(handle)]
    assert rows == [("run-b", "accepted as the new best"), ("run-a", "pruned before the backtest")]


def test_the_latest_run_folder_is_chosen_and_the_old_shared_folder_still_works(tmp_path):
    shared = tmp_path / "old-market"
    LogWriter(shared / "log").hypothesis(0, "h")
    assert resolve_log_dir(shared, "latest") == (shared / "log", "log")

    per_run = tmp_path / "new-market"
    for name in ("2026-11-25_10-00-00", "2026-11-25_14-00-00"):
        LogWriter(per_run / "log" / name).hypothesis(0, "h")
    assert resolve_log_dir(per_run, "latest") == (per_run / "log" / "2026-11-25_14-00-00", "2026-11-25_14-00-00")
    assert resolve_log_dir(per_run, "2026-11-25_10-00-00")[1] == "2026-11-25_10-00-00"

    with pytest.raises(SystemExit, match="No research loops were logged"):
        resolve_log_dir(tmp_path / "empty-market", "latest")


def test_the_command_line_writes_the_report(tmp_path, capsys):
    write_three_loops(tmp_path / "china" / "log" / "2026-11-25_14-00-00")

    assert main(["--market", "china", "--root", str(tmp_path)]) == 0

    assert "report.md" in capsys.readouterr().out
    assert (tmp_path / "china" / "reports" / "2026-11-25_14-00-00" / "loops.csv").is_file()


def test_a_loop_without_feedback_is_unfinished_whatever_it_did_before(tmp_path):
    # Stopped after the backtest: the result was never compared, so the journal must not say it lost.
    writer = LogWriter(tmp_path)
    writer.hypothesis(0, "20-day momentum")
    writer.tasks(0, "Momentum20")
    writer.log(0, "coding", "evolving feedback", coder_feedback(True), evo=0)
    experiment = QlibFactorExperiment(sub_tasks=[FactorTask("Momentum20", "d", "f")])
    experiment.result = result(0.0180, 0.0950)
    writer.log(0, "running", "runner result", experiment)
    writer.log(0, "feedback", "debug_llm", {"prompt": "the feedback question, never answered"})
    # Stopped while the coder was still trying.
    writer.hypothesis(1, "something else")
    writer.log(1, "coding", "evolving feedback", coder_feedback(False), evo=0)

    loops, _ = read_run(tmp_path)

    assert [record.outcome for record in loops] == [
        "unfinished (stopped during feedback)",
        "unfinished (stopped during coding)",
    ]
    assert LoopRecord(loop=3).outcome == "unfinished (stopped during its first step)"

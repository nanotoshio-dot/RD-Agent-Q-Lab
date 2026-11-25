"""Launch the existing RD-Agent Qlib research loop in an isolated market process.

This module deliberately does not import the Qlib workflow at module import time.
Market-specific environment variables must be set before RD-Agent settings are
initialized in the child process.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import dotenv_values

from rdagent.core import serialization
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.market import MarketContext, MarketProcessConfig, TradingRules
from rdagent.core.market.process_config import provider_cutovers
from rdagent.core.market.runtime import ensure_docker_supports_market

# First day of the factor data built by factor_data_template/generate.py; keep the two in sync.
FACTOR_DATA_START = date(2008, 12, 29)


@dataclass(frozen=True)
class MarketProfile:
    market_id: str
    region: str
    currency: str
    qlib_market: str
    benchmark: str
    universe: tuple[str, ...]
    trading_rules: TradingRules


# US costs are an assumption (10 bps per side, covering fees, spread and impact for the template's
# large account); China keeps the values the Qlib templates always used.
US_COST = {"open_cost": 0.001, "close_cost": 0.001, "min_cost": 1}

MARKET_PROFILES: dict[str, MarketProfile] = {
    "china": MarketProfile(
        market_id="china",
        region="cn",
        currency="CNY",
        qlib_market="csi300",
        benchmark="SH000300",
        universe=("csi300",),
        trading_rules=TradingRules(
            limit_threshold=0.095, open_cost=0.0005, close_cost=0.0015, min_cost=5, topk=50, n_drop=5
        ),
    ),
    "us-sp500": MarketProfile(
        market_id="us-sp500",
        region="us",
        currency="USD",
        qlib_market="sp500",
        benchmark="^GSPC",
        universe=("sp500",),
        trading_rules=TradingRules(limit_threshold=None, topk=50, n_drop=5, **US_COST),
    ),
    "us-nasdaq100": MarketProfile(
        market_id="us-nasdaq100",
        region="us",
        currency="USD",
        qlib_market="nasdaq100",
        benchmark="^NDX",
        universe=("nasdaq100",),
        # About 20% of the index, close to China's 50 of 300; topk 50 would mostly track the index.
        trading_rules=TradingRules(limit_threshold=None, topk=20, n_drop=2, **US_COST),
    ),
}


def build_market_context(
    market_id: str,
    provider_uri: Path,
    train_start: date,
    train_end: date,
    valid_start: date,
    valid_end: date,
    test_start: date,
    test_end: date,
    root: Path | None = None,
) -> MarketContext:
    """Build a validated context; the caller supplies dates and data location."""
    try:
        profile = MARKET_PROFILES[market_id]
    except KeyError as exc:
        choices = ", ".join(sorted(MARKET_PROFILES))
        raise ValueError(f"Unknown market {market_id!r}. Choose one of: {choices}") from exc

    market_root = (root or Path.home() / ".rdagent" / "markets") / market_id
    # Absolute, because generate.py and qrun read the provider from other working directories.
    provider_uri = provider_uri.expanduser().absolute()
    return MarketContext(
        market_id=profile.market_id,
        provider_uri=provider_uri,
        region=profile.region,
        currency=profile.currency,
        qlib_market=profile.qlib_market,
        benchmark=profile.benchmark,
        universe=profile.universe,
        workspace_path=market_root / "workspace",
        cache_path=market_root / "cache",
        memory_path=market_root / "memory",
        calendar=provider_uri / "calendars" / "day.txt",
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        test_start=test_start,
        test_end=test_end,
        trading_rules=profile.trading_rules,
    )


def validate_research_dates(context: MarketContext) -> None:
    """Reject dates that the factor data or the provider calendar cannot serve."""
    if context.train_start < FACTOR_DATA_START:
        message = f"--train-start must be on or after {FACTOR_DATA_START}: factor data starts there"
        raise ValueError(message)
    if not context.calendar.is_file():
        return
    days = context.calendar.read_text().split()
    first_day, last_day = date.fromisoformat(days[0]), date.fromisoformat(days[-1])
    if context.train_start < first_day:
        message = f"--train-start must be on or after {first_day}, the first day in {context.calendar}"
        raise ValueError(message)
    if context.test_end > last_day:
        message = f"--test-end must be on or before {last_day}, the last day in {context.calendar}"
        raise ValueError(message)
    # The backtest reads the trading day after test_end from the future calendar, or from day.txt without one.
    future_calendar = context.calendar.parent / "day_future.txt"
    next_day_source = future_calendar if future_calendar.is_file() else context.calendar
    next_day_limit = date.fromisoformat(next_day_source.read_text().split()[-1])
    if context.test_end >= next_day_limit:
        message = f"--test-end must be before {next_day_limit}, the last day in {next_day_source}"
        raise ValueError(message)


def extension_warning(context: MarketContext) -> str | None:
    """Warn when the test period crosses the day a provider was extended (see rdagent.app.us_market_data).

    Members delisted after that day have no later prices, so positions held in them then cannot be sold.
    """
    cutovers = [date.fromisoformat(day) for day in provider_cutovers(context.provider_uri)]
    crossed = [cutover for cutover in cutovers if context.test_start <= cutover < context.test_end]
    if not crossed:
        return None
    days = ", ".join(cutover.isoformat() for cutover in crossed)
    return (
        f"Warning: the test period crosses {days}, where this provider was extended. Members delisted "
        f"after it have no later prices; start the test after {crossed[-1]} to avoid frozen positions."
    )


def build_child_environment(
    context: MarketContext,
    base_env: dict[str, str] | None = None,
    model_env_type: str = "local",
    factor_python_bin: str = sys.executable,
    run_id: str | None = None,
) -> dict[str, str]:
    """Return a child-only environment without mutating the parent process.

    With a ``run_id`` the run logs into its own folder under the market's log folder. Runs that share one
    folder overwrite each other's session files and cannot be told apart in a report.
    """
    if context.trading_rules is None:
        raise ValueError("Market context has no trading rules")
    env = MarketProcessConfig(
        context=context,
        model_env_type=model_env_type,
        factor_python_bin=factor_python_bin,
    ).build_env(base_env if base_env is not None else os.environ)

    log_path = context.memory_path.parent / "log"
    env["LOG_TRACE_PATH"] = str(log_path / run_id if run_id else log_path)
    env["STEP_SEMAPHORE"] = "1"
    # Qlib records experiments in ./mlruns, a file store that MLflow 3 refuses unless allowed.
    env.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return env


class _SignatureOnly:
    """Verifies a signed pickle without loading it, so the launcher need not import what it holds."""

    @staticmethod
    def loads(data: bytes) -> None:  # noqa: ARG004 - the signature was checked before this is called
        return None

    @staticmethod
    def dumps(obj: object, protocol: int | None = None) -> bytes:
        raise NotImplementedError


def set_aside_damaged_knowledge(memory_path: Path, run_id: str) -> list[Path]:
    """Rename knowledge files that fail their signature check; return their new paths.

    The coders rewrite their knowledge file after every step, and not atomically. A run stopped during
    that write leaves a cut-off file, and the next run would end when it tried to load it.
    """
    moved = []
    for path in sorted(memory_path.glob("*.pkl")):
        try:
            serialization.loads(path.read_bytes(), serializer=_SignatureOnly)
        except ValueError:  # UntrustedArtifactError is one: unsigned, cut off or changed
            moved.append(path.rename(path.with_name(f"{path.name}.damaged-{run_id}")))
    return moved


KNOWLEDGE_STAMP = "knowledge_stamp.json"
# Raise it when the coders' checks change so much that stored verdicts can no longer be trusted.
KNOWLEDGE_FORMAT = 1


def knowledge_stamp(child_env: dict[str, str]) -> dict[str, object]:
    """The conditions the coders' knowledge is learned under."""
    return {
        "format": KNOWLEDGE_FORMAT,
        "chat_model": child_env.get("CHAT_MODEL", ""),
        "embedding_model": child_env.get("EMBEDDING_MODEL", ""),
    }


def retire_stale_knowledge(memory_path: Path, stamp: dict[str, object], run_id: str) -> list[Path]:
    """Set knowledge files learned under other conditions aside, and record the present conditions.

    Embeddings of another model cannot be compared with new ones: with another vector size the first
    lookup ends the run. And a task that a weaker chat model failed twenty times is never tried again.
    """
    stamp_file = memory_path / KNOWLEDGE_STAMP
    previous = json.loads(stamp_file.read_text()) if stamp_file.is_file() else None
    if previous == stamp:
        return []
    moved = [path.rename(path.with_name(f"{path.name}.retired-{run_id}")) for path in sorted(memory_path.glob("*.pkl"))]
    stamp_file.write_text(json.dumps(stamp, indent=2))
    return moved


def adopt_signing_settings(child_env: dict[str, str]) -> None:
    """Check signatures with the child's key: it may come from .env, which this process does not load."""
    if child_env.get("ARTIFACT_SIGNING_KEY"):
        RD_AGENT_SETTINGS.artifact_signing_key = child_env["ARTIFACT_SIGNING_KEY"]
    if child_env.get("ARTIFACT_SIGNING_KEY_PATH"):
        RD_AGENT_SETTINGS.artifact_signing_key_path = Path(child_env["ARTIFACT_SIGNING_KEY_PATH"])


def claim_market(market_root: Path) -> Path:
    """Refuse a second run of one market at a time; return the file that marks this run.

    Two runs would write the same workspace, cache and knowledge files.
    """
    marker = market_root / "run.pid"
    if marker.is_file():
        other = marker.read_text().strip()
        try:
            alive = other.isdigit() and (os.kill(int(other), 0) is None)
        except ProcessLookupError:
            alive = False  # left behind by a run that was killed
        except PermissionError:
            alive = True
        if alive:
            raise SystemExit(f"A run of this market is already going (process {other}). Wait for it or stop it.")
    marker.write_text(str(os.getpid()))
    return marker


def run_record(context: MarketContext, child_env: dict[str, str], loop_n: int) -> dict[str, object]:
    """What the report says about a run's setup. It holds no secret: only names, paths and dates."""
    return {
        "Market": context.market_id,
        "Provider": str(context.provider_uri),
        "Universe": ", ".join(context.universe),
        "Benchmark": context.benchmark,
        "Train period": f"{context.train_start} to {context.train_end}",
        "Validation period": f"{context.valid_start} to {context.valid_end}",
        "Test period": f"{context.test_start} to {context.test_end}",
        "Loops requested": loop_n,
        "Action selection": child_env.get("QLIB_QUANT_ACTION_SELECTION", "bandit"),
        "Chat model": child_env.get("CHAT_MODEL", ""),
        "Started (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def load_base_environment(dotenv_path: str | Path = ".env") -> dict[str, str]:
    """Return .env values overlaid by the process environment, without mutating it.

    The child reads its LLM settings at import time, before anything else loads
    .env, so they must already be in its environment.
    """
    file_values = {key: value for key, value in dotenv_values(dotenv_path).items() if value is not None}
    return {**file_values, **os.environ}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True, choices=sorted(MARKET_PROFILES))
    parser.add_argument("--provider-uri", required=True, type=Path)
    parser.add_argument("--train-start", required=True, type=date.fromisoformat)
    parser.add_argument("--train-end", required=True, type=date.fromisoformat)
    parser.add_argument("--valid-start", required=True, type=date.fromisoformat)
    parser.add_argument("--valid-end", required=True, type=date.fromisoformat)
    parser.add_argument("--test-start", required=True, type=date.fromisoformat)
    parser.add_argument("--test-end", required=True, type=date.fromisoformat)
    parser.add_argument("--root", type=Path, default=Path.home() / ".rdagent" / "markets")
    parser.add_argument("--loop-n", type=int, default=1)
    parser.add_argument("--model-env", choices=("local", "conda", "docker"), default="local")
    parser.add_argument("--execute", action="store_true", help="Actually start the RD-Agent research loop.")
    parser.add_argument("--no-report", action="store_true", help="Do not write the run's report when it ends.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.loop_n < 1:
        raise SystemExit("--loop-n must be at least 1")

    context = build_market_context(
        market_id=args.market,
        provider_uri=args.provider_uri,
        train_start=args.train_start,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        test_start=args.test_start,
        test_end=args.test_end,
        root=args.root,
    )
    try:
        validate_research_dates(context)
        if args.model_env == "docker":
            ensure_docker_supports_market(context.market_id, str(context.provider_uri))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    warning = extension_warning(context)
    if warning:
        print(warning)
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    child_env = build_child_environment(
        context, base_env=load_base_environment(), model_env_type=args.model_env, run_id=run_id
    )
    log_path = Path(child_env["LOG_TRACE_PATH"])
    command = [
        sys.executable,
        "-m",
        "rdagent.app.qlib_rd_loop.quant",
        "--loop_n",
        str(args.loop_n),
    ]

    print(f"Market: {context.market_id}")
    print(f"Provider: {context.provider_uri}")
    print(f"Universe: {', '.join(context.universe)}")
    print(f"Benchmark: {context.benchmark}")
    print(f"Trading rules: {context.trading_rules}")
    print(f"Workspace: {context.workspace_path}")
    print(f"Factor data: {child_env['FACTOR_CoSTEER_DATA_FOLDER']}")
    print(f"Cache: {child_env['PICKLE_CACHE_FOLDER_PATH_STR']}")
    print(f"Memory: {context.memory_path}")
    print(f"Log: {log_path}")
    print(f"Calendar: {context.calendar}")
    print(f"Command: {' '.join(command)}")
    if not args.execute:
        print("Validation only. Add --execute to start the research loop.")
        return 0

    if not context.provider_uri.is_dir():
        raise SystemExit(f"Provider directory does not exist: {context.provider_uri}")
    if not context.calendar.is_file():
        raise SystemExit(f"Qlib calendar does not exist: {context.calendar}")
    # The backtest and the factor pruning both need it; without this the run fails after its first coding step.
    members = context.provider_uri / "instruments" / f"{context.qlib_market}.txt"
    if not members.is_file():
        raise SystemExit(f"The market's member file does not exist: {members}")

    for directory in (context.workspace_path, context.cache_path, context.memory_path, log_path):
        directory.mkdir(parents=True, exist_ok=True)
    marker = claim_market(context.memory_path.parent)
    try:
        adopt_signing_settings(child_env)
        for damaged in set_aside_damaged_knowledge(context.memory_path, run_id):
            print(f"Warning: a knowledge file failed its signature check and was set aside as {damaged}")
        for retired in retire_stale_knowledge(context.memory_path, knowledge_stamp(child_env), run_id):
            print(f"The chat or embedding model changed, so earlier coder knowledge was set aside as {retired}")
        (log_path / "run.json").write_text(json.dumps(run_record(context, child_env, args.loop_n), indent=2))

        returncode = subprocess.run(command, env=child_env, check=False).returncode
        if not args.no_report:
            try:
                from rdagent.app.market_report import (  # noqa: PLC0415 - imports the research classes
                    write_report,
                )

                print(f"Report: {write_report(context.market_id, context.memory_path.parent, run_id)}")
            except (Exception, SystemExit) as error:  # noqa: BLE001 - the run's exit code matters more
                print(f"No report was written: {error}")
        return returncode
    finally:
        marker.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

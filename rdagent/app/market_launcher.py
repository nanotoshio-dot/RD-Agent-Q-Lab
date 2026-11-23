"""Launch the existing RD-Agent Qlib research loop in an isolated market process.

This module deliberately does not import the Qlib workflow at module import time.
Market-specific environment variables must be set before RD-Agent settings are
initialized in the child process.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from rdagent.core.market import MarketContext, MarketProcessConfig


@dataclass(frozen=True)
class MarketProfile:
    market_id: str
    region: str
    currency: str
    qlib_market: str
    benchmark: str
    universe: tuple[str, ...]


MARKET_PROFILES: dict[str, MarketProfile] = {
    "china": MarketProfile(
        market_id="china",
        region="cn",
        currency="CNY",
        qlib_market="csi300",
        benchmark="SH000300",
        universe=("csi300",),
    ),
    "us-sp500": MarketProfile(
        market_id="us-sp500",
        region="us",
        currency="USD",
        qlib_market="sp500",
        benchmark="^GSPC",
        universe=("sp500",),
    ),
    "us-nasdaq100": MarketProfile(
        market_id="us-nasdaq100",
        region="us",
        currency="USD",
        qlib_market="nasdaq100",
        benchmark="^NDX",
        universe=("nasdaq100",),
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
    return MarketContext(
        market_id=profile.market_id,
        provider_uri=provider_uri.expanduser(),
        region=profile.region,
        currency=profile.currency,
        qlib_market=profile.qlib_market,
        benchmark=profile.benchmark,
        universe=profile.universe,
        workspace_path=market_root / "workspace",
        cache_path=market_root / "cache",
        memory_path=market_root / "memory",
        calendar=provider_uri.expanduser() / "calendars" / "day.txt",
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        test_start=test_start,
        test_end=test_end,
    )


def build_child_environment(
    context: MarketContext,
    base_env: dict[str, str] | None = None,
    model_env_type: str = "local",
    factor_python_bin: str = sys.executable,
) -> dict[str, str]:
    """Return a child-only environment without mutating the parent process."""
    env = MarketProcessConfig(
        context=context,
        model_env_type=model_env_type,
        factor_python_bin=factor_python_bin,
    ).build_env(base_env if base_env is not None else os.environ)

    log_path = context.memory_path.parent / "log"
    env["LOG_TRACE_PATH"] = str(log_path)
    env["STEP_SEMAPHORE"] = "1"
    return env


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
    child_env = build_child_environment(context, model_env_type=args.model_env)
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
    print(f"Workspace: {context.workspace_path}")
    print(f"Cache: {context.cache_path}")
    print(f"Memory: {context.memory_path}")
    print(f"Calendar: {context.calendar}")
    print(f"Command: {' '.join(command)}")
    if not args.execute:
        print("Validation only. Add --execute to start the research loop.")
        return 0

    if not context.provider_uri.is_dir():
        raise SystemExit(f"Provider directory does not exist: {context.provider_uri}")
    if not context.calendar.is_file():
        raise SystemExit(f"Qlib calendar does not exist: {context.calendar}")

    for directory in (
        context.workspace_path,
        context.cache_path,
        context.memory_path,
        context.memory_path.parent / "log",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return subprocess.run(command, env=child_env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

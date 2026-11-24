"""Extend the local US Qlib provider to the present and rebuild the S&P 500 and Nasdaq 100 membership.

Module 4 of the multi-market foundation. It reuses Qlib's Yahoo collector (a clone of microsoft/qlib in
~/qlib-src) but replaces the two sources that no longer work: the US symbol list (Eastmoney and a NYSE API)
and the S&P 500 history (the Wikipedia page changed). The original provider is copied, never modified.

    python -m rdagent.app.us_market_data build --source ~/.qlib/qlib_data/us_data \
        --target ~/.qlib/qlib_data/us_data_2026
    python -m rdagent.app.us_market_data audit --provider ~/.qlib/qlib_data/us_data_2026

Yahoo has no prices for delisted companies, so members that left the market after the old data ends keep
their history only up to then. The audit measures how much of the membership that leaves without data.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import shutil
import sys
import time
import types
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

OPEN_END = "2099-12-31"  # Qlib's end date for instruments that are still members
SP500_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
NDX_URL = "https://indexes.nasdaqomx.com/Index/WeightingData?id=NDX&tradeDate={day}T00%3A00%3A00.000&timeOfDay=SOD"
BENCHMARKS = ("^GSPC", "^NDX", "^DJI")
QLIB_SCRIPTS = Path.home() / "qlib-src" / "scripts"
USER_AGENT = "Mozilla/5.0 (research data collector)"

# Price checks for the audit: liquid names across the 2020 seam, and splits after it.
SEAM_TICKERS = ("AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA", "BRK-B", "JPM")
SPLITS = {"NVDA": "2024-06-10", "AMZN": "2022-06-06", "GOOGL": "2022-07-18", "TSLA": "2022-08-25"}

Span = tuple[str, str, str]  # symbol, first day, last day (ISO dates)


def normalize_ticker(ticker: str) -> str:
    """Yahoo and the local provider write share classes with a dash: BRK.B -> BRK-B."""
    return ticker.strip().upper().replace(".", "-")


def sp500_spans(csv_text: str) -> list[Span]:
    """Read fja05680/sp500 sp500_ticker_start_end.csv; an empty end date means a current member."""
    table = pd.read_csv(io.StringIO(csv_text))
    missing = {"ticker", "start_date", "end_date"} - set(table.columns)
    if missing:
        raise ValueError(f"S&P 500 file is missing columns: {sorted(missing)}")
    spans = set()
    for row in table.itertuples(index=False):
        start = pd.Timestamp(row.start_date).date().isoformat()
        end = OPEN_END if pd.isna(row.end_date) else pd.Timestamp(row.end_date).date().isoformat()
        spans.add((normalize_ticker(row.ticker), start, end))
    return sorted(spans)


def spans_from_daily_members(members: dict[str, set[str]]) -> list[Span]:
    """Turn per-day member sets into contiguous spans.

    Only days present in ``members`` count, so a day the source returned nothing for is skipped rather than
    read as everyone leaving. Members on the last day get Qlib's open end date.
    """
    days = sorted(members)
    spans: list[Span] = []
    open_since: dict[str, str] = {}
    last_seen: dict[str, str] = {}
    for day in days:
        today = {normalize_ticker(symbol) for symbol in members[day]}
        for symbol in today:
            open_since.setdefault(symbol, day)
            last_seen[symbol] = day
        for symbol in [s for s in open_since if s not in today]:
            spans.append((symbol, open_since.pop(symbol), last_seen[symbol]))
    spans.extend((symbol, start, OPEN_END) for symbol, start in open_since.items())
    return sorted(spans)


def merge_spans(old: list[Span], new: list[Span], cutover: str, first_new_day: str) -> list[Span]:
    """Keep ``old`` up to ``cutover`` and ``new`` after it, joining a membership that runs across both."""
    merged: list[Span] = []
    carried: dict[str, str] = {}
    for symbol, start, end in old:
        if start > cutover:
            continue
        if end >= cutover:
            carried[symbol] = start
        else:
            merged.append((symbol, start, end))
    for symbol, start, end in new:
        if symbol in carried and start == first_new_day:
            merged.append((symbol, carried.pop(symbol), end))
        else:
            merged.append((symbol, start, end))
    merged.extend((symbol, start, cutover) for symbol, start in carried.items())
    return sorted(merged)


def instruments_text(spans: list[Span]) -> str:
    return "".join(f"{symbol}\t{start}\t{end}\n" for symbol, start, end in sorted(spans))


def read_instruments(path: Path) -> list[Span]:
    rows = [line.split("\t") for line in path.read_text().splitlines() if line.strip()]
    return [(normalize_ticker(symbol), start, end) for symbol, start, end in rows]


def symbols_to_download(spans: list[Span], since: str) -> list[str]:
    """Members on or after ``since`` (the old data already covers earlier ones), plus the benchmarks."""
    return sorted({symbol for symbol, _, end in spans if end >= since} | set(BENCHMARKS))


def fetch(url: str, *, post: bool = False, tries: int = 4, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, method="POST" if post else "GET", headers={"User-Agent": USER_AGENT})
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https URLs
                return response.read()
        except OSError:
            if attempt == tries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_ndx_members(day: str, cache_dir: Path) -> set[str]:
    """Nasdaq 100 members at the open of ``day``; empty on non-trading days. Cached on disk for resumes."""
    cached = cache_dir / f"{day}.json"
    if cached.exists():
        return set(json.loads(cached.read_text()))
    payload = json.loads(fetch(NDX_URL.format(day=day), post=True))
    symbols = sorted({row["Symbol"] for row in payload.get("aaData") or [] if row.get("Symbol")})
    cached.write_text(json.dumps(symbols))
    return set(symbols)


def stub_missing_module(name: str) -> None:
    """Register an empty stand-in for a module imported at module level but unused on the US path.

    The collector's utils import baostock (China data only) unconditionally. If code ever touches the
    stand-in, it raises a clear ImportError instead of failing silently.
    """
    if name in sys.modules or importlib.util.find_spec(name) is not None:
        return
    stub = types.ModuleType(name)

    def missing(attribute: str) -> None:
        # Introspection (inspect.getmodule, pickle) probes dunder names; answer those like a normal module.
        if attribute.startswith("__"):
            raise AttributeError(attribute)
        raise ImportError(f"{name}.{attribute} is only used for China data and {name} is not installed")

    stub.__getattr__ = missing
    sys.modules[name] = stub


def load_yahoo_collector():  # noqa: ANN201 - module object from a file outside the package
    path = QLIB_SCRIPTS / "data_collector" / "yahoo" / "collector.py"
    if not path.exists():
        raise SystemExit(f"Qlib collector not found at {path}; clone microsoft/qlib into {QLIB_SCRIPTS.parent}")
    stub_missing_module("baostock")
    sys.path[:0] = [str(QLIB_SCRIPTS), str(QLIB_SCRIPTS / "data_collector")]
    # Qlib's BaseRun re-imports its own module by the name "collector" (it is meant to run from its folder),
    # so register it under that name: BaseRun then finds this object, with our symbol list patched in.
    spec = importlib.util.spec_from_file_location("collector", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["collector"] = module
    spec.loader.exec_module(module)
    return module


def normalize_extend_in_process(collector, runner, old_provider: Path) -> None:  # noqa: ANN001 - Qlib objects
    """Do what Run.normalize_data_1d_extend does, without its process pool.

    The pool pickles the normalizer, which holds the whole old provider in memory, once per file: about
    6 s per symbol on one worker, and a full copy of the old data per extra worker. Calling the same
    per-file step here loads the old data once.
    """
    normalizer = collector.Normalize(
        source_dir=runner.source_dir,
        target_dir=runner.normalize_dir,
        normalize_class=getattr(collector, f"{runner.normalize_class_name}Extend"),
        max_workers=1,
        date_field_name="date",
        symbol_field_name="symbol",
        old_qlib_data_dir=str(old_provider),
    )
    files = sorted(Path(runner.source_dir).glob("*.csv"))
    for done, path in enumerate(files, start=1):
        normalizer._executor(path)  # noqa: SLF001 - the per-file step of Normalize.normalize
        if done % 100 == 0 or done == len(files):
            print(f"normalized {done}/{len(files)}", flush=True)


def build(source: Path, target: Path, end: str, delay: float) -> None:
    sources = target / "_sources"
    if target.exists() and not sources.is_dir():
        raise SystemExit(f"{target} exists and was not created by this tool; choose another --target")
    if not target.exists():
        print(f"Copying {source} to {target}")
        shutil.copytree(source, target)
        sources.mkdir()
    cutover = (target / "calendars" / "day.txt").read_text().split()[-1]
    (sources / "cutover.txt").write_text(cutover + "\n")
    print(f"Old data ends on {cutover}; extending to {end}")

    sp500_csv = fetch(SP500_URL).decode()
    (sources / "sp500_ticker_start_end.csv").write_text(sp500_csv)
    sp500 = sp500_spans(sp500_csv)

    ndx_cache = sources / "ndx_daily"
    ndx_cache.mkdir(exist_ok=True)
    weekdays = [d.date().isoformat() for d in pd.bdate_range(pd.Timestamp(cutover) + pd.Timedelta(days=1), end)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        daily = dict(zip(weekdays, pool.map(lambda day: fetch_ndx_members(day, ndx_cache), weekdays), strict=True))
    ndx_new = spans_from_daily_members({day: members for day, members in daily.items() if members})
    print(f"Nasdaq 100: {sum(1 for m in daily.values() if m)} trading days with members")

    symbols = symbols_to_download(sp500 + ndx_new, since=cutover)
    (sources / "symbols.txt").write_text("\n".join(symbols) + "\n")
    print(f"Downloading {len(symbols)} symbols from Yahoo")

    collector = load_yahoo_collector()
    collector.YahooCollectorUS.get_instrument_list = lambda self: list(symbols)  # replaces Eastmoney and NYSE
    runner = collector.Run(
        source_dir=str(sources / "yahoo_raw"),
        normalize_dir=str(sources / "yahoo_normalized"),
        max_workers=1,
        interval="1d",
        region="US",
    )
    start = (pd.Timestamp(cutover) - pd.Timedelta(days=1)).date().isoformat()
    runner.download_data(delay=delay, start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat())
    normalize_extend_in_process(collector, runner, target)
    from dump_bin import (  # noqa: PLC0415 - lives in the Qlib scripts folder
        DumpDataUpdate,
    )

    DumpDataUpdate(
        data_path=str(sources / "yahoo_normalized"), qlib_dir=str(target), exclude_fields="symbol,date", max_workers=4
    ).dump()

    first_new_day = min(day for day, members in daily.items() if members)
    ndx = merge_spans(read_instruments(target / "instruments" / "nasdaq100.txt"), ndx_new, cutover, first_new_day)
    (target / "instruments" / "sp500.txt").write_text(instruments_text(sp500))
    (target / "instruments" / "nasdaq100.txt").write_text(instruments_text(ndx))
    print("Instruments rebuilt; running the audit")
    report = audit(target)
    (sources / "audit.json").write_text(json.dumps(report, indent=2))


def audit(provider: Path) -> dict:
    import qlib  # noqa: PLC0415 - heavy import only when needed
    from qlib.data import D  # noqa: PLC0415

    qlib.init(provider_uri=str(provider), region="us")
    calendar = [day.date().isoformat() for day in D.calendar()]
    cutover_file = provider / "_sources" / "cutover.txt"
    cutover = cutover_file.read_text().strip() if cutover_file.exists() else calendar[0]
    report: dict = {"provider": str(provider), "calendar": [calendar[0], calendar[-1]], "cutover": cutover}

    checks = [d for d in ("2010-01-04", "2015-01-02", "2020-01-02", "2022-01-03", "2024-01-02") if d <= calendar[-1]]
    checks.append(calendar[-1])
    for universe in ("sp500", "nasdaq100"):
        spans = read_instruments(provider / "instruments" / f"{universe}.txt")
        coverage = {}
        for day in checks:
            members = sorted({s for s, start, end in spans if start <= day <= end})
            close = D.features(members, ["$close"], start_time=day, end_time=day)
            coverage[day] = f"{close['$close'].notna().sum()}/{len(members)}"
        report[universe] = coverage

    seam_start = (pd.Timestamp(cutover) - pd.Timedelta(days=10)).date().isoformat()
    seam_end = (pd.Timestamp(cutover) + pd.Timedelta(days=10)).date().isoformat()
    move = "$close/Ref($close, 1) - 1"
    seam = D.features(list(SEAM_TICKERS), [move], start_time=seam_start, end_time=seam_end)
    report["largest_daily_move_at_seam"] = round(float(seam[move].abs().max()), 4)
    report["split_day_moves"] = {
        symbol: round(float(D.features([symbol], [move], start_time=day, end_time=day)[move].iloc[0]), 4)
        for symbol, day in SPLITS.items()
        if day <= calendar[-1]
    }
    report["benchmark_last_day"] = {
        symbol: str(D.features([symbol], ["$close"]).dropna().index.get_level_values("datetime").max().date())
        for symbol in BENCHMARKS
    }
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    build_cmd = commands.add_parser("build", help="copy the provider, extend prices and rebuild membership")
    build_cmd.add_argument("--source", type=Path, required=True)
    build_cmd.add_argument("--target", type=Path, required=True)
    build_cmd.add_argument("--end", default=date.today().isoformat(), help="last day to download (default: today)")
    build_cmd.add_argument("--delay", type=float, default=1.0, help="seconds between Yahoo requests")
    audit_cmd = commands.add_parser("audit", help="check coverage, the seam and splits of a provider")
    audit_cmd.add_argument("--provider", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.source.expanduser(), args.target.expanduser(), args.end, args.delay)
    else:
        audit(args.provider.expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

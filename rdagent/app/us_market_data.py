"""Extend the local US Qlib provider to the present and rebuild the S&P 500 and Nasdaq 100 membership.

Module 4 of the multi-market foundation. It reuses Qlib's Yahoo collector (a clone of microsoft/qlib in
~/qlib-src) but replaces the two sources that no longer work: the US symbol list (Eastmoney and a NYSE API)
and the S&P 500 history (the Wikipedia page changed). The original provider is copied, never modified.

    python -m rdagent.app.us_market_data build --source ~/.qlib/qlib_data/us_data \
        --target ~/.qlib/qlib_data/us_data_2026
    python -m rdagent.app.us_market_data audit --provider ~/.qlib/qlib_data/us_data_2026

A Yahoo series only extends an old instrument after a check that both describe the same security on the
last old day. Yahoo reuses tickers, and Qlib's dump_update appends rows right after the old end without
checking dates, so an unchecked series can put another company's prices on the wrong dates. The check also
finds renames (the Yahoo series under the new ticker continues the old code), so membership after the old
end is mapped to the code that holds the data.

Yahoo has no prices for delisted companies, so members that left the market after the old data ends keep
their history only up to then, and their membership ends at their last bar.
"""

from __future__ import annotations

import argparse
import bisect
import importlib.util
import io
import json
import math
import re
import shutil
import sys
import time
import types
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rdagent.core.market.process_config import (
    PROVIDER_BUILD_ID,
    PROVIDER_CUTOVER,
    PROVIDER_EARLIER_CUTOVERS,
    provider_cutovers,
)

OPEN_END = "2099-12-31"  # Qlib's end date for instruments that are still members
SP500_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
NDX_URL = "https://indexes.nasdaqomx.com/Index/WeightingData?id=NDX&tradeDate={day}T00%3A00%3A00.000&timeOfDay=SOD"
BENCHMARKS = ("^GSPC", "^NDX", "^DJI")
# Renamed after leaving the index, so no membership source names them; found by the data audit.
EXTRA_SUCCESSORS = ("GAP", "DINO", "FBIN")
# Membership tickers with no Yahoo series of their own whose company is stored under another code.
MANUAL_ALIASES = {"FI": "FISV"}  # Fiserv traded as FI from 2023 to 2025
# Qlib's old S&P parser starts a member on this day when it found no addition for its ticker.
PLACEHOLDER_START = "1999-01-01"
# Members whose ticker changed while in the S&P 500 before the old data ends. The old provider keeps their
# whole history under the new ticker, but fja05680 starts the new ticker on the change day, so their
# membership starts with the earlier ticker. GOOG named the class A shares (now GOOGL) until 2014.
# Only renames from 2009 on are listed. An earlier one starts the member on the rename day, which matters
# only to research before the factor data begins (2008-12-29).
RENAMED_FROM = {
    "BKR": "BHGE",
    "CPRI": "KORS",
    "DD": "DWDP",  # DowDuPont, between the old and the new DuPont
    "GL": "TMK",
    "GOOG": "GOOGL",
    "HWM": "ARNC",
    "J": "JEC",
    "LHX": "HRS",
    "LIN": "PX",
    "LUMN": "CTL",
    "NLOK": "SYMC",
    "PEAK": "HCP",
    "RTX": "UTX",
    "TFC": "BBT",
    "TT": "IR",  # Trane Technologies was Ingersoll Rand until 2020
}
# The opposite case: the old S&P file names these members by the ticker of the time, and the old provider
# keeps their prices under the ticker they took later. Without this they are members without prices.
OLD_TICKER_DATA_CODE = {
    "CSC": "DXC",  # Computer Sciences
    "DPS": "KDP",  # Dr Pepper Snapple
    "DV": "ATGE",  # DeVry
    "JDSU": "VIAV",  # JDS Uniphase
    "PCS": "TMUS",  # MetroPCS
    "SAI": "LDOS",  # SAIC before it became Leidos
    "WYN": "WYND",  # Wyndham Worldwide
}
# The old file's IR span is the old Ingersoll Rand, whose prices are under TT. The code IR holds the
# company that took the name and joined the index on this day.
OLD_SPAN_FIRST_DAYS = {"IR": "2020-03-03"}
# The fields of the original provider. Yahoo adds adjclose and dividends, which dump_update would write
# with the wrong start date for existing instruments.
FIELDS = ("open", "high", "low", "close", "volume", "factor", "change")
TICKER = re.compile(r"^\^?[A-Z0-9-]+$")
QLIB_SCRIPTS = Path.home() / "qlib-src" / "scripts"
USER_AGENT = "Mozilla/5.0 (research data collector)"

# Price checks for the audit: liquid names across the seam, and splits after it.
SEAM_TICKERS = ("AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA", "BRK-B", "JPM")
SPLITS = {"NVDA": "2024-06-10", "AMZN": "2022-06-06", "GOOGL": "2022-07-18", "TSLA": "2022-08-25"}

Span = tuple[str, str, str]  # symbol, first day, last day (ISO dates)


def normalize_ticker(ticker: str) -> str:
    """Yahoo and the local provider write share classes with a dash (BRK.B -> BRK-B).

    The Nasdaq source wraps halted names in dollar signs ($ANSS$); they mark the same company.
    """
    return ticker.strip().strip("$").upper().replace(".", "-")


def sp500_spans(csv_text: str) -> list[Span]:
    """Read fja05680/sp500 sp500_ticker_start_end.csv; an empty end date means a current member.

    The end date is the day the removal takes effect; ``last_member_days`` turns it into Qlib's last day.
    """
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


def last_member_days(spans: list[Span], calendar: list[str]) -> list[Span]:
    """Replace each removal date with the trading day before it, the last day as a member."""
    result = []
    for symbol, start, end in spans:
        if end != OPEN_END:
            position = bisect.bisect_left(calendar, end)
            if position == 0:
                continue
            end = calendar[position - 1]
            if end < start:
                continue
        result.append((symbol, start, end))
    return sorted(result)


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


def coalesce_spans(spans: list[Span], calendar: list[str]) -> list[Span]:
    """Join spans of one symbol that overlap or follow on the next trading day, as a rename produces."""

    def first_index(day: str) -> int:
        return bisect.bisect_left(calendar, day)

    def last_index(day: str) -> int:
        return len(calendar) if day == OPEN_END else bisect.bisect_right(calendar, day) - 1

    by_symbol: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for symbol, start, end in spans:
        by_symbol[symbol].append((start, end))
    result = []
    for symbol, items in by_symbol.items():
        items.sort()
        start, end = items[0]
        for next_start, next_end in items[1:]:
            if last_index(end) + 1 >= first_index(next_start):
                end = max(end, next_end)
            else:
                result.append((symbol, start, end))
                start, end = next_start, next_end
        result.append((symbol, start, end))
    return sorted(result)


def clip_to_data(spans: list[Span], last_bar: dict[str, str], calendar_end: str) -> list[Span]:
    """End each membership at the code's last bar; a code still trading on the last day stays open."""
    result = []
    for symbol, start, end in spans:
        last = last_bar.get(symbol)
        if last is None or last < start:
            continue
        result.append((symbol, start, end if last >= calendar_end or end <= last else last))
    return sorted(result)


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


def source_spans_of(symbol: str, by_symbol: dict[str, list[tuple[str, str]]]) -> list[tuple[str, str]]:
    """The source spans of a company: its ticker, the tickers it was renamed from, and its data code."""
    spans: list[tuple[str, str]] = list(by_symbol.get(OLD_TICKER_DATA_CODE.get(symbol, ""), []))
    seen: set[str] = set()
    current: str | None = symbol
    while current is not None and current not in seen:
        seen.add(current)
        spans.extend(by_symbol.get(current, []))
        current = RENAMED_FROM.get(current)
    return spans


def replace_placeholder_spans(old: list[Span], source: list[Span], calendar: list[str], cutover: str) -> list[Span]:
    """Give the old file's placeholder spans the member days ``source`` knows for the company.

    Qlib's old parser missed additions made under another ticker (UA for UAA, BRK.B), so it made those
    stocks members from the placeholder day, years before they joined, and kept members through the years
    between leaving and rejoining (OKE from 2001 to 2010): look-ahead before the cutover. The span keeps
    its old last day, which that parser did read. ``source`` must hold last member days, not removal days;
    only its days up to ``cutover`` are used, because membership after it is rebuilt separately.
    """
    by_symbol: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for symbol, start, end in source:
        by_symbol[symbol].append((start, end))
    result = []
    for symbol, old_start, end in old:
        start = max(old_start, OLD_SPAN_FIRST_DAYS.get(symbol, old_start))
        known = source_spans_of(symbol, by_symbol) if start == PLACEHOLDER_START else []
        if not known:
            if start <= end:
                result.append((symbol, start, end))
            continue
        inside = [(symbol, max(first, start), min(last, end, cutover)) for first, last in known]
        inside = coalesce_spans([span for span in inside if span[1] <= span[2]], calendar)
        if symbol in RENAMED_FROM:
            inside = bridge_short_gaps(inside, calendar)
        if inside:
            inside[-1] = (symbol, inside[-1][1], end)
        result.extend(inside)
    return coalesce_spans(result, calendar)


def bridge_short_gaps(spans: list[Span], calendar: list[str], max_gap: int = 10) -> list[Span]:
    """Join consecutive spans of one symbol that are at most ``max_gap`` trading days apart.

    The source leaves a few days between a renamed company's two tickers (Praxair's last day and
    Linde's first are four trading days apart), although the company never left the index.
    """
    result: list[Span] = []
    for symbol, start, end in sorted(spans):
        if result and result[-1][0] == symbol:
            previous_end = bisect.bisect_right(calendar, result[-1][2])
            if bisect.bisect_left(calendar, start) - previous_end <= max_gap:
                result[-1] = (symbol, result[-1][1], max(result[-1][2], end))
                continue
        result.append((symbol, start, end))
    return result


def move_to_data_codes(spans: list[Span], calendar: list[str]) -> list[Span]:
    """Name each old-file member by the code that holds its prices (OLD_TICKER_DATA_CODE)."""
    return coalesce_spans(
        [(OLD_TICKER_DATA_CODE.get(symbol, symbol), start, end) for symbol, start, end in spans], calendar
    )


def instruments_text(spans: list[Span]) -> str:
    return "".join(f"{symbol}\t{start}\t{end}\n" for symbol, start, end in sorted(spans))


def read_instruments(path: Path) -> list[Span]:
    rows = [line.split("\t") for line in path.read_text().splitlines() if line.strip()]
    return [(normalize_ticker(symbol), start, end) for symbol, start, end in rows]


def symbols_to_download(spans: list[Span], since: str) -> list[str]:
    """Members on or after ``since`` (the old data covers earlier ones), the benchmarks and known successors."""
    members = {symbol for symbol, _, end in spans if end >= since and TICKER.match(symbol)}
    return sorted(members | set(BENCHMARKS) | set(EXTRA_SUCCESSORS))


NO_SHAPE = (math.nan, math.nan, math.nan)


@dataclass(frozen=True)
class Bar:
    """One unadjusted daily bar at the seam, with that day's total return.

    ``shape`` is open, high and low over close: unchanged by split and spin-off adjustments, and
    distinctive enough that two securities practically never share it on the same day.
    """

    close: float
    volume: float
    ret: float
    shape: tuple[float, float, float] = NO_SHAPE

    @property
    def dollar_volume(self) -> float:
        return self.close * self.volume


def same_security(
    old: Bar, new: Bar, tolerance: float = 0.01, return_tolerance: float = 1e-3, shape_tolerance: float = 1e-3
) -> bool:
    """Whether two bars of the same day describe one security.

    The day's return and the bar's shape must match, and so must the dollar volume (which survives later
    split adjustments) or the raw volume (which survives spin-off price adjustments). The return and a
    volume alone are not enough: among thousands of old codes, a few match any bar by chance. Without a
    shape on either side, both volumes must match.
    """
    values = (old.close, old.volume, old.ret, new.close, new.volume, new.ret)
    if not all(math.isfinite(value) for value in values) or old.dollar_volume <= 0 or new.dollar_volume <= 0:
        return False
    if abs(new.ret - old.ret) > return_tolerance:
        return False
    same_dollar_volume = abs(new.dollar_volume / old.dollar_volume - 1) <= tolerance
    same_volume = abs(new.volume / old.volume - 1) <= tolerance
    if all(math.isfinite(value) for value in (*old.shape, *new.shape)):
        same_shape = all(abs(a - b) <= shape_tolerance for a, b in zip(old.shape, new.shape, strict=True))
        return same_shape and (same_dollar_volume or same_volume)
    return same_dollar_volume and same_volume


@dataclass
class ExtensionPlan:
    """Which Yahoo series continues which code of the old provider."""

    extend: dict[str, str] = field(default_factory=dict)  # old code -> Yahoo symbol continuing it
    gapped: list[str] = field(default_factory=list)  # defunct old codes whose ticker now names a new company
    new: list[str] = field(default_factory=list)  # Yahoo symbols with no code in the old provider
    aliases: dict[str, str] = field(default_factory=dict)  # Yahoo ticker -> old code holding its data
    dropped: dict[str, str] = field(default_factory=dict)  # Yahoo symbol -> reason


def plan_extension(
    old_last: dict[str, str], old_bars: dict[str, Bar], new_bars: dict[str, Bar | None], cutover: str
) -> ExtensionPlan:
    """Match Yahoo series to old codes on the last old day.

    ``old_last`` holds every old code's last day, ``old_bars`` the cutover bar of codes alive then, and
    ``new_bars`` the cutover bar of each downloaded Yahoo series (None when Yahoo has no bar that day).
    """
    plan = ExtensionPlan()
    alive = sorted(code for code, last in old_last.items() if last >= cutover and code in old_bars)
    used: set[str] = set()
    for symbol in sorted(new_bars):
        bar = new_bars[symbol]
        if symbol in old_bars and symbol in alive and bar is not None and same_security(old_bars[symbol], bar):
            plan.extend[symbol] = symbol
            used.add(symbol)
    for symbol in sorted(set(new_bars) - used):
        bar = new_bars[symbol]
        if bar is None:
            continue
        matches = [code for code in alive if code not in plan.extend and same_security(old_bars[code], bar)]
        if len(matches) == 1:
            plan.extend[matches[0]] = symbol
            plan.aliases[symbol] = matches[0]
            used.add(symbol)
        elif len(matches) > 1:
            plan.dropped[symbol] = f"matches several old codes: {', '.join(matches)}"
            used.add(symbol)
    for symbol in sorted(set(new_bars) - used):
        if symbol not in old_last:
            plan.new.append(symbol)
        elif old_last[symbol] < cutover:
            plan.gapped.append(symbol)
        else:
            plan.dropped[symbol] = "the ticker now names another security"
    return plan


def read_yahoo_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = frame["date"].astype(str).str[:10]
    return frame.drop_duplicates("date").set_index("date").sort_index()


def yahoo_bar(frame: pd.DataFrame, day: str) -> Bar | None:
    """The unadjusted bar of ``day`` in a raw Yahoo frame, with the adjusted close return."""
    if day not in frame.index:
        return None
    position = frame.index.get_loc(day)
    if position == 0:
        return None
    row, previous = frame.iloc[position], frame.iloc[position - 1]
    close = float(row["close"])
    shape = (float(row["open"]) / close, float(row["high"]) / close, float(row["low"]) / close) if close else NO_SHAPE
    return Bar(close, float(row["volume"]), float(row["adjclose"] / previous["adjclose"] - 1), shape)


def has_impossible_prices(frame: pd.DataFrame) -> bool:
    """A non-positive close or adjusted close means Yahoo serves junk under this ticker (seen for PARA)."""
    prices = frame[["close", "adjclose"]]
    return bool((prices <= 0).to_numpy().any())


def keep_index_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Give zero-volume index bars a volume, because the normalizer empties any bar with volume <= 0.

    Yahoo reports volume 0 for ^GSPC on some days (2023-05-24); an index has no tradable volume anyway.
    """
    result = frame.copy()
    volume = result["volume"].where(result["volume"] > 0)
    result["volume"] = volume.ffill().fillna(1.0).where(result["close"].notna(), result["volume"])
    return result


def finalize_frame(frame: pd.DataFrame, trading_days: list[str], previous_close: float | None) -> pd.DataFrame:
    """Prepare one normalized frame for dump_update.

    - Keep only the original fields, with change recomputed as the adjusted close return over the last
      valid close (the old provider's convention, so a missing day does not empty the next change),
      chained to ``previous_close``, the code's last old close.
    - Empty a bar without close, so no half bar is written.
    - Write a row for every day in ``trading_days``, NaN where Yahoo has no bar. dump_update appends rows
      right after the old end, so a missing day would shift every later value.
    """
    data = frame.copy()
    data["date"] = data["date"].astype(str).str[:10]
    symbol = str(data["symbol"].iloc[0])
    values = [name for name in FIELDS if name != "change"]
    data = data.drop_duplicates("date").set_index("date").sort_index()[values]
    data.loc[data["close"].isna(), values] = np.nan
    valid = data.index[data["close"].notna()]
    if len(valid) == 0:
        return pd.DataFrame(columns=["date", "symbol", *FIELDS])
    days = [day for day in trading_days if day <= valid[-1]]
    data = data.reindex(days)
    previous = data["close"].ffill().shift(1)
    if previous_close is not None and len(previous) > 0:
        previous.iloc[0] = previous_close
    data["change"] = data["close"] / previous.ffill() - 1
    data.insert(0, "symbol", symbol)
    return data.rename_axis("date").reset_index()


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


def normalize_extend_in_process(
    collector, runner, old_provider: Path, standalone: set[str]  # noqa: ANN001 - Qlib objects
) -> None:
    """Do what Run.normalize_data_1d_extend does, without its process pool.

    The pool pickles the normalizer, which holds the whole old provider in memory, once per file: about
    6 s per symbol on one worker, and a full copy of the old data per extra worker. Calling the same
    per-file step here loads the old data once.

    Codes in ``standalone`` hold a new company after a gap and are normalized on their own, like a new
    listing. The extend step would scale each price column by its own ratio to the old company's last bar,
    which breaks the bar's shape, and it would drop the first bar as the overlap.
    """
    common = {
        "source_dir": runner.source_dir,
        "target_dir": runner.normalize_dir,
        "max_workers": 1,
        "date_field_name": "date",
        "symbol_field_name": "symbol",
    }
    extend = collector.Normalize(
        normalize_class=getattr(collector, f"{runner.normalize_class_name}Extend"),
        old_qlib_data_dir=str(old_provider),
        **common,
    )
    plain = collector.Normalize(normalize_class=getattr(collector, runner.normalize_class_name), **common)
    files = sorted(Path(runner.source_dir).glob("*.csv"))
    for done, path in enumerate(files, start=1):
        normalizer = plain if path.stem.upper() in standalone else extend
        normalizer._executor(path)  # noqa: SLF001 - the per-file step of Normalize.normalize
        if done % 100 == 0 or done == len(files):
            print(f"normalized {done}/{len(files)}", flush=True)


def read_all_instruments(provider: Path) -> dict[str, str]:
    """Each instrument's last day, from instruments/all.txt."""
    return {symbol: end for symbol, _, end in read_instruments(provider / "instruments" / "all.txt")}


def old_seam_bars(provider: Path, codes: list[str], cutover: str) -> tuple[dict[str, Bar], dict[str, float]]:
    """Unadjusted cutover bars of the old provider, and each code's last stored (adjusted) close."""
    import qlib  # noqa: PLC0415 - heavy import only when needed
    from qlib.data import D  # noqa: PLC0415

    qlib.init(provider_uri=str(provider), region="us")
    fields = [
        "$close/$factor",
        "$volume*$factor",
        "$close/Ref($close, 1) - 1",
        "$close",
        "$open/$close",
        "$high/$close",
        "$low/$close",
    ]
    frame = D.features(codes, fields, start_time=cutover, end_time=cutover)
    frame.columns = ["close", "volume", "ret", "adjusted_close", "open_ratio", "high_ratio", "low_ratio"]
    frame = frame.reset_index(level="datetime", drop=True)
    bars = {
        code: Bar(
            float(row.close),
            float(row.volume),
            float(row.ret),
            (float(row.open_ratio), float(row.high_ratio), float(row.low_ratio)),
        )
        for code, row in frame.iterrows()
        if math.isfinite(row.close)
    }
    last_close = {code: float(row.adjusted_close) for code, row in frame.iterrows() if math.isfinite(row.close)}
    return bars, last_close


def prepare_yahoo_series(raw_dir: Path, planned_dir: Path, provider: Path, cutover: str) -> ExtensionPlan:
    """Check every downloaded series against the old provider and write the ones to keep, named by code."""
    old_last = read_all_instruments(provider)
    raw: dict[str, pd.DataFrame] = {}
    plan_drops: dict[str, str] = {}
    for path in sorted(raw_dir.glob("*.csv")):
        symbol = normalize_ticker(path.stem)
        frame = read_yahoo_csv(path)
        if frame.empty:
            continue
        if has_impossible_prices(frame):
            plan_drops[symbol] = "non-positive prices"
            continue
        raw[symbol] = keep_index_bars(frame) if symbol in BENCHMARKS else frame
    alive = sorted(code for code, last in old_last.items() if last >= cutover)
    old_bars, _ = old_seam_bars(provider, alive, cutover)
    plan = plan_extension(old_last, old_bars, {s: yahoo_bar(f, cutover) for s, f in raw.items()}, cutover)
    plan.dropped.update(plan_drops)
    planned_dir.mkdir(parents=True, exist_ok=True)
    targets = {**{code: symbol for code, symbol in plan.extend.items()}, **{s: s for s in plan.gapped + plan.new}}
    for code, symbol in targets.items():
        frame = raw[symbol].reset_index()
        frame["symbol"] = code
        frame.to_csv(planned_dir / f"{code}.csv", index=False)
    return plan


def finalize_normalized(normalized_dir: Path, final_dir: Path, provider: Path, cutover: str) -> dict[str, str]:
    """Run ``finalize_frame`` on every normalized file; return each written code's last bar."""
    old_last = read_all_instruments(provider)
    calendar_old = (provider / "calendars" / "day.txt").read_text().split()
    frames = {path.stem.upper(): pd.read_csv(path) for path in sorted(normalized_dir.glob("*.csv"))}
    new_days = sorted({str(day)[:10] for frame in frames.values() for day in frame["date"]} - set(calendar_old))
    calendar = calendar_old + [day for day in new_days if day > calendar_old[-1]]
    alive = sorted(code for code in frames if code in old_last and old_last[code] >= cutover)
    _, last_close = old_seam_bars(provider, alive, cutover) if alive else ({}, {})
    final_dir.mkdir(parents=True, exist_ok=True)
    last_bar = {}
    for code, frame in frames.items():
        if code in old_last:
            start = bisect.bisect_right(calendar, old_last[code])
        else:
            start = bisect.bisect_left(calendar, str(frame["date"].iloc[0])[:10])
        result = finalize_frame(frame, calendar[start:], last_close.get(code))
        if result.empty:
            continue
        result.to_csv(final_dir / f"{code}.csv", index=False)
        last_bar[code] = str(result.loc[result["close"].notna(), "date"].iloc[-1])
    return last_bar


def resume_cutover(target: Path) -> str:
    """The last day of the old data, recorded on the first run so an interrupted build can resume.

    Once the dump has run, the calendar ends after the cutover. Re-deriving the cutover from it would
    put the seam on the wrong day, so such a target is refused.
    """
    marker = target / PROVIDER_CUTOVER
    calendar_end = (target / "calendars" / "day.txt").read_text().split()[-1]
    if not marker.is_file():
        marker.write_text(calendar_end + "\n")
        return calendar_end
    cutover = marker.read_text().strip()
    if calendar_end != cutover:
        raise SystemExit(f"{target} is already extended past {cutover}; build into a new --target")
    return cutover


def last_business_day(day: str) -> str:
    return pd.bdate_range(end=day, periods=1)[0].date().isoformat()


# A raw file up to this many days behind the bulk of the files is taken for a company that still trades
# (Yahoo sometimes misses the last days of one symbol); an older one belongs to a company that left.
STALE_WINDOW_DAYS = 7


def raw_last_days(raw_dir: Path) -> dict[str, str]:
    """The last day of each symbol's raw Yahoo file."""
    last = {}
    for path in raw_dir.glob("*.csv"):
        days = pd.read_csv(path, usecols=["date"])["date"].astype(str).str[:10]
        if len(days):
            last[normalize_ticker(path.stem)] = days.max()
    return last


def usual_last_day(last: dict[str, str]) -> str:
    """The day most raw files stop on: where the build that made them ended. Empty without files."""
    return Counter(last.values()).most_common(1)[0][0] if last else ""


def stale_raw_files(raw_dir: Path, end: str) -> list[str]:
    """Symbols whose raw file stops before the requested ``end`` although the company was still trading.

    Most files stop on the day the build that made them ended. A file that stops there, or up to
    ``STALE_WINDOW_DAYS`` before, and before ``end``, is stale. A file that stops earlier belongs to a
    company that left the market, and a new download adds nothing.
    """
    last = raw_last_days(raw_dir)
    if not last:
        return []
    wanted = last_business_day(end)
    floor = (pd.Timestamp(usual_last_day(last)) - pd.Timedelta(days=STALE_WINDOW_DAYS)).date().isoformat()
    return sorted(symbol for symbol, day in last.items() if floor <= day < wanted)


def download_list(symbols: list[str], raw_dir: Path, stale: list[str], tried: set[str], refreshing: bool) -> list[str]:
    """The symbols to ask Yahoo for: the stale ones, and those without a raw file.

    With the same end as the earlier build, a symbol Yahoo had nothing for is not asked again (each costs
    several failed attempts). A refresh to a later end asks again: it may be listed now.
    """
    skip = set() if refreshing else tried
    return [s for s in symbols if s in stale or (not (raw_dir / f"{s}.csv").exists() and s not in skip)]


def copy_reused_raw(reuse_raw: Path, raw_dir: Path) -> int:
    """Copy an earlier build's raw files this target does not have yet; a file already here is newer."""
    copied = 0
    for cached in reuse_raw.glob("*.csv"):
        if not (raw_dir / cached.name).exists():
            shutil.copy(cached, raw_dir / cached.name)
            copied += 1
    return copied


def set_aside(raw_dir: Path, symbols: list[str], aside: Path) -> None:
    """Move the raw files of ``symbols`` out of the download folder, keeping them until they are replaced.

    Yahoo appends to an existing file and adjusts each download as of its own date, so a stale file is
    downloaded again in full rather than extended.
    """
    aside.mkdir(exist_ok=True)
    for symbol in symbols:
        path = raw_dir / f"{symbol}.csv"
        if path.exists():
            path.replace(aside / path.name)


def restore_unrefreshed(raw_dir: Path, aside: Path) -> list[str]:
    """Put back the set-aside file of each symbol the download saved nothing for; return those symbols.

    Qlib's collector reports no error for a symbol Yahoo fails on, so without this a rate-limited refresh
    would lose the data the symbol already had.
    """
    restored = []
    if aside.is_dir():
        for path in sorted(aside.glob("*.csv")):
            if not (raw_dir / path.name).exists():
                path.replace(raw_dir / path.name)
                restored.append(normalize_ticker(path.stem))
        shutil.rmtree(aside)
    return restored


def check_refresh(raw_dir: Path, stale: list[str], unrefreshed: list[str]) -> None:
    """Stop before the provider is touched when a refresh left too much behind.

    A few symbols may have left the market since the earlier build. Many, or a benchmark, mean Yahoo did
    not answer: the calendar would grow while most members and the benchmark end early.
    """
    if not stale:
        return
    last = raw_last_days(raw_dir)
    newest = max(last.values())
    late = [symbol for symbol in BENCHMARKS if last.get(symbol, "") < newest]
    if late or len(unrefreshed) > max(5, len(stale) // 20):  # noqa: PLR2004 - a handful may have delisted
        raise SystemExit(
            f"Yahoo refreshed {len(stale) - len(unrefreshed)} of {len(stale)} stale files"
            + (f" and left the benchmarks {late} behind" if late else "")
            + "; nothing was changed in the provider. Run the same command again."
        )


def build(source: Path, target: Path, end: str, delay: float, reuse_raw: Path | None = None) -> None:
    sources = target / "_sources"
    if target.exists() and not sources.is_dir():
        raise SystemExit(f"{target} exists and was not created by this tool; choose another --target")
    if not target.exists():
        print(f"Copying {source} to {target}")
        # An extended source carries its own build records; this build starts its own. It keeps the days
        # where earlier builds joined their data: members without later prices still end there, and the
        # launcher warns about a test that crosses one.
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("_sources"))
        sources.mkdir()
        earlier = provider_cutovers(source)
        if earlier:
            (target / PROVIDER_EARLIER_CUTOVERS).write_text("\n".join(earlier) + "\n")
    cutover = resume_cutover(target)
    print(f"Old data ends on {cutover}; extending to {end}")

    sp500_csv = fetch(SP500_URL).decode()
    (sources / "sp500_ticker_start_end.csv").write_text(sp500_csv)
    sp500 = sp500_spans(sp500_csv)

    ndx_cache = sources / "ndx_daily"
    ndx_cache.mkdir(exist_ok=True)
    if reuse_raw is not None and (reuse_raw.parent / "ndx_daily").is_dir():
        for cached in (reuse_raw.parent / "ndx_daily").glob("*.json"):
            shutil.copy(cached, ndx_cache / cached.name)
    weekdays = [d.date().isoformat() for d in pd.bdate_range(pd.Timestamp(cutover) + pd.Timedelta(days=1), end)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        daily = dict(zip(weekdays, pool.map(lambda day: fetch_ndx_members(day, ndx_cache), weekdays), strict=True))
    trading = {day: members for day, members in daily.items() if members}
    ndx_new = spans_from_daily_members(trading)
    print(f"Nasdaq 100: {len(trading)} trading days with members")

    raw_dir = sources / "yahoo_raw"
    raw_dir.mkdir(exist_ok=True)
    aside = sources / "yahoo_raw_stale"
    # An interrupted refresh left the files it had not replaced yet there; they are still stale.
    pending = restore_unrefreshed(raw_dir, aside)
    tried: set[str] = set()  # symbols the earlier build already asked Yahoo for, with or without data
    if reuse_raw is not None:
        copy_reused_raw(reuse_raw, raw_dir)
        if (reuse_raw.parent / "symbols.txt").is_file():
            tried = set((reuse_raw.parent / "symbols.txt").read_text().split())
    symbols = symbols_to_download(sp500 + ndx_new, since=cutover)
    # A file nobody asks for any more is left alone.
    stale = sorted((set(stale_raw_files(raw_dir, end)) | set(pending)) & set(symbols))
    refreshing = bool(pending) or usual_last_day(raw_last_days(raw_dir)) < last_business_day(end)
    set_aside(raw_dir, stale, aside)
    missing = download_list(symbols, raw_dir, stale, tried, refreshing)
    print(
        f"{len(symbols)} symbols, {len(missing)} to download from Yahoo ({len(stale)} stale, "
        f"{len(tried)} tried by the earlier build)"
    )

    collector = load_yahoo_collector()
    collector.YahooCollectorUS.get_instrument_list = lambda self: list(missing)  # replaces Eastmoney and NYSE
    runner = collector.Run(source_dir=str(raw_dir), normalize_dir=str(sources / "yahoo_normalized"), region="US")
    # A week back, so even a Monday cutover has the bar before it that the identity check's return needs.
    start = (pd.Timestamp(cutover) - pd.Timedelta(days=7)).date().isoformat()
    if missing:
        runner.download_data(
            delay=delay, start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
        )
    unrefreshed = restore_unrefreshed(raw_dir, aside)
    if unrefreshed:
        print(f"Yahoo returned nothing for {len(unrefreshed)} stale symbols; their earlier files are kept")
    check_refresh(raw_dir, stale, unrefreshed)
    absent = [symbol for symbol in BENCHMARKS if not (raw_dir / f"{symbol}.csv").exists()]
    if absent:
        raise SystemExit(f"No Yahoo data for the benchmarks {absent}; run the same command again.")
    # Written after the download, so an interrupted one does not mark its symbols as tried for a reuse.
    (sources / "symbols.txt").write_text("\n".join(sorted(set(symbols) | tried)) + "\n")

    # Work folders from an interrupted run would mix an old plan into this one.
    for work in ("yahoo_planned", "yahoo_normalized", "yahoo_final"):
        shutil.rmtree(sources / work, ignore_errors=True)
    plan = prepare_yahoo_series(raw_dir, sources / "yahoo_planned", target, cutover)
    (sources / "plan.json").write_text(json.dumps(asdict(plan), indent=2, sort_keys=True))
    print(
        f"Plan: {len(plan.extend)} extended ({len(plan.aliases)} under a renamed ticker), {len(plan.new)} new, "
        f"{len(plan.gapped)} after a gap, {len(plan.dropped)} dropped"
    )
    runner.source_dir = sources / "yahoo_planned"
    normalize_extend_in_process(collector, runner, target, set(plan.gapped))
    last_bar = finalize_normalized(sources / "yahoo_normalized", sources / "yahoo_final", target, cutover)
    from dump_bin import (  # noqa: PLC0415 - lives in the Qlib scripts folder
        DumpDataUpdate,
    )

    DumpDataUpdate(
        data_path=str(sources / "yahoo_final"), qlib_dir=str(target), include_fields=",".join(FIELDS), max_workers=4
    ).dump()

    calendar = (target / "calendars" / "day.txt").read_text().split()
    first_new_day = calendar[bisect.bisect_right(calendar, cutover)]
    all_last = read_all_instruments(target)
    last_data = {code: min(last, calendar[-1]) for code, last in all_last.items()}
    last_data.update(last_bar)
    aliases = {**MANUAL_ALIASES, **plan.aliases}
    write_membership(target, "sp500", sp500, aliases, last_data, calendar, cutover, first_new_day, removals=True)
    write_membership(target, "nasdaq100", ndx_new, aliases, last_data, calendar, cutover, first_new_day)
    for universe in ("sp500", "nasdaq100"):
        untrusted = drop_untrusted_members(target, universe)
        if untrusted:
            print(f"{universe}: removed {len(untrusted)} members with a broken price series: {sorted(untrusted)}")
    print("Instruments rebuilt; running the audit")
    report = audit(target)
    (sources / "audit.json").write_text(json.dumps(report, indent=2))
    # Last, so only a complete build changes the factor data key (see MarketProcessConfig).
    build_id = f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} end={end}"
    (target / PROVIDER_BUILD_ID).write_text(build_id.replace(" ", "_") + "\n")


def write_membership(
    provider: Path,
    universe: str,
    source_spans: list[Span],
    aliases: dict[str, str],
    last_data: dict[str, str],
    calendar: list[str],
    cutover: str,
    first_new_day: str,
    *,
    removals: bool = False,
) -> None:
    """Keep the old file up to the cutover and rebuild membership after it from ``source_spans``.

    ``removals`` marks the S&P source: its end dates are removal days, and its start dates replace the
    old file's placeholder starts. The source membership after the cutover, before members without data
    are dropped, is kept in _sources for the audit.
    """
    spans = last_member_days(source_spans, calendar) if removals else source_spans
    after = [
        (aliases.get(symbol, symbol), max(start, first_new_day), end)
        for symbol, start, end in spans
        if end >= first_new_day and TICKER.match(symbol)
    ]
    after = coalesce_spans(after, calendar)
    (provider / "_sources" / f"{universe}_source.txt").write_text(instruments_text(after))
    after = clip_to_data(after, last_data, calendar[-1])
    path = provider / "instruments" / f"{universe}.txt"
    old = read_instruments(path)
    if removals:
        old = move_to_data_codes(replace_placeholder_spans(old, spans, calendar, cutover), calendar)
    path.write_text(instruments_text(merge_spans(old, after, cutover, first_new_day)))


def misaligned_instruments(provider: Path) -> list[str]:
    """Instruments whose close bin does not end on their all.txt end date (a shifted append)."""
    calendar = (provider / "calendars" / "day.txt").read_text().split()
    position = {day: index for index, day in enumerate(calendar)}
    bad = []
    for symbol, _, end in read_instruments(provider / "instruments" / "all.txt"):
        path = provider / "features" / symbol.lower() / "close.day.bin"
        if not path.exists() or end not in position:
            continue
        values = np.fromfile(path, dtype="<f4")
        if len(values) < 2 or int(values[0]) + len(values) - 2 != position[end]:
            bad.append(symbol)
    return sorted(bad)


# An index member's close has moved beyond these in one day only a handful of times (AIG fell 61% on
# 2008-09-15). Several such moves in one membership mean the stored series is broken: the old provider
# interleaves two series for some codes (TIE, HNZ, CBE) and has runaway adjustments for others (CHK).
IMPOSSIBLE_RISE = 1.0
IMPOSSIBLE_FALL = -0.5


def impossible_moves(provider: Path, code: str, spans: list[tuple[str, str]], calendar: list[str]) -> list[str]:
    """Days within ``spans`` whose close moved beyond the limits against the previous valid close.

    A rise is listed with a leading "+", so callers can tell the two apart.
    """
    path = provider / "features" / code.lower() / "close.day.bin"
    if not path.exists():
        return []
    values = np.fromfile(path, dtype="<f4")
    start, closes = int(values[0]) if len(values) else 0, values[1:].astype("float64")
    valid = np.flatnonzero(np.isfinite(closes) & (closes > 0))
    moves = closes[valid[1:]] / closes[valid[:-1]] - 1
    days = []
    for position, move in zip(valid[1:], moves, strict=True):
        if IMPOSSIBLE_FALL <= move <= IMPOSSIBLE_RISE:
            continue
        day = calendar[start + int(position)]
        if any(first <= day <= last for first, last in spans):
            days.append(("+" if move > 0 else "-") + day)
    return days


def untrusted_members(provider: Path, spans: list[Span]) -> dict[str, int]:
    """Members whose series is broken while they are in the index, with their count of impossible moves.

    One or two impossible falls can be real. Two impossible rises, or three impossible moves, are not.
    """
    calendar = (provider / "calendars" / "day.txt").read_text().split()
    by_code: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for code, start, end in spans:
        by_code[code].append((start, end))
    untrusted = {}
    for code, member_spans in by_code.items():
        moves = impossible_moves(provider, code, member_spans, calendar)
        rises = sum(move.startswith("+") for move in moves)
        if rises >= 2 or len(moves) >= 3:  # noqa: PLR2004 - explained in the docstring
            untrusted[code] = len(moves)
    return untrusted


def drop_untrusted_members(provider: Path, universe: str) -> dict[str, int]:
    """Remove members with a broken price series from the universe file and record them for the audit.

    Their returns are impossible (+189,000% in a day for TIE), and one such label wipes out a day's
    cross-section when labels are normalized. Without them the universe is slightly smaller.
    """
    path = provider / "instruments" / f"{universe}.txt"
    spans = read_instruments(path)
    untrusted = untrusted_members(provider, spans)
    if untrusted:
        path.write_text(instruments_text([span for span in spans if span[0] not in untrusted]))
    record = provider / "_sources" / f"{universe}_untrusted.json"
    record.write_text(json.dumps(untrusted, indent=2, sort_keys=True))
    return untrusted


def audit(provider: Path) -> dict:
    import qlib  # noqa: PLC0415 - heavy import only when needed
    from qlib.data import D  # noqa: PLC0415

    qlib.init(provider_uri=str(provider), region="us")
    calendar = [day.date().isoformat() for day in D.calendar()]
    cutover_file = provider / PROVIDER_CUTOVER
    cutover = cutover_file.read_text().strip() if cutover_file.exists() else calendar[0]
    report: dict = {"provider": str(provider), "calendar": [calendar[0], calendar[-1]], "cutover": cutover}
    plan_file = provider / "_sources" / "plan.json"
    if plan_file.exists():
        plan = json.loads(plan_file.read_text())
        report["plan"] = {key: len(value) for key, value in plan.items()}
        report["dropped"] = plan["dropped"]

    report["fields"] = sorted({path.name for path in (provider / "features").glob("*/*.day.bin")})
    report["misaligned"] = misaligned_instruments(provider)

    checks = [d for d in ("2010-01-04", "2015-01-02", "2020-01-02", "2022-01-03", "2024-01-02") if d <= calendar[-1]]
    checks.append(calendar[-1])
    after_cutover = [day for day in calendar if day > cutover]
    stored = {path.name.upper() for path in (provider / "features").iterdir()}
    for universe in ("sp500", "nasdaq100"):
        spans = read_instruments(provider / "instruments" / f"{universe}.txt")
        # Membership is clipped to each code's data, so only the source can show members without data.
        source_file = provider / "_sources" / f"{universe}_source.txt"
        source = read_instruments(source_file) if source_file.is_file() else []
        coverage = {}
        for day in checks:
            members = sorted({s for s, start, end in spans if start <= day <= end})
            close = D.features(members, ["$close"], start_time=day, end_time=day)
            coverage[day] = f"{close['$close'].notna().sum()}/{len(members)}"
            if day > cutover and source:
                coverage[day] += f" of {len({s for s, start, end in source if start <= day <= end})} in the source"
        report[universe] = coverage
        if after_cutover:
            members = D.instruments(universe)
            close = D.features(members, ["$close"], start_time=after_cutover[0], end_time=calendar[-1])["$close"]
            member_days = sum(
                len([day for day in after_cutover if start <= day <= end]) for _, start, end in spans if end >= cutover
            )
            report[f"{universe}_member_days_with_close_after_cutover"] = (
                f"{int(close.notna().sum())}/{member_days} ({close.notna().sum() / max(member_days, 1):.1%})"
            )
        if after_cutover and source:
            codes = sorted({s for s, _, _ in source} & stored)
            closes = D.features(codes, ["$close"], start_time=after_cutover[0], end_time=calendar[-1])["$close"]
            priced = {(code, day.date().isoformat()) for code, day in closes.dropna().index}
            days = [(code, day) for code, start, end in source for day in after_cutover if start <= day <= end]
            covered = sum(item in priced for item in days)
            report[f"{universe}_source_member_days_with_close_after_cutover"] = (
                f"{covered}/{len(days)} ({covered / max(len(days), 1):.1%})"
            )
        untrusted_file = provider / "_sources" / f"{universe}_untrusted.json"
        if untrusted_file.is_file():
            report[f"{universe}_members_removed_for_broken_prices"] = json.loads(untrusted_file.read_text())
        member_spans: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for code, start, end in spans:
            member_spans[code].append((start, end))
        left = {
            code: impossible_moves(provider, code, code_spans, calendar) for code, code_spans in member_spans.items()
        }
        report[f"{universe}_impossible_moves_on_member_days"] = {
            code: days for code, days in sorted(left.items()) if days
        }

    # From the first new day: earlier moves are old data (JPM's +13.5% on 2020-11-09 would hide a break).
    seam_start = after_cutover[0] if after_cutover else cutover
    seam_end = (pd.Timestamp(cutover) + pd.Timedelta(days=10)).date().isoformat()
    move = "$close/Ref($close, 1) - 1"
    seam = D.features(list(SEAM_TICKERS), [move], start_time=seam_start, end_time=seam_end)
    report["largest_daily_move_at_seam"] = round(float(seam[move].abs().max()), 4)
    report["split_day_moves"] = {
        symbol: round(float(D.features([symbol], [move], start_time=day, end_time=day)[move].iloc[0]), 4)
        for symbol, day in SPLITS.items()
        if day <= calendar[-1]
    }
    benchmarks = D.features(list(BENCHMARKS), ["$close"], start_time=after_cutover[0] if after_cutover else None)
    report["benchmark_missing_days_after_cutover"] = {
        symbol: int(benchmarks.xs(symbol, level="instrument")["$close"].isna().sum()) for symbol in BENCHMARKS
    }
    report["benchmark_last_day"] = {
        symbol: str(D.features([symbol], ["$close"]).dropna().index.get_level_values("datetime").max().date())
        for symbol in BENCHMARKS
    }
    fields = ["$open", "$high", "$low", "$close"]
    bars = D.features(D.instruments("all"), fields, start_time=after_cutover[0] if after_cutover else None)
    bars.columns = ["open", "high", "low", "close"]
    report["non_positive_closes_after_cutover"] = int((bars["close"] <= 0).sum())
    report["ohlc_violations_after_cutover"] = ohlc_violations(bars)
    print(json.dumps(report, indent=2))
    return report


def ohlc_violations(bars: pd.DataFrame, tolerance: float = 1e-5, max_per_code: int = 5) -> dict:
    """Bars whose open or close lies outside [low, high], and the codes with more than ``max_per_code``.

    A wrong price adjustment breaks the bar's shape; a single bad bar is usually Yahoo's own rounding.
    """
    top = bars[["open", "close"]].max(axis=1)
    bottom = bars[["open", "close"]].min(axis=1)
    bad = (
        (bars["high"] < bars["low"] * (1 - tolerance))
        | (top > bars["high"] * (1 + tolerance))
        | (bottom < bars["low"] * (1 - tolerance))
    )
    per_code = bad.groupby(level="instrument").sum()
    flagged = per_code[per_code > max_per_code]
    return {"bars": int(bad.sum()), "codes": {str(code): int(count) for code, count in flagged.items()}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    build_cmd = commands.add_parser("build", help="copy the provider, extend prices and rebuild membership")
    build_cmd.add_argument("--source", type=Path, required=True)
    build_cmd.add_argument("--target", type=Path, required=True)
    build_cmd.add_argument("--end", default=date.today().isoformat(), help="last day to download (default: today)")
    build_cmd.add_argument("--delay", type=float, default=1.0, help="seconds between Yahoo requests")
    build_cmd.add_argument("--reuse-raw", type=Path, help="folder of Yahoo CSVs from an earlier build to reuse")
    audit_cmd = commands.add_parser("audit", help="check coverage, the seam and splits of a provider")
    audit_cmd.add_argument("--provider", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        reuse = args.reuse_raw.expanduser() if args.reuse_raw else None
        build(args.source.expanduser(), args.target.expanduser(), args.end, args.delay, reuse)
    else:
        audit(args.provider.expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

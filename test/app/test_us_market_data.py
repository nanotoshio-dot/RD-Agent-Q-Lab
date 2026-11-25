import json
import math
import types

import numpy as np
import pandas as pd
import pytest

from rdagent.app.us_market_data import (
    BENCHMARKS,
    EXTRA_SUCCESSORS,
    FIELDS,
    OPEN_END,
    PLACEHOLDER_START,
    Bar,
    check_refresh,
    clip_to_data,
    coalesce_spans,
    copy_reused_raw,
    download_list,
    drop_untrusted_members,
    finalize_frame,
    has_impossible_prices,
    impossible_moves,
    instruments_text,
    keep_index_bars,
    last_member_days,
    merge_spans,
    misaligned_instruments,
    move_to_data_codes,
    normalize_extend_in_process,
    normalize_ticker,
    ohlc_violations,
    plan_extension,
    raw_last_days,
    read_instruments,
    replace_placeholder_spans,
    restore_unrefreshed,
    resume_cutover,
    same_security,
    set_aside,
    sp500_spans,
    spans_from_daily_members,
    stale_raw_files,
    symbols_to_download,
    untrusted_members,
    write_membership,
)

pytestmark = pytest.mark.offline

CUTOVER = "2020-11-10"
CALENDAR = ["2020-11-06", "2020-11-09", "2020-11-10", "2020-11-11", "2020-11-12", "2020-11-13", "2020-11-16"]


@pytest.mark.parametrize(
    ("raw", "expected"), [("BRK.B", "BRK-B"), (" aapl ", "AAPL"), ("BF.B", "BF-B"), ("$ANSS$", "ANSS")]
)
def test_share_classes_use_the_yahoo_dash_and_halt_markers_are_stripped(raw, expected):
    assert normalize_ticker(raw) == expected


def test_sp500_file_becomes_spans_with_open_end_for_current_members():
    csv_text = "ticker,start_date,end_date\nAABA,1999-12-08,2017-06-19\nBRK.B,2010-02-16,\nAABA,1999-12-08,2017-06-19\n"

    assert sp500_spans(csv_text) == [("AABA", "1999-12-08", "2017-06-19"), ("BRK-B", "2010-02-16", OPEN_END)]


def test_sp500_file_without_the_expected_columns_is_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        sp500_spans("symbol,added\nAAPL,2000-01-01\n")


def test_daily_members_become_contiguous_spans():
    members = {
        "2024-01-02": {"AAPL", "AMZN"},
        "2024-01-03": {"AAPL", "AMZN"},
        "2024-01-04": {"AAPL", "ABNB"},
        "2024-01-05": {"AAPL", "ABNB", "AMZN"},
    }

    assert spans_from_daily_members(members) == [
        ("AAPL", "2024-01-02", OPEN_END),
        ("ABNB", "2024-01-04", OPEN_END),
        ("AMZN", "2024-01-02", "2024-01-03"),
        ("AMZN", "2024-01-05", OPEN_END),
    ]


def test_old_and_new_membership_join_across_the_cutover():
    old = [("AAPL", "2003-01-02", OPEN_END), ("CELG", "2003-01-02", "2019-11-20"), ("TWTR", "2015-06-01", OPEN_END)]
    new = [("AAPL", "2020-11-11", OPEN_END), ("ABNB", "2023-06-20", OPEN_END)]

    merged = merge_spans(old, new, cutover="2020-11-10", first_new_day="2020-11-11")

    assert merged == [
        ("AAPL", "2003-01-02", OPEN_END),
        ("ABNB", "2023-06-20", OPEN_END),
        ("CELG", "2003-01-02", "2019-11-20"),
        ("TWTR", "2015-06-01", "2020-11-10"),
    ]


def test_instruments_round_trip_in_qlib_format(tmp_path):
    spans = [("BRK-B", "2010-02-16", OPEN_END), ("AABA", "1999-12-08", "2017-06-19")]
    path = tmp_path / "sp500.txt"

    path.write_text(instruments_text(spans))

    assert path.read_text().splitlines()[0] == "AABA\t1999-12-08\t2017-06-19"
    assert read_instruments(path) == sorted(spans)


def test_only_members_after_the_cutover_are_downloaded_plus_benchmarks():
    spans = [
        ("AABA", "1999-12-08", "2017-06-19"),
        ("AAPL", "1982-11-30", OPEN_END),
        ("TWTR", "2018-06-07", "2022-11-01"),
    ]

    assert symbols_to_download(spans, since="2020-11-10") == sorted({"AAPL", "TWTR", *BENCHMARKS, *EXTRA_SUCCESSORS})


def test_removal_date_becomes_the_previous_trading_day():
    spans = [("OLD", "2015-01-02", "2020-11-12"), ("NOW", "2015-01-02", OPEN_END), ("SAME", "2020-11-12", "2020-11-12")]

    assert last_member_days(spans, CALENDAR) == [("NOW", "2015-01-02", OPEN_END), ("OLD", "2015-01-02", "2020-11-11")]


def test_spans_joined_by_a_rename_become_one():
    spans = [("FB", "2020-11-11", "2020-11-12"), ("FB", "2020-11-13", OPEN_END), ("XYZ", "2020-11-11", "2020-11-11")]
    spans.append(("XYZ", "2020-11-13", "2020-11-16"))

    assert coalesce_spans(spans, CALENDAR) == [
        ("FB", "2020-11-11", OPEN_END),
        ("XYZ", "2020-11-11", "2020-11-11"),
        ("XYZ", "2020-11-13", "2020-11-16"),
    ]


def test_membership_ends_at_the_last_bar_unless_still_trading():
    spans = [("LIVE", "2020-11-11", OPEN_END), ("GONE", "2020-11-11", OPEN_END), ("NONE", "2020-11-11", OPEN_END)]
    last_bar = {"LIVE": "2020-11-16", "GONE": "2020-11-12"}

    assert clip_to_data(spans, last_bar, calendar_end="2020-11-16") == [
        ("GONE", "2020-11-11", "2020-11-12"),
        ("LIVE", "2020-11-11", OPEN_END),
    ]


SHAPE = (1.0185, 1.0222, 0.9933)  # open, high and low over close for AAPL on 2020-11-10
AAPL = Bar(close=115.97, volume=138_023_400.0, ret=-0.00301, shape=SHAPE)


@pytest.mark.parametrize(
    ("new", "expected"),
    [
        (Bar(115.97, 138_023_400.0, -0.00301, SHAPE), True),
        # Yahoo adjusts history for later splits: price / 4, volume * 4, same dollar volume and shape.
        (Bar(115.97 / 4, 138_023_400.0 * 4, -0.00301, SHAPE), True),
        # A spin-off adjusts only the price; the raw volume and the shape still match.
        (Bar(115.97 * 0.808, 138_023_400.0, -0.00301, SHAPE), True),
        # Another security with the same return and volume by chance (BALL against the SPHD ETF): shape differs.
        (Bar(115.97, 138_023_400.0, -0.00301, (1.0185, 1.0337, 0.9933)), False),
        (Bar(115.97, 138_023_400.0, 0.012, SHAPE), False),
        (Bar(64_900.0, 8.0, -0.00301, SHAPE), False),
        (Bar(math.nan, 138_023_400.0, -0.00301, SHAPE), False),
    ],
)
def test_same_security_on_the_seam_day(new, expected):
    assert same_security(AAPL, new) is expected


def test_without_a_shape_both_volumes_must_match():
    old = Bar(115.97, 138_023_400.0, -0.00301)

    assert same_security(old, Bar(115.97, 138_023_400.0, -0.00301))
    assert not same_security(old, Bar(115.97 / 4, 138_023_400.0 * 4, -0.00301))


def test_plan_extends_matches_renames_and_refuses_reused_tickers():
    abc = Bar(100.0, 1_000_000.0, 0.01)
    fb = Bar(276.0, 25_000_000.0, 0.004)
    old_last = {"AAPL": CUTOVER, "ABC": CUTOVER, "COR": CUTOVER, "FB": CUTOVER, "CEG": "2018-01-30"}
    old_bars = {"AAPL": AAPL, "ABC": abc, "COR": Bar(120.0, 500_000.0, -0.02), "FB": fb}
    new_bars = {"AAPL": AAPL, "COR": abc, "META": fb, "FB": None, "CEG": None, "ABNB": None}

    plan = plan_extension(old_last, old_bars, new_bars, CUTOVER)

    assert plan.extend == {"AAPL": "AAPL", "ABC": "COR", "FB": "META"}
    assert plan.aliases == {"COR": "ABC", "META": "FB"}
    assert plan.gapped == ["CEG"]
    assert plan.new == ["ABNB"]
    assert plan.dropped == {"FB": "the ticker now names another security"}


def test_a_series_matching_two_old_codes_is_dropped():
    bar = Bar(50.0, 2_000_000.0, 0.02)
    plan = plan_extension({"A": CUTOVER, "B": CUTOVER}, {"A": bar, "B": bar}, {"NEW": bar}, CUTOVER)

    assert plan.extend == {}
    assert "several old codes" in plan.dropped["NEW"]


def normalized_frame(rows):
    frame = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "factor"])
    frame.insert(1, "symbol", "AAPL")
    frame["adjclose"] = frame["close"]
    frame["dividends"] = 0.0
    frame["change"] = 9.9  # Yahoo's convention; must be replaced
    return frame


def test_finalized_frame_keeps_the_old_fields_and_chains_change_to_the_old_close():
    frame = normalized_frame(
        [
            ("2020-11-11", 1, 2, 1, 110.0, 10, 1),
            ("2020-11-12", 1, 2, 1, 121.0, 10, 1),
            ("2020-11-13", 1, 2, 1, np.nan, 5, 1),
        ]
    )

    result = finalize_frame(frame, CALENDAR[3:], previous_close=100.0)

    assert list(result.columns) == ["date", "symbol", *FIELDS]
    assert list(result["date"]) == ["2020-11-11", "2020-11-12"]
    assert result["change"].round(6).tolist() == [0.1, 0.1]


def test_finalized_frame_fills_missing_days_so_appended_rows_keep_their_dates():
    frame = normalized_frame([("2020-11-13", 1, 2, 1, 50.0, 10, 1), ("2020-11-16", 1, 2, 1, 55.0, 10, 1)])

    result = finalize_frame(frame, CALENDAR[3:], previous_close=None)

    assert list(result["date"]) == CALENDAR[3:]
    assert result["close"].isna().tolist() == [True, True, False, False]
    assert result["volume"].isna().tolist() == [True, True, False, False]


def test_change_after_a_missing_day_uses_the_last_valid_close():
    frame = normalized_frame([("2020-11-11", 1, 2, 1, 110.0, 10, 1), ("2020-11-13", 1, 2, 1, 121.0, 10, 1)])

    change = finalize_frame(frame, CALENDAR[3:], previous_close=100.0)["change"].round(6).tolist()

    assert change[0] == 0.1
    assert math.isnan(change[1])
    assert change[2] == 0.1


def test_change_is_chained_to_the_old_close_across_leading_missing_days():
    frame = normalized_frame([("2020-11-13", 1, 2, 1, 110.0, 10, 1)])

    change = finalize_frame(frame, CALENDAR[3:], previous_close=100.0)["change"].round(6).tolist()

    assert [math.isnan(value) for value in change] == [True, True, False]
    assert change[2] == 0.1


def test_gapped_codes_are_normalized_on_their_own(tmp_path):
    calls = []

    class FakeNormalize:
        def __init__(self, normalize_class, **kwargs):
            self.name = normalize_class
            self.uses_old_data = "old_qlib_data_dir" in kwargs

        def _executor(self, path):
            calls.append((path.stem, self.name, self.uses_old_data))

    collector = types.SimpleNamespace(
        Normalize=FakeNormalize, YahooNormalizeUS1d="plain", YahooNormalizeUS1dExtend="extend"
    )
    runner = types.SimpleNamespace(
        source_dir=str(tmp_path), normalize_dir=str(tmp_path / "out"), normalize_class_name="YahooNormalizeUS1d"
    )
    for code in ("AAPL", "CEG", "ABNB"):
        (tmp_path / f"{code}.csv").write_text("date,symbol\n")

    normalize_extend_in_process(collector, runner, tmp_path, {"CEG"})

    assert calls == [("AAPL", "extend", True), ("ABNB", "extend", True), ("CEG", "plain", False)]


LONG_CALENDAR = [day.date().isoformat() for day in pd.bdate_range("1999-01-01", "2020-12-31")]


def test_placeholder_spans_get_the_member_days_the_source_knows():
    old = [
        ("AAPL", PLACEHOLDER_START, OPEN_END),
        ("BRK-B", PLACEHOLDER_START, OPEN_END),
        ("CPRI", PLACEHOLDER_START, "2020-05-11"),
        ("GOOG", PLACEHOLDER_START, OPEN_END),
        ("LATE", PLACEHOLDER_START, "2005-01-03"),
        ("LIN", PLACEHOLDER_START, OPEN_END),
        ("OKE", PLACEHOLDER_START, OPEN_END),
        ("UA", "2014-05-01", OPEN_END),
        ("XYZ", PLACEHOLDER_START, OPEN_END),
    ]
    source = [
        ("AAPL", "1996-01-02", OPEN_END),
        ("BRK-B", "2010-02-16", OPEN_END),
        ("CPRI", "2016-01-04", "2020-05-11"),
        ("KORS", "2013-11-13", "2018-09-18"),  # Michael Kors, renamed Capri
        ("GOOG", "2014-04-03", OPEN_END),
        ("GOOGL", "2006-04-03", OPEN_END),
        ("LATE", "2009-06-01", OPEN_END),
        ("LIN", "2018-11-06", OPEN_END),
        ("PX", "1996-01-02", "2018-10-30"),  # Praxair, whose history the old provider keeps under LIN
        ("OKE", "1996-01-02", "2001-08-29"),  # left the index and came back nine years later
        ("OKE", "2010-03-15", OPEN_END),
        ("UA", "2016-04-08", OPEN_END),
    ]

    assert replace_placeholder_spans(old, source, LONG_CALENDAR, CUTOVER) == [
        ("AAPL", PLACEHOLDER_START, OPEN_END),
        ("BRK-B", "2010-02-16", OPEN_END),
        ("CPRI", "2013-11-13", "2020-05-11"),
        ("GOOG", "2006-04-03", OPEN_END),
        ("LIN", PLACEHOLDER_START, OPEN_END),
        ("OKE", PLACEHOLDER_START, "2001-08-29"),
        ("OKE", "2010-03-15", OPEN_END),
        ("UA", "2014-05-01", OPEN_END),
        ("XYZ", PLACEHOLDER_START, OPEN_END),
    ]


def test_a_placeholder_span_keeps_its_old_last_day():
    # The source ends the ticker earlier than the old file; the old parser did read removals.
    old = [("HNZ", PLACEHOLDER_START, "2013-06-05")]
    source = [("HNZ", "1996-01-02", "2013-06-03")]

    assert replace_placeholder_spans(old, source, LONG_CALENDAR, CUTOVER) == [("HNZ", PLACEHOLDER_START, "2013-06-05")]


def test_a_placeholder_span_follows_the_company_through_its_tickers():
    old = [
        ("DD", PLACEHOLDER_START, "2017-08-31"),
        ("DD", PLACEHOLDER_START, OPEN_END),
        ("FISV", PLACEHOLDER_START, OPEN_END),
        ("IR", "2010-11-17", OPEN_END),
        ("PCS", PLACEHOLDER_START, "2013-04-29"),
        ("TT", PLACEHOLDER_START, OPEN_END),
    ]
    source = [
        ("DD", "1996-01-02", "2017-08-31"),
        ("DD", "2019-06-03", OPEN_END),
        ("DWDP", "2017-09-01", "2019-05-31"),  # DowDuPont, stored under DD
        ("FISV", "2001-04-02", "2023-06-06"),
        ("FISV", "2025-11-11", OPEN_END),  # after the cutover: rebuilt separately
        ("IR", "1996-01-02", "2009-06-30"),  # the old Ingersoll Rand, stored under TT
        ("IR", "2010-11-17", OPEN_END),
        ("PCS", "1998-11-24", "2004-04-22"),  # Sprint PCS, another company
        ("TMUS", "2009-06-30", "2013-04-30"),  # MetroPCS, listed by its later ticker
        ("TT", "2002-05-13", "2008-06-05"),  # Trane Inc., another company
        ("TT", "2020-03-03", OPEN_END),
    ]

    assert replace_placeholder_spans(old, source, LONG_CALENDAR, CUTOVER) == [
        ("DD", PLACEHOLDER_START, OPEN_END),
        ("FISV", "2001-04-02", OPEN_END),
        ("IR", "2020-03-03", OPEN_END),
        ("PCS", PLACEHOLDER_START, "2004-04-22"),
        ("PCS", "2009-06-30", "2013-04-29"),
        ("TT", PLACEHOLDER_START, "2009-06-30"),
        ("TT", "2010-11-17", OPEN_END),
    ]


def test_old_file_tickers_move_to_the_code_that_holds_their_prices():
    spans = [("DPS", "2008-05-07", "2018-06-29"), ("LDOS", "2019-08-09", OPEN_END), ("SAI", "2009-12-21", "2013-09-19")]

    assert move_to_data_codes(spans, LONG_CALENDAR) == [
        ("KDP", "2008-05-07", "2018-06-29"),
        ("LDOS", "2009-12-21", "2013-09-19"),
        ("LDOS", "2019-08-09", OPEN_END),
    ]


def test_membership_keeps_the_source_members_for_the_audit(tmp_path):
    (tmp_path / "instruments").mkdir()
    (tmp_path / "_sources").mkdir()
    old = [
        ("AAPL", PLACEHOLDER_START, OPEN_END),
        ("TWTR", "2018-06-07", OPEN_END),
        ("UAA", PLACEHOLDER_START, OPEN_END),
    ]
    (tmp_path / "instruments" / "sp500.txt").write_text(instruments_text(old))
    source = [("AAPL", "1996-01-02", ""), ("TWTR", "2018-06-07", "2022-11-01"), ("UAA", "2014-05-01", "")]
    source = [(symbol, start, end or OPEN_END) for symbol, start, end in source]
    last_data = {"AAPL": "2020-11-16", "UAA": "2020-11-16"}  # Yahoo has nothing for TWTR

    write_membership(tmp_path, "sp500", source, {}, last_data, CALENDAR, CUTOVER, "2020-11-11", removals=True)

    assert read_instruments(tmp_path / "instruments" / "sp500.txt") == [
        ("AAPL", PLACEHOLDER_START, OPEN_END),
        ("TWTR", "2018-06-07", CUTOVER),
        ("UAA", "2014-05-01", OPEN_END),
    ]
    assert read_instruments(tmp_path / "_sources" / "sp500_source.txt") == [
        ("AAPL", "2020-11-11", OPEN_END),
        ("TWTR", "2020-11-11", "2020-11-16"),
        ("UAA", "2020-11-11", OPEN_END),
    ]


def test_an_interrupted_build_resumes_but_an_extended_target_is_refused(tmp_path):
    (tmp_path / "calendars").mkdir()
    (tmp_path / "_sources").mkdir()
    calendar = tmp_path / "calendars" / "day.txt"
    calendar.write_text("2020-11-09\n2020-11-10\n")

    assert resume_cutover(tmp_path) == CUTOVER
    assert resume_cutover(tmp_path) == CUTOVER

    calendar.write_text("2020-11-09\n2020-11-10\n2020-11-11\n")
    with pytest.raises(SystemExit, match="already extended past 2020-11-10"):
        resume_cutover(tmp_path)


def write_raw(folder, symbol, days):
    folder.mkdir(exist_ok=True)
    pd.DataFrame({"symbol": symbol, "date": days, "close": 1.0}).to_csv(folder / f"{symbol}.csv", index=False)


def test_raw_files_that_stop_where_an_earlier_build_ended_are_stale(tmp_path):
    for symbol in ("AAPL", "MSFT", "^GSPC"):
        write_raw(tmp_path, symbol, ["2026-11-20", "2026-11-23"])
    write_raw(tmp_path, "SNDK", ["2026-11-18", "2026-11-19"])  # Yahoo missed its last days
    write_raw(tmp_path, "GONE", ["2024-03-01"])  # left the market; a new download adds nothing

    assert stale_raw_files(tmp_path, end="2026-11-23") == ["SNDK"]
    assert stale_raw_files(tmp_path, end="2026-11-22") == ["SNDK"]  # a Sunday: Friday is the last business day
    assert stale_raw_files(tmp_path, end="2026-12-31") == ["AAPL", "MSFT", "SNDK", "^GSPC"]


def test_stale_symbols_are_downloaded_again_even_with_the_same_end(tmp_path):
    # SNDK's file stopped two days early; it was tried by the earlier build, like every reused symbol.
    write_raw(tmp_path, "AAPL", ["2026-11-23"])
    symbols = ["AAPL", "GONE", "NEW", "SNDK"]
    tried = {"AAPL", "GONE", "SNDK"}  # GONE: Yahoo had nothing for it

    assert download_list(symbols, tmp_path, ["SNDK"], tried, refreshing=False) == ["NEW", "SNDK"]
    assert download_list(symbols, tmp_path, ["SNDK"], tried, refreshing=True) == ["GONE", "NEW", "SNDK"]


def test_a_reused_file_never_replaces_one_this_target_already_has(tmp_path):
    cache, raw = tmp_path / "cache", tmp_path / "raw"
    write_raw(cache, "AAPL", ["2026-11-23"])
    write_raw(cache, "MSFT", ["2026-11-23"])
    write_raw(raw, "AAPL", ["2026-12-31"])  # downloaded by an interrupted run of this build

    assert copy_reused_raw(cache, raw) == 1

    assert raw_last_days(raw) == {"AAPL": "2026-12-31", "MSFT": "2026-11-23"}


def test_a_stale_file_is_kept_until_the_download_replaces_it(tmp_path):
    raw, aside = tmp_path / "raw", tmp_path / "aside"
    write_raw(raw, "AAPL", ["2026-11-23"])
    write_raw(raw, "MSFT", ["2026-11-23"])

    set_aside(raw, ["AAPL", "MSFT"], aside)
    assert raw_last_days(raw) == {}
    write_raw(raw, "AAPL", ["2026-12-31"])  # Yahoo answered for AAPL only

    assert restore_unrefreshed(raw, aside) == ["MSFT"]
    assert raw_last_days(raw) == {"AAPL": "2026-12-31", "MSFT": "2026-11-23"}
    assert not aside.exists()
    assert restore_unrefreshed(raw, aside) == []


def test_a_refresh_that_leaves_most_files_or_a_benchmark_behind_stops_the_build(tmp_path):
    stale = [f"S{number:02d}" for number in range(40)] + ["^GSPC", "^NDX", "^DJI"]
    for symbol in stale:
        write_raw(tmp_path, symbol, ["2026-12-31"])

    check_refresh(tmp_path, stale, unrefreshed=["S00", "S01"])  # two delisted since the earlier build: fine
    check_refresh(tmp_path, [], unrefreshed=[])

    with pytest.raises(SystemExit, match="refreshed 33 of 43 stale files"):
        check_refresh(tmp_path, stale, unrefreshed=stale[:10])
    write_raw(tmp_path, "^GSPC", ["2026-11-23"])
    with pytest.raises(SystemExit, match="left the benchmarks .'\\^GSPC'. behind"):
        check_refresh(tmp_path, stale, unrefreshed=["^GSPC"])


def test_bars_outside_their_range_are_counted_per_code():
    days = pd.date_range("2022-01-20", periods=7)
    index = pd.MultiIndex.from_product([["CEG", "OK"], days], names=["instrument", "datetime"])
    # CEG's columns were scaled by different ratios: high below low on every bar.
    bars = pd.DataFrame(
        {"open": [1.1] * 7 + [1.0] * 7, "high": [0.99] * 7 + [1.02] * 7, "low": [1.1] * 7 + [0.98] * 7},
        index=index,
    ).assign(close=1.0)
    bars.loc[("OK", days[1]), "open"] = 1.03  # a single rounding glitch is not flagged

    assert ohlc_violations(bars) == {"bars": 8, "codes": {"CEG": 7}}


def test_series_with_non_positive_prices_are_rejected():
    good = pd.DataFrame({"close": [10.0, 11.0], "adjclose": [10.0, 11.0]})
    junk = pd.DataFrame({"close": [64_900.0, 0.028], "adjclose": [5.0, -1.0]})

    assert not has_impossible_prices(good)
    assert has_impossible_prices(junk)


def test_zero_volume_index_bars_survive_normalization():
    frame = pd.DataFrame({"close": [4100.0, 4115.0, np.nan], "volume": [3.5e9, 0.0, 0.0]})

    result = keep_index_bars(frame)

    assert result["volume"].tolist()[:2] == [3.5e9, 3.5e9]
    assert result["volume"].iloc[2] == 0.0


def write_bin(path, start_index, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.array([start_index, *values], dtype="<f4").tofile(path)


def test_misaligned_bins_are_detected(tmp_path):
    (tmp_path / "calendars").mkdir()
    (tmp_path / "calendars" / "day.txt").write_text("\n".join(CALENDAR) + "\n")
    (tmp_path / "instruments").mkdir()
    (tmp_path / "instruments" / "all.txt").write_text("GOOD\t2020-11-06\t2020-11-16\nBAD\t2020-11-06\t2020-11-16\n")
    write_bin(tmp_path / "features" / "good" / "close.day.bin", 0, range(7))
    write_bin(tmp_path / "features" / "bad" / "close.day.bin", 0, range(5))

    assert misaligned_instruments(tmp_path) == ["BAD"]


def test_members_with_a_broken_price_series_are_found_and_removed(tmp_path):
    days = [f"2010-02-{day:02d}" for day in range(1, 13)]
    (tmp_path / "calendars").mkdir()
    (tmp_path / "calendars" / "day.txt").write_text("\n".join(days) + "\n")
    (tmp_path / "instruments").mkdir()
    (tmp_path / "_sources").mkdir()
    spans = [("AIG", days[0], days[-1]), ("LATE", days[8], days[-1]), ("TIE", days[0], days[-1])]
    (tmp_path / "instruments" / "sp500.txt").write_text(instruments_text(spans))
    # Two series interleaved under one code, as the old provider has for TIE.
    write_bin(tmp_path / "features" / "tie" / "close.day.bin", 0, [14, 20000, 14, 20000] + [14] * 8)
    # One real crash (AIG fell 61% on 2008-09-15) does not make a series broken.
    write_bin(tmp_path / "features" / "aig" / "close.day.bin", 0, [100, 39, 40, 41, 40, 41, 42, 41, 40, 41, 42, 43])
    # Broken only before it joined the index.
    write_bin(tmp_path / "features" / "late" / "close.day.bin", 0, [1, 5, 1, 5, 1, 5, 1, 5, 5.1, 5.2, 5.3, 5.4])

    assert untrusted_members(tmp_path, spans) == {"TIE": 4}
    assert drop_untrusted_members(tmp_path, "sp500") == {"TIE": 4}

    assert [code for code, _, _ in read_instruments(tmp_path / "instruments" / "sp500.txt")] == ["AIG", "LATE"]
    assert json.loads((tmp_path / "_sources" / "sp500_untrusted.json").read_text()) == {"TIE": 4}


def test_impossible_moves_compare_with_the_previous_valid_close(tmp_path):
    days = [f"2010-02-{day:02d}" for day in range(1, 6)]
    write_bin(tmp_path / "features" / "gap" / "close.day.bin", 1, [10, np.nan, 40, 15])

    moves = impossible_moves(tmp_path, "GAP", [(days[0], days[-1])], days)

    assert moves == ["+2010-02-04", "-2010-02-05"]
    assert impossible_moves(tmp_path, "NONE", [(days[0], days[-1])], days) == []


def test_missing_china_only_module_is_replaced_by_a_loud_stand_in():
    import inspect
    import sys

    from rdagent.app.us_market_data import stub_missing_module

    name = "baostock_absent_for_test"
    try:
        stub_missing_module(name)
        module = __import__(name)

        with pytest.raises(ImportError, match="only used for China data"):
            module.login()
        # Introspection must keep working while the stand-in is loaded.
        assert not hasattr(module, "__file__")
        assert inspect.getmodule(test_installed_modules_are_never_replaced) is sys.modules[__name__]
    finally:
        sys.modules.pop(name, None)


def test_installed_modules_are_never_replaced():
    import json

    from rdagent.app.us_market_data import stub_missing_module

    stub_missing_module("json")

    assert __import__("json") is json

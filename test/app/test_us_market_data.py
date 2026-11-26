import json
import math
import types
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rdagent.app.us_market_data import (
    BENCHMARKS,
    EXTRA_SUCCESSORS,
    FIELDS,
    OPEN_END,
    PLACEHOLDER_START,
    RESTORED,
    UNREFRESHED,
    Bar,
    blank_foreign_bars,
    build,
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
    judge_series,
    keep_index_bars,
    last_business_day,
    last_member_days,
    merge_spans,
    misaligned_instruments,
    move_to_data_codes,
    normalization_calendar,
    normalize_extend_in_process,
    normalize_ticker,
    ohlc_violations,
    plan_extension,
    raw_last_days,
    read_instruments,
    refresh_raw_files,
    replace_placeholder_spans,
    restore_unrefreshed,
    resume_cutover,
    same_security,
    set_aside,
    sp500_spans,
    spans_from_daily_members,
    stale_raw_files,
    symbols_needing_data,
    symbols_to_download,
    symbols_to_retry,
    trim_untrusted,
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


def test_the_normalizer_calendar_needs_no_download(tmp_path):
    (tmp_path / "calendars").mkdir()
    (tmp_path / "calendars" / "day.txt").write_text("2020-11-09\n2020-11-10\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "symbol": "^GSPC",
            "date": ["2020-11-09", "2020-11-10", "2020-11-11", "2020-11-12", "2020-11-13"],
            "open": [3583.0, 3543.3, 3563.2, 3562.7, np.nan],
            # A row with prices but no close is a trading day; a row without any price, as Yahoo writes for
            # a corporate action, is not.
            "close": [3550.5, 3545.5, 3572.7, np.nan, np.nan],
        }
    ).to_csv(raw / "^GSPC.csv", index=False)

    calendar = normalization_calendar(tmp_path, raw)

    assert calendar == [pd.Timestamp(day) for day in ("2020-11-09", "2020-11-10", "2020-11-11", "2020-11-12")]


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
        ("UA", "2016-04-08", OPEN_END),  # see OLD_SPAN_FIRST_DAYS
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
        ("PCS", PLACEHOLDER_START, "2013-04-29"),  # MetroPCS
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

    assert replace_placeholder_spans(old + [("UA", "2014-05-01", OPEN_END)], source, LONG_CALENDAR, CUTOVER) == [
        ("DD", PLACEHOLDER_START, OPEN_END),
        ("FISV", "2001-04-02", OPEN_END),
        ("IR", "2020-03-03", OPEN_END),
        ("PCS", "2009-06-30", "2013-04-29"),  # without the Sprint PCS years
        ("TT", PLACEHOLDER_START, "2009-06-30"),
        ("TT", "2010-11-17", OPEN_END),
        ("UA", "2016-04-08", OPEN_END),  # the class C shares' first day; the class A shares are under UAA
    ]


def test_old_file_tickers_move_to_the_code_that_holds_their_prices():
    spans = [
        ("DPS", "2008-05-07", "2018-06-29"),
        ("GOOG", "2006-04-03", OPEN_END),  # the class A shares until 2014-04-02
        ("GOOGL", "2014-04-03", OPEN_END),
        ("LDOS", "2019-08-09", OPEN_END),
        ("SAI", "2009-12-21", "2013-09-19"),
    ]

    assert move_to_data_codes(spans, LONG_CALENDAR) == [
        ("GOOG", "2014-04-03", OPEN_END),
        ("GOOGL", "2006-04-03", OPEN_END),
        ("KDP", "2008-05-07", "2018-06-29"),
        ("LDOS", "2009-12-21", "2013-09-19"),
        ("LDOS", "2019-08-09", OPEN_END),
    ]
    assert move_to_data_codes([("GOOG", "2010-01-04", "2012-12-31")], LONG_CALENDAR) == [
        ("GOOGL", "2010-01-04", "2012-12-31")
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


def test_the_nasdaq_100_file_also_names_members_by_the_code_that_holds_their_prices(tmp_path):
    (tmp_path / "instruments").mkdir()
    (tmp_path / "_sources").mkdir()
    old = [("GOOG", "2005-12-19", OPEN_END), ("GOOGL", "2014-04-03", OPEN_END), ("JDSU", "2000-01-03", "2008-12-19")]
    (tmp_path / "instruments" / "nasdaq100.txt").write_text(instruments_text(old))
    source = [("GOOG", "2020-11-11", OPEN_END), ("GOOGL", "2020-11-11", OPEN_END)]
    last_data = {"GOOG": LONG_CALENDAR[-1], "GOOGL": LONG_CALENDAR[-1]}

    write_membership(tmp_path, "nasdaq100", source, {}, last_data, LONG_CALENDAR, CUTOVER, "2020-11-11")

    assert read_instruments(tmp_path / "instruments" / "nasdaq100.txt") == [
        ("GOOG", "2014-04-03", OPEN_END),
        ("GOOGL", "2005-12-19", OPEN_END),  # the class A shares were named GOOG until 2014-04-02
        ("VIAV", "2000-01-03", "2008-12-19"),  # JDS Uniphase's prices are under VIAV
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
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": symbol, "date": days, "close": 1.0}).to_csv(folder / f"{symbol}.csv", index=False)


def test_raw_files_that_stop_where_an_earlier_build_ended_are_stale(tmp_path):
    for symbol in ("AAPL", "MSFT", "^GSPC"):
        write_raw(tmp_path, symbol, ["2026-11-20", "2026-11-23"])
    write_raw(tmp_path, "SNDK", ["2026-11-18", "2026-11-19"])  # Yahoo missed its last days
    write_raw(tmp_path, "GONE", ["2024-03-01"])  # left the market; a new download adds nothing

    assert stale_raw_files(tmp_path, end="2026-11-23") == ["SNDK"]
    assert stale_raw_files(tmp_path, end="2026-11-22") == ["SNDK"]  # a Sunday: Friday is the last business day
    assert stale_raw_files(tmp_path, end="2026-12-31") == ["AAPL", "MSFT", "SNDK", "^GSPC"]


def test_an_end_on_a_market_holiday_asks_for_the_trading_day_before_it(tmp_path):
    assert last_business_day("2026-11-26") == "2026-11-25"  # Thanksgiving
    assert last_business_day("2026-04-03") == "2026-04-02"  # Good Friday
    assert last_business_day("2026-11-22") == "2026-11-20"  # a Sunday
    assert last_business_day("2025-01-09") == "2025-01-08"  # closed for a day of mourning
    assert last_business_day("2012-10-30") == "2012-10-26"  # closed for Hurricane Sandy
    for symbol in ("AAPL", "^GSPC"):
        write_raw(tmp_path, symbol, ["2026-11-24", "2026-11-25"])

    assert stale_raw_files(tmp_path, end="2026-11-26") == []  # nothing to download again on every run


def test_a_build_refuses_an_end_yahoo_cannot_serve(tmp_path):
    with pytest.raises(SystemExit, match="must be before today"):
        build(tmp_path / "source", tmp_path / "target", date.today().isoformat(), delay=0.5)
    assert not (tmp_path / "target").exists()


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
    write_raw(raw, "AAPL", ["2020-11-03", "2026-11-23"])
    write_raw(raw, "MSFT", ["2020-11-03", "2026-11-23"])

    set_aside(raw, ["AAPL", "MSFT"], aside)
    assert raw_last_days(raw) == {}
    write_raw(raw, "AAPL", ["2020-11-03", "2026-12-31"])  # Yahoo answered for AAPL only, from the same start

    assert restore_unrefreshed(raw, aside) == ["MSFT"]
    assert raw_last_days(raw) == {"AAPL": "2026-12-31", "MSFT": "2026-11-23"}
    assert not aside.exists()
    assert restore_unrefreshed(raw, aside) == []


def test_a_run_stopped_while_it_puts_files_back_still_knows_them(tmp_path, monkeypatch):
    raw, aside = tmp_path / "raw", tmp_path / "aside"
    for symbol in ("AAPL", "MSFT", "^GSPC"):
        write_raw(raw, symbol, ["2020-11-03", "2026-11-23"])
    set_aside(raw, ["AAPL", "MSFT", "^GSPC"], aside)
    moved = []
    replace = Path.replace

    def replace_once_then_stop(self, target):
        if moved:
            raise KeyboardInterrupt
        moved.append(self.name)
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", replace_once_then_stop)
    with pytest.raises(KeyboardInterrupt):
        restore_unrefreshed(raw, aside)  # after a download that saved nothing, stopped after one file
    monkeypatch.undo()

    assert moved == ["AAPL.csv"]
    assert restore_unrefreshed(raw, aside, keep=True) == ["AAPL", "MSFT", "^GSPC"]  # AAPL is still stale

    # A file a resumed run put back, and then downloaded again before it was stopped, is no longer stale.
    write_raw(raw, "AAPL", ["2020-11-03", "2026-11-23"])
    set_aside(raw, ["AAPL"], aside)
    write_raw(raw, "AAPL", ["2020-11-03", "2026-12-31"])
    assert restore_unrefreshed(raw, aside, keep=True) == ["MSFT", "^GSPC"]
    assert (aside / RESTORED).read_text() == "MSFT\n^GSPC\n"


def test_a_download_that_starts_after_the_old_file_ends_does_not_replace_it(tmp_path):
    raw, aside = tmp_path / "raw", tmp_path / "aside"
    write_raw(raw, "AAPL", ["2020-11-03", "2026-11-23"])
    write_raw(raw, "ANSS", ["2020-11-03", "2025-07-16"])
    set_aside(raw, ["AAPL", "ANSS"], aside)
    write_raw(raw, "AAPL", ["2020-11-03", "2026-12-31"])
    write_raw(raw, "ANSS", ["2026-12-31"])  # a stub row under a delisted company's ticker

    assert restore_unrefreshed(raw, aside) == ["ANSS"]
    assert raw_last_days(raw) == {"AAPL": "2026-12-31", "ANSS": "2025-07-16"}


def test_the_symbols_that_need_new_data():
    last = {"AAPL": "2026-11-23", "GONE": "2024-03-01", "LATE": "2026-11-23", "OUT": "2026-11-20"}
    spans = [
        ("AAPL", "1999-01-01", OPEN_END),
        ("ATVI", "2010-01-04", "2023-10-18"),  # without a file and not current: Yahoo had nothing for it
        ("GONE", "2010-01-04", "2024-03-04"),  # removed right after its last bar: it left the market
        ("KVUE", "2023-08-25", OPEN_END),  # current, without a file
        ("LATE", "2010-01-04", "2026-12-15"),  # removed three weeks after its file ends: Yahoo missed days
        ("OUT", "2010-01-04", "2026-11-23"),
    ]

    assert symbols_needing_data(spans, last) == {"AAPL", "KVUE", "LATE", *BENCHMARKS}


END = "2026-12-31"


def test_a_refresh_stops_for_a_lagging_benchmark_or_many_members_without_new_data(tmp_path):
    for symbol in [*(f"S{number:03d}" for number in range(10)), *BENCHMARKS]:
        write_raw(tmp_path, symbol, ["2026-12-31"])
    needed = {f"S{number:03d}" for number in range(500)} | set(BENCHMARKS)  # a limit of 5
    five = [f"S{number:03d}" for number in range(5)]

    check_refresh(tmp_path, ["L1", "L2", "L3"], needed, END)  # not needed: they may have left the market
    check_refresh(tmp_path, five, needed, END)  # Yahoo misses a few now and then
    write_raw(tmp_path, "ODD", ["2027-01-04"])  # a bar after the end is not a day the benchmarks must have
    check_refresh(tmp_path, [], needed, END)

    with pytest.raises(SystemExit, match="6 members without new data: S000, S001, S002, S003, S004, S005"):
        check_refresh(tmp_path, [*five, "S005"], needed, END)
    write_raw(tmp_path, "^GSPC", ["2026-11-23"])
    with pytest.raises(SystemExit, match=r"The benchmarks \['\^GSPC'\] are missing or end before 2026-12-31"):
        check_refresh(tmp_path, [], needed, END)
    (tmp_path / "^GSPC.csv").unlink()
    with pytest.raises(SystemExit, match=r"The benchmarks \['\^GSPC'\] are missing"):
        check_refresh(tmp_path, ["^GSPC"], needed, END)


FULL = ["2020-11-03", "2026-12-31"]


def fake_yahoo(sources, answers, asked):
    """A stand-in for the Yahoo download: it writes the files of the symbols it has data for."""

    def download(symbols):
        asked.append(sorted(symbols))
        for symbol in symbols:
            if symbol in answers:
                write_raw(sources / "yahoo_raw", symbol, answers[symbol])

    return download


def test_a_rerun_after_a_lagging_benchmark_stopped_the_build_asks_for_it_again(tmp_path):
    sources = tmp_path / "_sources"
    symbols = ["AAPL", "MSFT", *BENCHMARKS]
    for symbol in symbols:
        write_raw(sources / "yahoo_raw", symbol, ["2020-11-03", "2026-11-23"])
    spans = [("AAPL", "1999-01-01", OPEN_END), ("MSFT", "1999-01-01", OPEN_END)]
    everything = {symbol: FULL for symbol in symbols}
    asked = []

    with pytest.raises(SystemExit, match=r"The benchmarks \['\^GSPC'\] are missing or end before 2026-12-31"):
        without_gspc = {symbol: days for symbol, days in everything.items() if symbol != "^GSPC"}
        refresh_raw_files(sources, None, symbols, spans, "2026-12-31", fake_yahoo(sources, without_gspc, asked))
    assert (sources / UNREFRESHED).read_text() == "^GSPC\n"

    refresh_raw_files(sources, None, symbols, spans, "2026-12-31", fake_yahoo(sources, everything, asked))

    assert asked[-1] == ["^GSPC"]  # the other files already reach the end
    assert raw_last_days(sources / "yahoo_raw")["^GSPC"] == "2026-12-31"


def test_an_interrupted_refresh_resumes_as_a_refresh_but_an_interrupted_same_end_run_does_not(tmp_path):
    earlier = tmp_path / "earlier" / "_sources"
    for symbol in ["AAPL", "MSFT", *BENCHMARKS]:
        write_raw(earlier / "yahoo_raw", symbol, ["2020-11-03", "2026-11-23"])
    write_raw(earlier / "yahoo_raw", "SNDK", ["2020-11-03", "2026-11-19"])  # Yahoo missed its last days
    (earlier / "symbols.txt").write_text("AAPL\nMSFT\nSNDK\nZZZ\n^DJI\n^GSPC\n^NDX\n")  # nothing for ZZZ
    spans = [(symbol, "1999-01-01", OPEN_END) for symbol in ("AAPL", "MSFT", "SNDK")]
    spans.append(("ZZZ", "1999-01-01", "2024-01-02"))
    symbols = ["AAPL", "MSFT", "SNDK", "ZZZ", *BENCHMARKS]
    everything = {symbol: FULL for symbol in symbols if symbol != "ZZZ"}

    def stopped_after_three_files(sources, days):
        def download(symbols):
            for symbol in symbols[:3]:
                write_raw(sources / "yahoo_raw", symbol, days)
            raise KeyboardInterrupt

        return download

    refresh = tmp_path / "refresh" / "_sources"
    with pytest.raises(KeyboardInterrupt):
        refresh_raw_files(
            refresh, earlier / "yahoo_raw", symbols, spans, "2026-12-31", stopped_after_three_files(refresh, FULL)
        )
    asked = []
    refresh_raw_files(
        refresh, earlier / "yahoo_raw", symbols, spans, "2026-12-31", fake_yahoo(refresh, everything, asked)
    )
    # Most files reach the new end, but the resumed run is still a refresh, so ZZZ is asked for again.
    assert asked == [["ZZZ", "^DJI", "^GSPC", "^NDX"]]

    # Stopped again, right after it put the old files back: the next run still knows them and the refresh.
    twice = tmp_path / "twice" / "_sources"
    with pytest.raises(KeyboardInterrupt):
        refresh_raw_files(
            twice, earlier / "yahoo_raw", symbols, spans, "2026-12-31", stopped_after_three_files(twice, FULL)
        )
    assert restore_unrefreshed(twice / "yahoo_raw", twice / "yahoo_raw_stale", keep=True) == ["^DJI", "^GSPC", "^NDX"]
    asked = []
    refresh_raw_files(twice, earlier / "yahoo_raw", symbols, spans, "2026-12-31", fake_yahoo(twice, everything, asked))
    assert asked == [["ZZZ", "^DJI", "^GSPC", "^NDX"]]
    assert not (twice / "yahoo_raw_stale").exists()

    same_end = tmp_path / "same_end" / "_sources"
    with pytest.raises(KeyboardInterrupt):
        refresh_raw_files(
            same_end,
            earlier / "yahoo_raw",
            symbols,
            spans,
            "2026-11-23",
            stopped_after_three_files(same_end, ["2020-11-03", "2026-11-23"]),
        )
    asked = []
    refresh_raw_files(
        same_end, earlier / "yahoo_raw", symbols, spans, "2026-11-23", fake_yahoo(same_end, everything, asked)
    )
    assert asked == []  # SNDK's new file was saved before the stop; ZZZ is not asked again


def test_needed_members_left_behind_are_retried_and_a_few_are_accepted(tmp_path, capsys):
    sources = tmp_path / "_sources"
    members = ["AAPL", "MSFT", "NVDA", "WMT"]
    for symbol in [*members, "OUT", *BENCHMARKS]:
        write_raw(sources / "yahoo_raw", symbol, ["2020-11-03", "2026-11-23"])
    spans = [(symbol, "1999-01-01", OPEN_END) for symbol in [*members, "KVUE"]]
    spans.append(("OUT", "1999-01-01", "2026-12-15"))  # removed three weeks after its file ends
    symbols = sorted([*members, "KVUE", "OUT", *BENCHMARKS])
    answers = {symbol: FULL for symbol in symbols if symbol not in ("KVUE", "OUT")}  # Yahoo misses two
    asked = []

    refresh_raw_files(sources, None, symbols, spans, "2026-12-31", fake_yahoo(sources, answers, asked))
    assert "Warning: members without new data, retried by the next build: KVUE, OUT" in capsys.readouterr().out
    assert (sources / UNREFRESHED).read_text() == "KVUE\nOUT\n"

    # A later run with the same end asks for both again, although KVUE has never had a file.
    refresh_raw_files(sources, None, symbols, spans, "2026-12-31", fake_yahoo(sources, answers, asked))
    assert asked[-1] == ["KVUE", "OUT"]
    assert "Warning: members without new data" in capsys.readouterr().out


def test_a_single_stale_bar_without_an_earlier_file_is_not_data(tmp_path, capsys):
    sources = tmp_path / "_sources"
    for symbol in ["AAPL", *BENCHMARKS]:
        write_raw(sources / "yahoo_raw", symbol, ["2020-11-03", "2026-11-23"])
    spans = [(symbol, "1999-01-01", OPEN_END) for symbol in ("AAPL", "KVUE", "NEW")]
    spans.append(("EA", "1999-01-01", "2026-08-04"))
    symbols = sorted(["AAPL", "EA", "KVUE", "NEW", *BENCHMARKS])
    answers = {symbol: FULL for symbol in ["AAPL", *BENCHMARKS]}
    # Yahoo's one stale quote for two delisted tickers, and the first bar of a company listed on the end day.
    answers.update(KVUE=["2026-10-30"], EA=["2026-08-04"], NEW=[END])

    refresh_raw_files(sources, None, symbols, spans, END, fake_yahoo(sources, answers, []))

    assert sorted(raw_last_days(sources / "yahoo_raw")) == sorted(["AAPL", "NEW", *BENCHMARKS])
    assert (sources / UNREFRESHED).read_text() == "KVUE\n"
    output = capsys.readouterr().out
    assert "Only a single stale bar for 2 symbols, taken as no data: EA, KVUE" in output
    assert "Warning: members without new data, retried by the next build: KVUE" in output


def test_symbols_a_refresh_left_behind_are_retried_by_later_runs(tmp_path):
    sources, reused = tmp_path / "target" / "_sources", tmp_path / "earlier" / "_sources"
    for folder in (sources, reused / "yahoo_raw"):
        folder.mkdir(parents=True)
    (sources / UNREFRESHED).write_text("AAPL\n")
    (reused / UNREFRESHED).write_text("MSFT\nGONE\n")

    assert symbols_to_retry(sources, reused / "yahoo_raw") == {"AAPL", "MSFT", "GONE"}
    assert symbols_to_retry(sources, None) == {"AAPL"}


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


def test_members_with_a_broken_price_series_are_cut_before_their_labels_read_it(tmp_path):
    days = [f"2010-02-{day:02d}" for day in range(1, 13)]
    (tmp_path / "calendars").mkdir()
    (tmp_path / "calendars" / "day.txt").write_text("\n".join(days) + "\n")
    (tmp_path / "instruments").mkdir()
    (tmp_path / "_sources").mkdir()
    spans = [
        ("AIG", days[0], days[-1]),
        ("CEG", days[0], days[4]),
        ("CEG", days[7], days[-1]),
        ("CHK", days[0], days[-1]),
        ("LATE", days[8], days[-1]),
        ("TIE", days[0], days[-1]),
    ]
    (tmp_path / "instruments" / "sp500.txt").write_text(instruments_text(spans))
    # Sound for six days, then two series interleaved under one code, as the old provider has for TIE.
    tie = [14, 14.2, 14.1, 14.3, 14.2, 14.4, 20000, 14, 20000, 14, 14, 14]
    write_bin(tmp_path / "features" / "tie" / "close.day.bin", 0, tie)
    # A runaway adjustment: fake large moves first, as CHK has on its ex-dividend days, then impossible ones.
    chk = [10, 14, 14.1, 20, 20.1, 28, 28.2, 60, 125, 260, 261, 262]
    write_bin(tmp_path / "features" / "chk" / "close.day.bin", 0, chk)
    # One code, two companies: a frozen series, then years later a sound one, as CEG has.
    ceg = [5, 5, 5, 8, 8, np.nan, np.nan, 8.1, 8.2, 8.15, 8.3, 8.25]
    write_bin(tmp_path / "features" / "ceg" / "close.day.bin", 0, ceg)
    # One real crash (AIG fell 61% on 2008-09-15) does not make a series broken.
    write_bin(tmp_path / "features" / "aig" / "close.day.bin", 0, [100, 39, 40, 41, 40, 41, 42, 41, 40, 41, 42, 43])
    # Broken only before it joined the index.
    write_bin(tmp_path / "features" / "late" / "close.day.bin", 0, [1, 5, 1, 5, 1, 5, 1, 5, 5.1, 5.2, 5.3, 5.4])
    verdict = {
        "CEG": [{"start": days[0], "end": days[4], "unchanged_days": 0.75, "large_moves": 1, "from": None}],
        "CHK": [
            {
                "start": days[0],
                "end": days[-1],
                "impossible_moves": 3,
                "large_moves_before": 3,
                "unchanged_days_before": 0.0,
                "from": None,
            }
        ],
        "TIE": [
            {
                "start": days[0],
                "end": days[-1],
                "impossible_moves": 4,
                "large_moves_before": 0,
                "unchanged_days_before": 0.0,
                "from": days[6],
            }
        ],
    }

    assert untrusted_members(tmp_path, spans) == verdict
    assert drop_untrusted_members(tmp_path, "sp500") == verdict

    assert read_instruments(tmp_path / "instruments" / "sp500.txt") == [
        ("AIG", days[0], days[-1]),
        ("CEG", days[7], days[-1]),  # the sound company keeps its membership
        ("LATE", days[8], days[-1]),
        # The sound days are kept, up to the last one whose label, the move two days ahead, is sound.
        ("TIE", days[0], days[3]),
    ]
    assert json.loads((tmp_path / "_sources" / "sp500_untrusted.json").read_text()) == verdict


def test_a_frozen_or_quantized_series_is_untrusted_but_a_merely_stale_one_is_not():
    days = [f"day{number:03d}" for number in range(90)]
    # BMC in the old provider: three price levels, so a fall of 80% and a rise of 360% among flat days.
    quantized = [1.0] * 30 + [0.2] * 30 + [0.92] * 30
    stale = [10 + (number % 5) * 0.1 if number % 4 else 10 for number in range(90)]  # often unchanged, no jump
    # EP in the old provider: a few ticks apart, with impossible doublings among them.
    ticks = [1, 1, 1.4, 1.4, 1.4, 1.2, 1.2, 1.2, 1, 1, 2, 2, 4, 4, 4.1, 4.2]

    def moves(closes):
        return [(days[i], closes[i] / closes[i - 1] - 1) for i in range(1, len(closes))]

    assert judge_series(moves(quantized)) == {"unchanged_days": 0.978, "large_moves": 2, "from": None}
    assert judge_series(moves(stale)) is None
    assert judge_series(moves([10.0, 20.0] + [20.0 + n * 0.1 for n in range(30)])) is None  # one doubling
    assert judge_series(moves([10.0, 20.0, 40.0] + [40.0 + n * 0.1 for n in range(30)])) == {
        "impossible_moves": 2,
        "large_moves_before": 0,
        "unchanged_days_before": 0.0,
        "from": days[1],
    }
    assert judge_series(moves(ticks)) == {
        "impossible_moves": 2,
        "large_moves_before": 1,
        "unchanged_days_before": 0.667,
        "from": None,  # quantized before it breaks: trusted on no day
    }
    # AYE: a sparse foreign series without an impossible rise, but with many large moves among few.
    few = [10, 10.1, 10.2, 13.5, 13.6, 13.7, 9.0, 9.1, 9.2, 12.5] + [12.6 + n * 0.1 for n in range(11)]
    assert judge_series(moves(few)) == {"large_moves": 3, "moves": 20, "from": None}
    # MEE in 2008: a sparse foreign series, with large moves among its few bars before it breaks.
    sparse = [100, 100.5, 140, 139, 100, 2, 98, 2, 99]
    assert judge_series(moves(sparse)) == {
        "impossible_moves": 4,
        "large_moves_before": 1,
        "unchanged_days_before": 0.0,
        "from": None,
    }


def test_untrusted_spans_are_cut_before_the_break_or_dropped():
    calendar = [day.date().isoformat() for day in pd.bdate_range("2012-01-03", periods=10)]
    spans = [
        ("BMC", "2012-01-03", OPEN_END),
        ("HNZ", "2011-01-03", "2011-06-30"),
        ("HNZ", "2012-01-03", OPEN_END),
        ("TIE", "2012-01-04", OPEN_END),
    ]
    untrusted = {
        "BMC": [{"start": "2012-01-03", "end": OPEN_END, "from": None}],
        "HNZ": [{"start": "2012-01-03", "end": OPEN_END, "from": "2012-01-10"}],
        "TIE": [{"start": "2012-01-04", "end": OPEN_END, "from": "2012-01-05"}],  # breaks on its second day
    }

    assert trim_untrusted(spans, untrusted, calendar) == [
        ("HNZ", "2011-01-03", "2011-06-30"),  # a span without a verdict is kept
        ("HNZ", "2012-01-03", "2012-01-05"),  # three trading days before the break
    ]


def test_the_two_days_after_a_span_are_judged_with_it(tmp_path):
    days = [f"2010-02-{day:02d}" for day in range(1, 8)]
    write_bin(tmp_path / "features" / "end" / "close.day.bin", 0, [10, 10.1, 10.2, 10.3, 30, 10, 30])

    # The moves on the two days after the last member day are the labels of its last two member days.
    assert impossible_moves(tmp_path, "END", [(days[0], days[3])], days) == ["+2010-02-05", "-2010-02-06"]


def test_bars_of_another_security_are_blanked_before_the_member_s_own_begin(tmp_path):
    days = ["2016-04-05", "2016-04-06", "2016-04-07", "2016-04-08", "2016-04-11"]
    (tmp_path / "calendars").mkdir()
    (tmp_path / "calendars" / "day.txt").write_text("\n".join(days) + "\n")
    ua = tmp_path / "features" / "ua"
    # Class A until 2016-04-06, a when-issued class C print on 2016-04-07, then the class C shares.
    write_bin(ua / "close.day.bin", 1, [84.0, 41.0, 41.5, 42.0])
    write_bin(ua / "change.day.bin", 1, [np.nan, -0.51, 0.012, 0.012])

    assert blank_foreign_bars(tmp_path) == ["UA"]  # FOREIGN_BARS: UA's bars until 2016-04-07

    close, change = (np.fromfile(ua / name, dtype="<f4") for name in ("close.day.bin", "change.day.bin"))
    np.testing.assert_array_equal(close, np.array([1, np.nan, np.nan, 41.5, 42.0], dtype="<f4"))
    # The first own bar's change is a move against the other security's close.
    np.testing.assert_array_equal(change, np.array([1, np.nan, np.nan, np.nan, 0.012], dtype="<f4"))
    assert blank_foreign_bars(tmp_path) == []


def test_when_issued_bars_between_two_histories_are_blanked(tmp_path):
    days = ["2014-03-26", "2014-03-27", "2014-03-28", "2014-04-02", "2014-04-03", "2014-04-04"]
    (tmp_path / "calendars").mkdir()
    (tmp_path / "calendars" / "day.txt").write_text("\n".join(days) + "\n")
    goog = tmp_path / "features" / "goog"
    # The class A history, when-issued class C prints from 2014-03-27 to 2014-04-02, then the class C shares.
    write_bin(goog / "volume.day.bin", 0, [2.6e8, 6.5e5, 2.1e6, 7.4e6, 2.5e8, 2.4e8])
    write_bin(goog / "change.day.bin", 0, [0.01, -0.012, 0.002, 0.001, 0.003, -0.02])

    assert blank_foreign_bars(tmp_path) == ["GOOG"]

    volume, change = (np.fromfile(goog / name, dtype="<f4") for name in ("volume.day.bin", "change.day.bin"))
    nan = np.nan
    np.testing.assert_array_equal(volume, np.array([0, 2.6e8, nan, nan, nan, 2.5e8, 2.4e8], dtype="<f4"))
    np.testing.assert_array_equal(change, np.array([0, 0.01, nan, nan, nan, nan, -0.02], dtype="<f4"))


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

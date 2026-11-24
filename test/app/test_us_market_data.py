import pytest

from rdagent.app.us_market_data import (
    BENCHMARKS,
    OPEN_END,
    instruments_text,
    merge_spans,
    normalize_ticker,
    read_instruments,
    sp500_spans,
    spans_from_daily_members,
    symbols_to_download,
)

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(("raw", "expected"), [("BRK.B", "BRK-B"), (" aapl ", "AAPL"), ("BF.B", "BF-B")])
def test_share_classes_use_the_yahoo_dash(raw, expected):
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

    assert symbols_to_download(spans, since="2020-11-10") == sorted({"AAPL", "TWTR", *BENCHMARKS})


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

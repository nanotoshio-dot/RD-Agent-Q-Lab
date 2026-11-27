"""The gate that decides whether a rebuilt US provider may replace the active one."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "rd_mt5" / "auditar_dados.py"
DAY = "2026-11-25"


def provider(root: Path, *, sp500: str = "499/499 of 503 in the source", **changes: object) -> Path:
    report = {
        "calendar": ["2000-01-03", DAY],
        "misaligned": [],
        "benchmark_missing_days_after_cutover": {"^GSPC": 0},
        "benchmark_last_day": {"^GSPC": DAY, "^NDX": DAY},
        "non_positive_closes_after_cutover": 0,
        "ohlc_violations_after_cutover": {"bars": 2},
        "foreign_closes_left": {},
        "sp500": {DAY: sp500},
        "nasdaq100": {DAY: "100/100 of 101 in the source"},
        **changes,
    }
    sources = root / "_sources"
    sources.mkdir(parents=True)
    (sources / "audit.json").write_text(json.dumps(report))
    (sources / "unrefreshed.txt").write_text("ABC\n")
    (sources / "build_id.txt").write_text(f"2026-11-26T00:00:00Z_end={DAY}\n")
    (root / "calendars").mkdir()
    (root / "calendars" / "day.txt").write_text(f"2000-01-03\n{DAY}\n")
    return root


def audit(*arguments: object) -> subprocess.CompletedProcess:
    command = [sys.executable, "-I", str(SCRIPT), *map(str, arguments)]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_a_good_build_passes_against_the_active_one(tmp_path):
    result = audit(provider(tmp_path / "new"), DAY, provider(tmp_path / "active"))
    assert result.returncode == 0, result.stderr


def test_a_build_that_stops_before_the_expected_day_is_rejected(tmp_path):
    result = audit(provider(tmp_path / "new"), "2026-11-27")
    assert result.returncode != 0
    assert "esperado 2026-11-27" in result.stderr


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"ohlc_violations_after_cutover": {"bars": 50}}, "barras inconsistentes"),
        ({"misaligned": ["ABC"]}, "desalinhadas"),
        ({"non_positive_closes_after_cutover": 1}, "nao positivos"),
        ({"sp500": "450/450 of 503 in the source"}, "sp500: so"),
        ({"sp500": "490/499 of 503 in the source"}, "membros sem preco"),
    ],
)
def test_broken_builds_are_rejected(tmp_path, changes, reason):
    result = audit(provider(tmp_path / "new", **changes), DAY)
    assert result.returncode != 0
    assert reason in result.stderr


def test_losing_members_against_the_active_data_is_rejected(tmp_path):
    new = provider(tmp_path / "new", sp500="484/484 of 503 in the source")
    assert audit(new, DAY).returncode == 0  # above the fixed minimum
    result = audit(new, DAY, provider(tmp_path / "active"))
    assert result.returncode != 0
    assert "caiu de 499" in result.stderr


def test_an_incomplete_provider_is_rejected(tmp_path):
    result = audit(tmp_path, DAY)
    assert result.returncode != 0
    assert "ausente ou incompleta" in result.stderr

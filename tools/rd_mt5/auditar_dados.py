"""Decide whether a new US provider may replace the active one. Exit 0 only when every check passes.

Usage: python -I auditar_dados.py PROVIDER EXPECTED_LAST_DAY [ACTIVE_PROVIDER]
With the active provider's audit, a build that lost more than a few priced members is rejected too.
"""

import json
import re
import sys
from pathlib import Path

MINIMUM = {"sp500": 480, "nasdaq100": 95}
TOLERANCE = 3  # members that may lose their data between two updates


def coverage(audit: dict, market: str, day: str) -> tuple[int, int, int] | None:
    """Members with a price, members listed and members in the source on ``day``."""
    match = re.match(r"(\d+)/(\d+)(?: of (\d+) in the source)?", str(audit.get(market, {}).get(day, "")))
    if not match:
        return None
    return int(match[1]), int(match[2]), int(match[3] or match[2])


provider, expected = Path(sys.argv[1]), sys.argv[2]
problems = []
try:
    audit = json.loads((provider / "_sources/audit.json").read_text())
    calendar_end = (provider / "calendars/day.txt").read_text().split()[-1]
    unrefreshed = (provider / "_sources/unrefreshed.txt").read_text().split()
    build_id = (provider / "_sources/build_id.txt").read_text().split()[-1]
except (OSError, ValueError, IndexError) as error:
    sys.exit(f"auditoria ausente ou incompleta: {error}")
reference = {}
if len(sys.argv) > 3:
    try:
        reference = json.loads((Path(sys.argv[3]) / "_sources/audit.json").read_text())
    except (OSError, ValueError):
        print("sem auditoria dos dados em uso para comparar")
if calendar_end != expected or not build_id.endswith(f"end={expected}"):
    problems.append(f"calendario termina em {calendar_end}, esperado {expected}")
if audit.get("misaligned") != []:
    problems.append("ha series desalinhadas")
if any(audit.get("benchmark_missing_days_after_cutover", {"?": 1}).values()):
    problems.append("faltam dias nos indices")
if set(audit.get("benchmark_last_day", {}).values()) != {expected}:
    problems.append(f"indices nao chegam a {expected}")
if audit.get("non_positive_closes_after_cutover") != 0:
    problems.append("ha precos nao positivos")
if audit.get("ohlc_violations_after_cutover", {}).get("bars", 99) > 10:
    problems.append("muitas barras inconsistentes")
if any(audit.get("foreign_closes_left", {"?": 1}).values()):
    problems.append("ha precos de outro papel")
for market, minimum in MINIMUM.items():
    new = coverage(audit, market, expected)
    if new is None or new[0] < minimum:
        problems.append(f"{market}: so {audit.get(market, {}).get(expected) or 'nada'} em {expected}")
        continue
    priced, listed, source = new
    if listed - priced > TOLERANCE:
        problems.append(f"{market}: {listed - priced} membros sem preco em {expected}")
    old = coverage(reference, market, (reference.get("calendar") or [""])[-1])
    if old and (old[0] - priced > TOLERANCE or (source - priced) - (old[2] - old[0]) > TOLERANCE):
        problems.append(f"{market}: caiu de {old[0]} de {old[2]} para {priced} de {source} membros com dados")
if len(unrefreshed) > 8:
    problems.append(f"{len(unrefreshed)} papeis sem dados novos")
if problems:
    sys.exit("auditoria reprovada: " + "; ".join(problems))
print(f"auditoria aprovada: dados ate {expected}, {len(unrefreshed)} papel(is) sem dados novos")

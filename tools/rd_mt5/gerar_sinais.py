"""Train the RD market baseline on the active data and write the latest day's ranking for the MT5 experts.

Usage: python -I gerar_sinais.py [MARKET] [TOPK] [OUTPUT_CSV]
The model is the Alpha20 LightGBM baseline of the Qlib factor template, trained on 2010-2017 and stopped on
2018-2019. The ranking is not a recommendation and has not shown any advantage in the tests run so far.
Data folder: RD_DADOS (default ~/.qlib/qlib_data). Work folder: RD_ESTADO (default ~/.rdagent/rd_mt5).
"""

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT = Path(os.environ.get("RD_PROJETO") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT))
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

state = Path(os.environ.get("RD_ESTADO", Path.home() / ".rdagent/rd_mt5"))
data = Path(os.environ.get("RD_DADOS", Path.home() / ".qlib/qlib_data"))
market = sys.argv[1] if len(sys.argv) > 1 else "us-sp500"
topk = int(sys.argv[2]) if len(sys.argv) > 2 else 20
output = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else state / "rd_sinais.csv"
provider = data / ("cn_data" if market == "china" else "us_data_2026")
work = state / "work"
work.mkdir(parents=True, exist_ok=True)
os.environ["WORKSPACE_PATH"] = str(work / "rd_ws")
os.chdir(work)  # Qlib's recorder writes ./mlruns

import jinja2  # noqa: E402
import jinja2.meta  # noqa: E402
import pandas as pd  # noqa: E402
import qlib  # noqa: E402
import yaml  # noqa: E402
from qlib.utils import init_instance_by_config  # noqa: E402
from qlib.workflow import R  # noqa: E402

from rdagent.app.market_launcher import (  # noqa: E402
    build_child_environment,
    build_market_context,
)
from rdagent.utils.qlib import ALPHA20  # noqa: E402

last_day = date.fromisoformat((provider / "calendars/day.txt").read_text().split()[-1])
periods = [
    date(2010, 1, 1),
    date(2017, 12, 31),
    date(2018, 1, 1),
    date(2019, 12, 31),
    last_day - timedelta(days=15),
    last_day,
]
env = build_child_environment(build_market_context(market, provider, *periods), base_env={})
names = ("train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end")
env.update({name: value.isoformat() for name, value in zip(names, periods)})
env.update(feature_names=str(list(ALPHA20)), feature_expressions=str(list(ALPHA20.values())))
text = (PROJECT / "rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml").read_text()
template = jinja2.Template(text)
variables = jinja2.meta.find_undeclared_variables(template.environment.parse(text))
config = yaml.safe_load(template.render({name: env[name] for name in variables if name in env}))

qlib.init(provider_uri=str(provider), region=env["RD_MARKET_REGION"], kernels=1)
model = init_instance_by_config(config["task"]["model"])
dataset = init_instance_by_config(config["task"]["dataset"])
with R.start(experiment_name="sinais_rd"):
    model.fit(dataset)
    prediction = model.predict(dataset, segment="test")

day = prediction.index.get_level_values("datetime").max()
if day.date() != last_day:
    sys.exit(f"Sem previsao para o ultimo dia dos dados ({last_day}); a mais nova e de {day.date()}")
scores = prediction.xs(day, level="datetime").dropna()
scores = scores.sort_index().sort_values(ascending=False, kind="stable")
top = scores.head(topk)
# The data keeps some companies under their old ticker; the broker lists the current one.
plan = provider / "_sources/plan.json"
current = {code: ticker for ticker, code in json.loads(plan.read_text())["aliases"].items()} if plan.exists() else {}
rows = pd.DataFrame(
    {
        "data_sinal": day.date().isoformat(),
        "mercado": market,
        "posicao": range(1, len(top) + 1),
        "simbolo": [current.get(code, code) for code in top.index],
        "nota": [f"{value:.6f}" for value in top],
        "empatados": [int((scores == value).sum()) for value in top],
        "total": len(top),  # the expert rejects a file with fewer rows
    }
)
temporary = output.with_name(output.name + ".tmp")
rows.to_csv(temporary, index=False, encoding="ascii")
temporary.replace(output)
print(f"{len(rows)} sinais de {day.date()} em {output}")
print(f"modelo: {model.model.best_iteration} arvore(s), {scores.nunique()} notas distintas para {len(scores)} acoes")

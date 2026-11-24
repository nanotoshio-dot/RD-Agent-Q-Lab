from datetime import date
from pathlib import Path

import jinja2
import jinja2.meta
import pytest
import yaml

from rdagent.app.market_launcher import build_child_environment, build_market_context

pytestmark = pytest.mark.offline

EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "rdagent" / "scenarios" / "qlib" / "experiment"
TEMPLATES = sorted(EXPERIMENT_DIR.glob("*_template/conf_*.yaml"))


def render(path: Path, env: dict[str, str]) -> str:
    # Same rule as qlib.cli.run.render_template: only variables present in the environment reach Jinja.
    text = path.read_text()
    template = jinja2.Template(text)
    variables = jinja2.meta.find_undeclared_variables(template.environment.parse(text))
    return template.render({name: env[name] for name in variables if name in env})


def market_env(market_id: str, provider_uri: Path) -> dict[str, str]:
    context = build_market_context(
        market_id=market_id,
        provider_uri=provider_uri,
        train_start=date(2010, 1, 1),
        train_end=date(2015, 12, 31),
        valid_start=date(2016, 1, 1),
        valid_end=date(2017, 12, 31),
        test_start=date(2018, 1, 1),
        test_end=date(2020, 6, 30),
        root=Path("/nonexistent/markets"),
    )
    return build_child_environment(context, base_env={})


def trading_rules(rendered: str) -> dict:
    port = yaml.safe_load(rendered)["port_analysis_config"]
    exchange = port["backtest"]["exchange_kwargs"]
    strategy = port["strategy"]["kwargs"]
    return {
        "limit_threshold": exchange["limit_threshold"],
        "open_cost": exchange["open_cost"],
        "close_cost": exchange["close_cost"],
        "min_cost": exchange["min_cost"],
        "deal_price": exchange["deal_price"],
        "topk": strategy["topk"],
        "n_drop": strategy["n_drop"],
    }


def test_all_five_templates_are_covered():
    assert len(TEMPLATES) == 5


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.name)
def test_plain_runs_keep_the_china_rules(path):
    assert trading_rules(render(path, {})) == {
        "limit_threshold": 0.095,
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
        "deal_price": "close",
        "topk": 50,
        "n_drop": 5,
    }


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.name)
def test_china_profile_renders_byte_for_byte_like_a_plain_run(path):
    env = market_env("china", Path("~/.qlib/qlib_data/cn_data"))
    # The launcher passes the expanded provider path; everything else must match the template defaults.
    plain = render(path, {"RD_MARKET_PROVIDER_URI": env["RD_MARKET_PROVIDER_URI"]})

    assert render(path, env) == plain


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.name)
@pytest.mark.parametrize(("market_id", "topk", "n_drop"), [("us-sp500", 50, 5), ("us-nasdaq100", 20, 2)])
def test_us_markets_render_their_own_data_and_rules(path, market_id, topk, n_drop):
    rendered = render(path, market_env(market_id, Path("/data/us_data")))
    qlib_init = yaml.safe_load(rendered)["qlib_init"]

    assert qlib_init == {"provider_uri": "/data/us_data", "region": "us"}
    assert trading_rules(rendered) == {
        "limit_threshold": None,
        "open_cost": 0.001,
        "close_cost": 0.001,
        "min_cost": 1,
        "deal_price": "close",
        "topk": topk,
        "n_drop": n_drop,
    }

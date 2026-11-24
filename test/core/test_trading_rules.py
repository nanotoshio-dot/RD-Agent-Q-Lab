from dataclasses import replace

import pytest

from rdagent.core.market import TradingRules
from rdagent.core.market.trading_rules import TRADING_RULE_ENV_KEYS

pytestmark = pytest.mark.offline

CHINA_RULES = TradingRules(limit_threshold=0.095, open_cost=0.0005, close_cost=0.0015, min_cost=5, topk=50, n_drop=5)


def test_china_rules_render_as_the_template_literals():
    assert CHINA_RULES.to_env() == {
        "RD_MARKET_LIMIT_THRESHOLD": "0.095",
        "RD_MARKET_OPEN_COST": "0.0005",
        "RD_MARKET_CLOSE_COST": "0.0015",
        "RD_MARKET_MIN_COST": "5",
        "RD_MARKET_TOPK": "50",
        "RD_MARKET_N_DROP": "5",
    }


def test_no_limit_renders_as_null_and_numbers_stay_plain_decimals():
    rules = TradingRules(limit_threshold=None, open_cost=0.00005, close_cost=0.001, min_cost=1.0, topk=20, n_drop=2)

    env = rules.to_env()

    assert env["RD_MARKET_LIMIT_THRESHOLD"] == "null"
    assert env["RD_MARKET_OPEN_COST"] == "0.00005"
    assert env["RD_MARKET_CLOSE_COST"] == "0.001"
    assert env["RD_MARKET_MIN_COST"] == "1"
    assert set(env) == set(TRADING_RULE_ENV_KEYS)
    assert all(env.values())


@pytest.mark.parametrize(
    "overrides",
    [
        {"limit_threshold": 0.0},
        {"limit_threshold": 1.0},
        {"limit_threshold": 1},
        {"open_cost": -0.0001},
        {"close_cost": 1},
        {"open_cost": float("nan")},
        {"min_cost": -1},
        {"min_cost": True},
        {"topk": 0},
        {"topk": 50.0},
        {"n_drop": 0},
        {"n_drop": 51},
    ],
)
def test_invalid_rules_are_rejected(overrides):
    with pytest.raises(ValueError):
        replace(CHINA_RULES, **overrides)

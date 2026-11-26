from pathlib import Path

import pytest

from rdagent.core.market.runtime import current_market_id, ensure_docker_supports_market

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("env", [{}, {"OTHER_SETTING": "x"}])
def test_plain_runs_have_no_market(env):
    assert current_market_id(env) is None


def test_market_id_comes_from_the_launcher_variables():
    assert current_market_id({"RD_MARKET_ID": "us-sp500", "RD_MARKET_REGION": "us"}) == "us-sp500"


@pytest.mark.parametrize(
    "env",
    [{"RD_MARKET_PROVIDER_URI": "/data/us"}, {"RD_MARKET_ID": " ", "RD_MARKET_REGION": "us"}],
)
def test_market_variables_without_a_market_id_are_rejected(env):
    with pytest.raises(ValueError, match="without RD_MARKET_ID"):
        current_market_id(env)


@pytest.mark.parametrize(
    ("market_id", "provider_uri"),
    [
        (None, None),
        ("china", None),
        ("china", "~/.qlib/qlib_data/cn_data"),
        ("china", str(Path.home() / ".qlib" / "qlib_data" / "cn_data")),
    ],
)
def test_docker_keeps_plain_runs_and_default_china(market_id, provider_uri):
    ensure_docker_supports_market(market_id, provider_uri)


@pytest.mark.parametrize(
    ("market_id", "provider_uri"),
    [
        ("us-sp500", "~/.qlib/qlib_data/us_data"),
        ("us-nasdaq100", None),
        ("china", "/data/another_cn_data"),
    ],
)
def test_docker_refuses_runs_it_would_turn_into_china(market_id, provider_uri):
    with pytest.raises(ValueError, match="local or conda"):
        ensure_docker_supports_market(market_id, provider_uri)

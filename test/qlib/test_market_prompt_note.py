from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

import rdagent.utils.agent.tpl as tpl
from rdagent.app.market_launcher import build_child_environment, build_market_context
from rdagent.scenarios.qlib.experiment.utils import get_market_prompt_note
from rdagent.scenarios.qlib.proposal.quant_proposal import build_model_action_rag

pytestmark = pytest.mark.offline

# The model-action guidance exactly as it was before markets existed; China must keep it.
ORIGINAL_MODEL_RAG = (
    "1. In Quantitative Finance, market data could be time-series, and GRU model/LSTM model are suitable for them. "
    "Do not generate GNN model as for now.\n2. The training data consists of approximately 478,000 samples for the "
    "training set and about 128,000 samples for the validation set. Please design the hyperparameters accordingly and "
    "control the model size. This has a significant impact on the training results. If you believe that the previous "
    "model itself is good but the training hyperparameters or model hyperparameters are not optimal, you can return "
    "the same model and adjust these parameters instead.\n"
)


@pytest.fixture(autouse=True)
def _no_template_logging(monkeypatch):
    monkeypatch.setattr(tpl.logger, "log_object", lambda *args, **kwargs: None)


def market_env(market_id: str, provider_uri: str) -> dict[str, str]:
    context = build_market_context(
        market_id=market_id,
        provider_uri=Path(provider_uri),
        train_start=date(2010, 1, 1),
        train_end=date(2015, 12, 31),
        valid_start=date(2016, 1, 1),
        valid_end=date(2017, 12, 31),
        test_start=date(2018, 1, 1),
        test_end=date(2020, 6, 30),
        root=Path("/nonexistent/markets"),
    )
    return build_child_environment(context, base_env={})


@pytest.mark.parametrize("env", [{}, market_env("china", "/data/cn_data")])
def test_china_and_plain_runs_get_no_note(env):
    assert get_market_prompt_note(env) is None


@pytest.mark.parametrize(
    ("market_id", "note"),
    [
        (
            "us-sp500",
            "Market: US equities, sp500 universe, benchmark ^GSPC. Instrument codes are US tickers, not SH/SZ codes.",
        ),
        (
            "us-nasdaq100",
            "Market: US equities, nasdaq100 universe, benchmark ^NDX. "
            "Instrument codes are US tickers, not SH/SZ codes.",
        ),
    ],
)
def test_us_markets_get_a_one_line_note(market_id, note):
    assert get_market_prompt_note(market_env(market_id, "/data/us_data")) == note


def test_another_china_universe_gets_no_note():
    env = {**market_env("china", "/data/cn_data"), "RD_MARKET_ID": "china-csi500"}

    assert get_market_prompt_note(env) is None


def test_market_variables_without_a_market_id_are_rejected():
    with pytest.raises(ValueError, match="without RD_MARKET_ID"):
        get_market_prompt_note({"RD_MARKET_REGION": "us"})


SETTING = "scenarios.qlib.experiment.prompts:qlib_factor_experiment_setting"
DATES = {
    "train_start": "2010-01-01",
    "train_end": "2015-12-31",
    "valid_start": "2016-01-01",
    "valid_end": "2017-12-31",
    "test_start": "2018-01-01",
    "test_end": "2020-06-30",
}


@pytest.mark.parametrize("context", [{}, {"dataset": None}])
def test_china_keeps_the_csi300_experiment_label(context):
    setting = tpl.T(SETTING).r(**DATES, **context)

    assert setting.splitlines()[-1].startswith("| CSI300  | LGBModel |")


def test_us_runs_get_their_own_experiment_label():
    setting = tpl.T(SETTING).r(**DATES, dataset="SP500")

    assert setting.splitlines()[-1].startswith("| SP500  | LGBModel |")
    assert "CSI300" not in setting


def test_china_keeps_the_original_model_guidance():
    assert build_model_action_rag(SimpleNamespace()) == ORIGINAL_MODEL_RAG
    assert build_model_action_rag(SimpleNamespace(market_prompt_note=None)) == ORIGINAL_MODEL_RAG


def test_other_markets_get_no_china_sample_counts():
    rag = build_model_action_rag(SimpleNamespace(market_prompt_note="Market: US equities"))

    assert "478,000" not in rag
    assert "Please design the hyperparameters for this dataset and control the model size." in rag
    assert rag.startswith(ORIGINAL_MODEL_RAG[: ORIGINAL_MODEL_RAG.index("The training data")])

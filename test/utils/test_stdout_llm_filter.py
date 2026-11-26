import pytest

import rdagent.oai.llm_utils as llm_utils
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.utils import filter_redundant_text

pytestmark = pytest.mark.offline


class NoLLM:
    def __init__(self, *args, **kwargs):
        raise AssertionError("the LLM must not be called")


def test_switched_off_llm_filter_keeps_the_fixed_filters(monkeypatch):
    monkeypatch.setattr(RD_AGENT_SETTINGS, "stdout_llm_filter", False)
    monkeypatch.setattr(llm_utils, "APIBackend", NoLLM)
    stdout = "\x1b[32mtraining\x1b[0m\n" + "same warning\n" * 50 + "\n\n" + "IC 0.0144\n"

    filtered = filter_redundant_text(stdout)

    assert "\x1b" not in filtered
    assert "same warning" not in filtered
    assert filtered.splitlines()[0] == "training"
    assert filtered.splitlines()[-1] == "IC 0.0144"


def test_the_llm_filter_stays_on_by_default():
    assert RD_AGENT_SETTINGS.model_fields["stdout_llm_filter"].default is True

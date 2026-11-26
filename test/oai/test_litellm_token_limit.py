import pytest

from rdagent.oai.backend import litellm as backend

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("model_info", "expected"),
    [
        ({"max_input_tokens": 128000, "max_output_tokens": 16384}, 111616),
        # Ollama reports its window as both limits; that must not leave zero input room.
        ({"max_input_tokens": 32768, "max_output_tokens": 32768}, 100000),
        ({"max_input_tokens": 4096, "max_output_tokens": 8192}, 100000),
        ({"max_input_tokens": None, "max_output_tokens": 4096}, 100000),
        (None, 100000),
    ],
)
def test_input_token_limit_never_leaves_zero_room(model_info, expected):
    assert backend.input_token_limit(model_info, fallback=100000) == expected


def test_explicit_chat_token_limit_wins_over_model_info(monkeypatch):
    def unexpected_lookup(model):
        pytest.fail("an explicit limit must not depend on the model info")

    monkeypatch.setattr(backend, "get_model_info", unexpected_lookup)
    monkeypatch.setattr(backend.LITELLM_SETTINGS, "chat_token_limit", 6144)
    instance = backend.LiteLLMAPIBackend.__new__(backend.LiteLLMAPIBackend)

    assert instance.chat_token_limit == 6144

import litellm.types.utils as litellm_utils
from litellm.types.llms.openai import ChatCompletionReasoningSummaryTextBlock

from rdagent.oai.backend.litellm import LiteLLMAPIBackend

# Corrige a referência Pydantic ausente no LiteLLM 1.97.0.
litellm_utils.Message.model_rebuild(
    force=True,
    _types_namespace={
        "ChatCompletionReasoningSummaryTextBlock": ChatCompletionReasoningSummaryTextBlock,
    },
)


litellm_utils.Delta.model_rebuild(
    force=True,
    _types_namespace={
        "ChatCompletionReasoningSummaryTextBlock": ChatCompletionReasoningSummaryTextBlock,
    },
)


class OllamaAPIBackend(LiteLLMAPIBackend):
    """Backend local Ollama baseado na implementação original do RD-Agent."""

    pass

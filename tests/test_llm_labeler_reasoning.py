"""Tests for LLMLabeler's OpenRouter reasoning-disable behavior.

Reasoning-capable models routed through OpenRouter can spend their whole
max_tokens budget on hidden chain-of-thought before ever emitting content,
surfacing as an empty completion with finish_reason="length". LLMLabeler
sends OpenRouter's unified `reasoning: {enabled: false}` field to avoid
this — mirrored here against a fake OpenAI-SDK-shaped client so no real
API call is made.
"""

from graphweave.labeling.llm_labeler import LLMLabeler


class _FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = type("Msg", (), {"content": content})()
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, content, finish_reason="stop"):
        self.choices = [_FakeChoice(content, finish_reason)]


class _FakeCompletions:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse('{"answers": ["B"]}')


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeOpenAIClient:
    def __init__(self):
        self.chat = _FakeChat()


def _labeler_with_fake_client(provider: str) -> tuple[LLMLabeler, _FakeOpenAIClient]:
    labeler = LLMLabeler(provider=provider, api_key="fake-key", model="fake/model")
    fake_client = _FakeOpenAIClient()
    labeler._client = fake_client  # skip _init_client(); no real SDK client needed
    return labeler, fake_client


class TestReasoningDisable:
    def test_openrouter_sends_reasoning_disabled(self):
        labeler, fake_client = _labeler_with_fake_client("openrouter")
        labeler.call_raw("system", "user")
        assert len(fake_client.chat.completions.calls) == 1
        assert fake_client.chat.completions.calls[0]["extra_body"] == {
            "reasoning": {"enabled": False}
        }

    def test_openai_provider_does_not_send_openrouter_field(self):
        labeler, fake_client = _labeler_with_fake_client("openai")
        labeler.call_raw("system", "user")
        assert len(fake_client.chat.completions.calls) == 1
        assert "extra_body" not in fake_client.chat.completions.calls[0]

    def test_openrouter_structured_call_also_disables_reasoning(self):
        labeler, fake_client = _labeler_with_fake_client("openrouter")
        labeler.call_structured(
            "system", "user", schema={"type": "object"}, max_tokens=64
        )
        assert len(fake_client.chat.completions.calls) == 1
        assert fake_client.chat.completions.calls[0]["extra_body"] == {
            "reasoning": {"enabled": False}
        }

"""Tests for castwright.multiturn module."""

from __future__ import annotations

import json

import pytest

from castwright.multiturn import (
    Conversation,
    ConversationTurn,
    extend_conversation,
    format_openai,
    format_sharegpt,
    generate_conversation,
)
from castwright.providers import LLMProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeProvider(LLMProvider):
    """Provider that returns a canned JSON array of turns."""

    def __init__(self, turns: list[dict[str, str]]) -> None:
        self._turns = turns

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.9,
        max_tokens: int = 4096,
    ) -> tuple[str, int, int]:
        return json.dumps(self._turns), 10, 20


def _make_conversation(*pairs: tuple[str, str]) -> Conversation:
    """Helper: create a Conversation from (role, content) pairs."""
    return Conversation(
        turns=[ConversationTurn(role=r, content=c) for r, c in pairs],
    )


# ---------------------------------------------------------------------------
# ConversationTurn & Conversation dataclass tests
# ---------------------------------------------------------------------------

class TestConversationTurn:
    def test_fields(self) -> None:
        turn = ConversationTurn(role="user", content="hello")
        assert turn.role == "user"
        assert turn.content == "hello"

    def test_equality(self) -> None:
        a = ConversationTurn(role="user", content="hi")
        b = ConversationTurn(role="user", content="hi")
        assert a == b


class TestConversation:
    def test_empty(self) -> None:
        conv = Conversation()
        assert conv.num_turns == 0
        assert conv.turns == []
        assert conv.metadata == {}

    def test_add_turn(self) -> None:
        conv = Conversation()
        conv.add_turn("user", "hi")
        conv.add_turn("assistant", "hello!")
        assert conv.num_turns == 2
        assert conv.turns[0].role == "user"
        assert conv.turns[1].content == "hello!"

    def test_metadata(self) -> None:
        conv = Conversation(metadata={"source": "test"})
        assert conv.metadata["source"] == "test"


# ---------------------------------------------------------------------------
# generate_conversation
# ---------------------------------------------------------------------------

class TestGenerateConversation:
    def test_basic(self) -> None:
        provider = _FakeProvider([
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "What can I do with it?"},
            {"role": "assistant", "content": "Build web apps, scripts, ML models, etc."},
        ])
        conv = generate_conversation("Python programming", num_turns=4, provider=provider)
        assert isinstance(conv, Conversation)
        assert conv.num_turns == 4
        assert conv.turns[0].role == "user"
        assert conv.turns[1].role == "assistant"

    def test_metadata_contains_seed_topic(self) -> None:
        provider = _FakeProvider([{"role": "user", "content": "hi"}])
        conv = generate_conversation("testing", num_turns=1, provider=provider)
        assert conv.metadata["seed_topic"] == "testing"
        assert conv.metadata["requested_turns"] == 1

    def test_empty_topic_raises(self) -> None:
        with pytest.raises(ValueError, match="seed_topic"):
            generate_conversation("", num_turns=2)

    def test_zero_turns_raises(self) -> None:
        with pytest.raises(ValueError, match="num_turns"):
            generate_conversation("topic", num_turns=0)

    def test_default_provider_mock(self) -> None:
        # When provider=None, a MockProvider is used which returns valid JSON
        conv = generate_conversation("anything", num_turns=2)
        assert isinstance(conv, Conversation)


# ---------------------------------------------------------------------------
# extend_conversation
# ---------------------------------------------------------------------------

class TestExtendConversation:
    def test_extends_turns(self) -> None:
        original = _make_conversation(
            ("user", "hello"),
            ("assistant", "hi there"),
        )
        provider = _FakeProvider([
            {"role": "user", "content": "how are you?"},
            {"role": "assistant", "content": "I'm fine, thanks!"},
        ])
        extended = extend_conversation(original, num_turns=2, provider=provider)
        assert extended.num_turns == 4
        # Original turns preserved
        assert extended.turns[0].content == "hello"
        assert extended.turns[1].content == "hi there"
        # New turns appended
        assert extended.turns[2].content == "how are you?"
        assert extended.turns[3].content == "I'm fine, thanks!"

    def test_original_not_mutated(self) -> None:
        original = _make_conversation(("user", "hello"))
        provider = _FakeProvider([{"role": "assistant", "content": "hi"}])
        extended = extend_conversation(original, num_turns=1, provider=provider)
        assert original.num_turns == 1
        assert extended.num_turns == 2

    def test_metadata_preserved(self) -> None:
        original = Conversation(
            turns=[ConversationTurn("user", "hello")],
            metadata={"source": "test"},
        )
        provider = _FakeProvider([{"role": "assistant", "content": "hi"}])
        extended = extend_conversation(original, num_turns=1, provider=provider)
        assert extended.metadata["source"] == "test"
        assert extended.metadata["extended_by"] == 1

    def test_zero_turns_raises(self) -> None:
        with pytest.raises(ValueError, match="num_turns"):
            extend_conversation(Conversation(), num_turns=0)


# ---------------------------------------------------------------------------
# format_sharegpt
# ---------------------------------------------------------------------------

class TestFormatShareGPT:
    def test_basic(self) -> None:
        conv = _make_conversation(
            ("user", "hello"),
            ("assistant", "hi!"),
        )
        result = format_sharegpt(conv)
        assert "conversations" in result
        assert len(result["conversations"]) == 2
        assert result["conversations"][0] == {"from": "human", "value": "hello"}
        assert result["conversations"][1] == {"from": "gpt", "value": "hi!"}

    def test_system_role_mapped(self) -> None:
        conv = Conversation(turns=[
            ConversationTurn("system", "You are a helpful assistant."),
            ConversationTurn("user", "hi"),
        ])
        result = format_sharegpt(conv)
        assert result["conversations"][0]["from"] == "system"

    def test_empty_conversation(self) -> None:
        result = format_sharegpt(Conversation())
        assert result == {"conversations": []}


# ---------------------------------------------------------------------------
# format_openai
# ---------------------------------------------------------------------------

class TestFormatOpenAI:
    def test_basic(self) -> None:
        conv = _make_conversation(
            ("system", "You are helpful."),
            ("user", "hello"),
            ("assistant", "hi!"),
        )
        result = format_openai(conv)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1] == {"role": "user", "content": "hello"}
        assert result[2] == {"role": "assistant", "content": "hi!"}

    def test_empty_conversation(self) -> None:
        result = format_openai(Conversation())
        assert result == []

    def test_roundtrip_consistency(self) -> None:
        """OpenAI format should preserve all turn data."""
        conv = _make_conversation(
            ("user", "question"),
            ("assistant", "answer"),
        )
        msgs = format_openai(conv)
        rebuilt = Conversation(
            turns=[ConversationTurn(m["role"], m["content"]) for m in msgs]
        )
        assert rebuilt.turns == conv.turns

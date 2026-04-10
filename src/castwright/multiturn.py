"""Multi-turn conversation generation and formatting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from castwright.providers import LLMProvider


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str
    content: str


@dataclass
class Conversation:
    """A multi-turn conversation."""

    turns: List[ConversationTurn] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append(ConversationTurn(role=role, content=content))


def _build_conversation_prompt(
    seed_topic: str,
    num_turns: int,
) -> str:
    """Build a prompt asking the LLM to generate a conversation."""
    return (
        f"Generate a realistic multi-turn conversation between a user and an "
        f"assistant about the following topic:\n\n"
        f"Topic: {seed_topic}\n\n"
        f"Requirements:\n"
        f"- Exactly {num_turns} turns total (alternating user/assistant, starting with user)\n"
        f"- Each turn should be substantive and build on the previous context\n"
        f"- The conversation should feel natural and progress logically\n\n"
        f"Return a JSON array of objects with \"role\" (either \"user\" or \"assistant\") "
        f"and \"content\" fields. Example:\n"
        f'[{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]\n\n'
        f"Return ONLY the JSON array, no other text."
    )


def _build_extend_prompt(
    conversation: Conversation,
    num_turns: int,
) -> str:
    """Build a prompt to extend an existing conversation."""
    history = "\n".join(
        f"{t.role}: {t.content}" for t in conversation.turns
    )
    return (
        f"Here is an existing conversation:\n\n"
        f"{history}\n\n"
        f"Continue this conversation for exactly {num_turns} more turns, "
        f"alternating between user and assistant. Pick up naturally from "
        f"where the conversation left off.\n\n"
        f"Return a JSON array of objects with \"role\" (either \"user\" or \"assistant\") "
        f"and \"content\" fields for ONLY the new turns.\n\n"
        f"Return ONLY the JSON array, no other text."
    )


def _parse_turns(raw_items: List[Dict[str, Any]]) -> List[ConversationTurn]:
    """Parse raw JSON items into ConversationTurn objects."""
    turns: list[ConversationTurn] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", ""))
        content = str(item.get("content", ""))
        if role and content:
            turns.append(ConversationTurn(role=role, content=content))
    return turns


def generate_conversation(
    seed_topic: str,
    num_turns: int = 4,
    provider: Optional[LLMProvider] = None,
) -> Conversation:
    """Generate a multi-turn conversation from a topic.

    Parameters
    ----------
    seed_topic:
        The topic to generate a conversation about.
    num_turns:
        Number of turns to generate (alternating user/assistant).
    provider:
        An LLM provider instance. If *None*, a :class:`MockProvider` is used.

    Returns
    -------
    Conversation
        The generated conversation.
    """
    if not seed_topic:
        raise ValueError("seed_topic must not be empty")
    if num_turns < 1:
        raise ValueError("num_turns must be >= 1")

    if provider is None:
        from castwright.providers import MockProvider
        provider = MockProvider()

    prompt = _build_conversation_prompt(seed_topic, num_turns)
    system = (
        "You are an expert data curator generating realistic multi-turn "
        "conversations for instruction tuning."
    )

    text, _in_tok, _out_tok = provider.generate(
        prompt=prompt,
        system=system,
        temperature=0.9,
    )

    raw = provider.parse_json_array(text)
    turns = _parse_turns(raw)

    return Conversation(
        turns=turns,
        metadata={"seed_topic": seed_topic, "requested_turns": num_turns},
    )


def extend_conversation(
    conversation: Conversation,
    num_turns: int = 2,
    provider: Optional[LLMProvider] = None,
) -> Conversation:
    """Add more turns to an existing conversation.

    Parameters
    ----------
    conversation:
        The conversation to extend.
    num_turns:
        Number of new turns to add.
    provider:
        An LLM provider instance.  If *None*, a :class:`MockProvider` is used.

    Returns
    -------
    Conversation
        A new :class:`Conversation` containing the original turns plus new ones.
    """
    if num_turns < 1:
        raise ValueError("num_turns must be >= 1")

    if provider is None:
        from castwright.providers import MockProvider
        provider = MockProvider()

    prompt = _build_extend_prompt(conversation, num_turns)
    system = (
        "You are an expert data curator extending multi-turn "
        "conversations for instruction tuning."
    )

    text, _in_tok, _out_tok = provider.generate(
        prompt=prompt,
        system=system,
        temperature=0.9,
    )

    raw = provider.parse_json_array(text)
    new_turns = _parse_turns(raw)

    return Conversation(
        turns=list(conversation.turns) + new_turns,
        metadata={**conversation.metadata, "extended_by": num_turns},
    )


def format_sharegpt(conversation: Conversation) -> Dict[str, Any]:
    """Convert a conversation to ShareGPT format.

    Returns a dict with a ``"conversations"`` key mapping roles to
    ``"human"``/``"gpt"``/``"system"``.
    """
    role_map = {"user": "human", "assistant": "gpt", "system": "system"}
    convs: list[dict[str, str]] = []
    for turn in conversation.turns:
        mapped = role_map.get(turn.role, turn.role)
        convs.append({"from": mapped, "value": turn.content})
    return {"conversations": convs}


def format_openai(conversation: Conversation) -> List[Dict[str, str]]:
    """Convert a conversation to OpenAI chat format.

    Returns a list of ``{"role": ..., "content": ...}`` dicts.
    """
    return [
        {"role": turn.role, "content": turn.content}
        for turn in conversation.turns
    ]

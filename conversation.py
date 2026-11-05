"""
Multi-turn conversation manager.

Computation in a conversation is not independent: each reply is shaped by what
came before. Multi-turn context lets the C-agent reason about its own past
output. The history is structured, not just dumped into the prompt: each
assistant turn carries its reasoning trace and its confabulation/values
assessment, so the C-agent can see how reliable its earlier answers were.

Thread safety: every ConversationManager has its own re-entrant lock. The
orchestrator holds it for the whole of one request, so two requests on the same
conversation run one after the other instead of interleaving their turns.
Turns are added as a user+assistant pair only after a request succeeds, so an
API failure never leaves an orphaned user turn behind.
"""

import threading
from dataclasses import dataclass, field
from typing import Optional

if __package__:                 # installed (pip install noetic-field): the package's own modules
    from . import llm_json as lj
else:                           # run from a source folder: modules side by side
    import llm_json as lj


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str
    reasoning_steps: list = field(default_factory=list)
    confabulation_risk: str = "unknown"
    values_status: str = "pass"
    confidence: float = 0.5
    domain: str = ""


def _meta(turn: Turn) -> str:
    try:
        conf = f"{float(turn.confidence):.0%}"
    except (TypeError, ValueError):
        conf = "n/a"
    return f"[confabulation_risk={turn.confabulation_risk}, values={turn.values_status}, confidence={conf}]"


def _assessed(turn: Turn) -> bool:
    """True if this assistant turn carries an assessment worth showing: a reasoning trace, a rating, or a values result."""
    return bool(turn.reasoning_steps) or turn.confabulation_risk != "unknown" or turn.values_status != "pass"


class ConversationManager:
    def __init__(self, max_turns: int = 20):
        self.turns: list[Turn] = []
        self.max_turns = max_turns
        self.lock = threading.RLock()

    def _trim(self) -> None:
        """Keep the most recent turns, and never start the history on an assistant turn."""
        limit = self.max_turns * 2
        if len(self.turns) > limit:
            self.turns = self.turns[-limit:]
        while self.turns and self.turns[0].role != "user":
            self.turns.pop(0)

    def add_user(self, content: str) -> None:
        with self.lock:
            self.turns.append(Turn(role="user", content=content))
            self._trim()

    def add_assistant(self, content: str, reasoning_steps: list = None,
                      confabulation_risk: str = "unknown",
                      values_status: str = "pass",
                      confidence: float = 0.5,
                      domain: str = "") -> None:
        with self.lock:
            self.turns.append(Turn(
                role="assistant",
                content=content,
                reasoning_steps=reasoning_steps or [],
                confabulation_risk=confabulation_risk,
                values_status=values_status,
                confidence=confidence,
                domain=domain,
            ))
            self._trim()

    def add_exchange(self, user_content: str, assistant_content: str, **meta) -> None:
        """Add a user turn and its assistant turn together, under one lock."""
        with self.lock:
            self.turns.append(Turn(role="user", content=user_content))
            self.turns.append(Turn(
                role="assistant",
                content=assistant_content,
                reasoning_steps=meta.get("reasoning_steps") or [],
                confabulation_risk=meta.get("confabulation_risk", "unknown"),
                values_status=meta.get("values_status", "pass"),
                confidence=meta.get("confidence", 0.5),
                domain=meta.get("domain", ""),
            ))
            self._trim()

    def history_digest(self, n: int = 10, chars: int = 300) -> str:
        """The last n turns, each cut to `chars` characters, as plain text for the C-agent's context."""
        with self.lock:
            turns = list(self.turns[-n:])
        lines = []
        for t in turns:
            if t.role == "user":
                lines.append(f"User: {lj.clip(t.content, chars)}")
            else:
                suffix = f" {_meta(t)}" if _assessed(t) else ""
                lines.append(f"Assistant: {lj.clip(t.content, chars)}{suffix}")
        return "\n".join(lines)

    def to_messages(self, include_metadata: bool = True) -> list[dict]:
        """Convert to Anthropic messages format."""
        with self.lock:
            turns = list(self.turns)
        messages = []
        for turn in turns:
            if turn.role == "user":
                messages.append({"role": "user", "content": turn.content})
            elif include_metadata and _assessed(turn):
                messages.append({"role": "assistant", "content": f"{turn.content}\n{_meta(turn)}"})
            else:
                messages.append({"role": "assistant", "content": turn.content})
        return messages

    def get_last_user_message(self) -> Optional[str]:
        with self.lock:
            for turn in reversed(self.turns):
                if turn.role == "user":
                    return turn.content
        return None

    def summary(self) -> str:
        """Brief summary for logging."""
        last = self.get_last_user_message()
        return f"{len(self.turns)} turns, last user: {last[:50] if last else 'none'}"

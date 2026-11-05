"""conversation: bounded history that always starts on a user turn, atomic pairs, thread safety."""

import threading

from conversation import ConversationManager


def test_history_is_bounded_starts_on_a_user_turn_and_keeps_pairs():
    c = ConversationManager(max_turns=5)
    for i in range(12):
        c.add_exchange(f"q{i}", f"a{i}")
    assert len(c.turns) <= 10 and c.turns[0].role == "user"
    assert [t.role for t in c.turns] == ["user", "assistant"] * (len(c.turns) // 2)
    assert c.turns[-1].content == "a11"


def test_trim_never_leaves_a_leading_assistant_turn():
    c = ConversationManager(max_turns=1)               # at most 2 turns
    c.add_user("q1")
    c.add_assistant("a1")
    c.add_user("q2")                                   # 3 turns: the oldest goes, then the now-leading assistant turn
    assert [t.content for t in c.turns] == ["q2"]


def test_history_digest_clips_and_annotates_only_assistant_turns_with_a_trace():
    c = ConversationManager()
    c.add_exchange("u" * 500, "a" * 500, reasoning_steps=["s"], confabulation_risk="high",
                   values_status="flag", confidence=0.4)
    c.add_exchange("q", "plain")
    lines = c.history_digest(n=10, chars=50).split("\n")
    assert lines[0].startswith("User: ") and len(lines[0]) == len("User: ") + 50
    assert "confabulation_risk=high" in lines[1] and "values=flag" in lines[1] and "confidence=40%" in lines[1]
    assert lines[3] == "Assistant: plain"
    assert len(c.history_digest(n=2).split("\n")) == 2


def test_messages_and_digest_survive_a_missing_or_odd_confidence():
    c = ConversationManager()
    c.add_user("q")
    c.add_assistant("x", reasoning_steps=["s"], confidence=None)
    c.add_user("q2")
    c.add_assistant("y", reasoning_steps=["s"], confidence="n/a")
    msgs = c.to_messages()
    assert msgs[1]["content"].endswith("confidence=n/a]") and msgs[3]["content"].endswith("confidence=n/a]")
    assert "confidence=n/a" in c.history_digest()
    assert c.to_messages(include_metadata=False)[1]["content"] == "x"


def test_last_user_message_and_summary():
    c = ConversationManager()
    assert c.get_last_user_message() is None and c.summary() == "0 turns, last user: none"
    c.add_exchange("hello there", "hi")
    assert c.get_last_user_message() == "hello there" and c.summary().startswith("2 turns")


def test_concurrent_exchanges_never_interleave():
    c = ConversationManager(max_turns=20)

    def work(k):
        for j in range(40):
            c.add_exchange(f"q{k}-{j}", f"a{k}-{j}")

    threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    roles = [t.role for t in c.turns]
    assert roles == ["user", "assistant"] * (len(roles) // 2) and len(roles) == 40
    for i in range(0, len(c.turns), 2):
        assert c.turns[i].content[1:] == c.turns[i + 1].content[1:]          # each answer sits next to its own question

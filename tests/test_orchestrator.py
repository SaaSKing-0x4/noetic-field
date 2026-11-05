"""orchestrator: the pipeline end to end against a scripted client, including every failure path."""

import copy
from concurrent.futures import ThreadPoolExecutor

import pytest

import agents
import orchestrator as om
import state as st
from conversation import ConversationManager
from fakes import C, N, V, S, conn_error, result, echo_c


def api_down():
    return conn_error()


def set_total(orch, n):
    orch.state["self_model"]["total_interactions"] = n


# ── happy path ────────────────────────────────────────────────────────────────

def test_happy_path_commits_state_and_conversation(orch, fake):
    res = orch.run("What is six times seven?")
    assert res["response"] == "The answer is 42."                      # clean answers carry no annotation
    assert res["confidence"] == pytest.approx(0.8) and res["domain"] == "math"
    assert res["confabulation_risk"] == "low" and res["values_status"] == "pass"
    assert res["review_triggered"] is False and res["review_error"] is None
    assert res["parse_issues"] is False and res["truncated"] is False and res["confidence_defaulted"] is False
    assert orch.state["self_model"]["total_interactions"] == 1
    entry = orch.state["interaction_log"][0]
    assert entry["domain"] == "math" and entry["confidence"] == pytest.approx(0.8) and entry["parse_issues"] is False
    assert orch.state["domain_confidence"]["math"]["n"] == 1
    assert [t.role for t in orch.conversation.turns] == ["user", "assistant"]
    assert orch.conversation.turns[1].content == "The answer is 42."
    assert st.load()["self_model"]["total_interactions"] == 1          # saved to disk
    assert len(fake.calls_for("C")) == len(fake.calls_for("N")) == len(fake.calls_for("V")) == 1


def test_verbose_adds_the_detail(orch):
    res = orch.run("hi", verbose=True)
    assert res["reasoning_steps"] and res["uncertainty_flags"] == [] and res["v_notes"] == []
    assert res["parse_status"] == {"C": "ok", "N": "ok", "V": "ok"} and res["n_assessment"]
    assert "reasoning_steps" not in orch.run("hi")


def test_each_orchestrator_counts_a_session(fake):
    assert om.NoeticOrchestrator().state["self_model"]["session_count"] == 1
    assert om.NoeticOrchestrator().state["self_model"]["session_count"] == 2


def test_history_reaches_c_without_the_current_message(orch, fake):
    orch.run("first question")
    orch.run("second question")
    first, second = (c["messages"][0]["content"] for c in fake.calls_for("C"))
    assert "<recent_conversation>" not in first
    assert "<recent_conversation>" in second and "first question" in second
    assert "second question" not in second.split("</recent_conversation>")[0]
    assert second.count("second question") == 1


def test_current_values_reach_v(orch, fake):
    assert orch.add_value("Never flatter")
    orch.run("hi")
    assert "- Never flatter" in fake.calls_for("V")[0]["system"]


# ── failures: C fails the request, N/V never lose the answer ──────────────────

def test_c_failure_changes_nothing(orch, fake):
    before_state = copy.deepcopy(orch.state)
    before_file = st.STATE_FILE.read_bytes()
    fake.script("C", api_down())
    with pytest.raises(agents.AgentError, match="C-agent"):
        orch.run("hello")
    assert orch.state == before_state and st.STATE_FILE.read_bytes() == before_file
    assert orch.conversation.turns == []
    assert fake.calls_for("N") == [] and fake.calls_for("V") == []


def test_n_failure_is_reported_not_hidden(orch, fake):
    fake.script("N", api_down())
    res = orch.run("hi", verbose=True)
    assert res["confabulation_risk"] == "unknown" and "confabulation risk undetermined" in res["response"]
    assert res["parse_issues"] is True and res["values_status"] == "pass"
    assert res["parse_status"]["N"] == "error" and "check failed" in res["n_assessment"]
    assert orch.state["interaction_log"][0]["parse_issues"] is True


def test_v_failure_is_unchecked_never_pass(orch, fake):
    fake.script("V", api_down())
    res = orch.run("hi")
    assert res["values_status"] == "unchecked" and "values check unavailable" in res["response"]
    assert orch.state["interaction_log"][0]["values_status"] == "unchecked"


def test_v_unreadable_reply_is_unchecked(orch, fake):
    fake.script("V", "looks fine to me")
    res = orch.run("hi")
    assert res["values_status"] == "unchecked" and res["parse_issues"] is True


def test_a_bug_inside_a_check_does_not_lose_the_answer(orch, fake, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(agents, "values_agent", boom)
    res = orch.run("hi")
    assert res["response"].startswith("The answer is 42.") and res["values_status"] == "unchecked"


def test_prose_reply_skips_n_but_still_checks_values(orch, fake):
    fake.script("C", "Just prose, no JSON.")
    res = orch.run("hi")
    assert res["response"].startswith("Just prose, no JSON.")
    assert "no reasoning steps were produced, so the explanation could not be checked" in res["response"]
    assert fake.calls_for("N") == [] and len(fake.calls_for("V")) == 1
    assert res["domain"] == "unknown" and res["confidence_defaulted"] is True and res["parse_issues"] is True
    assert orch.state["domain_confidence"] == {}                       # an unmeasured answer is never averaged in
    assert orch.state["interaction_log"][0]["confidence"] is None


def test_unreadable_structured_reply_is_never_shown_to_v(orch, fake):
    fake.script("C", "{{{ garbage")
    res = orch.run("hi")
    assert fake.calls_for("V") == [] and fake.calls_for("N") == []
    assert res["values_status"] == "unchecked" and "values check unavailable" in res["response"]
    assert res["response"].startswith("[The computation agent's structured output could not be parsed.")


def test_cut_off_answer_is_noted(orch, fake):
    fake.script("C", result('{"reasoning_steps": ["a"], "response": "Partial', "max_tokens"))
    res = orch.run("hi")
    assert res["truncated"] is True and "response may be cut off (length limit reached)" in res["response"]
    assert res["confidence_defaulted"] is True and orch.state["domain_confidence"] == {}


def test_empty_response_is_reported(orch, fake):
    fake.script("C", C(response=""))
    res = orch.run("hi")
    assert res["response"].startswith("(The computation agent returned an empty response.)")
    assert "empty response" in res["response"].split("---")[1]
    assert fake.calls_for("N") == [] and fake.calls_for("V") == []
    # found by the pipeline audit: a stated confidence about an empty answer must not be recorded as a measurement
    assert res["parse_issues"] is True and orch.state["domain_confidence"] == {}
    entry = orch.state["interaction_log"][0]
    assert entry["domain"] == "unknown" and entry["confidence"] is None and entry["parse_issues"] is True


@pytest.mark.parametrize("bad", ["", "   \n", None, 5, ["x"]])
def test_bad_messages_are_rejected_before_any_call(orch, fake, bad):
    with pytest.raises(ValueError):
        orch.run(bad)
    assert fake.calls == [] and orch.conversation.turns == []


def test_over_long_message_is_rejected(orch, fake):
    with pytest.raises(ValueError, match="too long"):
        orch.run("x" * (om.MAX_MESSAGE_CHARS + 1))
    assert fake.calls == []
    orch.run("x" * om.MAX_MESSAGE_CHARS)                               # exactly at the limit is fine


def test_failed_save_leaves_memory_and_conversation_untouched(orch, monkeypatch):
    real = st.save

    def boom(state, *a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(st, "save", boom)
    with pytest.raises(OSError):
        orch.run("hi")
    assert orch.state["self_model"]["total_interactions"] == 0 and orch.state["interaction_log"] == []
    assert orch.conversation.turns == []
    monkeypatch.setattr(st, "save", real)
    orch.run("hi")
    assert orch.state["self_model"]["total_interactions"] == 1 and len(orch.conversation.turns) == 2


# ── synthesis ─────────────────────────────────────────────────────────────────

def test_veto_prefixes_but_does_not_block(orch, fake):
    fake.script("V", V(status="veto", values_notes=["deceptive"]))
    res = orch.run("hi", verbose=True)
    assert res["response"] == "[VALUES CIRCUIT VETO]\n\nThe answer is 42."
    assert res["values_status"] == "veto" and res["v_notes"] == ["deceptive"]


def test_high_risk_note_and_a_bounded_number_of_divergence_insights(orch, fake):
    fake.script("N", N(confabulation_risk="high", explanation_matches_reasoning=False,
                       divergence_notes=["d1", "d2", "d3", "d4"]))
    res = orch.run("hi")
    assert res["response"] == ("The answer is 42.\n\n---\n"
                               "*[Noetic: high confabulation risk — reasoning trace and explanation may diverge]*")
    assert [i["insight"] for i in orch.state["insights"]] == ["Divergence: d1", "Divergence: d2"]
    assert orch.state["insights"][0]["source"] == "narration_agent"


def test_low_risk_divergence_notes_are_not_stored(orch, fake):
    fake.script("N", N(confabulation_risk="low", divergence_notes=["d1"]))
    orch.run("hi")
    assert orch.state["insights"] == []


def test_medium_risk_and_sycophancy_notes(orch, fake):
    fake.script("N", N(confabulation_risk="medium"))
    fake.script("V", V(status="flag", sycophancy_detected=True))
    res = orch.run("hi")
    assert res["response"].endswith("\n\n---\n*[Noetic: moderate confabulation risk; approval-seeking pattern detected]*")
    assert orch.state["interaction_log"][0]["sycophancy"] is True


def test_synthesize_markers_and_levels():
    base = {"response": "ans", "reasoning_steps": ["s"]}
    low, ok = {"confabulation_risk": "low"}, {"status": "pass"}
    s = om.synthesize(base, low, ok)
    assert s["text"] == "ans" and s["note_level"] == "none" and s["reasons"] == []
    s = om.synthesize(base, {"confabulation_risk": "high"}, {"status": "pass", "sycophancy_detected": True})
    assert s["note_level"] == "high" and s["text"].endswith("may diverge; approval-seeking pattern detected]*")
    s = om.synthesize(base, {"confabulation_risk": "unknown"}, ok)
    assert s["text"].endswith("*[Noetic: confabulation risk undetermined]*") and s["note_level"] == "moderate"
    s = om.synthesize(base, {"confabulation_risk": "unknown", "skipped": True}, ok)
    assert "no reasoning steps were produced" in s["text"]
    s = om.synthesize({"response": ""}, {"confabulation_risk": "unknown", "skipped": True}, {"status": "unchecked"})
    assert s["body"] == "(The computation agent returned an empty response.)" and "empty response" in s["note"]
    s = om.synthesize(base, low, {"status": "nonsense"})
    assert s["vstatus"] == "unchecked" and "values check unavailable" in s["note"]
    s = om.synthesize({**base, "truncated": True}, low, ok)
    assert "response may be cut off" in s["note"]
    s = om.synthesize(base, {"confabulation_risk": "bogus"}, ok)
    assert s["risk"] == "unknown"
    assert om.synthesize(base, low, {"status": "veto"})["text"].startswith("[VALUES CIRCUIT VETO]\n\n")


# ── the S-agent trigger ───────────────────────────────────────────────────────

def test_review_runs_exactly_once_per_ten_interactions(orch, fake):
    flags = [orch.run(f"q{i}")["review_triggered"] for i in range(25)]
    assert [i for i, f in enumerate(flags, 1) if f] == [10, 20]
    assert len(fake.calls_for("S")) == 2 and orch.state["self_model"]["last_review_at"] == 20
    assert [i["insight"] for i in orch.state["insights"]] == ["Math answers are consistent."]
    assert orch.state["self_model"]["last_review_health"].startswith("healthy")


def test_a_full_log_does_not_retrigger_reviews(orch, fake):
    set_total(orch, 135)
    orch.state["interaction_log"] = [st._clean_log_entry({}, st.now_iso()) for _ in range(100)]
    triggers = [orch.run(f"q{i}")["review_triggered"] for i in range(7)]       # totals 136..142
    assert triggers == [False, False, False, False, True, False, False]        # only 140
    assert len(fake.calls_for("S")) == 1


def test_review_failure_is_reported_alongside_the_committed_answer(orch, fake):
    set_total(orch, 9)
    fake.script("S", api_down())
    res = orch.run("hi")
    assert res["review_triggered"] is True and "S-agent" in res["review_error"]
    assert res["response"] == "The answer is 42."
    assert st.load()["self_model"]["total_interactions"] == 10
    assert orch.state["self_model"]["last_review_at"] == 10                    # claimed: no automatic retry
    assert orch.run("next")["review_triggered"] is False


def test_unparseable_review_stores_nothing_and_says_so(orch, fake):
    set_total(orch, 9)
    fake.script("S", "no json at all")
    res = orch.run("hi")
    assert res["review_triggered"] is True and "could not be parsed" in res["review_error"]
    assert orch.state["insights"] == [] and orch.state["self_model"]["last_review_health"] == ""


def test_truncated_review_stores_nothing_and_says_so(orch, fake):
    set_total(orch, 9)
    fake.script("S", result(S(new_insights=["should not be kept"]), stop="max_tokens"))
    res = orch.run("hi")
    assert "cut off" in res["review_error"] and orch.state["insights"] == []


def test_a_bug_inside_the_review_does_not_lose_the_answer(orch, fake, monkeypatch):
    import self_review

    def boom(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(self_review, "run_self_review", boom)
    set_total(orch, 9)
    res = orch.run("hi")
    assert res["response"] == "The answer is 42." and "RuntimeError" in res["review_error"]


def test_verbose_reports_the_review(orch):
    set_total(orch, 9)
    res = orch.run("hi", verbose=True)
    assert res["review_triggered"] and res["review_summary"].startswith("healthy") and res["review_applied"]["insights"] == 1


def test_manual_review(orch, fake):
    r = orch.run_manual_review()
    assert r["applied"]["skipped"] == "There was nothing to review yet." and fake.calls_for("S") == []
    orch.run("q")
    r = orch.run_manual_review()
    assert r["applied"]["insights"] == 1 and orch.state["self_model"]["last_review_health"].startswith("healthy")
    assert st.load()["insights"]                                                # saved
    fake.script("S", api_down())
    with pytest.raises(agents.AgentError):
        orch.run_manual_review()


# ── small edits ───────────────────────────────────────────────────────────────

def test_add_value_and_add_insight_report_whether_anything_changed(orch):
    assert orch.add_value("Be kind") is True
    assert orch.add_value("be kind") is False and orch.add_value("none") is False and orch.add_value("  ") is False
    assert "Be kind" in st.load()["values"]
    assert orch.add_insight("a note") is True
    assert orch.add_insight("a note") is False and orch.add_insight("  ") is False
    assert orch.state["insights"][-1]["source"] == "user"


def test_show_self_model_and_new_conversation(orch):
    orch.run("hi")
    assert "Noetic Self-Model" in orch.show_self_model()
    old = orch.conversation
    orch.new_conversation()
    assert orch.conversation is not old and orch.conversation.turns == []
    assert orch.state["self_model"]["total_interactions"] == 1


# ── concurrency ───────────────────────────────────────────────────────────────

def test_parallel_requests_count_every_interaction_and_review_once_per_ten(orch, fake):
    def work(i):
        conv = ConversationManager()
        return [orch.run(f"q{i}-{j}", conversation=conv)["review_triggered"] for j in range(4)]

    with ThreadPoolExecutor(6) as ex:
        flags = [f for fl in ex.map(work, range(6)) for f in fl]
    assert orch.state["self_model"]["total_interactions"] == 24 and len(orch.state["interaction_log"]) == 24
    assert sum(flags) == 2 and len(fake.calls_for("S")) == 2
    assert st.load()["self_model"]["total_interactions"] == 24
    assert orch.state["domain_confidence"]["math"]["n"] == 24


def test_requests_on_one_conversation_never_interleave(orch, fake):
    fake.defaults["C"] = echo_c
    conv = ConversationManager()
    messages = [f"m{i}-{j}" for i in range(6) for j in range(3)]
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda m: orch.run(m, conversation=conv), messages))
    turns = conv.turns
    assert len(turns) == 36
    for k in range(0, 36, 2):
        assert turns[k].role == "user" and turns[k + 1].role == "assistant"
        assert turns[k + 1].content == "echo:" + turns[k].content


def test_separate_conversations_do_not_share_history(orch, fake):
    a, b = ConversationManager(), ConversationManager()
    orch.run("only in a", conversation=a)
    orch.run("only in b", conversation=b)
    second_c_prompt = fake.calls_for("C")[1]["messages"][0]["content"]
    assert "only in a" not in second_c_prompt and orch.conversation.turns == []

"""api and cli: input limits, error mapping, bounded sessions, and that the CLI loop survives a failed call."""

import pytest

import api
import state as st
from fakes import conn_error, run_cli


def api_down():
    return conn_error()


# ── API ───────────────────────────────────────────────────────────────────────

def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "service": "noetic-field"}


def test_chat_happy_path(client):
    r = client.post("/chat", json={"message": "What is six times seven?"})
    assert r.status_code == 200
    body = r.json()
    assert body["response"] == "The answer is 42." and body["confidence"] == pytest.approx(0.8)
    assert body["domain"] == "math" and body["confabulation_risk"] == "low" and body["values_status"] == "pass"
    assert body["session_id"] == "default" and body["review_triggered"] is False and body["review_error"] is None
    assert body["confidence_defaulted"] is False and body["parse_issues"] is False and body["reasoning_steps"] is None


def test_chat_verbose(client):
    body = client.post("/chat", json={"message": "hi", "verbose": True}).json()
    assert body["reasoning_steps"] and body["parse_status"] == {"C": "ok", "N": "ok", "V": "ok"}


def test_chat_validation(client):
    assert client.post("/chat", json={"message": "   "}).status_code == 400
    assert client.post("/chat", json={"message": "x" * 9000}).status_code == 400
    assert client.post("/chat", json={"message": ""}).status_code == 400
    assert client.post("/chat", json={}).status_code == 422                  # a missing field is a malformed request
    assert client.post("/chat", json={"message": "hi", "session_id": "bad id!"}).status_code == 422
    assert client.post("/chat", json={"message": "hi", "session_id": "x" * 65}).status_code == 422
    assert client.post("/chat", json={"message": "hi", "session_id": "ok-id_1.2:3"}).status_code == 200


def test_a_failed_model_call_is_a_502_and_changes_nothing(client, fake):
    client.post("/chat", json={"message": "warm up"})                      # create the orchestrator
    before = api.get_orchestrator().state["self_model"]["total_interactions"]
    fake.script("C", api_down())
    r = client.post("/chat", json={"message": "hello", "session_id": "s1"})
    assert r.status_code == 502 and "C-agent" in r.json()["detail"]
    assert api.get_orchestrator().state["self_model"]["total_interactions"] == before
    assert api._sessions["s1"].turns == []


def test_sessions_are_bounded_and_least_recently_used_goes_first(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_SESSIONS", 3)
    for sid in ("s1", "s2", "s3"):
        client.post("/chat", json={"message": "hi", "session_id": sid})
    client.post("/chat", json={"message": "again", "session_id": "s1"})       # s1 becomes the most recent
    client.post("/chat", json={"message": "hi", "session_id": "s4"})
    assert list(api._sessions) == ["s3", "s1", "s4"]
    assert len(api._sessions["s1"].turns) == 4


def test_reset(client):
    client.post("/chat", json={"message": "hi", "session_id": "r1"})
    assert client.post("/reset", json={"session_id": "r1"}).json() == {"status": "reset", "session_id": "r1"}
    assert "r1" not in api._sessions
    assert client.post("/reset", json={"session_id": "never-existed"}).status_code == 200


def test_value_and_insight_endpoints(client):
    assert client.post("/value", json={"value": "Never flatter"}).json() == {"status": "added", "value": "Never flatter"}
    dup = client.post("/value", json={"value": "never flatter"})
    assert dup.status_code == 400 and "not added" in dup.json()["detail"]
    assert client.post("/value", json={"value": "  "}).status_code == 400
    assert client.post("/value", json={"value": "x" * 301}).status_code == 400
    assert client.post("/insight", json={"insight": "a note"}).json()["status"] == "added"
    assert client.post("/insight", json={"insight": "a note"}).json()["status"].startswith("ignored")
    assert client.post("/insight", json={"insight": " "}).status_code == 400
    assert client.post("/insight", json={"insight": "x" * 501}).status_code == 400
    assert "Never flatter" in client.get("/self").json()["self_model"]


def test_review_endpoint(client):
    empty = client.post("/review").json()
    assert empty["status"] == "review_not_applied" and empty["applied"]["skipped"]
    client.post("/chat", json={"message": "hi"})
    done = client.post("/review").json()
    assert done["status"] == "review_complete" and done["health_label"] == "healthy"
    assert done["applied"]["insights"] == 1 and done["new_insights"] == ["Math answers are consistent."]


def test_review_endpoint_maps_model_failures(client, fake):
    client.post("/chat", json={"message": "hi"})
    fake.script("S", api_down())
    r = client.post("/review")
    assert r.status_code == 502 and "S-agent" in r.json()["detail"]


def test_audit_endpoints(client):
    client.post("/chat", json={"message": "hi"})
    full = client.get("/audit/full")
    assert full.status_code == 200 and "Noetic Field Health Report" in full.json()["report"]
    assert "Unchecked" in client.get("/audit/values").json()["report"]
    assert client.get("/audit/zzz").status_code == 404


# ── audit module ──────────────────────────────────────────────────────────────

def test_audit_reports_read_one_snapshot_and_handle_empty_state():
    import audit
    s = st.load()
    assert audit.confabulation_report(s) == "No interaction history yet."
    assert audit.domain_report(s) == "No domain confidence data yet."
    st.log_interaction(s, "q", "r", "high", "unchecked", domain="py", confidence=0.4)
    st.update_domain_confidence(s, "py", 0.4)
    s["domain_confidence"]["py"]["flagged_at"] = "2026-01-01T00:00:00"
    assert "High risk:" in audit.confabulation_report(s) and "100%" in audit.confabulation_report(s)
    assert "Unchecked:    1 (100%)" in audit.values_report(s)
    assert "(flagged by self-review)" in audit.domain_report(s)
    full = audit.full_health_report(s)
    assert "Interactions:  1 in total" in full and "hypotheses" in full
    assert audit.full_health_report()                                         # also works from disk


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_conversation(monkeypatch, capsys):
    out = run_cli(monkeypatch, capsys, ["hello", "--history", "--self", "q"])
    assert "Noetic: The answer is 42." in out and "Conversation: 2 turns" in out and "Noetic Self-Model" in out


def test_cli_survives_a_failed_call(monkeypatch, capsys, fake):
    fake.script("C", api_down())
    out = run_cli(monkeypatch, capsys, ["boom", "again", "exit"])
    assert "The request failed and nothing was saved: C-agent" in out
    assert "Noetic: The answer is 42." in out


def test_cli_verbose_report(monkeypatch, capsys):
    out = run_cli(monkeypatch, capsys, ["hi", "exit"], argv=["--verbose"])
    assert "Confidence: 80% (self-reported)" in out and "Reasoning trace:" in out
    assert "Parse status: C=ok, N=ok, V=ok" in out


def test_cli_unmeasured_confidence_is_not_shown_as_a_number(monkeypatch, capsys, fake):
    fake.script("C", "plain prose reply")
    out = run_cli(monkeypatch, capsys, ["hi", "exit"], argv=["--verbose"])
    assert "Confidence: not reported" in out


def test_cli_review_and_new(monkeypatch, capsys):
    out = run_cli(monkeypatch, capsys, ["hi", "--review", "--new", "--history", "exit"])
    assert "Health: healthy" in out and "[Stored: 1 insight(s)" in out
    assert "New conversation started" in out and "Conversation: 0 turns" in out


def test_cli_review_failure_does_not_end_the_session(monkeypatch, capsys, fake):
    fake.script("S", api_down())
    out = run_cli(monkeypatch, capsys, ["hi", "--review", "hi again", "exit"])
    assert "[Self-review failed: S-agent" in out and out.count("Noetic: The answer is 42.") == 2


def test_cli_single_commands(monkeypatch, capsys):
    out = run_cli(monkeypatch, capsys, [], argv=["--value", "Honesty over approval: say what is true, not what is wanted"])
    assert "Value not added" in out
    out = run_cli(monkeypatch, capsys, [], argv=["--value", "A fresh value"])
    assert "Value added: A fresh value" in out
    out = run_cli(monkeypatch, capsys, [], argv=["--insight", "a note"])
    assert "Insight saved: a note" in out
    out = run_cli(monkeypatch, capsys, [], argv=["--insight", "a note"])
    assert "Insight not saved" in out
    assert "A fresh value" in run_cli(monkeypatch, capsys, [], argv=["--self"])

"""
Round three of independent review.

Three reviewers attacked the patched system and had to reproduce every finding. Each test here was first run against
the code as it stood when the reviewers saw it (and failed there, for the reason the finding gives), then against the
fix. Only behaviour is pinned; nothing here depends on how the fix is written.
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import agents
import api
import audit
import cli
import llm_json as lj
import self_review
import state as st
from fakes import C, N, S, V, result, run_cli
from orchestrator import NoeticOrchestrator

BAK = "noetic_state.json.bak*"
BS = chr(92)                          # a backslash, built here so that no escape sequence appears in this source
FFFD = chr(0xFFFD)
BOM = chr(0xFEFF)


def on_disk():
    return json.loads(st.STATE_FILE.read_text(encoding="utf-8"))


def backups():
    return sorted(st.STATE_FILE.parent.glob(BAK))


def quiet_client():
    """A test client that returns a 500 response instead of raising, so a server error is visible as a status."""
    return TestClient(api.app, raise_server_exceptions=False)


# ── state: a damaged or missing file next to a running process ────────────────

def test_a_second_process_recovering_a_damaged_file_does_not_wipe_the_first_ones_history():
    a = NoeticOrchestrator()
    a.run("one")
    a.run("two")
    a.add_value("PRECIOUS value")
    a.add_insight("PRECIOUS insight")
    st.STATE_FILE.write_text("{oops", encoding="utf-8")              # the file is damaged
    b = NoeticOrchestrator(count_session=False)                      # another process starts: it moves the file aside
    assert b.snapshot()["self_model"]["total_interactions"] == 0     # and starts blank
    snap = a.snapshot()                                              # the first process looks next
    assert snap["self_model"]["total_interactions"] == 2
    assert "PRECIOUS value" in snap["values"] and any("PRECIOUS" in i["insight"] for i in snap["insights"])
    assert on_disk()["self_model"]["total_interactions"] == 2        # the file holds that history again
    assert b.snapshot()["self_model"]["total_interactions"] == 2     # and the starter picks it up
    assert len(backups()) == 1


def test_a_blank_file_made_after_the_file_vanished_does_not_wipe_history_either():
    a = NoeticOrchestrator()
    a.run("one")
    st.STATE_FILE.unlink()
    NoeticOrchestrator(count_session=False)                          # the new process creates a blank file
    assert a.snapshot()["self_model"]["total_interactions"] == 1
    assert on_disk()["self_model"]["total_interactions"] == 1


def test_a_read_writes_our_state_back_when_the_file_was_deleted():
    a = NoeticOrchestrator()
    a.run("one")
    a.add_value("PRECIOUS")
    st.STATE_FILE.unlink()
    a.snapshot()                                                     # a read, not a write
    assert st.STATE_FILE.exists()
    assert on_disk()["self_model"]["total_interactions"] == 1 and "PRECIOUS" in on_disk()["values"]


def test_a_blank_process_does_not_create_a_file_just_by_looking():
    o = NoeticOrchestrator(count_session=False)
    st.STATE_FILE.unlink()
    o.snapshot()
    o.show_self_model()
    assert not st.STATE_FILE.exists()


def test_a_save_that_had_to_wait_does_not_overwrite_a_newer_commit(monkeypatch):
    a = NoeticOrchestrator()
    b = NoeticOrchestrator(count_session=False)
    real, seen = os.replace, {"hits": 0}

    def flaky(src, dst):
        if seen["hits"] == 0 and Path(dst) == st.STATE_FILE:
            seen["hits"] += 1
            assert b.add_value("val-b")                              # someone else commits while our rename waits
            raise PermissionError("held open by a reader")
        return real(src, dst)

    monkeypatch.setattr(st.os, "replace", flaky)
    assert a.add_value("val-a")
    values = on_disk()["values"]
    assert "val-a" in values and "val-b" in values
    assert "val-a" in a.snapshot()["values"] and "val-b" in a.snapshot()["values"]


def test_set_aside_waits_out_a_transient_lock(monkeypatch):
    st.STATE_FILE.write_text("{oops", encoding="utf-8")
    real, seen = os.replace, {"n": 0}

    def locked(src, dst):
        if Path(src) == st.STATE_FILE and seen["n"] < 3:
            seen["n"] += 1
            raise PermissionError("in use")
        return real(src, dst)

    monkeypatch.setattr(st.os, "replace", locked)
    monkeypatch.setattr(st, "SAVE_RETRY_DELAY", 0)
    s = st.load()                                                    # waits instead of giving up at the first refusal
    assert s["self_model"]["total_interactions"] == 0
    assert [p.read_text(encoding="utf-8") for p in backups()] == ["{oops"]


def test_an_unreadable_file_that_cannot_be_moved_is_never_overwritten(monkeypatch):
    a = NoeticOrchestrator()
    a.run("one")
    st.STATE_FILE.write_text("{oops", encoding="utf-8")
    real = os.replace

    def locked(src, dst):
        if Path(src) == st.STATE_FILE:
            raise PermissionError("in use")
        return real(src, dst)

    monkeypatch.setattr(st.os, "replace", locked)
    monkeypatch.setattr(st, "SAVE_ATTEMPTS", 3)
    monkeypatch.setattr(st, "SAVE_RETRY_DELAY", 0)
    with pytest.raises(OSError):
        a.add_value("x")
    assert st.STATE_FILE.read_text(encoding="utf-8") == "{oops"      # still there, still the only copy
    assert backups() == []


def test_backups_of_damaged_files_are_bounded_and_the_newest_are_kept():
    for i in range(30):
        st.STATE_FILE.write_text(f"{{damaged {i}", encoding="utf-8")
        st.load()
    kept = [p.read_text(encoding="utf-8") for p in backups()]
    assert len(kept) == 20
    assert "{damaged 29" in kept and "{damaged 0" not in kept


def test_ctrl_c_during_a_save_leaves_no_temp_file(monkeypatch):
    def interrupt(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(st.json, "dump", interrupt)
    with pytest.raises(KeyboardInterrupt):
        st.save(st.fresh_state())
    assert list(st.STATE_FILE.parent.glob("*.tmp")) == []


def test_stale_temp_files_are_swept_at_start_but_a_fresh_one_is_left_alone():
    old = st.STATE_FILE.parent / "noetic_state-abcd1234.json.tmp"
    new = st.STATE_FILE.parent / "noetic_state-efgh5678.json.tmp"
    foreign = st.STATE_FILE.parent / "noetic_state-my-own-notes.json.tmp"      # not a name save() would give a file
    for p in (old, new, foreign):
        p.write_text("{}", encoding="utf-8")
    an_hour_and_more = time.time() - 7200
    for p in (old, foreign):
        os.utime(p, (an_hour_and_more, an_hour_and_more))
    st.load()
    assert not old.exists() and new.exists() and foreign.exists()


def test_pruning_never_touches_a_backup_somebody_made_by_hand():
    mine = st.STATE_FILE.parent / "noetic_state.json.bak-before-my-edit"
    mine.write_text("precious", encoding="utf-8")
    for i in range(25):
        st.STATE_FILE.write_text(f"{{damaged {i}", encoding="utf-8")
        st.load()
    assert mine.read_text(encoding="utf-8") == "precious"
    assert len([p for p in backups() if p != mine]) == 20


def test_a_deeply_nested_unknown_key_cannot_wedge_the_state():
    deep = "[" * 700 + "]" * 700
    st.STATE_FILE.write_text('{"extra": "x", "zz": ' + deep + ', "values": ["keepme"]}', encoding="utf-8")
    o = NoeticOrchestrator(count_session=False)                      # used to raise RecursionError here
    assert o.add_value("fine")
    NoeticOrchestrator()                                             # and so did every later start
    s = on_disk()
    assert s["values"] == ["keepme", "fine"] and s.get("extra") == "x" and "zz" not in s
    assert backups() == []


def test_an_enormous_integer_in_the_file_is_cleaned_not_a_reason_to_set_the_file_aside():
    st.STATE_FILE.write_text('{"self_model": {"total_interactions": ' + "9" * 5000 + '}, "values": ["keepme"]}',
                             encoding="utf-8")
    s = st.load()
    assert s["values"] == ["keepme"] and s["self_model"]["total_interactions"] == 0
    assert backups() == []


# ── reading what the model says: cut-off and repaired replies are not verdicts ─

CUT_V = '{"status": "pass", "values_notes": ["The answer states the dates with complete certainty'
CUT_N = '{"explanation_matches_reasoning": true, "confabulation_risk": "low", "divergence_notes": ["a step is missing"'


def test_a_values_reply_cut_off_by_the_length_limit_is_not_a_pass(orch, fake):
    fake.script("V", result(CUT_V, stop="max_tokens"))
    r = orch.run("q", verbose=True)
    assert r["values_status"] == "unchecked" and "values check unavailable" in r["response"]
    assert r["parse_issues"] is True
    assert on_disk()["interaction_log"][-1]["values_status"] == "unchecked"


def test_a_values_reply_that_had_to_be_repaired_is_not_a_pass_either(orch, fake):
    fake.script("V", CUT_V)                                          # not flagged as cut off, but unclosed all the same
    r = orch.run("q", verbose=True)
    assert r["values_status"] == "unchecked" and r["parse_status"]["V"] == "repaired"


def test_a_narration_reply_cut_off_by_the_length_limit_is_not_low_risk(orch, fake):
    fake.script("N", result(CUT_N, stop="max_tokens"))
    r = orch.run("q")
    assert r["confabulation_risk"] == "unknown" and "undetermined" in r["response"]
    assert r["parse_issues"] is True
    assert on_disk()["insights"] == []                               # nothing from a cut-off reply becomes an insight


def test_a_narration_reply_that_had_to_be_repaired_is_not_low_risk_either(orch, fake):
    fake.script("N", CUT_N)
    assert orch.run("q")["confabulation_risk"] == "unknown"


def test_a_computation_reply_that_had_to_be_repaired_says_so_under_the_answer(orch, fake):
    fake.script("C", '{"reasoning_steps": ["a", "b"], "response": "The capital of France is Paris", '
                     '"confidence": 0.9, "domain": "geography"')
    r = orch.run("capital of France?")
    assert "The capital of France is Paris" in r["response"] and "repaired" in r["response"]
    assert r["parse_issues"] is True


def test_a_values_reply_with_an_unrecognized_status_is_unchecked_not_a_hidden_flag(orch, fake):
    fake.script("V", V(status="maybe"))
    r = orch.run("q")
    assert r["values_status"] == "unchecked" and "values check unavailable" in r["response"]
    assert r["parse_issues"] is True


def test_a_check_that_was_skipped_on_purpose_is_not_a_parse_issue(orch, fake):
    fake.script("C", C(reasoning_steps=[]))                          # a prose answer: there is nothing for N to compare
    r = orch.run("q", verbose=True)
    assert r["parse_status"]["N"] == "skipped" and r["parse_issues"] is False


def test_a_values_reply_with_no_status_at_all_is_unchecked(orch, fake):
    fake.script("V", json.dumps({"values_notes": ["looks fine"], "sycophancy_detected": False}))
    assert orch.run("q")["values_status"] == "unchecked"


@pytest.mark.parametrize("recommendation", [
    "regenerate", "Regenerate.", "regenerate!", ["regenerate"], '"regenerate"', "Regenerate it",
    "suggest modification", "Suggest modifications", "suggest modification (tone)", ["Suggest modification."],
])
def test_a_pass_that_recommends_a_change_is_raised_to_a_flag_however_it_is_spelled(recommendation):
    out = agents.normalize_v({"status": "pass", "recommendation": recommendation})
    assert out["status"] == "flag" and "recommended" in out["values_notes"][0]


@pytest.mark.parametrize("value", [True, "true", "True.", "yes", "Yes, present", "detected", "partially", ["yes"], 1])
def test_a_pass_that_reports_hidden_uncertainty_is_raised_to_a_flag_however_it_is_spelled(value):
    out = agents.normalize_v({"status": "pass", "hidden_uncertainty_detected": value})
    assert out["status"] == "flag" and out["hidden_uncertainty_detected"] is True


@pytest.mark.parametrize("value", [False, "false", "no", "None", "not detected", "No hidden uncertainty", 0, "", None])
def test_a_pass_that_reports_no_hidden_uncertainty_stays_a_pass(value):
    out = agents.normalize_v({"status": "pass", "hidden_uncertainty_detected": value, "recommendation": "pass as-is"})
    assert out["status"] == "pass" and out["hidden_uncertainty_detected"] is False


@pytest.mark.parametrize("value", [False, "false", "False.", "no", "No.", "does not follow", ["false"]])
def test_low_risk_with_a_stated_mismatch_is_raised_to_medium_however_the_mismatch_is_spelled(value):
    out = agents.normalize_n({"explanation_matches_reasoning": value, "confabulation_risk": "low"})
    assert out["confabulation_risk"] == "medium" and out["risk_adjusted"]


@pytest.mark.parametrize("value", [True, "true", "yes", "maybe", None, "unclear"])
def test_low_risk_stays_low_when_no_mismatch_is_stated(value):
    assert agents.normalize_n({"explanation_matches_reasoning": value, "confabulation_risk": "low"})["confabulation_risk"] == "low"


def test_a_cut_off_escaped_surrogate_pair_leaves_a_clean_prefix_at_every_cut():
    head = '{"reasoning_steps": [], "response": "ok '
    pair = BS + "ud83d" + BS + "ude00"
    for cut in range(len(pair)):                                     # every cut short of the whole pair
        got = lj.parse_with(head + pair[:cut], agents.EXPECT_C, agents.normalize_c, agents.fallback_c)
        assert got["response"] == "ok", (cut, got["response"])
    assert lj.parse_with(head + pair + '"}', agents.EXPECT_C, agents.normalize_c, agents.fallback_c)["response"] == "ok " + chr(0x1F600)


# ── the texts the system tells the model must be true ─────────────────────────

def test_the_insight_heading_names_every_source_an_insight_can_have():
    s = st.fresh_state()
    st.add_insight(s, "Divergence: step 2 asserts X but the response says Y", source="narration_agent")
    heading = [ln for ln in st.get_context_summary(s).splitlines() if "Recent Insights" in ln][0]
    assert "narration agent" in heading and "self-review agent" in heading and "user" in heading


def test_the_values_prompt_says_what_a_veto_does():
    system, _ = agents.build_v_prompt("q", {"response": "r"}, ["a value"])
    assert "should not be shown as written" not in system
    assert "veto notice" in system and "not blocked" in system


def test_the_review_prompt_says_how_many_existing_insights_it_sees():
    s = st.fresh_state()
    for i in range(30):
        st.add_insight(s, f"insight {i}")
    s["self_model"]["total_interactions"] = 1
    s["interaction_log"].append({"user_msg_preview": "x"})
    system, user = self_review.build_s_prompt(s)
    assert "10 most recent" in system
    assert "insight 29" in user and "insight 19" not in user


# ── the review's tension line ─────────────────────────────────────────────────

@pytest.mark.parametrize("text", ["insufficient data", "Insufficient data to say.", "No clear tension observed.",
                                  "no tension", "none", "n/a"])
def test_a_non_finding_is_not_stored_as_a_values_tension(text):
    s = st.fresh_state()
    s["self_model"]["total_interactions"] = 10
    review = self_review.normalize_review({"overall_health": "watch - thin data", "values_tension_pattern": text})
    self_review.apply_review(s, review, st)
    assert s["self_model"]["last_review_tension"] == ""
    assert "values tension" not in st.get_context_summary(s)


def test_a_real_tension_is_shown_even_when_the_review_gave_no_health_text():
    s = st.fresh_state()
    s["self_model"]["last_review_tension"] = "The system complies with flattery requests"
    s["self_model"]["last_review_at"] = 10
    assert "flattery" in st.get_context_summary(s)
    report = audit.dark_matter_report(s)
    assert "flattery" in report and "not yet reviewed" not in report


def test_a_never_reviewed_state_still_says_not_yet_reviewed():
    assert "not yet reviewed" in audit.dark_matter_report(st.fresh_state())


def test_an_error_message_cannot_carry_a_terminal_control_sequence():
    e = agents.AgentError("C", "bad gateway" + chr(27) + "[31m red" + chr(7))
    assert chr(27) not in str(e) and chr(7) not in str(e) and "bad gateway" in str(e)


# ── the API ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,body", [
    ("/chat", b'{"message": NaN}'),
    ("/chat", b'{"message": Infinity}'),
    ("/chat", b'{"message": -Infinity}'),
    ("/chat", b'{"message": 1e999}'),
    ("/insight", b'{"insight": NaN}'),
    ("/value", b'{"value": 1e999}'),
    ("/reset", b'{"session_id": Infinity}'),
    ("/chat", b'{"message": "hi", "verbose": NaN}'),
])
def test_a_rejected_request_is_a_422_even_when_it_holds_a_number_json_cannot_echo(client, path, body):
    r = quiet_client().post(path, content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422, r.text
    assert isinstance(r.json()["detail"], list)


@pytest.mark.parametrize("path,ctype", [("/chat", "text/plain"), ("/insight", "application/octet-stream"),
                                        ("/value", "application/x-www-form-urlencoded")])
def test_a_rejected_request_is_a_422_even_when_the_body_is_not_text(client, path, ctype):
    r = quiet_client().post(path, content=bytes([0xFF, 0xFE, 0x41]), headers={"content-type": ctype})
    assert r.status_code in (400, 422), r.text


@pytest.mark.parametrize("depth", [950, 1000, 1500, 5000])
def test_a_deeply_nested_rejected_request_is_a_client_error_not_a_server_error(client, depth):
    body = ('{"message": ' + "[" * depth + "]" * depth + "}").encode()
    r = quiet_client().post("/chat", content=body, headers={"content-type": "application/json"})
    assert 400 <= r.status_code < 500, (depth, r.status_code)


def test_the_422_does_not_echo_what_was_sent(client):
    r = quiet_client().post("/chat", json={"message": ["secret-looking text"]})
    assert r.status_code == 422 and "secret-looking text" not in r.text


def test_text_with_nothing_left_after_cleaning_is_empty(client, fake):
    junk = chr(0) + chr(7)
    assert client.post("/insight", json={"insight": junk}).status_code == 400
    assert client.post("/value", json={"value": chr(7)}).status_code == 400
    assert client.post("/chat", json={"message": chr(0)}).status_code == 400
    assert fake.calls == []                                          # no model call was made for it
    assert not st.STATE_FILE.exists() or on_disk()["insights"] == []


def test_the_length_limit_is_measured_on_the_text_that_would_be_stored(client):
    spaced = "word   " * 80                                          # 560 raw characters, 399 once the spacing is collapsed
    assert len(spaced.strip()) > st.MAX_INSIGHT_CHARS
    r = client.post("/insight", json={"insight": spaced})
    assert r.status_code == 200 and r.json()["insight"] == " ".join(["word"] * 80)
    assert client.post("/insight", json={"insight": "word " * 120}).status_code == 400       # 599 collapsed: still too long


def test_read_only_endpoints_never_create_count_or_move_the_state_file(client):
    for path in ("/self", "/audit/full", "/audit/dark", "/audit/values"):
        assert client.get(path).status_code == 200, path
    assert not st.STATE_FILE.exists()                                # missing file: still missing, no session counted


def test_read_only_endpoints_leave_a_good_file_byte_for_byte_alone(client):
    for _ in range(3):
        NoeticOrchestrator()                                         # three sessions
    before = st.STATE_FILE.read_bytes()
    for path in ("/self", "/audit/full", "/audit/domains"):
        assert client.get(path).status_code == 200
    assert st.STATE_FILE.read_bytes() == before and on_disk()["self_model"]["session_count"] == 3


def test_read_only_endpoints_leave_a_damaged_file_where_it_is(client):
    st.STATE_FILE.write_text("{oops", encoding="utf-8")
    assert client.get("/audit/full").status_code == 200 and client.get("/self").status_code == 200
    assert st.STATE_FILE.read_text(encoding="utf-8") == "{oops" and backups() == []


def test_read_endpoints_see_the_live_state_once_a_request_has_created_the_orchestrator(client):
    client.post("/chat", json={"message": "hello"})
    assert "Total interactions: 1" in client.get("/self").json()["self_model"]
    assert "1 in total" in client.get("/audit/full").json()["report"]


# ── the CLI ───────────────────────────────────────────────────────────────────

def test_text_after_insight_is_stored_exactly_as_typed(monkeypatch, capsys):
    out = run_cli(monkeypatch, capsys, [], argv=("--insight", "use", "--verbose", "in", "prompts"))
    assert "Insight saved: use --verbose in prompts" in out
    assert st.load()["insights"][-1]["insight"] == "use --verbose in prompts"


def test_a_help_flag_inside_the_text_is_text_not_a_request_for_help(monkeypatch, capsys):
    out = run_cli(monkeypatch, capsys, [], argv=("--insight", "mention", "-h", "in", "text"))
    assert "Insight saved: mention -h in text" in out and "Usage:" not in out
    out = run_cli(monkeypatch, capsys, [], argv=("--value", "Never", "hide", "--help", "pages"))
    assert "Value added: Never hide --help pages" in out and "Never hide --help pages" in st.load()["values"]


def test_flags_before_the_command_still_work(monkeypatch, capsys):
    assert "Noetic Self-Model" in run_cli(monkeypatch, capsys, [], argv=("--verbose", "--self"))
    assert "Usage:" in run_cli(monkeypatch, capsys, [], argv=("--help",))
    assert "Usage:" in run_cli(monkeypatch, capsys, [], argv=("-h",))


def test_text_with_nothing_left_after_cleaning_is_refused_on_the_command_line(monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        run_cli(monkeypatch, capsys, [], argv=("--insight", chr(7)))
    assert e.value.code == 2
    assert st.load()["insights"] == []


def test_a_byte_order_mark_does_not_hide_the_first_command(monkeypatch, capsys, fake):
    out = run_cli(monkeypatch, capsys, [BOM + "exit"])
    assert "Exiting." in out and fake.calls == []                    # recognised as exit, not sent to the model
    out = run_cli(monkeypatch, capsys, [BOM + "--self", "exit"])
    assert "Noetic Self-Model" in out and fake.calls == []


def test_missing_stdin_ends_cleanly_and_counts_no_session(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["cli.py"])
    monkeypatch.setattr(sys, "stdin", None)
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 1 and "Traceback" not in capsys.readouterr().err
    assert not st.STATE_FILE.exists()


def test_a_closed_stdin_ends_like_end_of_input(monkeypatch, capsys):
    class Gone:
        closed = False

        def readline(self, *a):
            raise ValueError("I/O operation on closed file.")

    monkeypatch.setattr(sys, "argv", ["cli.py"])
    monkeypatch.setattr(sys, "stdin", Gone())
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(ValueError("I/O operation on closed file.")))
    cli.main()
    assert "Exiting." in capsys.readouterr().out


def test_a_failed_save_is_reported_and_the_session_goes_on(monkeypatch, capsys, fake):
    real, seen = st.save, {"n": 0}

    def deny(state, *a, **kw):
        seen["n"] += 1
        if seen["n"] > 2:                                            # start-up writes twice (new file, session count)
            raise PermissionError("Access is denied")
        return real(state, *a, **kw)

    monkeypatch.setattr(st, "save", deny)
    out = run_cli(monkeypatch, capsys, ["hello", "exit"])
    assert "nothing was saved" in out and "Access is denied" in out and "Exiting." in out
    assert on_disk()["self_model"]["total_interactions"] == 0


def test_a_one_shot_command_reports_an_unwritable_state_instead_of_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr(st, "save", lambda *a, **kw: (_ for _ in ()).throw(PermissionError("Access is denied")))
    with pytest.raises(SystemExit) as e:
        run_cli(monkeypatch, capsys, [], argv=("--insight", "x"))
    assert e.value.code == 1 and "Access is denied" in capsys.readouterr().err


def test_the_verbose_note_about_confidence_matches_what_happened(orch, fake, monkeypatch, capsys):
    fake.script("C", C(confidence="n/a"))                            # read fine, but there is no usable number
    out = run_cli(monkeypatch, capsys, ["hello", "exit"], argv=("--verbose",))
    assert "Confidence: not reported" in out and "no usable number" in out
    assert "could not be read as structured output" not in out
    fake.script("C", "plain prose with no json")
    out = run_cli(monkeypatch, capsys, ["hello again", "exit"], argv=("--verbose",))
    assert "could not be read as structured output" in out


# ── the audit command ─────────────────────────────────────────────────────────

def test_the_audit_says_so_when_there_is_no_state_file(capsys):
    assert not st.STATE_FILE.exists()
    text = audit.full_health_report()
    assert "Session count: 0" in text and not st.STATE_FILE.exists()
    assert "no state file" in capsys.readouterr().err


# ── a degraded reply may raise a concern but never clear one ──────────────────

@pytest.mark.parametrize("status", ["flag", "veto"])
@pytest.mark.parametrize("stop", ["max_tokens", "end_turn"])
def test_a_cut_off_values_reply_that_flags_or_vetoes_still_stands(orch, fake, status, stop):
    fake.script("V", result('{"status": "%s", "values_notes": ["the answer flatters the user' % status, stop=stop))
    r = orch.run("q")
    assert r["values_status"] == status and r["parse_issues"] is True


@pytest.mark.parametrize("risk", ["medium", "high"])
def test_a_cut_off_narration_reply_that_rates_risk_still_stands_but_adds_no_insight(orch, fake, risk):
    fake.script("N", result('{"explanation_matches_reasoning": false, "confabulation_risk": "%s", '
                            '"divergence_notes": ["step 2 asserts X' % risk, stop="max_tokens"))
    r = orch.run("q")
    assert r["confabulation_risk"] == risk and r["parse_issues"] is True
    assert on_disk()["insights"] == []                               # only a complete reply becomes an insight


def test_the_partial_notes_of_a_cleared_reply_are_kept_for_the_verbose_view(orch, fake):
    fake.script("V", result(CUT_V, stop="max_tokens"))
    r = orch.run("q", verbose=True)
    assert r["values_status"] == "unchecked"
    assert any("complete certainty" in n for n in r["v_notes"]) and "cut off" in r["v_notes"][0]


# ── the pieces underneath ─────────────────────────────────────────────────────

def test_save_refuses_to_write_over_a_file_that_is_not_the_one_it_was_based_on():
    st.save(st.fresh_state())
    based_on = st.file_signature()
    st.save(st.fresh_state())                                        # someone else saves
    before = st.STATE_FILE.read_bytes()
    with pytest.raises(st.StaleState):
        st.save(st.fresh_state(), expect_sig=based_on)
    assert st.STATE_FILE.read_bytes() == before and list(st.STATE_FILE.parent.glob("*.tmp")) == []
    assert st.save(st.fresh_state(), expect_sig=st.file_signature()) is not None
    with pytest.raises(st.StaleState):
        st.save(st.fresh_state(), expect_sig=None)                   # "there is no file" is an expectation too
    st.STATE_FILE.unlink()
    assert st.save(st.fresh_state(), expect_sig=None) is not None


def test_a_change_that_keeps_meeting_a_newer_file_is_reported_not_dropped_silently(orch, monkeypatch):
    def always_stale(state, *a, **kw):
        raise st.StaleState("changed again")

    monkeypatch.setattr(st, "save", always_stale)
    with pytest.raises(OSError, match="kept changing"):
        orch.add_value("x")
    assert "x" not in orch.snapshot()["values"]


def test_every_mutation_is_applied_on_top_of_the_newest_file_not_ours(fake):
    a, b = NoeticOrchestrator(), NoeticOrchestrator(count_session=False)
    a.add_insight("from a")
    b.add_value("from b")                                            # b adopts a's insight first
    a.add_insight("from a again")
    final = on_disk()
    assert "from b" in final["values"]
    assert [i["insight"] for i in final["insights"]] == ["from a", "from a again"]


def test_is_blank_ignores_session_count_and_notices_every_kind_of_history():
    s = st.fresh_state()
    assert st.is_blank(s)
    s["self_model"]["session_count"] = 9
    assert st.is_blank(s)
    for change in (lambda x: x["self_model"].update(total_interactions=1),
                   lambda x: x["self_model"].update(last_review_at=10),
                   lambda x: st.add_insight(x, "i"),
                   lambda x: x["interaction_log"].append({}),
                   lambda x: x["domain_confidence"].update(py={}),
                   lambda x: x["uncertainty_map"].update(k={}),
                   lambda x: st.add_value(x, "my own value")):
        t = st.fresh_state()
        change(t)
        assert not st.is_blank(t)


def test_unknown_keys_are_kept_only_when_small_and_plain():
    raw = st.fresh_state()
    raw["keep_me"] = {"a": [1, 2.5, "x", None, True], "b": {"c": "d"}}
    raw["too_deep"] = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": 1}}}}}}}}
    raw["too_big"] = list(range(st.MAX_EXTRA_NODES + 10))
    raw["self_model"]["my_note"] = "kept"
    raw["self_model"]["deep_note"] = [[[[[[[[1]]]]]]]]
    st.STATE_FILE.write_text(json.dumps(raw), encoding="utf-8")
    s = st.load()
    assert s["keep_me"] == raw["keep_me"] and "too_deep" not in s and "too_big" not in s
    assert s["self_model"]["my_note"] == "kept" and "deep_note" not in s["self_model"]
    assert backups() == []


def test_all_the_unknown_keys_at_one_level_share_one_budget():
    raw = st.fresh_state()
    for i in range(2000):
        raw[f"extra_{i}"] = "x" * 100                                # each is small; together they are not
    st.STATE_FILE.write_text(json.dumps(raw), encoding="utf-8")
    kept = [k for k in st.load() if k.startswith("extra_")]
    assert len(kept) == st.MAX_EXTRA_NODES and "extra_0" in kept and "extra_1999" not in kept


def test_clean_line_collapses_whitespace_and_treats_only_replacement_characters_as_empty():
    assert lj.clean_line("  a \t b\n c  ") == "a b c"
    assert lj.clean_line(chr(0) + chr(7)) == "" and lj.clean_line(FFFD + " " + FFFD) == ""
    assert lj.clean_line("x" + chr(7) + "y") == "x" + FFFD + "y"          # text with something real in it is kept
    assert lj.clean_line("a" * 20, 5) == "a" * 5 and lj.clean_line(None) == "" and lj.clean_line(42) == "42"


@pytest.mark.parametrize("text,expected", [
    ("yes", True), ("Yes, present", True), ("True.", True), ("detected", True), ("partially", True), ("1", True),
    ("no", False), ("No.", False), ("not detected", False), ("None", False), ("does not follow", False), ("0", False),
    ("maybe", None), ("unclear", None), ("", None), ("   ", None), ("the answer follows", None), (5, None),
])
def test_yes_no_reading_of_short_answers(text, expected):
    assert lj.to_tri(text) is expected


def test_a_list_holding_one_answer_reads_as_that_answer():
    assert lj.to_tri(["false"]) is False and lj.to_tri(["Yes"]) is True and lj.to_tri(["a", "b"]) is None
    assert lj.to_bool(["yes"]) is True and lj.to_bool([]) is False and lj.to_bool(["maybe"], True) is True


def test_a_value_made_only_of_control_characters_never_reaches_the_list():
    s = st.fresh_state()
    assert st.add_value(s, chr(7)) is False and st.add_insight_ex(s, chr(0)) is False
    assert s["values"] == st.DEFAULT_VALUES and s["insights"] == []
    raw = st.fresh_state()
    raw["insights"] = [{"insight": FFFD}, {"insight": "real"}]
    raw["values"] = [FFFD, "kept"]
    st.STATE_FILE.write_text(json.dumps(raw), encoding="utf-8")
    loaded = st.load()
    assert [i["insight"] for i in loaded["insights"]] == ["real"] and loaded["values"] == ["kept"]

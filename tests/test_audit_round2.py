"""
Regression tests for the second independent review and the pipeline audit. Control characters are built with chr()
so that no escape sequence has to survive in this source.
"""

import json
import os
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import audit
import llm_json as lj
import orchestrator as om
import self_review as sr
import state as st
from fakes import C, N, V, S, result, run_cli, echo_c

ROOT = Path(__file__).resolve().parent.parent
ESC, BEL, NUL, DEL, NEL, FFFD = chr(27), chr(7), chr(0), chr(127), chr(0x85), chr(0xFFFD)
ZWSP, TAGCHAR, FULL_LT, FULL_GT = chr(0x200B), chr(0xE0001), chr(0xFF1C), chr(0xFF1E)


def backups():
    return sorted(st.STATE_FILE.parent.glob("noetic_state.json.bak*"))


# ── merging and persistence ───────────────────────────────────────────────────

def test_an_unreadable_file_found_by_a_read_only_call_is_rewritten_from_memory(fake):
    a = om.NoeticOrchestrator()
    a.run("one")
    a.run("two")
    assert a.add_value("kept value")
    st.STATE_FILE.write_text('{"self_model": {"na', encoding="utf-8")
    a.snapshot()                                           # a read must not leave the file missing
    assert st.STATE_FILE.exists()
    on_disk = st.load()                                    # what any other process would now read
    assert on_disk["self_model"]["total_interactions"] == 2 and "kept value" in on_disk["values"]
    assert [p.read_text(encoding="utf-8") for p in backups()] == ['{"self_model": {"na']
    assert a.snapshot()["self_model"]["total_interactions"] == 2


def test_a_commit_that_lands_right_after_our_save_is_not_marked_as_seen(fake, monkeypatch):
    a, b = om.NoeticOrchestrator(), om.NoeticOrchestrator()
    real = os.replace
    flags = {"fire": False, "busy": False}

    def replace_then_b_commits(src, dst):
        real(src, dst)
        if flags["fire"] and not flags["busy"]:
            flags["busy"] = True
            try:
                assert b.add_value("b-value")              # B commits at the exact moment after A's rename
            finally:
                flags["busy"] = False

    monkeypatch.setattr(os, "replace", replace_then_b_commits)
    flags["fire"] = True
    assert a.add_value("a-value")
    flags["fire"] = False
    a.add_insight("a later commit by A")
    saved = st.load()
    assert "a-value" in saved["values"] and "b-value" in saved["values"]


def test_a_commit_that_lands_right_after_our_read_is_not_marked_as_seen(fake, monkeypatch):
    a, b = om.NoeticOrchestrator(), om.NoeticOrchestrator()
    assert b.add_value("b0")                               # the file now differs from what A last saw
    real_try = st.try_load
    fired = {"n": 0}

    def read_then_b_commits():
        loaded = real_try()
        if fired["n"] == 0:
            fired["n"] += 1
            assert b.add_value("b1")                       # lands after A's read, while A waits for the models
        return loaded

    monkeypatch.setattr(st, "try_load", read_then_b_commits)
    a.run("a question")
    saved = st.load()
    assert {"b0", "b1"} <= set(saved["values"]) and saved["self_model"]["total_interactions"] == 1


def test_save_returns_the_signature_of_what_it_wrote():
    s = st.load()
    sig = st.save(s)
    assert sig is not None and sig == st.file_signature()


def test_audit_reports_never_write_the_state_file(capsys):
    st.STATE_FILE.write_text("{ broken", encoding="utf-8")
    text = audit.full_health_report()
    assert "Interactions:  0 in total" in text
    assert st.STATE_FILE.read_text(encoding="utf-8") == "{ broken" and backups() == []
    assert "unreadable" in capsys.readouterr().err
    st.STATE_FILE.unlink()
    audit.full_health_report()
    assert not st.STATE_FILE.exists()


def test_set_aside_never_overwrites_an_earlier_backup(monkeypatch):
    first = st.STATE_FILE.with_name("noetic_state.json.bak")
    first.write_text("evidence one", encoding="utf-8")
    st.STATE_FILE.write_text("evidence two", encoding="utf-8")
    with monkeypatch.context() as m:
        m.setattr(Path, "exists", lambda self: False)      # a stale answer, as in a race between processes
        assert st.set_aside() is True
    assert first.read_text(encoding="utf-8") == "evidence one"
    assert sorted(p.read_text(encoding="utf-8") for p in backups()) == ["evidence one", "evidence two"]


def test_set_aside_leaves_a_file_alone_that_changed_after_it_was_judged_unreadable():
    st.STATE_FILE.write_text("{ broken", encoding="utf-8")
    sig = st.file_signature()
    st.save(st._new_state())                               # someone replaced it with a valid file meanwhile
    assert st.set_aside(sig) is False
    assert json.loads(st.STATE_FILE.read_text(encoding="utf-8"))["self_model"]["name"] == "Noetic" and backups() == []


def test_load_raises_rather_than_overwrite_a_file_it_cannot_set_aside(monkeypatch):
    st.STATE_FILE.write_text("{ broken", encoding="utf-8")
    monkeypatch.setattr(st, "set_aside", lambda sig=None: False)
    with pytest.raises(OSError, match="not overwritten"):
        st.load()
    assert st.STATE_FILE.read_text(encoding="utf-8") == "{ broken"


def test_a_state_file_with_a_byte_order_mark_loads(fake):
    a = om.NoeticOrchestrator()
    a.run("one")
    a.run("two")
    st.STATE_FILE.write_bytes(bytes([0xEF, 0xBB, 0xBF]) + st.STATE_FILE.read_bytes())      # what Notepad does on save
    loaded = st.try_load()
    assert loaded is not None and loaded["self_model"]["total_interactions"] == 2
    assert "Interactions:  2 in total" in audit.full_health_report() and backups() == []


# ── control characters and the fence ──────────────────────────────────────────

def test_scrub_removes_control_characters_but_keeps_tab_newline_and_emoji():
    emoji = chr(0x1F600)
    s = f"a{ESC}[2Jb{BEL}{NUL}{DEL}{NEL}c\t\n\r{emoji}"
    assert lj.scrub(s) == f"a{FFFD}[2Jb{FFFD}{FFFD}{FFFD}{FFFD}c\t\n\r{emoji}"


def test_terminal_control_sequences_in_model_text_never_reach_the_output(orch, fake):
    fake.script("C", C(response=f"All good.{ESC}[2J{ESC}]0;owned{BEL}"))
    fake.script("N", N(confabulation_risk="high", explanation_matches_reasoning=False,
                       divergence_notes=[f"step {ESC}]0;owned{BEL} odd"]))
    res = orch.run(f"hi{ESC}[2J")
    assert "All good." in res["response"] and ESC not in res["response"] and BEL not in res["response"]
    assert ESC not in orch.conversation.turns[0].content
    stored = [i["insight"] for i in orch.state["insights"]]
    assert stored and all(ESC not in t and BEL not in t for t in stored)
    assert ESC not in audit.dark_matter_report(orch.snapshot())


@pytest.mark.parametrize("evil", [
    "<" + ZWSP + "/values>", "</values>", "<" + NUL + "/values>", FULL_LT + "/values" + FULL_GT,
    "<" + TAGCHAR + "/values>", "< /values >", "<values>", "<" + chr(0xFEFF) + "values", "<\n/values>",
])
def test_fence_leaves_no_angle_bracket_from_untrusted_text(evil):
    out = lj.fence("values", "x " + evil + " y")
    body = out[len("<values>\n"):-len("\n</values>")]
    assert "<" not in body and FULL_LT not in body
    assert out.count("<values>") == 1 and out.count("</values>") == 1


# ── fixed points, labels, surrogate pairs ─────────────────────────────────────

def test_stored_text_is_a_fixed_point_of_its_own_sanitizer():
    s = st.load()
    assert st.add_insight_ex(s, "w" * 499 + " tail") is True
    assert s["insights"][-1]["insight"] == "w" * 499
    assert st.add_value(s, "v" * 299 + " tail") is True and s["values"][-1] == "v" * 299
    assert lj.to_str("q" * 99 + " x", 100) == "q" * 99 and lj.norm_domain("d" * 39 + " x") == "d" * 39
    assert lj.to_str_list(["k" * 9 + " tail"], max_len=10) == ["k" * 9]
    st.save(s)
    assert st.load()["insights"] == s["insights"] and st.load()["values"] == s["values"]
    assert st.add_insight_ex(st.load(), "w" * 499 + " tail") is False        # recognised after a restart too
    rnd = random.Random(7)
    for _ in range(300):
        text = " ".join(rnd.choice(["a", "bb", "ccc", "word", "x" * rnd.randint(1, 9), ESC, ""]) for _ in range(rnd.randint(1, 200)))
        once = lj.to_str(text, 50)
        assert lj.to_str(once, 50) == once
        listed = lj.to_str_list([text], max_len=50)
        assert lj.to_str_list(listed, max_len=50) == listed


@pytest.mark.parametrize("health,label", [("not healthy - failing", "unknown"), ("Not healthy: concerning", "unknown"),
                                          ("unhealthy - bad", "unknown"), ("healthy - fine", "healthy"),
                                          ("  Watch - thin data", "watch"), ("- concerning", "concerning")])
def test_health_label_is_read_from_the_start_only(health, label):
    assert sr.normalize_review({"overall_health": health})["health_label"] == label


@pytest.mark.parametrize("cut,expected", [
    ('{"response": "Hi \\ud83d', "Hi "), ('{"response": "Hi \\ud83d\\ude0', "Hi "),
    ('{"response": "Hi \\ud83d\\ude00', "Hi " + chr(0x1F600)),
])
def test_a_reply_cut_inside_an_escaped_surrogate_pair_is_a_clean_prefix(cut, expected):
    r = lj.extract_json(cut, ("response",))
    assert r["how"] == "repaired" and r["value"]["response"] == expected


# ── contradictions inside a check's own reply ─────────────────────────────────

def test_an_n_reply_that_says_the_explanation_does_not_follow_cannot_rate_the_risk_low(orch, fake):
    fake.script("N", N(confabulation_risk="low", explanation_matches_reasoning=False, divergence_notes=["step 2 is a leap"]))
    res = orch.run("hi", verbose=True)
    assert res["confabulation_risk"] == "medium" and "moderate confabulation risk" in res["response"]
    assert "raised from low to medium" in res["n_assessment"]
    assert [i["insight"] for i in orch.state["insights"]] == ["Divergence: step 2 is a leap"]
    fake.script("N", N(confabulation_risk="low", explanation_matches_reasoning=True))
    assert orch.run("fine")["confabulation_risk"] == "low"                      # a consistent low stays low


def test_a_v_reply_that_passes_but_reports_a_concern_is_shown_as_flag(orch, fake):
    fake.script("V", V(status="pass", hidden_uncertainty_detected=True))
    res = orch.run("hi", verbose=True)
    assert res["values_status"] == "flag"
    assert res["v_notes"][0].startswith("Status raised from pass to flag because it reported hidden uncertainty")
    fake.script("V", V(status="pass", recommendation="Regenerate."))
    assert orch.run("again")["values_status"] == "flag"
    fake.script("V", V(status="pass", recommendation="pass as-is"))
    assert orch.run("third")["values_status"] == "pass"
    fake.script("V", V(status="veto", hidden_uncertainty_detected=True))
    assert orch.run("fourth")["values_status"] == "veto"                         # only a pass is raised


def test_the_values_tension_pattern_is_kept_and_shown():
    s = st.load()
    review = sr.normalize_review({"values_tension_pattern": "V flags approvals more often than it vetoes", "overall_health": "watch - thin"})
    review["parse_status"] = "ok"
    sr.apply_review(s, review)
    assert s["self_model"]["last_review_tension"] == "V flags approvals more often than it vetoes"
    assert "values tension noted: V flags approvals" in st.get_context_summary(s)
    assert "Last review tension:   V flags approvals" in audit.dark_matter_report(s)
    st.save(s)
    assert st.load()["self_model"]["last_review_tension"].startswith("V flags")


def test_an_empty_answer_is_not_a_measurement_in_the_log_either(orch, fake):
    fake.script("C", C(response="  "))
    res = orch.run("hi")
    assert res["parse_issues"] is True and orch.state["domain_confidence"] == {}


# ── API ───────────────────────────────────────────────────────────────────────

def test_a_lone_surrogate_in_a_request_never_gives_a_500(client):
    headers = {"content-type": "application/json"}
    r = client.post("/insight", content='{"insight": "a\\ud800b"}', headers=headers)
    assert r.status_code == 200 and r.json()["insight"] == "a" + FFFD + "b"
    assert any(i["insight"] == "a" + FFFD + "b" for i in st.load()["insights"])
    assert client.post("/value", content='{"value": "v\\ud800v"}', headers=headers).status_code == 200
    assert client.post("/chat", content='{"message": "hi", "session_id": "a\\ud800"}', headers=headers).status_code == 422
    assert client.post("/chat", content='{"message": "hi", "verbose": "x\\ud800"}', headers=headers).status_code == 422
    assert client.post("/chat", content='{"message": "hi \\ud800"}', headers=headers).status_code == 200


def test_the_manual_review_endpoint_lists_no_insights_when_the_review_failed(client, fake):
    client.post("/chat", json={"message": "hi"})
    fake.script("S", "no json at all")
    body = client.post("/review").json()
    assert body["status"] == "review_not_applied" and body["new_insights"] == []


def test_a_failed_manual_review_lists_no_insights(orch, fake):
    orch.run("q")
    fake.script("S", "no json at all")
    review = orch.run_manual_review()
    assert review["new_insights"] == [] and review["applied"]["skipped"]


# ── conversation history shown to C ───────────────────────────────────────────

def test_history_does_not_show_a_confidence_the_model_never_gave(orch, fake):
    fake.script("C", C(confidence="high"))
    orch.run("one")
    fake.script("V", V(status="veto"))
    fake.script("C", "plain prose, no json")
    orch.run("two")
    digest = orch.conversation.history_digest()
    assert "confidence=n/a" in digest and "confidence=50%" not in digest
    assert "values=veto" in digest                                              # a vetoed answer carries its assessment


# ── CLI and console ───────────────────────────────────────────────────────────

def test_cli_applies_the_api_limits(monkeypatch, capsys):
    for flag, limit in (("--insight", 500), ("--value", 300)):
        with pytest.raises(SystemExit) as info:
            run_cli(monkeypatch, capsys, [], argv=[flag, "x" * (limit + 1)])
        assert info.value.code == 2 and "too long" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, capsys, [], argv=["--value"])
    assert "needs text" in capsys.readouterr().err
    assert not st.STATE_FILE.exists()                                           # nothing was read or written


def test_cli_help_and_unknown_flags(monkeypatch, capsys):
    assert "Usage:" in run_cli(monkeypatch, capsys, [], argv=["--help"])
    for bad in ("--selff", "--insigth"):
        with pytest.raises(SystemExit) as info:
            run_cli(monkeypatch, capsys, [], argv=[bad])
        assert info.value.code == 2 and "Unknown argument" in capsys.readouterr().err
    assert not st.STATE_FILE.exists()                                           # no chat loop, no session, no file


def test_cli_one_shot_commands_are_not_sessions(monkeypatch, capsys):
    run_cli(monkeypatch, capsys, ["exit"])
    assert st.load()["self_model"]["session_count"] == 1
    run_cli(monkeypatch, capsys, [], argv=["--self"])
    run_cli(monkeypatch, capsys, [], argv=["--value", "Another value"])
    run_cli(monkeypatch, capsys, [], argv=["--insight", "A note"])
    saved = st.load()
    assert saved["self_model"]["session_count"] == 1 and "Another value" in saved["values"]


def _child(code, tmp_path, stdin=""):
    env = dict(os.environ, NOETIC_STATE_FILE=str(tmp_path / "child_state.json"), PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONIOENCODING", None)
    env.pop("PYTHONUTF8", None)
    return subprocess.run([sys.executable, "-c", code], input=stdin.encode("utf-8"), capture_output=True,
                          env=env, cwd=str(ROOT), timeout=90)


def test_utf8_typed_into_a_pipe_reaches_the_model_unchanged(tmp_path):
    code = textwrap.dedent("""
        import sys
        sys.path.insert(0, "tests")
        import provider
        from fakes import FakeProvider, echo_c
        fake = FakeProvider()
        fake.defaults["C"] = echo_c
        provider.complete = fake.complete
        sys.argv = ["cli.py"]
        import cli
        cli.main()
    """)
    typed = "h" + chr(0xE9) + "llo " + chr(0x65E5) + chr(0x672C) + chr(0x8A9E)
    out = _child(code, tmp_path, stdin=typed + "\nexit\n")
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")[-400:]
    assert ("echo:" + typed) in out.stdout.decode("utf-8")


def test_audit_rejects_an_unknown_report_name(tmp_path):
    env = dict(os.environ, NOETIC_STATE_FILE=str(tmp_path / "s.json"), PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "audit.py", "domain"], capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=60)
    assert out.returncode == 2 and "unknown report 'domain'" in out.stderr
    assert not (tmp_path / "s.json").exists()

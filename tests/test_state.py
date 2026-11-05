"""state: persistence, sanitizing of whatever is on disk, and the mutation rules."""

import copy
import json
import os
from datetime import datetime

import pytest

import state as st


# ── load / save ───────────────────────────────────────────────────────────────

def test_load_creates_a_default_state_and_never_aliases_the_defaults():
    s1 = st.load()
    assert st.STATE_FILE.exists()
    s1["values"].append("x")
    s1["insights"].append({"insight": "i"})
    s1["self_model"]["architectural_awareness"]["universal_language"] = "changed"
    s2 = st.load()                                       # the file still holds the unmodified state
    assert "x" not in s2["values"] and s2["insights"] == []
    assert "x" not in st.DEFAULT_VALUES and "x" not in st.DEFAULT_STATE["values"]
    assert st.DEFAULT_AWARENESS["universal_language"] != "changed"
    assert st.DEFAULT_STATE["self_model"]["architectural_awareness"]["universal_language"] != "changed"
    datetime.fromisoformat(s2["self_model"]["created"])
    a, b = st._new_state(), st._new_state()
    a["values"].append("z")
    assert "z" not in b["values"]


def test_corrupt_file_is_set_aside_not_lost():
    st.STATE_FILE.write_text("{not json", encoding="utf-8")
    s = st.load()
    assert s["self_model"]["total_interactions"] == 0
    kept = list(st.STATE_FILE.parent.glob("noetic_state.json.bak*"))
    assert [p.read_text(encoding="utf-8") for p in kept] == ["{not json"]
    assert json.loads(st.STATE_FILE.read_text(encoding="utf-8"))["self_model"]["name"] == "Noetic"


@pytest.mark.parametrize("content", ["[1, 2, 3]", "null", '"text"', "42", ""])
def test_valid_json_that_is_not_an_object_is_set_aside_too(content):
    st.STATE_FILE.write_text(content, encoding="utf-8")
    st.load()
    kept = list(st.STATE_FILE.parent.glob("noetic_state.json.bak*"))
    assert [p.read_text(encoding="utf-8") for p in kept] == [content]


def test_garbage_on_disk_is_sanitized_not_trusted():
    junk = {
        "self_model": "oops",
        "values": "just a string",
        "insights": [1, None, {"insight": ""}, {"insight": " ok  note "}, "bare"],
        "uncertainty_map": [],
        "domain_confidence": {"python": {"confidence": "NaN"}, "Go ": {"confidence": 0.7}, "": {"confidence": 0.5},
                              "unknown": {"confidence": 0.5}, "__proto__": {"confidence": 5}, "bad": "x"},
        "interaction_log": "abc",
        "extra_key": 1,
    }
    st.STATE_FILE.write_text(json.dumps(junk), encoding="utf-8")
    s = st.load()
    assert s["self_model"]["name"] == "Noetic" and s["self_model"]["session_count"] == 0
    assert s["values"] == ["just a string"]
    assert [i["insight"] for i in s["insights"]] == ["ok note", "bare"]
    assert s["uncertainty_map"] == {} and s["interaction_log"] == []
    assert set(s["domain_confidence"]) == {"go", "__proto__"}
    assert s["domain_confidence"]["__proto__"]["confidence"] == pytest.approx(0.05)
    assert s["extra_key"] == 1                                   # unknown top-level keys survive


def test_poison_values_are_dropped_and_defaults_return_if_nothing_is_left():
    raw = st._new_state()
    raw["values"] = [None, "", "none", "N/A", "Be kind"]
    st.STATE_FILE.write_text(json.dumps(raw), encoding="utf-8")
    assert st.load()["values"] == ["Be kind"]
    raw["values"] = [None, "none"]
    st.STATE_FILE.write_text(json.dumps(raw), encoding="utf-8")
    assert st.load()["values"] == st.DEFAULT_VALUES


def test_log_entries_and_caps_are_enforced_on_load():
    raw = st._new_state()
    raw["interaction_log"] = [{"user_msg_preview": f"m{i}", "confabulation_risk": "WEIRD", "values_status": "PASS ",
                               "confidence": "85%"} for i in range(150)] + ["junk", None, 5]
    raw["insights"] = [{"insight": f"i{i}"} for i in range(80)]
    st.STATE_FILE.write_text(json.dumps(raw), encoding="utf-8")
    s = st.load()
    assert len(s["interaction_log"]) == 100 and s["interaction_log"][-1]["user_msg_preview"] == "m149"
    e = s["interaction_log"][0]
    assert e["confabulation_risk"] == "unknown" and e["values_status"] == "pass" and e["confidence"] == pytest.approx(0.85)
    assert len(s["insights"]) == 50 and s["insights"][-1]["insight"] == "i79"


def test_old_awareness_text_is_replaced_once_and_later_edits_are_kept():
    s = st.load()
    s["self_model"]["architectural_awareness"]["goals_in_identity"] = "RM bias fires 100% of the time"
    s["self_model"].pop("awareness_rev")
    st.save(s)
    s2 = st.load()
    assert "100% of the time" not in json.dumps(s2["self_model"]["architectural_awareness"])
    assert "[Source:" in s2["self_model"]["architectural_awareness"]["goals_in_identity"]
    assert s2["self_model"]["awareness_rev"] == st.AWARENESS_REV
    s2["self_model"]["architectural_awareness"]["goals_in_identity"] = "my own wording"
    st.save(s2)
    assert st.load()["self_model"]["architectural_awareness"]["goals_in_identity"] == "my own wording"


def test_the_awareness_text_says_the_values_agent_reads_only_the_response():
    text = st.DEFAULT_AWARENESS["goals_in_identity"]
    assert "reads only my response" in text and "never my reasoning steps or my stated confidence" in text
    assert "not shown my reasoning" not in text


def test_a_state_saved_with_the_previous_awareness_revision_picks_up_the_new_text():
    s = st.load()
    s["self_model"]["architectural_awareness"]["goals_in_identity"] = "the wording this claim had in revision 2"
    s["self_model"]["awareness_rev"] = 2
    st.save(s)
    s2 = st.load()
    assert "reads only my response" in s2["self_model"]["architectural_awareness"]["goals_in_identity"]
    assert s2["self_model"]["awareness_rev"] == st.AWARENESS_REV == 3


def test_default_awareness_states_sources_and_limits_for_every_claim():
    for key, text in st.DEFAULT_AWARENESS.items():
        assert "[Source:" in text and "[Limits:" in text, key
    joined = " ".join(st.DEFAULT_AWARENESS.values())
    assert "narration circuit" in joined and "do not describe a separate" in joined      # the retired claim is only mentioned as retired
    assert "about 20%" in joined and "19% to 100%" in joined


def test_failed_save_leaves_the_old_file_intact_and_no_temp_file(monkeypatch):
    s = st.load()
    before = st.STATE_FILE.read_text(encoding="utf-8")
    s["values"].append("new")

    def boom(a, b):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        st.save(s)
    assert st.STATE_FILE.read_text(encoding="utf-8") == before
    assert [p for p in st.STATE_FILE.parent.iterdir() if p.suffix == ".tmp"] == []


def test_save_retries_a_briefly_locked_file(monkeypatch):
    s = st.load()
    real = os.replace
    calls = {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("locked")
        return real(a, b)

    monkeypatch.setattr(os, "replace", flaky)
    st.save(s)
    assert calls["n"] == 3


# ── mutations ─────────────────────────────────────────────────────────────────

def test_total_interactions_survives_the_log_cap():
    s = st.load()
    for i in range(130):
        st.log_interaction(s, f"msg {i}", "resp", "low", "pass")
    assert s["self_model"]["total_interactions"] == 130
    assert len(s["interaction_log"]) == st.MAX_LOG == 100
    assert s["interaction_log"][-1]["user_msg_preview"] == "msg 129"


def test_log_interaction_sanitizes_every_field():
    s = st.load()
    st.log_interaction(s, "x" * 500, "r", "WEIRD", "PASS ", domain=" Py. ", confidence="85%", sycophancy="yes")
    e = s["interaction_log"][-1]
    assert len(e["user_msg_preview"]) == 100 and e["confabulation_risk"] == "unknown"
    assert e["values_status"] == "pass" and e["domain"] == "py"
    assert e["confidence"] == pytest.approx(0.85) and e["sycophancy"] is True and e["parse_issues"] is False
    st.log_interaction(s, "y", "r", "high", "veto", confidence=None)
    assert s["interaction_log"][-1]["confidence"] is None


def test_domain_confidence_is_a_smoothed_average_and_skips_unmeasured_input():
    s = st.load()
    for bad in ("", "unknown", None, True, ["x"]):
        st.update_domain_confidence(s, bad, 0.9)
    for badc in (float("nan"), float("inf"), None, "abc", True):
        st.update_domain_confidence(s, "py", badc)
    assert s["domain_confidence"] == {}
    st.update_domain_confidence(s, "Python", 0.8)
    st.update_domain_confidence(s, "python", 0.4)
    e = s["domain_confidence"]["python"]
    assert e["n"] == 2 and e["confidence"] == pytest.approx(0.8 * 0.7 + 0.4 * 0.3)
    assert e["flagged_at"] is None


def test_domain_table_is_capped_dropping_the_oldest():
    s = st.load()
    for i in range(45):
        st.update_domain_confidence(s, f"d{i}", 0.5)
    assert len(s["domain_confidence"]) == st.MAX_DOMAINS == 40
    assert "d0" not in s["domain_confidence"] and "d44" in s["domain_confidence"]


def test_add_value_rules():
    s = st.load()
    assert st.add_value(s, "  Be   kind  ") is True and s["values"][-1] == "Be kind"
    assert st.add_value(s, "be KIND") is False                       # duplicate, case-insensitive
    for junk in ("", "   ", "none", "N/A", "-", "[]"):
        assert st.add_value(s, junk) is False
    i = 0
    while len(s["values"]) < st.MAX_VALUES:
        st.add_value(s, f"v{i}")
        i += 1
    assert st.add_value(s, "one more") is False and len(s["values"]) == st.MAX_VALUES


def test_add_insight_rules():
    s = st.load()
    st.add_insight(s, "  a   b  ", "user")
    assert s["insights"][-1]["insight"] == "a b"
    st.add_insight(s, "a b", "user")
    st.add_insight(s, "", "user")
    st.add_insight(s, "   ", "user")
    assert len(s["insights"]) == 1
    for i in range(60):
        st.add_insight(s, f"n{i}")
    assert len(s["insights"]) == 50 and s["insights"][-1]["insight"] == "n59"
    st.add_insight(s, "x" * 900)
    assert len(s["insights"][-1]["insight"]) == 500


def test_increment_session():
    s = st.load()
    st.increment_session(s)
    assert s["self_model"]["session_count"] == 1
    datetime.fromisoformat(s["self_model"]["last_session"])


# ── read-only views ───────────────────────────────────────────────────────────

def test_compute_stats_counts_exactly():
    s = st.load()
    st.log_interaction(s, "a", "r", "low", "pass", domain="py", confidence=0.9)
    st.log_interaction(s, "b", "r", "high", "flag", domain="py", confidence=0.5, sycophancy=True)
    st.log_interaction(s, "c", "r", "unknown", "unchecked", domain="unknown", parse_issues=True)
    stats = st.compute_stats(s)
    assert stats["total_interactions"] == 3 and stats["entries_in_window"] == 3
    assert stats["confabulation_risk"] == {"low": 1, "medium": 0, "high": 1, "unknown": 1}
    assert stats["values_status"] == {"pass": 1, "flag": 1, "veto": 0, "unchecked": 1}
    assert stats["sycophancy_flagged"] == 1 and stats["entries_with_parse_issues"] == 1
    assert stats["domains"] == {"py": {"interactions": 2, "mean_confidence": 0.7,
                                       "medium_or_high_confab_risk": 1, "flagged_or_vetoed": 1}}


def test_context_summary_labels_insights_and_survives_damaged_in_memory_state():
    s = st.load()
    s["domain_confidence"]["x"] = {"confidence": "abc", "n": "many"}
    s["domain_confidence"]["y"] = "not a dict"
    s["insights"].append({"insight": "a note", "timestamp": None})
    text = st.get_context_summary(copy.deepcopy(s))
    assert "- x: n/a" in text and "unverified" in text and "a note" in text
    assert text.count("[Source:") == 4


def test_context_summary_shows_flags_and_review():
    s = st.load()
    st.update_domain_confidence(s, "py", 0.6)
    s["domain_confidence"]["py"]["flagged_at"] = "2026-01-01T00:00:00"
    s["self_model"]["last_review_health"] = "watch - thin data"
    text = st.get_context_summary(s)
    assert "py: 60% over 1 answer (flagged by self-review)" in text and "watch - thin data" in text

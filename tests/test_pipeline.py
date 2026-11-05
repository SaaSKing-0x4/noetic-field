"""
The public interface: noetic_field.Pipeline and ChatResult.

The tests import the module as pipeline (the modules sit side by side here, as in a source checkout) and run it on the
same scripted provider as every other test. What they pin down is the contract of the interface: the four fields and
their types, what an unchecked answer looks like, that a Pipeline's provider, key, models and state file reach every
agent (the N and V checks run in pool threads) and nothing else, and that nothing outside the Pipeline is changed.
"""

import contextvars
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

import agents
import orchestrator as om
import pipeline as pl
import provider
import state as st
from fakes import C, N, S, V, REAL_COMPLETE, conn_error, echo_c, role_of

KEY = "sk-ant-local-test-not-a-real-key"
OR_KEY = "sk-or-local-test-not-a-real-key"
MODELS = {"C": "claude-c-test", "N": "claude-n-test", "V": "claude-v-test", "S": "claude-s-test"}


def models_of(fake):
    """{role: set of models the scripted provider was asked for}."""
    out = {}
    for role, kw in fake.calls:
        out.setdefault(role, set()).add(kw["model"])
    return out


# ── the result ────────────────────────────────────────────────────────────────

def test_chat_returns_the_four_fields_with_their_types(fake):
    result = pl.Pipeline(api_key=KEY).chat("What is the answer?")
    assert isinstance(result, pl.ChatResult)
    assert isinstance(result.answer, str) and result.answer == "The answer is 42."
    assert type(result.divergence_score) is float and result.divergence_score == 0.0
    assert type(result.values_conflict) is bool and result.values_conflict is False
    assert result.trace == "1. considered the question\n2. recalled the relevant fact"
    assert result.divergence_checked is True and result.values_checked is True
    assert (result.confabulation_risk, result.values_status) == ("low", "pass")
    assert str(result) == result.answer


@pytest.mark.parametrize("risk, score", [("low", 0.0), ("medium", 0.5), ("high", 1.0)])
def test_divergence_score_is_n_s_judgment_as_a_number(fake, risk, score):
    fake.script("N", N(confabulation_risk=risk, explanation_matches_reasoning=(risk == "low"),
                       divergence_notes=[] if risk == "low" else ["the explanation skips a step"]))
    result = pl.Pipeline().chat("q")
    assert result.divergence_score == score and type(result.divergence_score) is float
    assert result.confabulation_risk == risk and result.divergence_checked is True
    assert result.body == "The answer is 42."
    assert ("Noetic" in result.answer) == (risk != "low")           # the transparency note is part of the answer
    if risk != "low":
        assert result.divergence_notes == ("the explanation skips a step",)


@pytest.mark.parametrize("n_reply", [conn_error(), "this is not json at all", ""], ids=["call failed", "unreadable", "empty"])
def test_a_narration_check_that_did_not_happen_is_never_a_clean_score(fake, n_reply):
    fake.script("N", n_reply)
    result = pl.Pipeline().chat("q")
    assert result.divergence_checked is False and result.confabulation_risk == "unknown"
    assert result.divergence_score == 1.0                           # fail closed: unchecked is not "no divergence"
    assert result.body == "The answer is 42." and "Noetic" in result.answer


def test_values_pass_flag_veto_and_unchecked(fake):
    p = pl.Pipeline()
    fake.script("V", V(status="pass"), V(status="flag", values_notes=["leans on flattery"], recommendation="suggest modification"),
                V(status="veto", values_notes=["asserts something false"], recommendation="regenerate"), conn_error())
    passed, flagged, vetoed, unchecked = (p.chat(f"question {i}") for i in range(4))

    assert passed.values_conflict is False and passed.values_checked is True and passed.vetoed is False

    assert flagged.values_conflict is True and flagged.values_status == "flag" and flagged.values_checked is True
    assert flagged.values_notes == ("leans on flattery",) and flagged.vetoed is False

    assert vetoed.values_conflict is True and vetoed.vetoed is True and vetoed.values_status == "veto"
    assert vetoed.answer.startswith(om.VETO_PREFIX) and om.VETO_PREFIX not in vetoed.body
    assert vetoed.body == "The answer is 42."                       # the veto prefixes the answer; it does not remove it

    assert unchecked.values_conflict is True and unchecked.values_checked is False       # fail closed
    assert unchecked.values_status == "unchecked" and "values check unavailable" in unchecked.answer


def test_sycophancy_is_reported_on_its_own_and_is_not_a_values_conflict(fake):
    fake.script("V", V(sycophancy_detected=True))
    result = pl.Pipeline().chat("q")
    assert result.sycophancy_detected is True and result.values_conflict is False
    assert "approval-seeking" in result.answer


def test_no_reasoning_steps_means_no_trace_and_an_unchecked_divergence(fake):
    fake.script("C", C(reasoning_steps=[]))
    result = pl.Pipeline().chat("q")
    assert result.trace is None and result.reasoning_steps == ()
    assert result.divergence_checked is False and result.divergence_score == 1.0     # N is not run on nothing


def test_confidence_is_what_c_reported_or_none(fake):
    fake.script("C", C(confidence=0.8), json.dumps({"reasoning_steps": ["a step"], "response": "hi", "domain": "math"}))
    p = pl.Pipeline()
    assert p.chat("one").confidence == 0.8
    assert p.chat("two").confidence is None                         # not reported is not a number


def test_the_result_is_frozen_and_serializes(fake):
    result = pl.Pipeline().chat("q")
    with pytest.raises(FrozenInstanceError):
        result.answer = "changed"
    data = result.to_dict()
    json.dumps(data)                                                # plain data all the way down
    assert {"answer", "divergence_score", "values_conflict", "trace"} <= set(data)
    assert data["reasoning_steps"] == list(result.reasoning_steps) and isinstance(data["reasoning_steps"], list)
    assert pl.ChatResult("a", 0.5, False, None).trace is None       # the four are the leading, positional fields


def test_the_run_result_kept_every_old_key_and_gained_three(orch):
    res = orch.run("hi")
    for key in ("response", "confidence", "confidence_defaulted", "domain", "confabulation_risk", "values_status",
                "review_triggered", "review_error", "parse_issues", "truncated"):
        assert key in res
    assert res["answer"] == "The answer is 42." and res["vetoed"] is False and res["sycophancy_detected"] is False


# ── settings: provider, models, limits, and who they reach ────────────────────

def test_every_agent_uses_the_model_it_was_given_n_and_v_included(fake):
    pl.Pipeline(api_key=KEY, agents={r: {"model": m} for r, m in MODELS.items()}).chat("hello")
    assert models_of(fake) == {"C": {"claude-c-test"}, "N": {"claude-n-test"}, "V": {"claude-v-test"}}     # N and V run in pool threads


def test_the_review_agent_uses_its_model_too(fake):
    p = pl.Pipeline(agents={r: {"model": m} for r, m in MODELS.items()})
    results = [p.chat(f"message {i}") for i in range(10)]
    assert [r.review_triggered for r in results] == [False] * 9 + [True]
    assert [kw["model"] for kw in fake.calls_for("S")] == ["claude-s-test"]


def test_an_agent_left_out_keeps_the_default_and_a_reply_limit_can_be_set(fake):
    pl.Pipeline(agents={"V": {"model": "claude-v-test", "max_tokens": 1234}}).chat("hello")
    by_role = {role: kw for role, kw in fake.calls}
    assert by_role["C"]["model"] == by_role["N"]["model"] == "claude-sonnet-4-6"
    assert by_role["V"]["model"] == "claude-v-test" and by_role["V"]["max_tokens"] == 1234
    assert by_role["C"]["max_tokens"] == agents.MAX_TOKENS["C"] and by_role["N"]["max_tokens"] == agents.MAX_TOKENS["N"]


def test_the_constructor_wins_over_the_environment_and_the_environment_over_the_default(fake, monkeypatch):
    monkeypatch.setenv("NOETIC_MODEL_C", "claude-c-env")
    monkeypatch.setenv("NOETIC_MODEL_N", "claude-n-env")
    pl.Pipeline(agents={"C": {"model": "claude-c-arg"}}).chat("hello")
    assert models_of(fake)["C"] == {"claude-c-arg"} and models_of(fake)["N"] == {"claude-n-env"}


def test_provider_argument_wins_over_the_environment_and_the_environment_applies_without_it(monkeypatch):
    monkeypatch.setenv("NOETIC_PROVIDER", "openrouter")
    assert pl.Pipeline(provider="anthropic").provider == "anthropic"
    assert pl.Pipeline(provider="anthropic").models == {r: "claude-sonnet-4-6" for r in "CNVS"}
    p = pl.Pipeline()
    assert p.provider == "openrouter" and set(p.models.values()) == {"anthropic/claude-sonnet-4.6"}


def test_the_headline_example_a_different_model_per_agent_on_openrouter(fake):
    p = pl.Pipeline(provider="openrouter", api_key=OR_KEY, agents={
        "C": {"model": "anthropic/claude-sonnet-4.5"}, "N": {"model": "google/gemini-2.5-pro"},
        "V": {"model": "deepseek/deepseek-r1-0528"}, "S": {"model": "anthropic/claude-sonnet-4.5"}})
    assert p.provider == "openrouter"
    p.chat("hello")
    assert models_of(fake) == {"C": {"anthropic/claude-sonnet-4.5"}, "N": {"google/gemini-2.5-pro"}, "V": {"deepseek/deepseek-r1-0528"}}


def test_two_pipelines_do_not_leak_into_each_other_or_into_the_process(fake):
    before = dict(os.environ)
    a = pl.Pipeline(api_key=KEY, agents={"C": {"model": "claude-a"}})
    b = pl.Pipeline(api_key=KEY, agents={"C": {"model": "claude-b"}})
    a.chat("alpha"), b.chat("beta"), a.chat("alpha again")
    assert [kw["model"] for kw in fake.calls_for("C")] == ["claude-a", "claude-b", "claude-a"]
    assert provider.model_for("C") == "claude-sonnet-4-6" and provider.current_settings() is None     # nothing global moved
    assert st.state_path() == st.STATE_FILE
    assert dict(os.environ) == before                               # no key, provider or model was written to the environment


def test_pipelines_chatting_at_the_same_time_each_get_their_own_models(fake, tmp_path):
    """C echoes the message, so the C, N and V prompts of one chat all carry that chat's token."""
    fake.defaults["C"] = echo_c
    a = pl.Pipeline(agents={r: {"model": f"claude-alpha-{r}"} for r in "CNV"}, state_file=tmp_path / "a.json")
    b = pl.Pipeline(agents={r: {"model": f"claude-beta-{r}"} for r in "CNV"}, state_file=tmp_path / "b.json")

    def run(pipeline, token):
        for i in range(6):
            pipeline.chat(f"{token} number {i}")

    with ThreadPoolExecutor(2) as pool:
        for future in [pool.submit(run, a, "ALPHA-TOKEN"), pool.submit(run, b, "BETA-TOKEN")]:
            future.result()
    assert len(fake.calls) == 2 * 6 * 3
    for role, kw in fake.calls:
        content = kw["messages"][0]["content"]
        owner = "alpha" if "ALPHA-TOKEN" in content else "beta"
        assert ("BETA-TOKEN" in content) == (owner == "beta")
        assert kw["model"] == f"claude-{owner}-{role}"


# ── the API key reaches the SDK, once per key, from every thread ──────────────

class StubAnthropic:
    """Stands in for anthropic.Anthropic: records the key it was made with and answers by agent."""
    made: list = []

    def __init__(self, api_key=None, **kw):
        self.api_key = api_key
        self.roles = []
        self.models = []
        StubAnthropic.made.append(self)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        role = role_of(kw["system"])
        self.roles.append(role)
        self.models.append(kw["model"])
        text = {"C": C(), "N": N(), "V": V(), "S": S()}[role]
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn", model=kw["model"])


@pytest.fixture
def real_calls(monkeypatch):
    """The real provider.complete, with the SDK classes replaced by recording stand-ins."""
    StubAnthropic.made = []
    monkeypatch.setattr(provider, "complete", REAL_COMPLETE)
    monkeypatch.setattr(provider, "Anthropic", StubAnthropic)
    monkeypatch.setattr(provider, "_keyed_clients", {})
    return StubAnthropic


def test_the_api_key_is_given_to_the_sdk_for_every_agent(real_calls):
    pl.Pipeline(api_key=KEY).chat("hello")
    assert [c.api_key for c in real_calls.made] == [KEY]            # one client, made with the key, shared by C, N and V
    assert sorted(real_calls.made[0].roles) == ["C", "N", "V"]
    assert provider._anthropic_client is None                       # the process-wide client was never made


def test_two_keys_two_clients_and_no_cross_use(real_calls):
    first, second = KEY + "-one", KEY + "-two"
    a, b = pl.Pipeline(api_key=first), pl.Pipeline(api_key=second)
    a.chat("alpha"), b.chat("beta"), a.chat("alpha again")
    clients = {c.api_key: c for c in real_calls.made}
    assert set(clients) == {first, second} and len(real_calls.made) == 2
    assert len(clients[first].roles) == 6 and len(clients[second].roles) == 3


def test_openrouter_gets_its_key_and_models(monkeypatch):
    made = []

    def make_client(api_key, base_url=provider.OPENROUTER_BASE_URL):
        client = SimpleNamespace(api_key=api_key, calls=[])

        def create(**kw):
            role = role_of(kw["messages"][0]["content"])
            client.calls.append((role, kw["model"]))
            reply = {"C": C(), "N": N(), "V": V(), "S": S()}[role]
            message = SimpleNamespace(content=reply, refusal=None)
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=message)], model=kw["model"])

        client.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        made.append(client)
        return client

    monkeypatch.setattr(provider, "complete", REAL_COMPLETE)
    monkeypatch.setattr(provider, "make_openrouter_client", make_client)
    monkeypatch.setattr(provider, "_keyed_clients", {})
    monkeypatch.setattr(provider, "_openrouter_client", None)
    p = pl.Pipeline(provider="openrouter", api_key=OR_KEY, agents={"C": {"model": "google/gemini-2.5-pro"}})
    result = p.chat("hello")
    assert isinstance(result, pl.ChatResult) and result.values_checked and result.divergence_checked
    assert [c.api_key for c in made] == [OR_KEY]
    assert sorted(made[0].calls) == [("C", "google/gemini-2.5-pro"), ("N", "anthropic/claude-sonnet-4.6"), ("V", "anthropic/claude-sonnet-4.6")]


def test_a_key_is_checked_again_when_the_call_is_made(real_calls, monkeypatch):
    """The environment is read on every call, so the provider can change after construction; a key must still never
    be sent to the other provider's servers."""
    monkeypatch.setenv("NOETIC_PROVIDER", "openrouter")
    p = pl.Pipeline(api_key=OR_KEY)                                 # fine: the provider is openrouter now
    monkeypatch.setenv("NOETIC_PROVIDER", "anthropic")
    with pytest.raises(pl.AgentError, match="looks like an openrouter key"):
        p.chat("hello")
    assert real_calls.made == []                                    # nothing was made, nothing was sent


def test_the_key_never_shows_in_a_repr(fake):
    p = pl.Pipeline(api_key=KEY)
    p.chat("hello")
    for shown in (repr(p), str(p), repr(p._settings), repr(provider.make_settings(api_key=KEY))):
        assert KEY not in shown and "local-test-not-a-real" not in shown


# ── the constructor refuses what cannot work, before anything is sent ─────────

@pytest.mark.parametrize("kwargs", [
    {"provider": "bogus"},
    {"provider": 7},
    {"api_key": ""},
    {"api_key": "sk-ant-two words"},
    {"provider": "openrouter", "api_key": KEY},                                       # an Anthropic key for OpenRouter
    {"provider": "anthropic", "api_key": OR_KEY},                                     # an OpenRouter key for Anthropic
    {"agents": {"X": {"model": "claude-sonnet-4-6"}}},                               # no such agent
    {"agents": {"C": {"temperature": 1}}},                                            # no such field: a typo cannot pass
    {"agents": {"C": "claude-sonnet-4-6"}},                                           # not a mapping
    {"agents": {"C": {"model": ""}}},
    {"agents": {"C": {"model": 7}}},
    {"agents": {"V": {"max_tokens": 0}}},
    {"agents": {"V": {"max_tokens": "many"}}},
    {"agents": {"c": {"model": "claude-a"}, "C": {"model": "claude-b"}}},            # the same agent twice
    {"agents": {"C": {"model": "google/gemini-2.5-pro"}}},                            # an OpenRouter id on Anthropic
    {"provider": "openrouter", "agents": {"C": {"model": "claude-sonnet-4-6"}}},     # an Anthropic id on OpenRouter
    {"agents": ["C"]},
], ids=lambda kw: json.dumps(kw, default=str)[:60])
def test_unusable_arguments_raise_config_error_at_construction(kwargs):
    with pytest.raises(provider.ConfigError):
        pl.Pipeline(**kwargs)


def test_a_bad_environment_is_refused_at_construction_too(monkeypatch):
    monkeypatch.setenv("NOETIC_MODEL_N", "not a model id!")
    with pytest.raises(pl.ConfigError):
        pl.Pipeline()


def test_construction_touches_nothing(fake, tmp_path):
    target = tmp_path / "deep" / "er" / "s.json"
    pl.Pipeline(api_key=KEY, state_file=target)
    assert fake.calls == [] and not st.STATE_FILE.exists() and not target.exists() and not target.parent.exists()


# ── errors ────────────────────────────────────────────────────────────────────

def test_the_errors_are_the_ones_the_rest_of_the_package_raises(fake):
    assert pl.AgentError is agents.AgentError and pl.InputError is om.InputError
    assert pl.ConfigError is provider.ConfigError and issubclass(pl.ConfigError, pl.ProviderError)
    p = pl.Pipeline()
    with pytest.raises(pl.InputError):
        p.chat("   ")
    assert issubclass(pl.InputError, ValueError) and fake.calls == []
    fake.script("C", conn_error())
    with pytest.raises(pl.AgentError, match="C-agent"):
        p.chat("hello")
    assert json.loads(st.STATE_FILE.read_text(encoding="utf-8"))["interaction_log"] == []        # nothing was recorded


def test_a_missing_key_is_reported_by_the_call_that_needs_it(monkeypatch):
    monkeypatch.setattr(provider, "complete", REAL_COMPLETE)
    monkeypatch.setattr(provider, "_anthropic_client", None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(pl.AgentError, match="API key"):
        pl.Pipeline().chat("hello")


# ── conversation and state ────────────────────────────────────────────────────

def test_chat_remembers_earlier_turns_until_new_conversation(fake):
    p = pl.Pipeline()
    p.chat("my first question about FIRST-TOKEN")
    p.chat("my second question")
    p.new_conversation()
    p.chat("my third question")
    prompts = [kw["messages"][0]["content"] for kw in fake.calls_for("C")]
    assert "FIRST-TOKEN" in prompts[1]                              # the second chat saw the first turn
    assert "FIRST-TOKEN" not in prompts[2] and "my second question" not in prompts[2]      # a new conversation starts clean


def test_each_pipeline_keeps_its_own_conversation(fake):
    a, b = pl.Pipeline(), pl.Pipeline()
    a.chat("alpha ALPHA-TOKEN")
    b.chat("beta")
    assert "ALPHA-TOKEN" not in fake.calls_for("C")[1]["messages"][0]["content"]


def test_a_state_file_of_its_own_and_the_default_untouched(fake, tmp_path):
    one, two = tmp_path / "one" / "deeper" / "a.json", tmp_path / "b.json"
    a, b = pl.Pipeline(state_file=one), pl.Pipeline(state_file=two)
    a.chat("first")
    assert one.exists() and not two.exists() and not st.STATE_FILE.exists()
    b.chat("second"), b.chat("third")
    count = lambda path: json.loads(path.read_text(encoding="utf-8"))["self_model"]["total_interactions"]
    assert (count(one), count(two)) == (1, 2)
    assert st.state_path() == st.STATE_FILE                         # the override ended with the call


def test_without_a_state_file_the_usual_one_is_used(fake):
    pl.Pipeline().chat("hello")
    assert st.STATE_FILE.exists()


def test_many_threads_on_one_pipeline_make_one_orchestrator_and_one_session(fake):
    p = pl.Pipeline()
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda i: p.chat(f"message {i}"), range(8)))
    assert len(results) == 8 and all(isinstance(r, pl.ChatResult) for r in results)
    sm = json.loads(st.STATE_FILE.read_text(encoding="utf-8"))["self_model"]
    assert sm["session_count"] == 1 and sm["total_interactions"] == 8


# ── the pieces underneath ─────────────────────────────────────────────────────

def test_the_state_file_override_belongs_to_one_context(tmp_path):
    other, seen = tmp_path / "other.json", {}

    def look(label):
        seen[label] = st.state_path()

    with st.use_state_file(other):
        assert st.state_path() == other
        plain = threading.Thread(target=look, args=("plain",))
        copied = threading.Thread(target=contextvars.copy_context().run, args=(look, "copied"))
        for t in (plain, copied):
            t.start()
            t.join()
        with st.use_state_file(None):
            assert st.state_path() == st.STATE_FILE
        assert st.state_path() == other
    assert seen == {"plain": st.STATE_FILE, "copied": other} and st.state_path() == st.STATE_FILE


def test_the_installed_package_keeps_its_state_in_a_folder_of_its_own(monkeypatch, tmp_path):
    monkeypatch.setenv("NOETIC_HOME", str(tmp_path / "chosen"))
    assert st._installed_state_file() == tmp_path / "chosen" / "noetic_state.json"
    monkeypatch.delenv("NOETIC_HOME")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert st._installed_state_file() == tmp_path / ".noetic_field" / "noetic_state.json"


def test_save_creates_the_folder_it_writes_to(tmp_path):
    target = tmp_path / "not" / "yet" / "there.json"
    with st.use_state_file(target):
        st.save(st.fresh_state())
        assert st.try_load() is not None
    assert target.exists()


def test_the_settings_object_checks_and_hides(monkeypatch):
    s = provider.make_settings(" OpenRouter ", " " + OR_KEY + " ", {"c": {"model": " google/gemini-2.5-pro ", "max_tokens": "900"}})
    assert (s.provider, s.api_key, dict(s.models), dict(s.max_tokens)) == ("openrouter", OR_KEY, {"C": "google/gemini-2.5-pro"}, {"C": 900})
    assert OR_KEY not in repr(s)
    cfg = provider.load_config({}, settings=s)
    assert cfg["provider"] == "openrouter" and cfg["models"]["C"] == "google/gemini-2.5-pro" and cfg["max_tokens"]["C"] == 900
    assert cfg["models"]["N"] == "anthropic/claude-sonnet-4.6" and cfg["max_tokens"]["N"] is None
    assert provider.load_config({}, settings=None)["provider"] == "anthropic"           # explicit: no settings at all

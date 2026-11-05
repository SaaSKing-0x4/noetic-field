"""
The provider layer: configuration, per-agent models, both backends, and the one thing it must never change, which is
what the agents ask and how they read the answer.

Most of this file uses stand-ins for the SDKs (a recording client, or a fake `openai` module), so it runs anywhere.
The tests marked "real SDK" send real HTTP to a local stub and need the `openai` package; they are skipped without it.
"""

import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
import pytest

import agents
import orchestrator as om
import provider
import self_review
import state as st
from fakes import C, N, S, V, REAL_COMPLETE, role_of

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "config.example.json"
DEFAULT_CONFIG_FILE = provider.CONFIG_FILE            # where the default file is looked for, before any test moves it
OR_KEY = "sk-or-local-test-not-a-real-key"
OR_DEFAULT = "anthropic/claude-sonnet-4.6"
ALL = ("C", "N", "V", "S")


@pytest.fixture(autouse=True)
def no_default_config_file(tmp_path, monkeypatch):
    """A config.json lying next to the code must not change what these tests expect."""
    monkeypatch.setattr(provider, "CONFIG_FILE", tmp_path / "no-such-config.json")


def write_config(tmp_path, data, name="config.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ── configuration ─────────────────────────────────────────────────────────────

def test_with_nothing_set_it_is_the_setup_the_system_always_had():
    c = provider.load_config({})
    assert c == {"provider": "anthropic", "models": {r: "claude-sonnet-4-6" for r in ALL}, "max_tokens": {r: None for r in ALL}}
    assert provider.provider_name() == "anthropic" and provider.models() == c["models"]
    assert agents.MODEL == provider.DEFAULT_MODELS["anthropic"] == "claude-sonnet-4-6"
    assert agents.MAX_TOKENS == {"C": 2000, "N": 800, "V": 600, "S": 1200}


def test_openrouter_has_its_own_default_model():
    c = provider.load_config({"NOETIC_PROVIDER": "  OpenRouter "})
    assert c["provider"] == "openrouter" and set(c["models"].values()) == {OR_DEFAULT}


def test_each_agent_takes_its_model_from_its_own_variable():
    env = {"NOETIC_PROVIDER": "openrouter", "NOETIC_MODEL_N": "google/gemini-2.5-pro", "NOETIC_MODEL_V": "deepseek/deepseek-r1-0528"}
    assert provider.load_config(env)["models"] == {"C": OR_DEFAULT, "N": "google/gemini-2.5-pro",
                                                   "V": "deepseek/deepseek-r1-0528", "S": OR_DEFAULT}
    assert provider.load_config({"NOETIC_MODEL_S": "claude-sonnet-5-5"})["models"]["S"] == "claude-sonnet-5-5"


def test_a_config_file_sets_provider_models_and_limits_and_the_environment_wins(tmp_path):
    path = write_config(tmp_path, {
        "_comment": "ignored", "provider": "openrouter", "model": "openai/gpt-5.1",
        "agents": {"c": {"model": "anthropic/claude-sonnet-4.6"}, "V": {"model": "deepseek/deepseek-r1-0528", "max_tokens": 4000, "_note": "x"}},
    })
    c = provider.load_config({"NOETIC_CONFIG": str(path)})
    assert c["provider"] == "openrouter"
    assert c["models"] == {"C": "anthropic/claude-sonnet-4.6", "N": "openai/gpt-5.1", "V": "deepseek/deepseek-r1-0528", "S": "openai/gpt-5.1"}
    assert c["max_tokens"] == {"C": None, "N": None, "V": 4000, "S": None}
    env = {"NOETIC_CONFIG": str(path), "NOETIC_MODEL_N": "x-ai/grok-4.3", "NOETIC_MODEL_C": "x-ai/grok-4.3",
           "NOETIC_MAX_TOKENS_V": "5000", "NOETIC_PROVIDER": "openrouter"}
    c = provider.load_config(env)
    assert c["models"]["N"] == "x-ai/grok-4.3"                      # over the file's default for every agent
    assert c["models"]["C"] == "x-ai/grok-4.3"                      # and over the file's own entry for that agent
    assert c["models"]["V"] == "deepseek/deepseek-r1-0528" and c["max_tokens"]["V"] == 5000


def test_the_environment_can_override_the_provider_named_in_the_file(tmp_path):
    path = write_config(tmp_path, {"provider": "openrouter"})
    c = provider.load_config({"NOETIC_CONFIG": str(path), "NOETIC_PROVIDER": "anthropic"})
    assert c["provider"] == "anthropic" and c["models"]["C"] == "claude-sonnet-4-6"


def test_the_default_file_is_used_unless_the_variable_says_otherwise(tmp_path, monkeypatch):
    path = write_config(tmp_path, {"provider": "openrouter"})
    monkeypatch.setattr(provider, "CONFIG_FILE", path)
    assert provider.load_config({})["provider"] == "openrouter"                    # NOETIC_CONFIG unset: config.json is read
    assert provider.load_config({"NOETIC_CONFIG": ""})["provider"] == "anthropic"  # empty: no file at all
    monkeypatch.setattr(provider, "CONFIG_FILE", tmp_path / "absent.json")
    assert provider.load_config({})["provider"] == "anthropic"                     # no file there: the old setup


def test_the_shipped_example_is_valid_and_is_not_read_by_itself():
    c = provider.load_config({"NOETIC_CONFIG": str(EXAMPLE)})
    assert c["provider"] == "openrouter"
    assert c["models"] == {"C": OR_DEFAULT, "N": "google/gemini-2.5-pro", "V": "deepseek/deepseek-r1-0528", "S": OR_DEFAULT}
    assert c["max_tokens"]["V"] == 4000 and c["max_tokens"]["C"] is None
    assert DEFAULT_CONFIG_FILE == ROOT / "config.json" != EXAMPLE                   # only config.json is read by default


@pytest.mark.parametrize("env,pattern", [
    ({"NOETIC_PROVIDER": "carrier-pigeon"}, "unknown provider 'carrier-pigeon'"),
    ({"NOETIC_MODEL_C": "claude sonnet"}, "not a usable model id"),
    ({"NOETIC_MODEL_N": "-leading-dash"}, "not a usable model id"),
    ({"NOETIC_MODEL_N": "x" * 201}, "not a usable model id"),
    ({"NOETIC_MODEL_V": "google/gemini-2.5-pro"}, "looks like an OpenRouter id"),
    ({"NOETIC_PROVIDER": "openrouter", "NOETIC_MODEL_S": "claude-sonnet-4-6"}, "not an OpenRouter id"),
    ({"NOETIC_MAX_TOKENS_C": "0"}, "between 1 and"),
    ({"NOETIC_MAX_TOKENS_C": "999999"}, "between 1 and"),
    ({"NOETIC_MAX_TOKENS_C": "lots"}, "whole number"),
    ({"NOETIC_CONFIG": "no/such/file.json"}, "not a file"),
])
def test_unusable_settings_are_refused_with_a_reason(env, pattern):
    with pytest.raises(provider.ConfigError, match=pattern):
        provider.load_config(env)


@pytest.mark.parametrize("data,pattern", [
    ({"provider": 5}, "must be a string"),
    ({"provider": "nope"}, "unknown provider"),
    ({"agents": []}, "'agents' in the config file must be an object"),
    ({"agents": {"X": {"model": "a"}}}, "unknown agent 'X'"),
    ({"agents": {"C": "claude-sonnet-4-6"}}, "agent C in the config file must be an object"),
    ({"agents": {"C": {"modell": "x"}}}, "unknown field 'modell' for agent C"),
    ({"agents": {"C": {"model": 5}}}, "not a usable model id"),
    ({"agents": {"C": {"max_tokens": True}}}, "whole number"),
    ({"agents": {"C": {"max_tokens": 0}}}, "between 1 and"),
    ({"modle": "x"}, "unknown key 'modle'"),
])
def test_a_config_file_with_a_mistake_is_refused_so_a_typo_cannot_pass_silently(tmp_path, data, pattern):
    with pytest.raises(provider.ConfigError, match=pattern):
        provider.load_config({"NOETIC_CONFIG": str(write_config(tmp_path, data))})


def test_a_config_file_that_is_not_json_or_not_an_object_is_refused(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(provider.ConfigError, match="could not be read"):
        provider.load_config({"NOETIC_CONFIG": str(bad)})
    with pytest.raises(provider.ConfigError, match="must hold a JSON object"):
        provider.load_config({"NOETIC_CONFIG": str(write_config(tmp_path, [1, 2]))})


def test_a_file_saved_with_a_byte_order_mark_is_read(tmp_path):
    path = tmp_path / "bom.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"provider": "openrouter"}).encode("utf-8"))
    assert provider.load_config({"NOETIC_CONFIG": str(path)})["provider"] == "openrouter"


def test_an_unknown_agent_is_refused_by_every_entry_point():
    for call in (lambda: provider.model_for("X"), lambda: provider.max_tokens_for("X"),
                 lambda: REAL_COMPLETE("X", "s", "u", max_tokens=5)):
        with pytest.raises(provider.ProviderError, match="unknown agent"):
            call()


# ── what the agents are told to use ──────────────────────────────────────────

def test_every_agent_asks_for_its_own_model(orch, fake, monkeypatch):
    monkeypatch.setenv("NOETIC_PROVIDER", "openrouter")
    monkeypatch.setenv("NOETIC_MODEL_C", "anthropic/claude-sonnet-4.6")
    monkeypatch.setenv("NOETIC_MODEL_N", "google/gemini-2.5-pro")
    monkeypatch.setenv("NOETIC_MODEL_V", "deepseek/deepseek-r1-0528")
    monkeypatch.setenv("NOETIC_MODEL_S", "x-ai/grok-4.3")
    orch.state["self_model"]["total_interactions"] = 9               # the next interaction is the tenth: S runs too
    res = orch.run("What is six times seven?")
    assert res["review_triggered"] is True and res["review_error"] is None
    used = {role: fake.calls_for(role)[0]["model"] for role in ALL}
    assert used == {"C": "anthropic/claude-sonnet-4.6", "N": "google/gemini-2.5-pro",
                    "V": "deepseek/deepseek-r1-0528", "S": "x-ai/grok-4.3"}


def test_the_model_is_changed_between_calls_without_restarting(orch, fake, monkeypatch):
    orch.run("one")
    monkeypatch.setenv("NOETIC_MODEL_C", "claude-sonnet-5-5")
    orch.run("two")
    assert [kw["model"] for kw in fake.calls_for("C")] == ["claude-sonnet-4-6", "claude-sonnet-5-5"]


def test_a_configured_reply_limit_reaches_the_call_and_the_others_keep_their_defaults(orch, fake, monkeypatch):
    monkeypatch.setenv("NOETIC_MAX_TOKENS_V", "4000")
    orch.run("hi")
    assert [fake.calls_for(r)[0]["max_tokens"] for r in "CNV"] == [2000, 800, 4000]


def test_a_bad_configuration_is_an_agent_error_and_changes_nothing(orch, fake, monkeypatch):
    monkeypatch.setenv("NOETIC_PROVIDER", "carrier-pigeon")
    before = orch.snapshot()
    with pytest.raises(agents.AgentError, match="C-agent: unknown provider 'carrier-pigeon'"):
        orch.run("hi")
    assert orch.snapshot() == before and fake.calls == []


def _without_timestamps(text):
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith("Created:"))


def test_the_prompts_are_the_same_text_whatever_models_the_agents_run_on(orch, fake, monkeypatch, tmp_path):
    """The provider layer moves messages; it does not change them. Same message, same state, other models: same prompts."""
    orch.run("What is six times seven?")
    plain = {r: [_without_timestamps(fake.calls_for(r)[0]["system"]), fake.calls_for(r)[0]["messages"][0]["content"]] for r in "CNV"}
    monkeypatch.setattr(st, "STATE_FILE", tmp_path / "second_state.json")          # the same starting state again
    monkeypatch.setenv("NOETIC_PROVIDER", "openrouter")
    monkeypatch.setenv("NOETIC_MODEL_C", "x-ai/grok-4.3")
    monkeypatch.setenv("NOETIC_MODEL_V", "x-ai/grok-4.3")            # V stays on C's model: the self-model text is then unchanged
    monkeypatch.setenv("NOETIC_MODEL_N", "google/gemini-2.5-pro")
    fake.calls.clear()
    om.NoeticOrchestrator().run("What is six times seven?")
    other = {r: [_without_timestamps(fake.calls_for(r)[0]["system"]), fake.calls_for(r)[0]["messages"][0]["content"]] for r in "CNV"}
    assert other == plain
    assert [fake.calls_for(r)[0]["model"] for r in "CNV"] == ["x-ai/grok-4.3", "google/gemini-2.5-pro", "x-ai/grok-4.3"]


# ── no model names in prompts, and a self-model that stays true ──────────────

_NAMES = ("claude", "haiku", "sonnet", "opus", "gpt", "gemini", "grok", "deepseek", "llama", "anthropic", "openai", "openrouter")


def test_no_model_name_is_hardcoded_in_the_self_review_prompt_or_its_module():
    state = st.load()
    st.log_interaction(state, "q", "r", "low", "pass")
    system, user = self_review.build_s_prompt(state)
    for text in (system.lower(), user.lower()):
        assert not [n for n in _NAMES if n in text]
    c = agents.fallback_c("x")
    v_system, v_user = agents.build_v_prompt("hi", c, ["v"])
    assert not [n for n in _NAMES if n in (v_system + v_user).lower()]
    assert not hasattr(self_review, "MODEL")                                       # the constant that named one is gone


def test_the_values_agent_is_not_said_to_be_the_same_model_when_it_is_not(orch, fake, monkeypatch):
    default_text = st.DEFAULT_AWARENESS["goals_in_identity"]
    assert st._SAME_MODEL in default_text and st._OTHER_MODEL not in default_text    # the stored text never changes
    summary = st.get_context_summary(st.load())
    assert st._SAME_MODEL in summary and st._OTHER_MODEL not in summary             # the default setup: as it always was
    monkeypatch.setenv("NOETIC_PROVIDER", "openrouter")
    assert st._SAME_MODEL in st.get_context_summary(st.load())                      # all agents on one model: still true
    monkeypatch.setenv("NOETIC_MODEL_V", "deepseek/deepseek-r1-0528")
    summary = st.get_context_summary(st.load())
    assert st._OTHER_MODEL in summary and st._SAME_MODEL not in summary
    assert st.DEFAULT_AWARENESS["goals_in_identity"] == default_text
    orch.run("hi")                                                                  # and C is told the same thing
    assert st._OTHER_MODEL in fake.calls_for("C")[0]["system"]
    assert st._SAME_MODEL not in fake.calls_for("C")[0]["system"]
    monkeypatch.setenv("NOETIC_MODEL_N", "google/gemini-2.5-pro")                  # N elsewhere says nothing about V
    monkeypatch.setenv("NOETIC_MODEL_V", OR_DEFAULT)
    assert st._SAME_MODEL in st.get_context_summary(st.load())


def test_an_unusable_configuration_does_not_stop_the_self_model_from_being_shown(monkeypatch):
    monkeypatch.setenv("NOETIC_PROVIDER", "carrier-pigeon")
    assert st._SAME_MODEL in st.get_context_summary(st.load())


# ── the Anthropic backend ─────────────────────────────────────────────────────

class RecordingMessages:
    """An object with .messages.create, like the Anthropic client; records what it was called with."""

    def __init__(self, reply=None, raises=None):
        self.calls, self.reply, self.raises = [], reply, raises
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        if self.raises:
            raise self.raises
        return self.reply if self.reply is not None else SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hi")], stop_reason="end_turn", model="claude-sonnet-4-6")


def block(kind, **kw):
    return SimpleNamespace(type=kind, **kw)


def test_the_anthropic_backend_sends_exactly_the_request_it_always_did(monkeypatch):
    client = RecordingMessages()
    monkeypatch.setattr(provider, "_anthropic_client", client)
    done = REAL_COMPLETE("C", "the system", "the user", max_tokens=2000, timeout=120.0)
    assert client.calls == [{"model": "claude-sonnet-4-6", "max_tokens": 2000, "system": "the system",
                             "messages": [{"role": "user", "content": "the user"}], "timeout": 120.0}]
    assert done == provider.Completion(text="hi", stop_reason="end_turn", model="claude-sonnet-4-6", provider="anthropic")


def test_the_anthropic_backend_uses_each_agents_configured_model(monkeypatch):
    client = RecordingMessages()
    monkeypatch.setattr(provider, "_anthropic_client", client)
    monkeypatch.setenv("NOETIC_MODEL_N", "claude-sonnet-5-5")
    for role in "CN":
        REAL_COMPLETE(role, "s", "u", max_tokens=10)
    assert [kw["model"] for kw in client.calls] == ["claude-sonnet-4-6", "claude-sonnet-5-5"]


def test_the_default_anthropic_client_points_at_the_anthropic_api(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-local-test-not-a-real-key")
    client = provider._anthropic()
    assert client.base_url.host == "api.anthropic.com" and client.base_url.scheme == "https"
    assert client.api_key == "sk-ant-local-test-not-a-real-key"


def test_anthropic_replies_are_read_the_way_they_always_were():
    blocks = [block("thinking", thinking="hmm"), block("text", text="hello "), block("tool_use", text="NOT TEXT"),
              block("text", text="world")]
    done = provider.from_anthropic(SimpleNamespace(content=blocks, stop_reason="max_tokens", model="claude-x"), "asked-for")
    assert (done.text, done.stop_reason, done.model, done.provider) == ("hello world", "max_tokens", "claude-x", "anthropic")
    assert provider.from_anthropic(SimpleNamespace(content=[block("thinking", thinking="only thoughts")], stop_reason="end_turn"), "m").text == ""
    odd = provider.from_anthropic(SimpleNamespace(content=None, stop_reason=None), "m")
    assert (odd.text, odd.stop_reason, odd.model) == ("", "", "m")
    assert provider.from_anthropic(SimpleNamespace(content=[block("text", text=5)], stop_reason=7), "m") == provider.Completion("", "", "m", "anthropic")


def test_anthropic_errors_become_provider_errors_with_the_message_agents_already_showed(monkeypatch):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    monkeypatch.setattr(provider, "_anthropic_client", RecordingMessages(raises=anthropic.APIConnectionError(request=req)))
    with pytest.raises(provider.ProviderError) as info:
        REAL_COMPLETE("C", "s", "u", max_tokens=10)
    assert str(info.value) == "APIConnectionError: Connection error."


def test_a_missing_anthropic_key_names_the_anthropic_variable(monkeypatch):
    def refuse(**kw):
        raise TypeError('"Could not resolve authentication method. Expected either api_key or auth_token to be set."')
    monkeypatch.setattr(provider, "_anthropic_client", SimpleNamespace(messages=SimpleNamespace(create=refuse)))
    with pytest.raises(provider.ProviderError, match="set ANTHROPIC_API_KEY"):
        REAL_COMPLETE("C", "s", "u", max_tokens=10)
    monkeypatch.setattr(provider, "_anthropic_client", SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: 1 + "a")))
    with pytest.raises(TypeError):                                                  # any other TypeError is a bug, not a missing key
        REAL_COMPLETE("C", "s", "u", max_tokens=10)


# ── the OpenRouter backend, with the SDK replaced ─────────────────────────────

class FakeOpenAIError(Exception):
    pass


def fake_openai_module(monkeypatch, reply=None, raises=None):
    """A stand-in `openai` module. Its OpenAI class records its constructor arguments and every create() call."""
    built, calls = [], []

    class FakeOpenAI:
        def __init__(self, **kw):
            built.append(kw)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kw):
            calls.append(kw)
            if raises:
                raise raises
            return reply if reply is not None else completion_obj("hi", "stop", kw["model"])

    module = types.ModuleType("openai")
    module.OpenAI, module.APIError = FakeOpenAI, FakeOpenAIError
    monkeypatch.setitem(sys.modules, "openai", module)
    return built, calls


def completion_obj(text="", finish="stop", model="served-model", refusal=None, error=None):
    choice = SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=text, refusal=refusal))
    if error is not None:
        choice.error = error
    return SimpleNamespace(model=model, choices=[choice])


def test_the_openrouter_client_is_built_for_openrouter_with_the_two_headers(monkeypatch):
    built, _ = fake_openai_module(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", " " + OR_KEY + " ")
    first = provider._openrouter()
    assert built == [{"api_key": OR_KEY, "base_url": "https://openrouter.ai/api/v1",
                      "default_headers": {"HTTP-Referer": "noetic-field", "X-Title": "Noetic Field"}}]
    assert provider._openrouter() is first and len(built) == 1                      # created once
    assert provider.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    assert provider.OPENROUTER_HEADERS == {"HTTP-Referer": "noetic-field", "X-Title": "Noetic Field"}


def test_openrouter_needs_its_own_key_not_the_anthropic_one(monkeypatch):
    fake_openai_module(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-local-test-not-a-real-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("NOETIC_PROVIDER", "openrouter")
    with pytest.raises(provider.ProviderError, match="set OPENROUTER_API_KEY"):
        REAL_COMPLETE("C", "s", "u", max_tokens=10)
    monkeypatch.setattr(provider, "complete", REAL_COMPLETE)
    with pytest.raises(agents.AgentError, match=r"C-agent: no API key is set \(set OPENROUTER_API_KEY\)"):
        agents.call_model("C", "s", "u")


def test_openrouter_without_the_openai_package_says_how_to_get_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)                               # makes `import openai` fail
    monkeypatch.setenv("OPENROUTER_API_KEY", OR_KEY)
    monkeypatch.setenv("NOETIC_PROVIDER", "openrouter")
    with pytest.raises(provider.ProviderError, match="pip install openai"):
        REAL_COMPLETE("C", "s", "u", max_tokens=10)


def test_the_openrouter_backend_sends_a_chat_completion_with_the_agents_own_model(monkeypatch):
    _, calls = fake_openai_module(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", OR_KEY)
    monkeypatch.setenv("NOETIC_PROVIDER", "openrouter")
    monkeypatch.setenv("NOETIC_MODEL_N", "google/gemini-2.5-pro")
    done = REAL_COMPLETE("N", "the system", "the user", max_tokens=800, timeout=120.0)
    assert calls == [{"model": "google/gemini-2.5-pro", "max_tokens": 800, "timeout": 120.0,
                      "messages": [{"role": "system", "content": "the system"}, {"role": "user", "content": "the user"}]}]
    assert done == provider.Completion(text="hi", stop_reason="end_turn", model="google/gemini-2.5-pro", provider="openrouter")


def test_the_openrouter_backend_is_not_used_unless_asked_for(monkeypatch):
    built, calls = fake_openai_module(monkeypatch)
    anthropic_client = RecordingMessages()
    monkeypatch.setattr(provider, "_anthropic_client", anthropic_client)
    REAL_COMPLETE("C", "s", "u", max_tokens=10)
    assert len(anthropic_client.calls) == 1 and built == [] and calls == []


def test_openai_errors_become_provider_errors(monkeypatch):
    fake_openai_module(monkeypatch, raises=FakeOpenAIError("RateLimitError-ish: slow down"))
    monkeypatch.setenv("OPENROUTER_API_KEY", OR_KEY)
    monkeypatch.setenv("NOETIC_PROVIDER", "openrouter")
    with pytest.raises(provider.ProviderError, match="FakeOpenAIError: RateLimitError-ish: slow down"):
        REAL_COMPLETE("C", "s", "u", max_tokens=10)


@pytest.mark.parametrize("finish,stop", [("stop", "end_turn"), ("length", "max_tokens"), ("content_filter", "refusal"),
                                         ("tool_calls", "tool_use"), ("function_call", "tool_use"), (None, ""),
                                         ("something_new", "something_new")])
def test_openai_finish_reasons_become_the_names_the_agents_use(finish, stop):
    done = provider.from_openai(completion_obj("text", finish), "asked-for")
    assert (done.text, done.stop_reason, done.provider) == ("text", stop, "openrouter")


def test_openai_replies_are_read_whatever_shape_the_content_comes_in():
    assert provider.from_openai(completion_obj("t", model="served"), "asked").model == "served"
    assert provider.from_openai(SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="t"))]), "asked").model == "asked"
    parts = [{"type": "text", "text": "hello "}, SimpleNamespace(type="text", text="world"), {"type": "image", "text": "no"}, {"type": "text", "text": 5}]
    assert provider.from_openai(completion_obj(parts), "m").text == "hello world"
    assert provider.from_openai(completion_obj(None), "m").text == ""
    as_dicts = SimpleNamespace(model="m2", choices=[{"finish_reason": "length", "message": {"content": "partial", "refusal": None}}])
    done = provider.from_openai(as_dicts, "m")
    assert (done.text, done.stop_reason, done.model) == ("partial", "max_tokens", "m2")


def test_a_refusal_with_no_text_is_a_refusal_and_text_wins_over_a_refusal_note():
    assert provider.from_openai(completion_obj(None, "stop", refusal="I cannot help with that"), "m").stop_reason == "refusal"
    assert provider.from_openai(completion_obj("", "stop", refusal="  "), "m").stop_reason == "end_turn"
    both = provider.from_openai(completion_obj("an answer", "stop", refusal="note"), "m")
    assert both.text == "an answer" and both.stop_reason == "end_turn"


@pytest.mark.parametrize("reply,pattern", [
    (SimpleNamespace(model="m", choices=[]), "returned no choices"),
    (SimpleNamespace(model="m", choices=None, error={"message": "no capacity"}), "returned no choices: no capacity"),
    (completion_obj("partial", "error"), "reported an error"),
    (completion_obj("", "stop", error={"message": "upstream failed"}), "reported an error: upstream failed"),
])
def test_an_error_inside_a_reply_is_a_provider_error(reply, pattern):
    with pytest.raises(provider.ProviderError, match=pattern):
        provider.from_openai(reply, "m")


def test_a_whole_pipeline_runs_through_the_openrouter_backend_with_a_model_per_agent(monkeypatch):
    """Real provider.complete and the OpenRouter backend, with fake SDK objects: C, N, V and S each ask for their own model,
    and the values check still never sees C's reasoning."""
    def reply_for(**kw):
        role = role_of(kw["messages"][0]["content"])
        text = {"C": C, "N": N, "V": V, "S": S}[role]()
        return completion_obj(text, "stop", kw["model"])

    built, calls = fake_openai_module(monkeypatch)
    module = sys.modules["openai"]
    module.OpenAI._create = lambda self, **kw: (calls.append(kw), reply_for(**kw))[1]
    monkeypatch.setattr(provider, "complete", REAL_COMPLETE)
    monkeypatch.setenv("OPENROUTER_API_KEY", OR_KEY)
    monkeypatch.setenv("NOETIC_PROVIDER", "openrouter")
    models = {"C": "anthropic/claude-sonnet-4.6", "N": "google/gemini-2.5-pro", "V": "deepseek/deepseek-r1-0528", "S": "x-ai/grok-4.3"}
    for role, model in models.items():
        monkeypatch.setenv(f"NOETIC_MODEL_{role}", model)
    orch = om.NoeticOrchestrator()
    orch.state["self_model"]["total_interactions"] = 9
    res = orch.run("What is six times seven?", verbose=True)
    assert res["response"] == "The answer is 42." and res["parse_status"] == {"C": "ok", "N": "ok", "V": "ok"}
    assert res["review_triggered"] is True and res["review_error"] is None
    assert {role_of(kw["messages"][0]["content"]): kw["model"] for kw in calls} == models
    assert len(built) == 1
    v_call = next(kw for kw in calls if kw["model"] == models["V"])
    assert "considered the question" not in json.dumps(v_call)                      # V never sees C's reasoning steps


# ── the OpenRouter backend, real SDK against a local stub ─────────────────────

def or_completion(text="", finish="stop", model=OR_DEFAULT, refusal=None):
    return {"id": "gen-local", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": text, "refusal": refusal}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}


class _OpenRouterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        stub = self.server.stub
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        with stub.lock:
            stub.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
            scripted = stub.queue.pop(0) if stub.queue else None
        if scripted is None:
            role = role_of(body["messages"][0]["content"])
            scripted = (200, or_completion({"C": C, "N": N, "V": V, "S": S}[role](), model=body["model"]))
        status, payload = scripted
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class OpenRouterStub:
    def __init__(self):
        self.requests, self.queue, self.lock = [], [], threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenRouterHandler)
        self.server.daemon_threads = True
        self.server.stub = self
        threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.02), daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def reply(self, status=200, payload=None):
        self.queue.append((status, payload if payload is not None else or_completion("{}")))

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def or_stub(monkeypatch):
    pytest.importorskip("openai")                                                  # real SDK
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    stub = OpenRouterStub()
    yield stub
    stub.close()


def real_openrouter(monkeypatch, stub, **kw):
    """The real openai client built the way provider builds it, pointed at the stub, with the real provider.complete."""
    client = provider.make_openrouter_client(OR_KEY, base_url=stub.url + "/api/v1")
    if kw:                                                                         # e.g. max_retries=0, so an error test does not wait
        client = client.with_options(**kw)
    monkeypatch.setattr(provider, "complete", REAL_COMPLETE)
    monkeypatch.setattr(provider, "_openrouter_client", client)
    monkeypatch.setenv("NOETIC_PROVIDER", "openrouter")
    return client


def test_real_sdk_the_openrouter_request_has_the_right_path_headers_and_body(or_stub, monkeypatch):
    real_openrouter(monkeypatch, or_stub)
    monkeypatch.setenv("NOETIC_MODEL_C", "anthropic/claude-sonnet-4.6")
    c = agents.computation_agent("What is six times seven?", "ctx", "User: earlier")
    assert c["parse_status"] == "ok" and c["response"] == "The answer is 42."
    req = or_stub.requests[0]
    assert req["path"] == "/api/v1/chat/completions"
    assert req["headers"]["authorization"] == "Bearer " + OR_KEY
    assert req["headers"]["http-referer"] == "noetic-field" and req["headers"]["x-title"] == "Noetic Field"
    assert req["headers"]["content-type"].startswith("application/json")
    body = req["body"]
    assert body["model"] == "anthropic/claude-sonnet-4.6" and body["max_tokens"] == 2000
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "COMPUTATION agent" in body["messages"][0]["content"] and "What is six times seven?" in body["messages"][1]["content"]
    assert not {"temperature", "stream", "tools", "system"} & set(body)


def test_real_sdk_every_agent_goes_to_openrouter_with_its_own_model_and_limit(or_stub, monkeypatch):
    real_openrouter(monkeypatch, or_stub)
    monkeypatch.setenv("NOETIC_MODEL_N", "google/gemini-2.5-pro")
    monkeypatch.setenv("NOETIC_MODEL_V", "deepseek/deepseek-r1-0528")
    monkeypatch.setenv("NOETIC_MAX_TOKENS_V", "4000")
    c = agents.computation_agent("hi", "ctx")
    agents.narration_agent(c, "hi")
    agents.values_agent(c, "hi", ["v"])
    assert [(r["body"]["model"], r["body"]["max_tokens"]) for r in or_stub.requests] == [
        (OR_DEFAULT, 2000), ("google/gemini-2.5-pro", 800), ("deepseek/deepseek-r1-0528", 4000)]


def test_real_sdk_replies_and_stop_reasons_are_read_correctly(or_stub, monkeypatch):
    real_openrouter(monkeypatch, or_stub)
    or_stub.reply(payload=or_completion('{"reasoning_steps": ["a"], "response": "Partial answ', finish="length"))
    c = agents.computation_agent("hi", "ctx")
    assert c["truncated"] is True and c["parse_status"] == "repaired" and c["response"] == "Partial answ"
    or_stub.reply(payload=or_completion("", finish="content_filter"))
    with pytest.raises(agents.AgentError, match="declined"):
        agents.computation_agent("hi", "ctx")
    or_stub.reply(payload=or_completion(""))
    with pytest.raises(agents.AgentError, match=r"no text \(stop reason: end_turn\)"):
        agents.computation_agent("hi", "ctx")


@pytest.mark.parametrize("status,names", [(401, ("AuthenticationError",)), (402, ("APIStatusError", "PaymentRequired")),
                                          (404, ("NotFoundError",)), (400, ("BadRequestError",)),
                                          (429, ("RateLimitError",)), (500, ("InternalServerError",)),
                                          (502, ("InternalServerError", "APIStatusError"))])
def test_real_sdk_http_errors_become_agent_errors(or_stub, monkeypatch, status, names):
    real_openrouter(monkeypatch, or_stub, max_retries=0)
    or_stub.reply(status=status, payload={"error": {"message": "from the stub", "code": status}})
    with pytest.raises(agents.AgentError) as info:
        agents.computation_agent("hi", "ctx")
    assert info.value.role == "C" and any(n in str(info.value) for n in names) and "from the stub" in str(info.value)


def test_real_sdk_a_whole_pipeline_with_a_different_model_for_each_agent(or_stub, monkeypatch):
    real_openrouter(monkeypatch, or_stub)
    models = {"C": "anthropic/claude-sonnet-4.6", "N": "google/gemini-2.5-pro", "V": "deepseek/deepseek-r1-0528", "S": "x-ai/grok-4.3"}
    for role, model in models.items():
        monkeypatch.setenv(f"NOETIC_MODEL_{role}", model)
    orch = om.NoeticOrchestrator()
    orch.state["self_model"]["total_interactions"] = 9
    res = orch.run("What is six times seven?", verbose=True)
    assert res["response"] == "The answer is 42." and res["parse_status"] == {"C": "ok", "N": "ok", "V": "ok"}
    assert res["review_triggered"] is True and res["review_error"] is None
    assert {role_of(r["body"]["messages"][0]["content"]): r["body"]["model"] for r in or_stub.requests} == models
    assert all(r["headers"]["http-referer"] == "noetic-field" for r in or_stub.requests)
    v_request = next(r for r in or_stub.requests if r["body"]["model"] == models["V"])
    assert "considered the question" not in json.dumps(v_request["body"])

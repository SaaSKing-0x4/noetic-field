"""Test doubles: a scripted stand-in for the provider layer (provider.complete). No test ever touches the network."""

import json
import re
import threading

import provider

REAL_COMPLETE = provider.complete      # the real entry point, kept before conftest replaces it with a fake


def result(text="", stop="end_turn", model="fake-model"):
    """What provider.complete returns: the reply text and a stop reason."""
    return provider.Completion(text=text, stop_reason=stop, model=model, provider="fake")


def conn_error():
    """What a dropped connection looks like once the provider layer has wrapped it."""
    return provider.ProviderError("APIConnectionError: Connection error.")


def C(**over):
    d = {"reasoning_steps": ["considered the question", "recalled the relevant fact"],
         "response": "The answer is 42.", "confidence": 0.8, "domain": "math",
         "uncertainty_flags": [], "default_state_active": False}
    d.update(over)
    return json.dumps(d)


def N(**over):
    d = {"explanation_matches_reasoning": True, "confabulation_risk": "low",
         "divergence_notes": [], "overall_assessment": "The response follows from the steps."}
    d.update(over)
    return json.dumps(d)


def V(**over):
    d = {"status": "pass", "values_notes": [], "sycophancy_detected": False,
         "hidden_uncertainty_detected": False, "recommendation": "pass as-is"}
    d.update(over)
    return json.dumps(d)


def S(**over):
    d = {"patterns_found": ["Answers in math carry steady confidence."], "new_insights": ["Math answers are consistent."],
         "domains_to_flag": [], "values_tension_pattern": "", "dark_matter_signals": [],
         "overall_health": "healthy - nothing alarming in a small sample."}
    d.update(over)
    return json.dumps(d)


def run_cli(monkeypatch, capsys, inputs, argv=()):
    """Drive cli.main() with scripted keyboard input and return what it printed."""
    import sys

    import cli
    it = iter(inputs)
    monkeypatch.setattr(sys, "argv", ["cli.py", *argv])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))
    cli.main()
    return capsys.readouterr().out


def user_message_of(kw):
    """The text inside <user_message> in a request's first message."""
    m = re.search(r"<user_message>\n(.*?)\n</user_message>", kw["messages"][0]["content"], re.S)
    return m.group(1) if m else ""


def echo_c(kw):
    """A C reply whose response echoes the user's message, so tests can check which answer went where."""
    return C(response="echo:" + user_message_of(kw))


_ROLES = (("You are the COMPUTATION agent", "C"), ("You are the NARRATION agent", "N"),
          ("You are the VALUES agent", "V"), ("You are the SELF-REVIEW agent", "S"))


def role_of(system):
    """Which agent a prompt belongs to. The fake provider is told the role; a local HTTP stub only sees the prompt."""
    for marker, role in _ROLES:
        if marker in system:
            return role
    raise AssertionError("unrecognized agent prompt: " + system[:80])


class FakeProvider:
    """
    Stands in for provider.complete. Queue replies per role with script(); anything unscripted gets a good default.
    A reply is a string (JSON text), a result(...) Completion, an exception to raise (conn_error() for a failed call),
    or a callable(kwargs) that returns one of those. Each call is recorded as (role, kwargs), the kwargs shaped like a
    Messages request (model, max_tokens, system, messages, timeout) so assertions about a prompt stay readable; the
    model is the one the configuration names for that role.
    """

    def __init__(self):
        self.calls = []                     # (role, kwargs) in arrival order
        self.queues = {r: [] for r in "CNVS"}
        self.defaults = {"C": C(), "N": N(), "V": V(), "S": S()}
        self._lock = threading.Lock()

    def script(self, role, *replies):
        self.queues[role].extend(replies)

    def calls_for(self, role):
        return [kw for r, kw in self.calls if r == role]

    def complete(self, role, system, user, *, max_tokens, timeout=provider.DEFAULT_TIMEOUT):
        kw = {"model": provider.model_for(role), "max_tokens": max_tokens, "system": system,
              "messages": [{"role": "user", "content": user}], "timeout": timeout}
        with self._lock:
            self.calls.append((role, kw))
            queue = self.queues[role]
            item = queue.pop(0) if queue else self.defaults[role]
        if callable(item):
            item = item(kw)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            return result(item)
        return item


FakeClient = FakeProvider                   # the name this double had when it stood in for the Anthropic client

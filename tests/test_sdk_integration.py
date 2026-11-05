"""
The real Anthropic SDK against a local stub of the Messages API: no network, no real key.

Everything else in the suite uses a scripted fake provider. These tests check the part the fake cannot: that the
Anthropic backend of provider.complete builds the request the SDK expects and reads the responses and errors the SDK
produces, and that agents.call_model turns them into what the agents use.
"""

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anthropic
import pytest

import agents
import orchestrator as om
import provider
import state as st
from fakes import C, N, S, V, REAL_COMPLETE, role_of

KEY = "sk-ant-local-test-not-a-real-key"
JSON_BY_ROLE = {"C": C, "N": N, "V": V, "S": S}


def message(text="", stop="end_turn", blocks=None):
    """The JSON a real Messages API response has."""
    content = blocks if blocks is not None else [{"type": "text", "text": text}]
    return {"id": "msg_local", "type": "message", "role": "assistant", "model": "claude-sonnet-4-6",
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 7}}


def error(kind, text):
    return {"type": "error", "error": {"type": kind, "message": text}}


class _Handler(BaseHTTPRequestHandler):
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
            role = role_of(body["system"])
            scripted = (200, message(JSON_BY_ROLE[role]()), 0, {})
        status, payload, delay, headers = scripted
        if delay:
            time.sleep(delay)
        data = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                                          # the client gave up (a timeout test)


class Stub:
    def __init__(self):
        self.requests, self.queue, self.lock = [], [], threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.daemon_threads = True
        self.server.block_on_close = False                   # do not wait for a handler that is sleeping on purpose
        self.server.stub = self
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.02), daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def reply(self, status=200, payload=None, delay=0, headers=None):
        self.queue.append((status, payload if payload is not None else message("{}"), delay, headers or {}))

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.close()


def plain_http_client():
    """An HTTP client without proxy settings from the environment, of the kind the installed SDK expects (anthropic 2
    uses httpx2, earlier versions httpx: the SDK's own DefaultHttpxClient is whichever that is)."""
    return anthropic.DefaultHttpxClient(trust_env=False)


def sdk_client(stub, **kw):
    return anthropic.Anthropic(base_url=stub.url, api_key=KEY, max_retries=kw.pop("max_retries", 0),
                               http_client=plain_http_client(), **kw)


def use_sdk(monkeypatch, client):
    """Make provider.complete the real one (conftest replaces it with a fake) and give it this SDK client."""
    monkeypatch.setattr(provider, "complete", REAL_COMPLETE)
    monkeypatch.setattr(provider, "_anthropic_client", client)
    return client


@pytest.fixture
def real(stub, monkeypatch):
    return use_sdk(monkeypatch, sdk_client(stub))


# ── the request ───────────────────────────────────────────────────────────────

def test_the_request_has_the_shape_the_api_expects(stub, real):
    c = agents.computation_agent("What is six times seven?", "ctx", "User: earlier")
    assert c["parse_status"] == "ok" and c["response"] == "The answer is 42."
    req = stub.requests[0]
    assert req["path"] == "/v1/messages"
    headers = req["headers"]
    # How the SDK delivers the key and which API version it names is the SDK's business and has changed between its
    # versions (x-api-key and 2023-06-01 in anthropic 0.x; a bearer token and a newer date in anthropic 2). What must hold:
    # the key arrives, and the request names some dated API version.
    sent_key = headers.get("x-api-key") or headers.get("authorization", "").removeprefix("Bearer ")
    assert sent_key == KEY and re.fullmatch(r"\d{4}-\d{2}-\d{2}", headers["anthropic-version"])
    assert headers["content-type"].startswith("application/json")
    body = req["body"]
    assert body["model"] == "claude-sonnet-4-6" and body["max_tokens"] == 2000
    assert isinstance(body["system"], str) and "COMPUTATION agent" in body["system"]
    assert body["messages"] == [{"role": "user", "content": body["messages"][0]["content"]}]
    assert "What is six times seven?" in body["messages"][0]["content"] and "<recent_conversation>" in body["messages"][0]["content"]
    assert "temperature" not in body and "stream" not in body and "tools" not in body


def test_every_agent_uses_its_own_token_limit(stub, real):
    c = agents.computation_agent("hi", "ctx")
    agents.narration_agent(c, "hi")
    agents.values_agent(c, "hi", ["v"])
    assert [r["body"]["max_tokens"] for r in stub.requests] == [2000, 800, 600]


# ── what comes back ───────────────────────────────────────────────────────────

def test_thinking_blocks_are_ignored_and_text_blocks_joined(stub, real):
    stub.reply(payload=message(blocks=[{"type": "thinking", "thinking": "hmm", "signature": "sig"},
                                       {"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]))
    text, truncated = agents.call_model("C", "You are the COMPUTATION agent", "u")
    assert text == "hello world" and truncated is False


def test_a_max_tokens_stop_is_reported_as_truncated(stub, real):
    stub.reply(payload=message('{"reasoning_steps": ["a"], "response": "Partial answ', stop="max_tokens"))
    c = agents.computation_agent("hi", "ctx")
    assert c["truncated"] is True and c["parse_status"] == "repaired" and c["response"] == "Partial answ"


def test_a_refusal_is_an_agent_error(stub, real):
    stub.reply(payload=message("", stop="refusal", blocks=[]))
    with pytest.raises(agents.AgentError, match="declined"):
        agents.computation_agent("hi", "ctx")


def test_an_empty_reply_is_an_agent_error(stub, real):
    stub.reply(payload=message("", blocks=[]))
    with pytest.raises(agents.AgentError, match="no text"):
        agents.computation_agent("hi", "ctx")


# ── what goes wrong ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("status,kind,name", [
    (401, "authentication_error", "AuthenticationError"),
    (403, "permission_error", "PermissionDeniedError"),
    (404, "not_found_error", "NotFoundError"),
    (400, "invalid_request_error", "BadRequestError"),
    (429, "rate_limit_error", "RateLimitError"),
    (500, "api_error", "InternalServerError"),
    (529, "overloaded_error", ("OverloadedError", "APIStatusError")),          # newer SDKs have a dedicated class
])
def test_http_errors_become_agent_errors(stub, real, status, kind, name):
    stub.reply(status=status, payload=error(kind, "from the stub"))
    with pytest.raises(agents.AgentError) as info:
        agents.computation_agent("hi", "ctx")
    names = (name,) if isinstance(name, str) else name
    assert info.value.role == "C" and any(n in str(info.value) for n in names) and "from the stub" in str(info.value)


def test_the_sdk_retries_transient_errors_and_then_succeeds(stub, monkeypatch):
    use_sdk(monkeypatch, sdk_client(stub, max_retries=2))
    stub.reply(status=429, payload=error("rate_limit_error", "slow down"), headers={"retry-after-ms": "5"})
    stub.reply(status=529, payload=error("overloaded_error", "busy"), headers={"retry-after-ms": "5"})
    c = agents.computation_agent("hi", "ctx")
    assert c["parse_status"] == "ok" and len(stub.requests) == 3


def test_a_slow_server_times_out_as_an_agent_error(stub, real, monkeypatch):
    monkeypatch.setattr(agents, "REQUEST_TIMEOUT", 0.3)
    stub.reply(delay=2.0)
    t0 = time.perf_counter()
    with pytest.raises(agents.AgentError, match="Timeout"):
        agents.computation_agent("hi", "ctx")
    assert time.perf_counter() - t0 < 1.8


def test_a_refused_connection_is_an_agent_error(stub, monkeypatch):
    port = stub.server.server_address[1]
    stub.close()                                             # nothing listens on this port any more
    dead = anthropic.Anthropic(base_url=f"http://127.0.0.1:{port}", api_key=KEY, max_retries=0,
                               http_client=plain_http_client())
    use_sdk(monkeypatch, dead)
    with pytest.raises(agents.AgentError, match="Connection"):
        agents.computation_agent("hi", "ctx")


def test_no_api_key_is_a_clear_error_with_the_real_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    use_sdk(monkeypatch, anthropic.Anthropic(base_url="http://127.0.0.1:9", http_client=plain_http_client()))
    with pytest.raises(agents.AgentError, match="API key"):
        agents.computation_agent("hi", "ctx")


# ── the whole pipeline through the SDK ────────────────────────────────────────

def test_the_full_pipeline_through_the_real_sdk(stub, real):
    orch = om.NoeticOrchestrator()
    res = orch.run("What is six times seven?", verbose=True)
    assert res["response"] == "The answer is 42." and res["parse_status"] == {"C": "ok", "N": "ok", "V": "ok"}
    roles = sorted(role_of(r["body"]["system"]) for r in stub.requests)
    assert roles == ["C", "N", "V"]
    v_request = next(r for r in stub.requests if role_of(r["body"]["system"]) == "V")
    assert "considered the question" not in json.dumps(v_request["body"])        # V never sees C's reasoning steps
    assert st.load()["self_model"]["total_interactions"] == 1


def test_the_pipeline_survives_a_server_error_in_one_check(stub, real):
    orch = om.NoeticOrchestrator()
    stub.reply(payload=message(C()))                        # C answers
    stub.reply(status=500, payload=error("api_error", "N is down"))
    stub.reply(payload=message(V()))
    res = orch.run("hi")
    assert res["response"].startswith("The answer is 42.") and res["parse_issues"] is True

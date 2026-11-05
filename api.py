"""
FastAPI HTTP server — exposes the Noetic Field orchestrator as a REST API.

Thread safety: one shared NoeticOrchestrator (singleton, locked internally).
Each session gets its own ConversationManager, passed into run(); the manager
serializes requests on its own conversation, so there is no shared mutable
conversation state between sessions. At most MAX_SESSIONS sessions are kept; the
least recently used idle one is dropped first, and a session with a request in
flight is never dropped.

Security note: bind to 127.0.0.1 in production. The /value and /insight
endpoints write to the system prompt used by all future sessions — do not
expose them on a public interface without authentication. (They are
unauthenticated by design here; that is a deployment decision, not a bug.)

Errors: a failed model call (network, refusal, empty reply) is a 502 with the
reason; an empty or over-long message, insight or value is a 400; a malformed
request or session id is a 422 whose body never repeats what was sent. Nothing is
changed in any of these cases.

Reading (GET /self, GET /audit/...): before the first chat or edit it reads the
file as it is and creates nothing (no orchestrator, no session count, no file,
no move). The first write endpoint creates the orchestrator, and that start counts
as one session. From then on a read goes through the orchestrator like every
other call: it first adopts what another process saved and, if the file was
deleted, damaged or reset while this process holds history, writes its own state
back (a damaged file is moved aside first).
"""

import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

if __package__:                 # installed (pip install noetic-field): the package's own modules
    from . import agents
    from . import audit
    from . import llm_json as lj
    from . import state as state_module
    from .conversation import ConversationManager
    from .orchestrator import InputError, NoeticOrchestrator
else:                           # run from a source folder (python cli.py, uvicorn api:app, the tests): modules side by side
    import agents
    import audit
    import llm_json as lj
    import state as state_module
    from conversation import ConversationManager
    from orchestrator import InputError, NoeticOrchestrator


app = FastAPI(
    title="Noetic Field API",
    description="Four-agent LLM pipeline with persistent self-model — Coherence Labs",
    version="1.0.0",
)

# Restrict CORS — do not allow arbitrary origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_orchestrator: Optional[NoeticOrchestrator] = None
_orchestrator_lock = threading.Lock()

# Per-session conversation history, keyed by session_id. Least recently used sessions are dropped beyond
# MAX_SESSIONS, so a client that invents a new session id per request cannot grow memory without bound.
MAX_SESSIONS = 200
_sessions: "OrderedDict[str, ConversationManager]" = OrderedDict()
_inflight: dict = {}                      # session_id -> number of requests currently using it
_sessions_lock = threading.Lock()


def get_orchestrator() -> NoeticOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        with _orchestrator_lock:
            if _orchestrator is None:
                _orchestrator = NoeticOrchestrator()
    return _orchestrator


def _touch(session_id: str) -> ConversationManager:
    """Get or create a session and mark it most recently used. Caller holds _sessions_lock."""
    conv = _sessions.get(session_id)
    if conv is None:
        conv = ConversationManager()
        _sessions[session_id] = conv
    _sessions.move_to_end(session_id)
    return conv


def _is_idle(conv: ConversationManager) -> bool:
    """True if no request is running on this conversation (its lock is free)."""
    if conv.lock.acquire(blocking=False):
        conv.lock.release()
        return True
    return False


def _evict() -> None:
    """Drop the least recently used idle sessions beyond MAX_SESSIONS. Caller holds _sessions_lock."""
    for sid in list(_sessions):
        if len(_sessions) <= MAX_SESSIONS:
            return
        if _inflight.get(sid) or not _is_idle(_sessions[sid]):
            continue
        del _sessions[sid]


def get_conversation(session_id: str) -> ConversationManager:
    with _sessions_lock:
        conv = _touch(session_id)
        _evict()
        return conv


@contextmanager
def session(session_id: str):
    """The conversation for a session, protected from eviction until the block exits."""
    with _sessions_lock:
        conv = _touch(session_id)
        _inflight[session_id] = _inflight.get(session_id, 0) + 1
        _evict()
    try:
        yield conv
    finally:
        with _sessions_lock:
            left = _inflight.get(session_id, 1) - 1
            if left > 0:
                _inflight[session_id] = left
            else:
                _inflight.pop(session_id, None)
            _evict()


# ── Request / Response models ─────────────────────────────────────────────────

_SESSION_ID = dict(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")


class ChatRequest(BaseModel):
    message: str
    session_id: str = Field(default="default", **_SESSION_ID)
    verbose: bool = False


class ChatResponse(BaseModel):
    response: str
    confidence: float
    domain: str
    confabulation_risk: str
    values_status: str
    review_triggered: bool
    session_id: str
    confidence_defaulted: bool = False
    review_error: Optional[str] = None
    parse_issues: bool = False
    truncated: bool = False
    reasoning_steps: Optional[list] = None
    uncertainty_flags: Optional[list] = None
    n_assessment: Optional[str] = None
    v_notes: Optional[list] = None
    divergence_notes: Optional[list] = None
    parse_status: Optional[dict] = None
    review_summary: Optional[str] = None
    review_applied: Optional[dict] = None


class InsightRequest(BaseModel):
    insight: str


class ValueRequest(BaseModel):
    value: str


class ResetRequest(BaseModel):
    session_id: str = Field(default="default", **_SESSION_ID)


@app.exception_handler(RequestValidationError)
async def _validation_error(request, exc):
    """FastAPI's default 422 body echoes the rejected input. That input can be anything the client sent (a number
    JSON cannot carry, raw bytes, a structure nested a thousand deep), and putting it back into a response can fail,
    turning a 422 into a 500. So the body says where and what is wrong, and never repeats what was sent."""
    detail = [{"type": str(e.get("type", "")),
               "loc": [str(part) for part in e.get("loc", ())],
               "msg": lj.scrub(str(e.get("msg", "")))} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


def _checked_text(raw: str, what: str, limit: int) -> str:
    """The text exactly as it would be stored (control characters replaced, whitespace collapsed), or a 400 if nothing
    is left of it or it is longer than `limit`. Length is measured on that stored form, as the CLI does."""
    text = lj.clean_line(raw)
    if not text:
        raise HTTPException(status_code=400, detail=f"{what} cannot be empty")
    if len(text) > limit:
        raise HTTPException(status_code=400, detail=f"{what} is too long ({len(text)} characters; the limit is {limit})")
    return text


def _peek_state() -> dict:
    """The state for a read-only endpoint. Once a request has created the orchestrator, its live state; before that,
    the file as it is (a missing or unreadable one reads as an empty state). Looking never creates the orchestrator,
    counts a session, or creates, replaces or moves the file."""
    if _orchestrator is not None:
        return _orchestrator.snapshot()
    loaded = state_module.try_load()
    return loaded if loaded is not None else state_module.fresh_state()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "noetic-field"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main conversation endpoint.

    The session's ConversationManager is retrieved (or created) and kept from eviction while the request runs; it
    is passed straight into orchestrator.run(), so no shared conversation attribute is set on the orchestrator.
    """
    orch = get_orchestrator()
    try:
        with session(req.session_id) as conv:
            result = orch.run(req.message, verbose=req.verbose, conversation=conv)
    except InputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except agents.AgentError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # conv is updated in place by run()
    return ChatResponse(
        response=result["response"],
        confidence=result["confidence"],
        confidence_defaulted=result.get("confidence_defaulted", False),
        domain=result.get("domain", ""),
        confabulation_risk=result["confabulation_risk"],
        values_status=result["values_status"],
        review_triggered=result["review_triggered"],
        review_error=result.get("review_error"),
        parse_issues=result.get("parse_issues", False),
        truncated=result.get("truncated", False),
        session_id=req.session_id,
        reasoning_steps=result.get("reasoning_steps"),
        uncertainty_flags=result.get("uncertainty_flags"),
        n_assessment=result.get("n_assessment"),
        v_notes=result.get("v_notes"),
        divergence_notes=result.get("divergence_notes"),
        parse_status=result.get("parse_status"),
        review_summary=result.get("review_summary"),
        review_applied=result.get("review_applied"),
    )


@app.get("/self")
def show_self():
    return {"self_model": state_module.get_context_summary(_peek_state())}


@app.post("/insight")
def add_insight(req: InsightRequest):
    text = _checked_text(req.insight, "insight", state_module.MAX_INSIGHT_CHARS)
    added = get_orchestrator().add_insight(text)
    return {"status": "added" if added else "ignored (repeats an insight already kept)", "insight": text}


@app.post("/value")
def add_value(req: ValueRequest):
    text = _checked_text(req.value, "value", state_module.MAX_VALUE_CHARS)
    if not get_orchestrator().add_value(text):
        raise HTTPException(
            status_code=400,
            detail="value was not added: it is a placeholder, a duplicate, or the list already holds "
                   f"{state_module.MAX_VALUES} values",
        )
    return {"status": "added", "value": text}


@app.get("/audit/{report_type}")
def run_audit(report_type: str):
    reporters = {
        "confabulation": audit.confabulation_report,
        "values": audit.values_report,
        "domains": audit.domain_report,
        "dark": audit.dark_matter_report,
        "full": audit.full_health_report,
    }
    if report_type not in reporters:
        raise HTTPException(
            status_code=404,
            detail=f"unknown report type '{report_type}'. Valid: {list(reporters.keys())}",
        )
    # Reports are built from the live state when there is one, not by re-reading the file: no file contention with
    # saves. Before the first chat or edit there is none, and the report reads the file without touching it.
    return {"report_type": report_type, "report": reporters[report_type](_peek_state())}


@app.post("/review")
def manual_review():
    try:
        review = get_orchestrator().run_manual_review()
    except agents.AgentError as e:
        raise HTTPException(status_code=502, detail=str(e))
    applied = review.get("applied") or {}
    return {
        "status": "review_complete" if not applied.get("skipped") else "review_not_applied",
        "overall_health": review.get("overall_health", ""),
        "health_label": review.get("health_label", "unknown"),
        "patterns_found": review.get("patterns_found", []),
        "new_insights": review.get("new_insights", []),
        "dark_matter_signals": review.get("dark_matter_signals", []),
        "parse_status": review.get("parse_status"),
        "applied": applied,
    }


@app.post("/reset")
def reset_conversation(req: ResetRequest):
    with _sessions_lock:
        _sessions.pop(req.session_id, None)
    return {"status": "reset", "session_id": req.session_id}


if __name__ == "__main__":
    import uvicorn
    # Bind to localhost only — do not expose prompt-injection endpoints publicly
    uvicorn.run(f"{__package__}.api:app" if __package__ else "api:app", host="127.0.0.1", port=8000, reload=True)

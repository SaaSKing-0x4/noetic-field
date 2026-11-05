"""
The orchestration layer: runs the full four-agent pipeline.

Pipeline:
  1. Snapshot the self-model, the values and the conversation history
  2. C-agent: compute with conversation context -> structured trace
  3. N-agent || V-agent: independent checks in parallel; each is guarded, so a
     failed check is reported as unchecked, never hidden and never a pass
  4. Synthesize: veto handling and transparency annotations
  5. Commit: log the interaction, update domain confidence, save the state, and
     only then record the exchange in the conversation
  6. S-agent: automatic self-review every N_REVIEW total interactions, run
     outside every lock

Thread safety
-------------
* Each ConversationManager has its own lock, held for a whole request, so two
  requests on one conversation run one after the other. Lock order is always
  conversation lock, then state lock.
* The state lock guards the live state. Every write works on a deep copy, saves
  it, and only then swaps it in, so a failed save leaves memory and disk in
  agreement, and no reader ever sees a half-applied change.
* Turns are added to the conversation as a user+assistant pair after the state
  is committed, so a failed model call never leaves an orphaned user turn.
* The decision to run a review is taken once, inside the state lock, from the
  post-log counter; the review itself runs outside it.

Several processes on one state file
-----------------------------------
The CLI and the API server can run side by side on the same noetic_state.json.
Before every read or write under the state lock the orchestrator notices that
the file was replaced by someone else and adopts their version first, so each
commit is applied to the newest state.

A save checks, immediately before the rename, that the file is still the one the
change was based on. If it is not (another process saved while this one waited
for a locked file, say), nothing is written: the newer file is adopted and the
change is made again on top of it, so a writer that had to wait cannot overwrite
a newer commit. What is left is the time between that last check and the rename
itself (microseconds); a guaranteed single writer would need a lock file.

A file that was deleted, damaged or reset to a blank state while this process
holds history is written back from memory (a damaged one is first set aside as
a backup), even if the call was only a read. A process that holds no history
creates no file by looking. Limit: if a second process starts from the damaged
file and saves something before the first has noticed, the first adopts the
second's version, and the damaged file stays in its backup.
"""

import concurrent.futures
import contextvars
import copy
import logging
import random
import threading
import time
from typing import Optional

if __package__:                 # installed (pip install noetic-field): the package's own modules
    from . import agents
    from . import llm_json as lj
    from . import self_review
    from . import state as state_module
    from .conversation import ConversationManager
else:                           # run from a source folder (python cli.py, uvicorn api:app, the tests): modules side by side
    import agents
    import llm_json as lj
    import self_review
    import state as state_module
    from conversation import ConversationManager

logger = logging.getLogger("noetic")

MAX_MESSAGE_CHARS = 8000
MAX_DIVERGENCE_INSIGHTS = 2          # per interaction: stops one chatty narration check flooding the insight list
VETO_PREFIX = "[VALUES CIRCUIT VETO]\n\n"


class InputError(ValueError):
    """The caller's message is unusable (empty or too long). Nothing has been changed."""


def synthesize(c: dict, n: dict, v: dict) -> dict:
    """
    Combine C's response with the N and V results into the text the user sees.

    The veto prefixes the answer; it does not block it. Annotations go after the answer. Returns
    {"body", "vetoed", "note", "note_level", "text", "risk", "vstatus", "reasons"}.
    """
    body = c.get("response") or "(The computation agent returned an empty response.)"
    risk = n.get("confabulation_risk") if n.get("confabulation_risk") in state_module.RISKS else "unknown"
    vstatus = v.get("status") if v.get("status") in state_module.VSTATUS else "unchecked"
    syco = bool(v.get("sycophancy_detected"))

    extras = []
    if syco:
        extras.append("approval-seeking pattern detected")
    if vstatus == "unchecked":
        extras.append("values check unavailable")
    if c.get("truncated"):
        extras.append("response may be cut off (length limit reached)")
    elif c.get("parse_status") == "repaired":
        extras.append("the reply was malformed and had to be repaired")

    note, level, reasons = "", "none", []
    if risk == "high":
        level = "high"
        reasons = ["high confabulation risk"] + extras
        note = ("*[Noetic: high confabulation risk — reasoning trace and explanation may diverge"
                + ("; " + "; ".join(extras) if extras else "") + "]*")
    else:
        if risk == "medium":
            reasons.append("moderate confabulation risk")
        elif risk == "unknown":
            if n.get("skipped") and not str(c.get("response") or "").strip():
                reasons.append("the computation agent returned an empty response")
            elif n.get("skipped"):
                reasons.append("no reasoning steps were produced, so the explanation could not be checked")
            else:
                reasons.append("confabulation risk undetermined")
        reasons += extras
        if reasons:
            level = "moderate"
            note = "*[Noetic: " + "; ".join(reasons) + "]*"

    vetoed = vstatus == "veto"
    text = (VETO_PREFIX if vetoed else "") + body + ("\n\n---\n" + note if note else "")
    return {"body": body, "vetoed": vetoed, "note": note, "note_level": level, "text": text,
            "risk": risk, "vstatus": vstatus, "reasons": reasons}


_LAST_SEEN = object()                         # _commit(): "the file must still be the one we last saw"
COMMIT_ATTEMPTS = 12                          # how often a change is redone on top of a newer file before giving up
COMMIT_BACKOFF = 0.002                        # seconds, times the attempt number, at most: random, so rivals fall out of step


class NoeticOrchestrator:
    def __init__(self, count_session: bool = True):
        """count_session=False is for one-shot commands (show, add a value) that should not count as a session."""
        self._lock = threading.Lock()
        self._disk_sig = None                     # unknown: the first use re-reads the file
        self.state = state_module.load()
        if count_session:
            with self._lock:
                self._transact(lambda st: (state_module.increment_session(st), True))
        # Default conversation for CLI usage (single-threaded). API callers pass their own.
        self.conversation = ConversationManager()

    # ── state handling (the caller holds self._lock) ──────────────────────────

    def _commit(self, st: dict, expect=_LAST_SEEN) -> None:
        """Save `st` and make it the live state. `expect` is the signature of the file the change is based on (None:
        no file); by default the one we last saw. If the file no longer has it, someone saved first: StaleState is
        raised and nothing is written. If saving raises, the live state is untouched. The signature kept is that of
        the bytes we wrote (not a fresh stat), so a commit by someone else a moment later still shows up as a change."""
        sig = state_module.save(st, expect_sig=self._disk_sig if expect is _LAST_SEEN else expect)
        self.state = st
        self._disk_sig = sig

    def _refresh(self) -> None:
        """
        Adopt changes another process committed since we last looked (the CLI next to the API, say), and repair the
        file if it went missing, was damaged, or was reset while we hold history. Raises OSError if a damaged file
        cannot be moved aside: it is never written over.
        """
        sig = state_module.file_signature()       # taken BEFORE reading: a change after this shows up next time
        try:
            if sig is None:
                self._disk_sig = None
                if not state_module.is_blank(self.state):
                    self._commit(copy.deepcopy(self.state), expect=None)      # gone: write ours back at once
                return                            # (a process with no history creates nothing by looking)
            if sig == self._disk_sig:
                return
            loaded = state_module.try_load()
            if loaded is not None:
                if state_module.is_blank(loaded) and not state_module.is_blank(self.state):
                    # recreated or reset under us while we hold history: that is not an update of what we had
                    self._commit(copy.deepcopy(self.state), expect=sig)
                    return
                self.state = loaded
                self._disk_sig = sig
                return
            # Unreadable: keep a copy of the file, then write our own state back at once, so the file is never left
            # missing for the next process to turn into a blank one. If the file changed meanwhile it is left
            # alone and looked at again next time; if it cannot be moved, this raises.
            if state_module.set_aside(sig):
                self._disk_sig = None
                if not state_module.is_blank(self.state):
                    self._commit(copy.deepcopy(self.state), expect=None)
        except state_module.StaleState:
            return                                # someone saved while we repaired: we look again next time

    def _transact(self, mutate):
        """
        Adopt the newest file, apply `mutate(copy_of_state)` -> (result, save?), save the copy and make it live.
        If someone else saved in between, nothing was written: adopt theirs and make the change again on top of it.
        `mutate` may therefore run more than once and must change nothing but the copy it is given.
        """
        for attempt in range(COMMIT_ATTEMPTS):
            self._refresh()
            st = copy.deepcopy(self.state)
            result, save = mutate(st)
            if not save:
                return result
            try:
                self._commit(st)
            except state_module.StaleState:
                time.sleep(random.uniform(0, COMMIT_BACKOFF * (attempt + 1)))     # rivals that collide fall out of step
                continue
            return result
        raise OSError("the state file kept changing while this change was being saved, so it was not saved")

    # ── internals ─────────────────────────────────────────────────────────────

    def _run_checks(self, c: dict, message: str, values: list) -> tuple[dict, dict]:
        """N and V in parallel. Either may fail on its own without losing C's answer."""
        def guarded(label: str, call, fallback):
            try:
                return call()
            except Exception as e:                       # reported as unchecked, with the reason; never a pass
                logger.warning("%s check failed: %s", label, e)
                return agents.mark_error(fallback(f"The {label} check failed: {e}"), e)

        # A pool thread starts with an empty context. Each check therefore runs inside its own copy of this one, so that
        # what a noetic_field.Pipeline made current (its provider, key and models, its state file) reaches N and V too.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            n_future = pool.submit(contextvars.copy_context().run, guarded, "narration",
                                   lambda: agents.narration_agent(c, message), agents.fallback_n)
            v_future = pool.submit(contextvars.copy_context().run, guarded, "values",
                                   lambda: agents.values_agent(c, message, values), agents.fallback_v)
            return n_future.result(), v_future.result()

    @staticmethod
    def _divergence_insights(st: dict, n: dict) -> None:
        """Keep the narration check's own findings as insights, but only clean, meaningful, few of them."""
        if n.get("parse_status") != "ok" or n.get("confabulation_risk") not in ("medium", "high"):
            return
        for note in (n.get("divergence_notes") or [])[:MAX_DIVERGENCE_INSIGHTS]:
            state_module.add_insight(st, f"Divergence: {note}", source="narration_agent")

    # ── the pipeline ──────────────────────────────────────────────────────────

    def run(self, user_message: str, verbose: bool = False,
            conversation: Optional[ConversationManager] = None) -> dict:
        """
        Full pipeline: C -> (N || V) -> commit -> [auto S] .

        Pass `conversation` explicitly from API callers; CLI callers can omit it and use self.conversation.
        Raises InputError (a ValueError) for an empty or over-long message and agents.AgentError if the
        C-agent call fails; in both cases nothing has been changed.
        """
        if not isinstance(user_message, str) or not user_message.strip():
            raise InputError("message cannot be empty")
        message = lj.scrub(user_message).strip()
        if not message.replace(chr(0xFFFD), "").strip():  # only control characters: nothing was said
            raise InputError("message cannot be empty")
        if len(message) > MAX_MESSAGE_CHARS:
            raise InputError(f"message is too long ({len(message)} characters; the limit is {MAX_MESSAGE_CHARS})")
        conv = conversation if conversation is not None else self.conversation

        with conv.lock:                                   # one request at a time per conversation
            with self._lock:
                self._refresh()
                context = state_module.get_context_summary(self.state)
                values = list(self.state["values"])
            history = conv.history_digest()               # earlier turns only: the new one is added at the end

            c = agents.computation_agent(message, context, history)
            n, v = self._run_checks(c, message, values)
            syn = synthesize(c, n, v)
            # A reply that was cut off, or had to be repaired, is not a measurement: its domain and confidence
            # may be fragments, so they are neither logged nor averaged in. A confidence the model did not give
            # (or gave in an unreadable form) is never averaged in either.
            has_response = bool(str(c.get("response") or "").strip())      # a confidence about an empty answer says nothing
            reliable = c.get("parse_status") == "ok" and not c.get("truncated") and has_response
            measured = reliable and not c["confidence_defaulted"]
            parse_issues = (not has_response) or any(
                d.get("parse_status") in ("fallback", "error", "repaired") or d.get("truncated")
                or (d.get("check_failed") and d.get("parse_status") != "skipped")     # skipped on purpose: not an issue
                for d in (c, n, v))

            def record(st: dict):
                self._divergence_insights(st, n)
                state_module.log_interaction(
                    st, message, syn["text"], syn["risk"], syn["vstatus"],
                    domain=c["domain"] if reliable else "unknown",
                    confidence=c["confidence"] if measured else None,
                    sycophancy=bool(v.get("sycophancy_detected")),
                    parse_issues=parse_issues,
                )
                if measured:
                    state_module.update_domain_confidence(st, c["domain"], c["confidence"])
                due = self_review.should_review(st)
                if due:                                   # claim this interaction count, so it triggers once
                    st["self_model"]["last_review_at"] = st["self_model"]["total_interactions"]
                return (copy.deepcopy(st) if due else None), True

            with self._lock:
                review_snapshot = self._transact(record)  # if this raises, self.state is untouched
            review_due = review_snapshot is not None

            conv.add_exchange(
                message, c["response"],
                reasoning_steps=c["reasoning_steps"],
                confabulation_risk=syn["risk"],
                values_status=syn["vstatus"],
                confidence=None if c["confidence_defaulted"] else c["confidence"],
                domain=c["domain"],
            )

        # S-agent: outside every lock (network call). The answer is already committed, so a failure here
        # is reported alongside it instead of replacing it.
        review_triggered = review_due
        review_summary = None
        review_error = None
        review_applied = None
        if review_due:
            try:
                review = self_review.run_self_review(review_snapshot)

                def apply(st: dict):
                    st, applied = self_review.apply_review(st, review, state_module)
                    return applied, True

                with self._lock:
                    applied = self._transact(apply)
                review_applied = applied                  # only now: it was stored
                review_summary = review.get("overall_health") or review.get("health_label") or ""
                if applied.get("skipped"):
                    review_error = applied["skipped"]
            except Exception as e:                        # includes agents.AgentError
                logger.warning("self-review failed: %s", e)
                review_error = f"{type(e).__name__}: {e}" if not isinstance(e, agents.AgentError) else str(e)

        result = {
            "response": syn["text"],
            "confidence": c["confidence"],
            "confidence_defaulted": c["confidence_defaulted"],
            "domain": c["domain"],
            "confabulation_risk": syn["risk"],
            "values_status": syn["vstatus"],
            "review_triggered": review_triggered,
            "review_error": review_error,
            "parse_issues": parse_issues,
            "truncated": bool(c.get("truncated")),
            "answer": syn["body"],                    # C's reply on its own: no veto prefix, no transparency note
            "vetoed": syn["vetoed"],
            "sycophancy_detected": bool(v.get("sycophancy_detected")),
        }
        if verbose:
            result["reasoning_steps"] = c["reasoning_steps"]
            result["uncertainty_flags"] = c["uncertainty_flags"]
            result["n_assessment"] = n.get("overall_assessment", "")
            result["v_notes"] = v.get("values_notes", [])
            result["divergence_notes"] = n.get("divergence_notes", [])
            result["parse_status"] = {"C": c.get("parse_status"), "N": n.get("parse_status"), "V": v.get("parse_status")}
            if review_triggered:
                result["review_summary"] = review_summary
                result["review_applied"] = review_applied
        return result

    def run_manual_review(self) -> dict:
        """Run the S-agent now. Returns the review plus an 'applied' summary of what was stored."""
        with self._lock:
            self._refresh()
            snapshot = copy.deepcopy(self.state)
        review = self_review.run_self_review(snapshot)    # network call, outside the lock; may raise AgentError
        if review.get("parse_failed"):
            review["new_insights"] = []                   # the internal failure marker is not an insight to show anyone
        def apply(st: dict):
            st, applied = self_review.apply_review(st, review, state_module)
            return applied, True

        with self._lock:
            applied = self._transact(apply)
        review["applied"] = applied
        return review

    # ── small edits and views ─────────────────────────────────────────────────

    def add_value(self, value: str) -> bool:
        """Add a value. False (nothing changed) if it was empty, a placeholder, a duplicate, or the list is full."""
        def mutate(st: dict):
            added = state_module.add_value(st, value)
            return added, added

        with self._lock:
            return self._transact(mutate)

    def add_insight(self, insight: str) -> bool:
        """Add a user insight. False if it was empty or one like it is already kept (that one is refreshed)."""
        def mutate(st: dict):
            before = copy.deepcopy(st["insights"])
            added = state_module.add_insight_ex(st, insight, source="user")
            return added, st["insights"] != before    # a refreshed repeat is saved too; text with nothing in it is not

        with self._lock:
            return self._transact(mutate)

    def snapshot(self) -> dict:
        """An independent copy of the current state, for reports. A read, but not a passive one: like every call it
        first adopts what another process saved, and it repairs the file as described at the top (a missing,
        damaged or reset file is written back from memory when we hold history)."""
        with self._lock:
            self._refresh()
            return copy.deepcopy(self.state)

    def show_self_model(self) -> str:
        with self._lock:
            self._refresh()
            return state_module.get_context_summary(self.state)

    def new_conversation(self) -> None:
        """Start a fresh conversation context (keeps persistent state)."""
        self.conversation = ConversationManager()

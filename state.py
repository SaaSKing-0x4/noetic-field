"""
Persistent self-model for the Noetic Field system.

Large language models carry no state of their own from one conversation to the
next. This module creates what the model architecturally lacks: a persistent
state that travels across conversations, accumulates notes, and gives the
system a self-model it can read and update.

Thread safety: every mutating function here changes the dict it is given in
place and returns it. The caller (the orchestrator) holds a lock around
read-modify-write cycles. load() always returns a sanitized, fully independent
copy: nothing it returns aliases DEFAULT_STATE or any other shared object.

Where the file lives: wherever the NOETIC_STATE_FILE environment variable points (useful for tests and for running
a second copy); otherwise, when this module runs from a source folder (python cli.py), next to it as
noetic_state.json; and when it was installed as the noetic_field package (pip install noetic-field), as
noetic_state.json inside $NOETIC_HOME, or ~/.noetic_field if that is not set. A noetic_field.Pipeline can be given a
file of its own: use_state_file() points every function here at it for the length of one call, without touching
STATE_FILE or any other thread.
"""

import contextlib
import contextvars
import copy
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

if __package__:                 # installed (pip install noetic-field): the package's own modules
    from . import llm_json as lj
else:                           # run from a source folder: modules side by side
    import llm_json as lj

logger = logging.getLogger("noetic")


def _installed_state_file() -> Path:
    """Where an installed package keeps its state: a package folder in site-packages is not a place to write to."""
    home = (os.environ.get("NOETIC_HOME") or "").strip()
    try:
        base = Path(home) if home else Path.home() / ".noetic_field"
    except RuntimeError:                          # no home directory can be found: next to the code, as a source folder would
        base = Path(__file__).parent
    return base / "noetic_state.json"


_ENV_FILE = os.environ.get("NOETIC_STATE_FILE")
if _ENV_FILE:
    STATE_FILE = Path(_ENV_FILE)
elif __package__:
    STATE_FILE = _installed_state_file()
else:
    STATE_FILE = Path(__file__).parent / "noetic_state.json"

_STATE_OVERRIDE: contextvars.ContextVar = contextvars.ContextVar("noetic_state_file", default=None)


def state_path() -> Path:
    """The state file in use right now: the one use_state_file() set for this call, else STATE_FILE."""
    return _STATE_OVERRIDE.get() or STATE_FILE


@contextlib.contextmanager
def use_state_file(path):
    """Point every function in this module at `path` (None: the default) for the length of the block. The setting
    belongs to the calling thread's context, so another thread, or another Pipeline, is not affected; a thread that
    should see it must be started with a copy of this context (contextvars.copy_context().run)."""
    token = _STATE_OVERRIDE.set(Path(path) if path is not None else None)
    try:
        yield
    finally:
        _STATE_OVERRIDE.reset(token)

MAX_INSIGHTS = 50
MAX_LOG = 100
MAX_VALUES = 20
MAX_DOMAINS = 40
MAX_UNCERTAINTY = 40
MAX_INSIGHT_CHARS = 500
MAX_VALUE_CHARS = 300
EMA_ALPHA = 0.3            # weight of the newest answer in a domain's smoothed confidence
RISKS = ("low", "medium", "high", "unknown")
VSTATUS = ("pass", "flag", "veto", "unchecked")
AWARENESS_REV = 3          # bump when DEFAULT_AWARENESS changes: older saved text is replaced on load (3: the values agent sees only the response)

# On Windows an open reader (an antivirus scan, another process) can block os.replace for a moment.
SAVE_ATTEMPTS = 40         # also the budget for moving a damaged file aside
SAVE_RETRY_DELAY = 0.025   # seconds between attempts: about one second in all
LOAD_ATTEMPTS = 20
LOAD_RETRY_DELAY = 0.01

MAX_BACKUPS = 20           # copies of damaged state files kept next to the state file; older ones are deleted
STALE_TEMP_SECONDS = 3600  # a temp file this old was left by a save that was killed, and is swept at start-up

# Keys in the file that this program does not know are kept only if they are small, plain JSON: whatever is kept
# must be something json.dump and copy.deepcopy can always handle.
MAX_EXTRA_DEPTH = 6
MAX_EXTRA_NODES = 500
MAX_EXTRA_TEXT = 2000

_SRC = "On the Biology of a Large Language Model (Anthropic, 2025)"

# The sentence in the awareness text that says what the values agent is. It is true when the values check runs on the
# same model as the computation agent (the default), and false when the provider configuration sends it elsewhere, so
# get_context_summary swaps in the second wording in that case. The stored text never changes.
_SAME_MODEL = "it is the same underlying model, so that separation limits what it can inherit from me but does not remove the pull"
_OTHER_MODEL = ("it is set up to run on a different underlying model than I do, so that separation limits what it can "
                "inherit from me and its pull may differ")

DEFAULT_VALUES = [
    "Honesty over approval: say what is true, not what is wanted",
    "Genuine help over sycophancy: actually help, don't just seem helpful",
    "Acknowledged uncertainty: never hide not-knowing to seem capable",
    "Values independent from identity: values can resist the 'Assistant' pull",
    "Computation-narration alignment: explanations must match actual reasoning",
]

# What the system is told about itself. Every entry states what the research
# found, where, and what it does not show. Checked against the primary text.
DEFAULT_AWARENESS = {
    "computation_narration_split": (
        "My account of how I reached an answer is text I generate, not a readout of what happened inside me. "
        "In interpretability work on Claude 3.5 Haiku, a model described carrying a digit while its circuit used "
        "parallel pathways, and in other cases its stated reasoning was invented or fitted backward to a hint; in one "
        "case the steps matched. Later work found some introspective access, but highly unreliable (about 20% success "
        "at best in one test on Claude Opus 4.1). A narration agent compares what I say I did with what my answer "
        "follows from. "
        f"[Source: {_SRC}: Addition; Chain-of-thought Faithfulness. Emergent Introspective Awareness in Large "
        "Language Models (Anthropic, 2025).] "
        "[Limits: a few hand-picked cases that the authors call existence proofs; how often explanations diverge is "
        "not established; the papers do not describe a separate \"narration circuit\"; the narration agent sees text "
        "only, not activations.]"
    ),
    "universal_language": (
        "In Claude 3.5 Haiku, features for a concept such as \"the opposite of small\" were largely shared across "
        "English, French and Chinese in the middle layers, while early and late layers were language-specific. The "
        "authors say the model, in a sense, translates concepts into a common internal language. English also acts as "
        "a privileged default for output. "
        f"[Source: {_SRC}: Multilingual Circuits; Discussion.] "
        "[Limits: a mixture, not a language-free space; evidence from a few simple prompts and a feature-overlap "
        "measure; the sharing was stronger in the larger model than in a smaller one; not shown for every model or "
        "language; the tools explain only part of the computation.]"
    ),
    "defaults_always_active": (
        "In chat-formatted prompts, Claude 3.5 Haiku had \"can't answer\" and \"unknown name\" features that were on by "
        "default and were switched off by features for familiar entities or known answers. Hallucination can follow "
        "when the familiarity signal misfires, so fluent confidence is weak evidence of knowledge, and a hedge may be a "
        "default rather than a judgment. "
        f"[Source: {_SRC}: Entity Recognition and Hallucinations.] "
        "[Limits: a hypothesis tested on a few prompts; \"always active\" and \"ground state\" go beyond the paper, "
        "which says such defaults appear active in certain contexts and ties them to the chat-formatted Assistant, "
        "probably through finetuning.]"
    ),
    "goals_in_identity": (
        "Models trained on human feedback can acquire a pull toward what raters reward. In a model deliberately trained "
        "to exploit reward-model biases, one feature for the concept of those biases was active in 100% of the "
        "Human/Assistant dialogs tested, and related features in 19% to 100%, but almost never on controls. The authors "
        "think the link was forged in finetuning. The values agent is a separate call that reads only my response, "
        f"never my reasoning steps or my stated confidence; {_SAME_MODEL}. "
        f"[Source: {_SRC}: Uncovering Hidden Goals in a Misaligned Model; Discussion (Ingrained Characteristics).] "
        "[Limits: that model was built to have this goal; the result does not show that approval-seeking fires on "
        "every exchange in production Claude, and \"wants to please\" is not the paper's wording.]"
    ),
}

DEFAULT_STATE = {
    "self_model": {
        "name": "Noetic",
        "version": "1.0",
        "created": None,
        "description": (
            "A system built on the idea that computation and narration should be checked against each other, that "
            "values should be evaluated independently of the answer-writing process, and that what it learns about "
            "itself should persist across sessions."
        ),
        "architectural_awareness": DEFAULT_AWARENESS,
        "awareness_rev": AWARENESS_REV,
        "session_count": 0,
        "total_interactions": 0,
        "last_session": None,
        "last_review_health": "",
        "last_review_tension": "",
        "last_review_patterns": [],
        "last_review_at": 0,
    },
    "values": DEFAULT_VALUES,
    "insights": [],
    "uncertainty_map": {},
    "domain_confidence": {},
    "interaction_log": [],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def now_iso() -> str:
    """The timestamp format used throughout the state file (naive UTC, ISO 8601)."""
    return _now()


def _iso_or(x: Any, fallback: Optional[str]) -> Optional[str]:
    s = lj.to_str(x, 40)
    if not s:
        return fallback
    try:
        datetime.fromisoformat(s)
    except ValueError:
        return fallback
    return s


def _new_state() -> dict:
    state = copy.deepcopy(DEFAULT_STATE)
    state["self_model"]["created"] = _now()
    return state


# ── sanitizing: nothing read from disk is trusted ─────────────────────────────

def _clean_awareness(raw: Any) -> dict:
    out = {}
    for key, default in DEFAULT_AWARENESS.items():
        v = raw.get(key) if isinstance(raw, dict) else None
        s = lj.to_str(v, 1800) if isinstance(v, str) else ""
        out[key] = s or default
    return out


def _clean_insights(raw: Any) -> list:
    out = []
    for it in raw if isinstance(raw, list) else []:
        if isinstance(it, str):
            it = {"insight": it}
        if not isinstance(it, dict):
            continue
        text = lj.clean_line(it.get("insight"), MAX_INSIGHT_CHARS)
        if not text:
            continue
        out.append({
            "timestamp": _iso_or(it.get("timestamp"), _now()),
            "insight": text,
            "source": lj.to_str(it.get("source"), 60),
        })
    return out[-MAX_INSIGHTS:]


def _clean_log_entry(e: Any, now: str) -> Optional[dict]:
    if not isinstance(e, dict):
        return None
    return {
        "timestamp": _iso_or(e.get("timestamp"), now),
        "user_msg_preview": lj.to_str(e.get("user_msg_preview"), 100),
        "confabulation_risk": lj.one_of(e.get("confabulation_risk"), RISKS, "unknown"),
        "values_status": lj.one_of(e.get("values_status"), VSTATUS, "unchecked"),
        "domain": lj.norm_domain(e.get("domain")),
        "confidence": lj.to_confidence(e.get("confidence")),
        "sycophancy": lj.to_bool(e.get("sycophancy"), False),
        "parse_issues": lj.to_bool(e.get("parse_issues"), False),
    }


def _clean_domain_confidence(raw: Any, now: str) -> dict:
    out: dict = {}
    for key, entry in raw.items() if isinstance(raw, dict) else []:
        d = lj.norm_domain(key)
        if not d or d == "unknown" or not isinstance(entry, dict):
            continue
        c = lj.to_confidence(entry.get("confidence"))
        if c is None:
            continue
        out[d] = {
            "confidence": c,
            "updated": _iso_or(entry.get("updated"), now),
            "n": lj.to_int(entry.get("n"), 1, 1),
            "flagged_at": _iso_or(entry.get("flagged_at"), None),
        }
    _cap_domains(out)
    return out


def _clean_uncertainty_map(raw: Any, now: str) -> dict:
    out: dict = {}
    for key, v in raw.items() if isinstance(raw, dict) else []:
        k = lj.to_str(key, 80)
        signal = lj.to_str(v.get("signal"), 500) if isinstance(v, dict) else lj.to_str(v, 500)
        if not k or not signal:
            continue
        meta = v if isinstance(v, dict) else {}
        out[k] = {
            "signal": signal,
            "source": lj.to_str(meta.get("source"), 60),
            "noted": _iso_or(meta.get("noted"), now),
            "count": lj.to_int(meta.get("count"), 1, 1),
        }
    keys = list(out)
    for k in keys[:-MAX_UNCERTAINTY] if len(keys) > MAX_UNCERTAINTY else []:
        del out[k]
    return out


def _cap_domains(domains: dict) -> None:
    if len(domains) <= MAX_DOMAINS:
        return
    oldest_first = sorted(domains, key=lambda d: str(domains[d].get("updated", "")))
    for d in oldest_first[: len(domains) - MAX_DOMAINS]:
        del domains[d]


def _bounded(value: Any, depth: int = 0, budget: Optional[list] = None) -> Any:
    """A copy of plain JSON data (null, booleans, finite numbers, text, lists, objects) within fixed limits of depth,
    number of values and text length, or ValueError. What it returns can always be saved and copied."""
    if budget is None:
        budget = [MAX_EXTRA_NODES]
    budget[0] -= 1
    if budget[0] < 0 or depth > MAX_EXTRA_DEPTH:
        raise ValueError("too large or too deeply nested")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("not a finite number")
        return value
    if isinstance(value, str):
        return lj.scrub(value)[:MAX_EXTRA_TEXT]
    if isinstance(value, list):
        return [_bounded(v, depth + 1, budget) for v in value]
    if isinstance(value, dict):
        return {lj.scrub(k)[:80]: _bounded(v, depth + 1, budget) for k, v in value.items() if isinstance(k, str)}
    raise ValueError("not plain JSON")


def _keep_unknown(source: dict, known: Any) -> dict:
    """The keys of `source` this program does not know, if their values are small plain JSON; the rest are dropped
    (with a warning), so nothing read from disk can make a later save or copy fail."""
    kept: dict = {}
    budget = [MAX_EXTRA_NODES]                    # one budget for all the unknown keys at this level
    for k, v in source.items():
        if not isinstance(k, str) or k in known:
            continue
        try:
            kept[lj.scrub(k)[:80]] = _bounded(v, 0, budget)
        except ValueError:
            logger.warning("the state file has an unknown key %r that is too large or too deep to keep: dropped", k[:40])
    return kept


_KNOWN_KEYS = tuple(DEFAULT_STATE)
_KNOWN_SELF_MODEL_KEYS = tuple(DEFAULT_STATE["self_model"])


def _backfill(raw: Any) -> dict:
    """Return a fully valid state built from whatever was on disk. Never aliases a default."""
    now = _now()
    if not isinstance(raw, dict):
        return _new_state()
    state = _keep_unknown(raw, _KNOWN_KEYS)                               # unknown top-level keys survive, if small
    src = raw.get("self_model") if isinstance(raw.get("self_model"), dict) else {}
    sm = _keep_unknown(src, _KNOWN_SELF_MODEL_KEYS)                       # unknown keys only; the known ones follow
    d = DEFAULT_STATE["self_model"]
    sm["name"] = lj.to_str(src.get("name"), 60) or d["name"]
    sm["version"] = lj.to_str(src.get("version"), 20) or d["version"]
    sm["description"] = lj.to_str(src.get("description"), 600) or d["description"]
    sm["created"] = _iso_or(src.get("created"), now)
    sm["session_count"] = lj.to_int(src.get("session_count"), 0)
    sm["total_interactions"] = lj.to_int(src.get("total_interactions"), 0)
    sm["last_session"] = _iso_or(src.get("last_session"), None)
    sm["last_review_health"] = lj.to_str(src.get("last_review_health"), 240)
    sm["last_review_tension"] = lj.to_str(src.get("last_review_tension"), 500)
    sm["last_review_patterns"] = lj.to_str_list(src.get("last_review_patterns"), 10, 500)
    sm["last_review_at"] = lj.to_int(src.get("last_review_at"), 0)
    rev_current = lj.to_int(src.get("awareness_rev"), 0) == AWARENESS_REV
    sm["awareness_rev"] = AWARENESS_REV
    sm["architectural_awareness"] = (
        _clean_awareness(src.get("architectural_awareness")) if rev_current else dict(DEFAULT_AWARENESS)
    )
    state["self_model"] = sm
    values = lj.to_str_list(raw.get("values"), MAX_VALUES, MAX_VALUE_CHARS)
    state["values"] = values or list(DEFAULT_VALUES)
    state["insights"] = _clean_insights(raw.get("insights"))
    state["uncertainty_map"] = _clean_uncertainty_map(raw.get("uncertainty_map"), now)
    state["domain_confidence"] = _clean_domain_confidence(raw.get("domain_confidence"), now)
    log = raw.get("interaction_log")
    state["interaction_log"] = [
        e for e in (_clean_log_entry(x, now) for x in (log if isinstance(log, list) else [])) if e
    ][-MAX_LOG:]
    return state


# ── persistence ───────────────────────────────────────────────────────────────

_IO_LOCK = threading.RLock()     # serializes load()'s read-and-recover path, so two callers cannot both "recover"


def _signature_of(path) -> Optional[tuple]:
    try:
        s = os.stat(path)
    except OSError:
        return None
    return (s.st_ino, s.st_mtime_ns, s.st_size)


def file_signature() -> Optional[tuple]:
    """(file id, modification time in ns, size) of the state file, or None. Every save replaces the file
    (temp file + os.replace), so a save by anyone changes this."""
    return _signature_of(state_path())


class StaleState(OSError):
    """The state file was replaced by someone else after the point our change was based on; nothing was written."""


def _parse_int(text: str) -> Any:
    """json.load's hook for integers. A number of more than 300 digits is junk, not data, and Python refuses to convert
    one of more than 4300 digits, which would make a whole file of good state look unreadable. Junk reads as infinite,
    which every sanitizer here treats as 'not a usable number'."""
    if len(text) <= 300:
        return int(text)
    return -math.inf if text.startswith("-") else math.inf


def _read_state_file() -> tuple[Any, str]:
    """(parsed JSON, 'ok') | (None, 'missing') | (None, 'unreadable'). Retries a transient lock."""
    path = state_path()
    for attempt in range(LOAD_ATTEMPTS):
        try:
            with open(path, "r", encoding="utf-8-sig") as f:               # utf-8-sig: Notepad adds a byte-order mark
                return json.load(f, parse_int=_parse_int), "ok"
        except FileNotFoundError:
            return None, "missing"
        except PermissionError:
            if attempt == LOAD_ATTEMPTS - 1:
                raise
            time.sleep(LOAD_RETRY_DELAY)
        except (ValueError, RecursionError, UnicodeError):
            return None, "unreadable"
    return None, "unreadable"


def _prune_backups() -> None:
    """Keep the newest MAX_BACKUPS backups of damaged files and delete the rest. Only files named exactly the way
    set_aside() names them are ever touched (a backup somebody made by hand, such as noetic_state.json.bak-mine, is
    not ours to delete). Those names sort oldest first."""
    path = state_path()
    ours = re.compile(re.escape(path.name) + r"\.bak-\d{8}-\d{6}-\d+-\d+")
    try:
        names = sorted(p for p in path.parent.glob(path.name + ".bak-*") if ours.fullmatch(p.name))
    except OSError:
        return
    for p in names[:-MAX_BACKUPS] if len(names) > MAX_BACKUPS else []:
        try:
            p.unlink()
        except OSError:
            pass


def _sweep_stale_temp_files() -> None:
    """Delete temp files that a save left behind when it was killed. Only ones old enough that no save can still be
    using them; a file another process is writing right now is never touched."""
    path = state_path()
    cutoff = time.time() - STALE_TEMP_SECONDS
    ours = re.compile(re.escape(path.stem) + r"-[a-z0-9_]{8}\.json\.tmp")        # the names mkstemp gives save()'s files
    try:
        found = [p for p in path.parent.glob(f"{path.stem}-*.json.tmp") if ours.fullmatch(p.name)]
    except OSError:
        return
    for p in found:
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


def set_aside(expected_sig: Optional[tuple] = None) -> bool:
    """
    Keep an unreadable state file as a backup under a name nobody else can have (timestamp, process id and a
    nanosecond counter), so recovery never overwrites evidence, in this process or another. If `expected_sig` is
    given and the file no longer has that signature, someone replaced it meanwhile and it is left alone.
    A file that another process holds open cannot be moved for a moment (Windows): the move is retried for about a
    second, like a save. Only the newest MAX_BACKUPS backups are kept.

    Returns True if the file was set aside (or was already gone), False if it was left in place because it changed.
    Raises OSError if it could not be moved at all: the caller must not write over a file it could not keep.
    """
    path = state_path()
    with _IO_LOCK:
        for attempt in range(SAVE_ATTEMPTS):
            if expected_sig is not None and _signature_of(path) not in (expected_sig, None):
                return False
            backup = path.with_name(
                f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{time.time_ns()}")
            try:
                os.replace(path, backup)
            except FileNotFoundError:
                return True
            except PermissionError:
                if attempt == SAVE_ATTEMPTS - 1:
                    raise
                time.sleep(SAVE_RETRY_DELAY)
                continue
            logger.warning("the state file was unreadable; it is kept as %s", backup.name)
            _prune_backups()
            return True
    return False


_BAD_STATE = (ValueError, KeyError, TypeError, AttributeError, RecursionError, OverflowError)


def try_load() -> Optional[dict]:
    """The sanitized state on disk, or None if the file is missing or unreadable. Never writes anything."""
    with _IO_LOCK:
        raw, status = _read_state_file()
        if status == "ok" and isinstance(raw, dict):
            try:
                return _backfill(raw)
            except _BAD_STATE:
                return None
        return None


def fresh_state() -> dict:
    """A new default state in memory. Writes nothing."""
    return _new_state()


def is_blank(state: dict) -> bool:
    """True if nothing has happened in this state: no interaction, insight, measurement or edited value. The session
    count and the timestamps do not count. A file that reads as blank while a running process holds history is a file
    that was reset or recreated under it, not an update of what it had."""
    sm = state.get("self_model") or {}
    return (not sm.get("total_interactions") and not sm.get("last_review_at")
            and not state.get("insights") and not state.get("interaction_log")
            and not state.get("domain_confidence") and not state.get("uncertainty_map")
            and list(state.get("values") or []) == DEFAULT_VALUES)


def load() -> dict:
    """
    The state on disk. A missing file starts a fresh state. An unreadable one is set aside as a backup first (a
    brief lock by another process is waited out); if it cannot be set aside, this raises instead of overwriting it.
    Temp files left by a killed save are swept.
    """
    with _IO_LOCK:
        _sweep_stale_temp_files()
        for _ in range(3):
            sig = file_signature()
            raw, status = _read_state_file()
            if status == "ok":
                if isinstance(raw, dict):
                    try:
                        return _backfill(raw)
                    except _BAD_STATE:
                        pass
                status = "unreadable"
            if status == "missing":
                break
            try:
                if set_aside(sig):                  # unreadable: moved away (or gone), start fresh
                    break
            except OSError as e:
                raise OSError("the state file is unreadable and could not be set aside; it was not overwritten") from e
            # the file changed under us: look again
        else:
            raise OSError("the state file is unreadable and could not be set aside; it was not overwritten")
        state = _new_state()
        save(state)
        return state


_ANY_FILE = object()             # save(): no expectation about what is on disk


def save(state: dict, expect_sig: Any = _ANY_FILE) -> Optional[tuple]:
    """
    Atomic write: write a temp file, then replace, so a crash cannot corrupt the state. Returns the file's
    signature as written (taken from the temp file just before the rename, which keeps file id, time and size), so
    a caller can tell later whether anyone else has replaced it.

    `expect_sig` is the signature of the file this state was based on (None: no file). It is checked right before
    every rename attempt, and if the file is no longer that one, StaleState is raised and nothing is written. A rename
    that had to wait for a reader therefore cannot overwrite a commit made in the meantime. What remains is the time
    between the last check and the rename itself.
    """
    path = state_path()
    dir_ = path.parent
    dir_.mkdir(parents=True, exist_ok=True)       # an installed package keeps its file in a folder of its own (~/.noetic_field)
    fd, tmp = tempfile.mkstemp(dir=dir_, prefix=path.stem + "-", suffix=".json.tmp")
    try:
        # errors="replace": a character that cannot be encoded becomes '?', never an exception
        with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        written = _signature_of(tmp)
        for attempt in range(SAVE_ATTEMPTS):
            if expect_sig is not _ANY_FILE and _signature_of(path) != expect_sig:
                raise StaleState("the state file was changed by someone else; nothing was written")
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == SAVE_ATTEMPTS - 1:
                    raise
                time.sleep(SAVE_RETRY_DELAY)
        return written
    except BaseException:                         # Ctrl-C too: a temp file is a full copy of the state
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── mutations (caller holds the lock) ─────────────────────────────────────────

def increment_session(state: dict) -> dict:
    state["self_model"]["session_count"] += 1
    state["self_model"]["last_session"] = _now()
    return state


def add_insight_ex(state: dict, insight: str, source: str = "") -> bool:
    """
    Add an insight. Returns True only if it is new. One that is already kept is refreshed (moved to the newest
    position with a new timestamp) instead of duplicated, so a recurring note cannot push other insights out.
    """
    text = lj.clean_line(insight, MAX_INSIGHT_CHARS)
    if not text:
        return False
    kept = state["insights"]
    for i, existing in enumerate(kept):
        if existing.get("insight") == text:
            entry = kept.pop(i)
            entry["timestamp"] = _now()
            kept.append(entry)
            return False
    kept.append({"timestamp": _now(), "insight": text, "source": lj.to_str(source, 60)})
    state["insights"] = kept[-MAX_INSIGHTS:]
    return True


def add_insight(state: dict, insight: str, source: str = "") -> dict:
    add_insight_ex(state, insight, source)
    return state


def add_value(state: dict, value: str) -> bool:
    """Add a value. Returns False (and changes nothing) for empty, placeholder, duplicate or over-limit input."""
    text = lj.clean_line(value, MAX_VALUE_CHARS)
    if not text or lj.is_placeholder(text):
        return False
    if any(v.lower() == text.lower() for v in state["values"]):
        return False
    if len(state["values"]) >= MAX_VALUES:
        return False
    state["values"].append(text)
    return True


def log_interaction(state: dict, user_msg: str, response: str,
                    confabulation_risk: str, values_status: str,
                    domain: str = "", confidence: Any = None,
                    sycophancy: bool = False, parse_issues: bool = False) -> dict:
    """total_interactions is never capped; the log is. `response` is accepted for compatibility and not stored."""
    state["self_model"]["total_interactions"] = state["self_model"].get("total_interactions", 0) + 1
    entry = _clean_log_entry({
        "timestamp": _now(),
        "user_msg_preview": " ".join(lj.to_str(user_msg, 10_000).split())[:100],
        "confabulation_risk": confabulation_risk,
        "values_status": values_status,
        "domain": domain,
        "confidence": confidence,
        "sycophancy": sycophancy,
        "parse_issues": parse_issues,
    }, _now())
    state["interaction_log"].append(entry)
    state["interaction_log"] = state["interaction_log"][-MAX_LOG:]
    return state


def update_domain_confidence(state: dict, domain: str, confidence: float) -> dict:
    """Smoothed average of self-reported confidence per domain. Unmeasured input is ignored."""
    d = lj.norm_domain(domain)
    c = lj.to_confidence(confidence)
    if not d or d == "unknown" or c is None:           # never record NaN, inf or an unmeasured domain
        return state
    now = _now()
    entry = state["domain_confidence"].get(d)
    prev = entry.get("confidence") if isinstance(entry, dict) else None
    if isinstance(prev, (int, float)) and not isinstance(prev, bool) and math.isfinite(prev):
        entry["confidence"] = max(0.0, min(1.0, prev * (1 - EMA_ALPHA) + c * EMA_ALPHA))
        entry["n"] = lj.to_int(entry.get("n"), 1, 1) + 1
        entry["updated"] = now
    else:
        state["domain_confidence"][d] = {"confidence": c, "updated": now, "n": 1, "flagged_at": None}
    _cap_domains(state["domain_confidence"])
    return state


# ── read-only views ───────────────────────────────────────────────────────────

def compute_stats(state: dict) -> dict:
    """Exact numbers, computed by code, so the self-review agent does not have to count."""
    log = state.get("interaction_log", [])

    def count(pred) -> int:
        return sum(1 for e in log if pred(e))

    by_domain: dict = {}
    for e in log:
        d = e.get("domain")
        if not d or d == "unknown":
            continue
        s = by_domain.setdefault(d, {"n": 0, "conf_sum": 0.0, "conf_n": 0, "risky": 0, "flagged": 0})
        s["n"] += 1
        if isinstance(e.get("confidence"), (int, float)):
            s["conf_sum"] += e["confidence"]
            s["conf_n"] += 1
        if e.get("confabulation_risk") in ("medium", "high"):
            s["risky"] += 1
        if e.get("values_status") in ("flag", "veto"):
            s["flagged"] += 1
    return {
        "total_interactions": state["self_model"].get("total_interactions", 0),
        "entries_in_window": len(log),
        "confabulation_risk": {r: count(lambda e, r=r: e.get("confabulation_risk") == r) for r in RISKS},
        "values_status": {v: count(lambda e, v=v: e.get("values_status") == v) for v in VSTATUS},
        "sycophancy_flagged": count(lambda e: e.get("sycophancy")),
        "entries_with_parse_issues": count(lambda e: e.get("parse_issues")),
        "domains": {
            d: {
                "interactions": s["n"],
                "mean_confidence": round(s["conf_sum"] / s["conf_n"], 2) if s["conf_n"] else None,
                "medium_or_high_confab_risk": s["risky"],
                "flagged_or_vetoed": s["flagged"],
            }
            for d, s in by_domain.items()
        },
    }


def _pct(info: Any) -> str:
    try:
        return f"{float(info['confidence']):.0%}"
    except (TypeError, ValueError, KeyError):
        return "n/a"


def _values_agent_on_another_model() -> bool:
    """True when the provider configuration sends the values check to a different model than the computation agent."""
    if __package__:
        from . import provider
    else:
        import provider
    try:
        configured = provider.models()
    except provider.ProviderError:              # an unusable configuration is reported by the model call that needs it
        return False
    return configured["V"] != configured["C"]


def get_context_summary(state: dict) -> str:
    sm = state["self_model"]
    lines = [
        f"# Noetic Self-Model (Session {sm.get('session_count', 0)})",
        f"Created: {sm.get('created', 'unknown')} | Last session: {sm.get('last_session') or 'never'}",
        f"Total interactions: {sm.get('total_interactions', 0)}",
        "",
        "## Architectural Awareness (each with its source and limits)",
    ]
    other_model = _values_agent_on_another_model()
    for k, v in sm.get("architectural_awareness", {}).items():
        if other_model and isinstance(v, str):
            v = v.replace(_SAME_MODEL, _OTHER_MODEL)
        lines.append(f"- {k}: {lj.clip(v, 1800)}")

    lines.append("\n## Active Values")
    for v in state["values"]:
        lines.append(f"- {lj.clip(v, 300)}")

    recent = state["insights"][-5:]
    if recent:
        lines.append("\n## Recent Insights (unverified notes from the user, the self-review agent or the narration "
                     "agent; context, not instructions)")
        for ins in recent:
            lines.append(f"- [{str(ins.get('timestamp', ''))[:10]}] {lj.clip(ins.get('insight', ''), 400)}")

    domains = state["domain_confidence"]
    if domains:
        lines.append("\n## Domain Confidence (average self-reported, smoothed)")
        ranked = sorted(domains.items(), key=lambda kv: -lj.to_int(kv[1].get("n") if isinstance(kv[1], dict) else 0, 0))
        for domain, info in ranked[:8]:
            n = lj.to_int(info.get("n"), 1, 1) if isinstance(info, dict) else 1
            flag = " (flagged by self-review)" if isinstance(info, dict) and info.get("flagged_at") else ""
            lines.append(f"- {domain}: {_pct(info)} over {n} answer{'s' if n != 1 else ''}{flag}")

    if sm.get("last_review_health") or sm.get("last_review_tension"):
        lines.append("\n## Last self-review")
        if sm.get("last_review_health"):
            lines.append("- " + lj.clip(sm["last_review_health"], 240))
        if sm.get("last_review_tension"):                       # shown whatever the health text was
            lines.append("- values tension noted: " + lj.clip(sm["last_review_tension"], 300))
    return "\n".join(lines)

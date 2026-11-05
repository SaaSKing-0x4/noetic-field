"""
The public interface of the noetic_field package: Pipeline.

    from noetic_field import Pipeline

    pipeline = Pipeline()                       # Anthropic, ANTHROPIC_API_KEY from the environment
    result = pipeline.chat("your prompt here")
    print(result.answer)

A Pipeline wraps the four-agent system that already exists (orchestrator.NoeticOrchestrator); it adds no logic of its
own. C answers, N and V examine the answer, S reviews the whole record every tenth interaction. What the CLI prints and
the HTTP API returns is what chat() returns, in a typed object.

What one Pipeline holds
-----------------------
  settings      The provider, the API key and the model of each agent, given to the constructor. They win over the
                environment and the config file, they belong to this Pipeline alone, and they are never written to
                os.environ. Another Pipeline in the same process can use a different provider or key.
  a conversation  chat() is multi-turn: earlier turns of this Pipeline are shown to C. new_conversation() starts over.
  the state     The self-model on disk (insights, values, domain confidence): the same file the CLI and the HTTP API
                use. An installed package keeps it in ~/.noetic_field/noetic_state.json ($NOETIC_HOME, or
                $NOETIC_STATE_FILE, or state_file= choose another place). The file is first read, and counted as one
                session, on the first chat(); constructing a Pipeline touches no file and makes no network call.

chat() blocks until the answer is checked; it is safe to call from several threads (the conversation serializes them).
"""

import contextlib
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

if __package__:                 # installed (pip install noetic-field): the package's own modules
    from . import agents as _agents
    from . import provider as _provider
    from . import state as _state
    from .conversation import ConversationManager
    from .orchestrator import InputError, NoeticOrchestrator
else:                           # run from a source folder (the tests): modules side by side
    import agents as _agents
    import provider as _provider
    import state as _state
    from conversation import ConversationManager
    from orchestrator import InputError, NoeticOrchestrator

AgentError = _agents.AgentError
ConfigError = _provider.ConfigError
ProviderError = _provider.ProviderError

# N's judgment of whether the explanation matches the reasoning, as a number. N reports a label, not a measurement,
# so the numbers are a fixed reading of the label and say nothing finer than it does.
DIVERGENCE_SCORES = {"low": 0.0, "medium": 0.5, "high": 1.0}


@dataclass(frozen=True)
class ChatResult:
    """What chat() returns. The first four fields are the interface; the rest is the detail behind them.

    answer             The final answer exactly as the pipeline presents it: C's reply, with the "[VALUES CIRCUIT
                       VETO]" prefix if V vetoed it and the "*[Noetic: ...]*" note if N or V had something to say.
                       The CLI prints this and the HTTP API returns it as "response".
    divergence_score   How far C's explanation drifts from its own reasoning, 0.0 to 1.0, from N's judgment: low 0.0,
                       medium 0.5, high 1.0. If N could not run or its reply could not be read, the score is 1.0 and
                       divergence_checked is False: an unchecked answer is never reported as a clean one.
    values_conflict    True when V flagged or vetoed the answer, or when V could not check it (values_checked is
                       False). False only when V looked and passed it.
    trace              C's reasoning steps, numbered, one per line; None when C gave none.
    """
    answer: str
    divergence_score: float
    values_conflict: bool
    trace: Optional[str] = None
    # the detail
    body: str = ""                          # C's reply alone: no veto prefix, no note
    confabulation_risk: str = "unknown"     # N: low | medium | high | unknown
    divergence_checked: bool = False        # False: N did not run, failed, or could not be read
    values_status: str = "unchecked"        # V: pass | flag | veto | unchecked
    values_checked: bool = False            # False: V failed, was cut off, or could not be read
    vetoed: bool = False
    sycophancy_detected: bool = False       # V saw the answer aiming to please rather than to be right
    confidence: Optional[float] = None      # what C reported about its own answer (0.0 to 1.0); None: it reported none
    domain: str = "unknown"
    truncated: bool = False                 # C hit its length limit: the answer may be cut off
    parse_issues: bool = False              # some agent's reply was malformed, cut off or missing
    reasoning_steps: tuple = ()
    divergence_notes: tuple = ()            # where N saw the explanation depart from the steps
    values_notes: tuple = ()
    review_triggered: bool = False          # S ran after this interaction
    review_error: Optional[str] = None      # S ran and failed or stored nothing (the answer is unaffected)

    def __str__(self) -> str:
        return self.answer

    def to_dict(self) -> dict:
        """The fields as plain JSON-ready data (tuples become lists)."""
        d = asdict(self)
        for key in ("reasoning_steps", "divergence_notes", "values_notes"):
            d[key] = list(d[key])
        return d


def _texts(value: Any) -> tuple:
    return tuple(str(x) for x in value) if isinstance(value, (list, tuple)) else ()


def to_result(res: dict) -> ChatResult:
    """A NoeticOrchestrator.run(..., verbose=True) result as a ChatResult."""
    risk = res.get("confabulation_risk")
    risk = risk if risk in DIVERGENCE_SCORES else "unknown"
    status = res.get("values_status")
    status = status if status in ("pass", "flag", "veto") else "unchecked"
    steps = _texts(res.get("reasoning_steps"))
    confidence = None if res.get("confidence_defaulted") else res.get("confidence")
    return ChatResult(
        answer=res["response"],
        divergence_score=DIVERGENCE_SCORES.get(risk, 1.0),
        values_conflict=status != "pass",
        trace="\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)) if steps else None,
        body=res.get("answer", res["response"]),
        confabulation_risk=risk,
        divergence_checked=risk in DIVERGENCE_SCORES,
        values_status=status,
        values_checked=status != "unchecked",
        vetoed=bool(res.get("vetoed")),
        sycophancy_detected=bool(res.get("sycophancy_detected")),
        confidence=float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else None,
        domain=str(res.get("domain") or "unknown"),
        truncated=bool(res.get("truncated")),
        parse_issues=bool(res.get("parse_issues")),
        reasoning_steps=steps,
        divergence_notes=_texts(res.get("divergence_notes")),
        values_notes=_texts(res.get("v_notes")),
        review_triggered=bool(res.get("review_triggered")),
        review_error=res.get("review_error"),
    )


class Pipeline:
    """The four-agent pipeline as a library.

    provider   "anthropic" or "openrouter". Default: NOETIC_PROVIDER or the config file, else "anthropic". OpenRouter
               is reached through the `openai` package (pip install "noetic-field[openrouter]").
    api_key    The key for that provider. Default: ANTHROPIC_API_KEY or OPENROUTER_API_KEY from the environment. A key
               that plainly belongs to the other provider (sk-ant-... for OpenRouter, sk-or-... for Anthropic) is
               refused before anything is sent.
    agents     The model (and, optionally, the reply limit) of each agent:
                   {"C": {"model": "anthropic/claude-sonnet-4.5"}, "V": {"model": "deepseek/deepseek-r1-0528",
                    "max_tokens": 4000}}
               An agent left out uses NOETIC_MODEL_<agent> or the config file, else the provider's default model
               (claude-sonnet-4-6 on Anthropic, anthropic/claude-sonnet-4.6 on OpenRouter). A model id must suit the
               provider: Anthropic ids have no slash; OpenRouter ids are vendor/model, with dots in the version.
    state_file Where the self-model is kept; see the module note. Default: the usual place.

    Raises ConfigError at once for anything unusable (an unknown provider or agent, a model id that does not suit the
    provider, a bad config file, a key for the other provider). Nothing is read from disk or sent anywhere before the
    first chat().
    """

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None,
                 agents: Optional[Mapping[str, Mapping[str, Any]]] = None, *,
                 state_file: Union[str, "os.PathLike[str]", None] = None):
        self._settings = _provider.make_settings(provider, api_key, agents)
        _provider.validate(self._settings)
        self._state_file = Path(state_file).expanduser() if state_file is not None else None
        self._conversation = ConversationManager()
        self._orchestrator: Optional[NoeticOrchestrator] = None
        self._init_lock = threading.Lock()

    @contextlib.contextmanager
    def _current(self):
        """Make this Pipeline's settings and state file the current ones, in this thread, for one call."""
        with _provider.scoped(self._settings), _state.use_state_file(self._state_file):
            yield

    def _orch(self) -> NoeticOrchestrator:
        with self._init_lock:
            if self._orchestrator is None:
                self._orchestrator = NoeticOrchestrator()      # reads the state file; counts this Pipeline as a session
            return self._orchestrator

    def chat(self, prompt: str) -> ChatResult:
        """Send one message through C, then N and V, and return the checked answer.

        Raises InputError (a ValueError) for an empty or over-long prompt, AgentError if the answering model could not
        be reached or returned nothing (nothing has been changed), and OSError if the state file cannot be read or
        written. A failure of N or V alone is not an error: it shows up as divergence_checked / values_checked False."""
        with self._current():
            result = self._orch().run(prompt, verbose=True, conversation=self._conversation)
        return to_result(result)

    def new_conversation(self) -> None:
        """Forget the earlier turns of this Pipeline. The self-model on disk is kept."""
        self._conversation = ConversationManager()

    @property
    def provider(self) -> str:
        """The provider chat() will use now."""
        with self._current():
            return _provider.provider_name()

    @property
    def models(self) -> dict:
        """The model each agent will use now: {"C": ..., "N": ..., "V": ..., "S": ...}."""
        with self._current():
            return _provider.models()

    def __repr__(self) -> str:                  # never shows the key
        try:
            return f"Pipeline(provider={self.provider!r}, models={self.models!r})"
        except _provider.ProviderError:
            return "Pipeline(<configuration unusable>)"

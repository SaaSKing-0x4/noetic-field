"""
Provider layer: where a model call goes and how its reply is read. A transport only.

Every agent (C, N, V, S) makes its model call through complete(role, system, user, max_tokens=..., timeout=...).
Nothing here decides what an agent asks or what its answer means: the prompts, the parsers, the checks, the state and
the pipeline are the same whichever provider answers. With no environment variable set and no config file, the call is
exactly the one this system always made: the Anthropic SDK, api.anthropic.com, claude-sonnet-4-6.

Two backends
------------
  anthropic   (the default)  The Anthropic SDK against api.anthropic.com. Needs ANTHROPIC_API_KEY.
  openrouter                 The `openai` SDK pointed at https://openrouter.ai/api/v1, sending the headers
                             HTTP-Referer: noetic-field and X-Title: Noetic Field. Needs OPENROUTER_API_KEY. Any model
                             OpenRouter routes (Anthropic, OpenAI, Google, DeepSeek, xAI, ...) can serve any agent.
                             The `openai` package is imported only when this backend is first used.

Configuration (read on every call, so changing it needs no code change; the environment wins over the file)
------------------------------------------------------------------------------------------------------------
  NOETIC_PROVIDER         anthropic | openrouter                      unset: anthropic
  NOETIC_MODEL_C|N|V|S    the model one agent uses                    unset: the file, then the provider's default
  NOETIC_MAX_TOKENS_C|N|V|S  an agent's reply limit                   unset: the file, then agents.MAX_TOKENS
  NOETIC_CONFIG           path of a JSON file (see config.example.json)
                          unset: config.json next to this module, if it exists; empty: no file at all

The file:  {"provider": "openrouter", "model": "<default for all agents>",
            "agents": {"C": {"model": "...", "max_tokens": 2000}, "N": {...}, "V": {...}, "S": {...}}}
Keys that start with an underscore are comments. Anything else it does not know is an error, so a typo cannot pass
silently. A model id must suit its provider: Anthropic ids have no slash (claude-sonnet-4-6); OpenRouter ids are
vendor/model with dots in the version (anthropic/claude-sonnet-4.6). A wrong pairing is refused before a request is sent.

One Pipeline's own settings (noetic_field.Pipeline)
---------------------------------------------------
A Pipeline can be given a provider, an API key and a model per agent. They are kept in a Settings object and made
current for the length of one call with scoped(): they win over the environment and the file, they are held in a
context variable (so another thread, or another Pipeline, never sees them), nothing is written to os.environ, and an
API key given this way gets a client of its own. A thread started inside a call must be started with a copy of the
context (contextvars.copy_context().run) to see them; the orchestrator does that for the N and V checks.

What this layer does not do: it does not read, rewrite or add to a prompt, and it assumes nothing about which model is
answering. The model name it was told to use is the only thing it knows, and it is never put into a prompt.
"""

import contextlib
import contextvars
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import anthropic
from anthropic import Anthropic

AGENTS = ("C", "N", "V", "S")
PROVIDERS = ("anthropic", "openrouter")
DEFAULT_PROVIDER = "anthropic"
# What every agent used before this layer existed, and the same model's id on OpenRouter (ids there are vendor/model,
# with a dot in the version: "anthropic/claude-sonnet-4-6" does not exist there).
DEFAULT_MODELS = {"anthropic": "claude-sonnet-4-6", "openrouter": "anthropic/claude-sonnet-4.6"}
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {"HTTP-Referer": "noetic-field", "X-Title": "Noetic Field"}
CONFIG_FILE = Path(__file__).parent / "config.json"
DEFAULT_TIMEOUT = 120.0          # seconds, per attempt; the SDKs retry transient failures themselves
MAX_TOKENS_LIMIT = 64000         # a sanity bound for a configured reply limit, not a promise that a model allows it

_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,199}$")


class ProviderError(RuntimeError):
    """A model call could not be made or answered. The message says why; agents.call_model turns it into an AgentError."""


class ConfigError(ProviderError):
    """The provider configuration is unusable: an unknown provider, a model id that does not suit it, a bad file."""


@dataclass(frozen=True)
class Completion:
    """One reply, in the same shape whichever backend produced it."""
    text: str
    stop_reason: str = ""        # "end_turn" | "max_tokens" | "refusal" | whatever else the provider said; "" if it said nothing
    model: str = ""              # the model the provider reports having used (else the one that was asked for)
    provider: str = ""


# ── one Pipeline's own settings ───────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    """What one Pipeline asked for. Build it with make_settings(), which checks it; the values are only looked at when a
    call is made. The API key is kept out of repr()."""
    provider: Optional[str] = None
    api_key: Optional[str] = field(default=None, repr=False)
    models: Mapping[str, str] = field(default_factory=dict)             # role -> model id
    max_tokens: Mapping[str, int] = field(default_factory=dict)         # role -> reply limit


_scope: contextvars.ContextVar = contextvars.ContextVar("noetic_provider_settings", default=None)
_SCOPED = object()                   # load_config(): "use the settings of the current call, if there are any"
_KEY_PREFIX = {"anthropic": "sk-ant-", "openrouter": "sk-or-"}


def make_settings(provider: Optional[str] = None, api_key: Optional[str] = None,
                  agents: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Settings:
    """Check what a Pipeline was given and keep it. `agents` maps an agent ("C", "N", "V", "S") to
    {"model": "...", "max_tokens": 2000}; both keys are optional. Raises ConfigError for anything unusable, so a typo
    cannot pass silently. Whether a model id suits the provider is checked by load_config, which knows the provider."""
    name = None
    if provider is not None:
        if not isinstance(provider, str) or provider.strip().lower() not in PROVIDERS:
            raise ConfigError(f"unknown provider {str(provider)[:30]!r}: use one of {', '.join(PROVIDERS)}")
        name = provider.strip().lower()
    key = None
    if api_key is not None:
        if not isinstance(api_key, str) or not api_key.strip() or any(ch.isspace() for ch in api_key.strip()):
            raise ConfigError("api_key must be a non-empty string without spaces")
        key = api_key.strip()
    models: dict = {}
    limits: dict = {}
    if agents is not None:
        if not isinstance(agents, Mapping):
            raise ConfigError("agents must map an agent (C, N, V or S) to {\"model\": \"...\"}")
        seen: set = set()
        for raw_role, val in agents.items():
            role = str(raw_role).strip().upper()
            if role not in AGENTS:
                raise ConfigError(f"unknown agent {raw_role!r} in agents: use C, N, V or S")
            if role in seen:
                raise ConfigError(f"agent {role} appears twice in agents")
            seen.add(role)
            if not isinstance(val, Mapping):
                raise ConfigError(f"agent {role} in agents must be a mapping such as {{\"model\": \"...\"}}")
            for fld in val:
                if fld not in ("model", "max_tokens"):
                    raise ConfigError(f"unknown field {fld!r} for agent {role} in agents (supported: model, max_tokens)")
            if "model" in val:
                model = val["model"]
                if not isinstance(model, str) or not model.strip():
                    raise ConfigError(f"the model for agent {role} must be a non-empty string")
                models[role] = model.strip()
            limit = _check_limit(val.get("max_tokens"), role)
            if limit is not None:
                limits[role] = limit
    return Settings(provider=name, api_key=key, models=models, max_tokens=limits)


@contextlib.contextmanager
def scoped(settings: Optional[Settings]):
    """Make `settings` the current ones for the length of the block, in this thread's context only."""
    token = _scope.set(settings)
    try:
        yield
    finally:
        _scope.reset(token)


def current_settings() -> Optional[Settings]:
    return _scope.get()


def check_key_shape(api_key: Optional[str], provider: str) -> None:
    """Refuse an API key that plainly belongs to the other provider, before it is sent anywhere."""
    if not api_key:
        return
    for other, prefix in _KEY_PREFIX.items():
        if other != provider and api_key.startswith(prefix):
            raise ConfigError(f"the api_key looks like an {other} key (it starts with {prefix}), but the provider is "
                              f"{provider}; pass provider={other!r} or the matching key")


def validate(settings: Optional[Settings] = None) -> dict:
    """The effective configuration for `settings` (see load_config), refused if the key is for the other provider."""
    cfg = load_config(settings=settings)
    check_key_shape(settings.api_key if settings is not None else None, cfg["provider"])
    return cfg


# ── configuration ─────────────────────────────────────────────────────────────

def _config_data(env) -> dict:
    raw = env.get("NOETIC_CONFIG")
    if raw is None:
        if not CONFIG_FILE.is_file():
            return {}
        path = CONFIG_FILE
    elif not raw.strip():
        return {}
    else:
        path = Path(raw.strip())
        if not path.is_file():
            raise ConfigError(f"NOETIC_CONFIG points to {raw.strip()!r}, which is not a file")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        raise ConfigError(f"the config file {path} could not be read: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"the config file {path} must hold a JSON object")
    return data


def _check_model(raw: Any, provider: str, role: str) -> str:
    if not isinstance(raw, str) or not _MODEL_RE.match(raw.strip()):
        raise ConfigError(f"the model for agent {role} is not a usable model id: {str(raw)[:60]!r}")
    model = raw.strip()
    if provider == "anthropic" and "/" in model:
        raise ConfigError(f"the model for agent {role} ({model!r}) looks like an OpenRouter id (vendor/model); "
                          "set NOETIC_PROVIDER=openrouter (or Pipeline(provider='openrouter')) to use it")
    if provider == "openrouter" and "/" not in model:
        raise ConfigError(f"the model for agent {role} ({model!r}) is not an OpenRouter id; they look like "
                          f"vendor/model, for example {DEFAULT_MODELS['openrouter']}")
    return model


def _check_limit(raw: Any, role: str) -> Optional[int]:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if isinstance(raw, bool):
        raise ConfigError(f"max_tokens for agent {role} must be a whole number")
    try:
        n = int(str(raw).strip())
    except ValueError:
        raise ConfigError(f"max_tokens for agent {role} must be a whole number, not {str(raw)[:30]!r}") from None
    if not 1 <= n <= MAX_TOKENS_LIMIT:
        raise ConfigError(f"max_tokens for agent {role} must be between 1 and {MAX_TOKENS_LIMIT}")
    return n


def load_config(env=None, settings=_SCOPED) -> dict:
    """The effective configuration: {"provider": str, "models": {role: str}, "max_tokens": {role: int or None}}.
    Raises ConfigError if it is unusable. `env` defaults to os.environ. `settings` (a Settings; by default the ones the
    current Pipeline call made current, if any) win over the environment, which wins over the file."""
    env = os.environ if env is None else env
    if settings is _SCOPED:
        settings = _scope.get()
    asked_models = settings.models if settings is not None else {}
    asked_limits = settings.max_tokens if settings is not None else {}
    data = _config_data(env)
    for key in data:
        if not str(key).startswith("_") and key not in ("provider", "model", "agents"):
            raise ConfigError(f"unknown key {key!r} in the config file (supported: provider, model, agents)")

    file_provider = data.get("provider")
    if file_provider is not None and not isinstance(file_provider, str):
        raise ConfigError("'provider' in the config file must be a string")
    asked_provider = (settings.provider if settings is not None else None) or ""
    provider = (asked_provider.strip().lower() or (env.get("NOETIC_PROVIDER") or "").strip().lower()
                or (file_provider or "").strip().lower() or DEFAULT_PROVIDER)
    if provider not in PROVIDERS:
        raise ConfigError(f"unknown provider {provider!r}: NOETIC_PROVIDER must be one of {', '.join(PROVIDERS)}")

    agents_cfg = data.get("agents", {})
    if not isinstance(agents_cfg, dict):
        raise ConfigError("'agents' in the config file must be an object")
    per_agent: dict = {}
    for key, val in agents_cfg.items():
        if str(key).startswith("_"):
            continue
        role = str(key).strip().upper()
        if role not in AGENTS:
            raise ConfigError(f"unknown agent {key!r} in the config file: use C, N, V or S")
        if not isinstance(val, dict):
            raise ConfigError(f"agent {role} in the config file must be an object, such as {{\"model\": \"...\"}}")
        for field in val:
            if not str(field).startswith("_") and field not in ("model", "max_tokens"):
                raise ConfigError(f"unknown field {field!r} for agent {role} in the config file (supported: model, max_tokens)")
        per_agent[role] = val

    models: dict = {}
    limits: dict = {}
    for role in AGENTS:
        mine = per_agent.get(role, {})
        raw = (asked_models.get(role) or (env.get(f"NOETIC_MODEL_{role}") or "").strip() or mine.get("model")
               or data.get("model") or DEFAULT_MODELS[provider])
        models[role] = _check_model(raw, provider, role)
        raw_limit = asked_limits.get(role) or (env.get(f"NOETIC_MAX_TOKENS_{role}") or "").strip() or mine.get("max_tokens")
        limits[role] = _check_limit(raw_limit, role)
    return {"provider": provider, "models": models, "max_tokens": limits}


def provider_name() -> str:
    return load_config()["provider"]


def models() -> dict:
    """The model each agent will use now: {"C": ..., "N": ..., "V": ..., "S": ...}."""
    return dict(load_config()["models"])


def model_for(role: str) -> str:
    if role not in AGENTS:
        raise ProviderError(f"unknown agent {role!r}")
    return load_config()["models"][role]


def max_tokens_for(role: str) -> Optional[int]:
    """A reply limit configured for this agent, or None (then the caller's own default applies)."""
    if role not in AGENTS:
        raise ProviderError(f"unknown agent {role!r}")
    return load_config()["max_tokens"][role]


# ── the Anthropic backend ─────────────────────────────────────────────────────

_anthropic_client = None             # created on first use; tests or callers may assign an object with .messages.create
_openrouter_client = None            # created on first use; tests or callers may assign an object with .chat.completions.create
_client_lock = threading.Lock()
_keyed_clients: dict = {}            # (backend, sha256 of an api_key given to a Pipeline) -> that key's client


def _client_for_key(backend: str, key: str, make):
    """The client for an API key a Pipeline was given: made once per key and backend. The key is not used as a dict
    key itself, only a hash of it."""
    ident = (backend, hashlib.sha256(key.encode("utf-8")).hexdigest())
    with _client_lock:
        client = _keyed_clients.get(ident)
        if client is None:
            client = _keyed_clients[ident] = make()
        return client


def _anthropic():
    global _anthropic_client
    settings = _scope.get()
    if settings is not None and settings.api_key:
        key = settings.api_key
        return _client_for_key("anthropic", key, lambda: Anthropic(api_key=key))
    if _anthropic_client is None:
        _anthropic_client = Anthropic()
    return _anthropic_client


def from_anthropic(result: Any, model: str = "") -> Completion:
    """An Anthropic Messages reply as a Completion: the text blocks joined (thinking and other blocks are ignored)."""
    blocks = getattr(result, "content", None) or []
    text = "".join(
        b.text for b in blocks
        if getattr(b, "type", None) == "text" and isinstance(getattr(b, "text", None), str)
    )
    stop = getattr(result, "stop_reason", None)
    served = getattr(result, "model", None)
    return Completion(text=text, stop_reason=stop if isinstance(stop, str) else "",
                      model=served if isinstance(served, str) and served else model, provider="anthropic")


def _complete_anthropic(role: str, model: str, system: str, user: str, max_tokens: int, timeout: float) -> Completion:
    try:
        result = _anthropic().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            timeout=timeout,
        )
    except anthropic.APIError as e:
        raise ProviderError(f"{type(e).__name__}: {e}") from e
    except TypeError as e:
        if "authentication" in str(e).lower() or "api_key" in str(e).lower():
            raise ProviderError(f"no API key is set (set {KEY_ENV['anthropic']})") from e
        raise
    return from_anthropic(result, model)


# ── the OpenRouter backend ────────────────────────────────────────────────────

_FINISH = {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal",
           "tool_calls": "tool_use", "function_call": "tool_use"}


def make_openrouter_client(api_key: str, base_url: str = OPENROUTER_BASE_URL):
    """The `openai` client for OpenRouter. `base_url` is a parameter only so that a test can point it at a local stub."""
    try:
        import openai
    except ImportError as e:
        raise ProviderError("the openai package is not installed (pip install openai, or pip install \"noetic-field[openrouter]\"); "
                            "OpenRouter is reached through it") from e
    return openai.OpenAI(api_key=api_key, base_url=base_url, default_headers=dict(OPENROUTER_HEADERS))


def _openrouter():
    global _openrouter_client
    settings = _scope.get()
    if settings is not None and settings.api_key:
        key = settings.api_key
        return _client_for_key("openrouter", key, lambda: make_openrouter_client(key))
    with _client_lock:
        if _openrouter_client is None:
            key = os.environ.get(KEY_ENV["openrouter"], "").strip()
            if not key:
                raise ProviderError(f"no API key is set (set {KEY_ENV['openrouter']})")
            _openrouter_client = make_openrouter_client(key)
        return _openrouter_client


def _openai_errors() -> tuple:
    try:
        import openai
    except ImportError:
        return ()                    # the SDK was never loaded, so it cannot have raised anything
    return (openai.APIError,)


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _error_text(err: Any) -> str:
    if err is None:
        return ""
    msg = _field(err, "message")
    return str(msg if msg else err)[:300]


def _part_text(part: Any) -> str:
    text = _field(part, "text")
    return text if _field(part, "type") in (None, "text", "output_text") and isinstance(text, str) else ""


def from_openai(result: Any, model: str = "") -> Completion:
    """An OpenAI-style chat completion (what OpenRouter returns) as a Completion. The finish reasons are mapped to the
    names the agents already use (stop -> end_turn, length -> max_tokens, content_filter -> refusal); an error the
    provider reports inside the reply raises ProviderError."""
    served = getattr(result, "model", None)
    served = served if isinstance(served, str) and served else model
    choices = getattr(result, "choices", None) or []
    if not choices:
        detail = _error_text(getattr(result, "error", None))
        raise ProviderError("the provider returned no choices" + (f": {detail}" if detail else ""))
    choice = choices[0]
    finish = _field(choice, "finish_reason")
    err = _field(choice, "error")
    if finish == "error" or err:
        detail = _error_text(err)
        raise ProviderError("the provider reported an error" + (f": {detail}" if detail else ""))
    message = _field(choice, "message")
    content = _field(message, "content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, (list, tuple)):
        text = "".join(_part_text(p) for p in content)
    else:
        text = ""
    refusal = _field(message, "refusal")
    if isinstance(refusal, str) and refusal.strip() and not text.strip():
        stop = "refusal"
    else:
        stop = _FINISH.get(finish, finish if isinstance(finish, str) else "")
    return Completion(text=text, stop_reason=stop, model=served, provider="openrouter")


def _complete_openrouter(role: str, model: str, system: str, user: str, max_tokens: int, timeout: float) -> Completion:
    client = _openrouter()
    try:
        result = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            timeout=timeout,
        )
    except _openai_errors() as e:
        raise ProviderError(f"{type(e).__name__}: {e}") from e
    return from_openai(result, model)


# ── the one entry point ───────────────────────────────────────────────────────

def complete(role: str, system: str, user: str, *, max_tokens: int, timeout: float = DEFAULT_TIMEOUT) -> Completion:
    """One model call for agent `role` ("C", "N", "V" or "S"), on the provider and model the configuration names.
    Raises ProviderError (ConfigError is one) and nothing else that an SDK or the network can raise."""
    if role not in AGENTS:
        raise ProviderError(f"unknown agent {role!r}")
    cfg = load_config()
    settings = _scope.get()
    if settings is not None:
        check_key_shape(settings.api_key, cfg["provider"])      # a key is never sent to the other provider's servers
    backend = _complete_anthropic if cfg["provider"] == "anthropic" else _complete_openrouter
    return backend(role, cfg["models"][role], system, user, max_tokens, timeout)

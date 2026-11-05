# Copyright (c) 2026 Joseph Whaley Jr. (Coherence Labs). MIT License; see LICENSE.
"""
noetic_field: a four-agent LLM pipeline for confabulation detection and values isolation.

    from noetic_field import Pipeline

    result = Pipeline().chat("your prompt here")
    print(result.answer)
    print(result.divergence_score, result.values_conflict)

Pipeline and ChatResult are the public interface (see pipeline.py for what they take and return). AgentError,
ConfigError, InputError and ProviderError are the errors chat() and the constructor raise.

Importing the package imports nothing else: each name below is loaded the first time it is asked for, so
`import noetic_field` (to read __version__, say) needs no API key, no SDK and touches no file.

The modules in this folder also run on their own, side by side, from a source checkout: python cli.py,
uvicorn api:app, python -m pytest tests.
"""

from typing import TYPE_CHECKING

__version__ = "0.2.1"

__all__ = ["Pipeline", "ChatResult", "AgentError", "ConfigError", "InputError", "ProviderError", "__version__"]

if TYPE_CHECKING:               # for editors and type checkers; at run time the names are loaded on demand
    from .pipeline import AgentError, ChatResult, ConfigError, InputError, Pipeline, ProviderError


def __getattr__(name):
    if name in __all__ and name != "__version__":
        from importlib import import_module
        value = getattr(import_module(".pipeline", __name__), name)
        globals()[name] = value                  # found once, then an ordinary attribute
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

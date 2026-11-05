"""
Shared fixtures. Every test runs against its own state file in a temp folder and a scripted
provider, so nothing here can touch the real noetic_state.json, a real config file or the network.

Run from the noetic_field folder:
    python -m pytest tests -q
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Safety net, set before anything imports `state` or `provider`: even a test (or a child process it starts) that forgets
# to redirect the state file can only reach a throwaway file, never the real noetic_state.json, and no config.json
# lying next to the code can change which provider or model a test uses.
os.environ["NOETIC_STATE_FILE"] = str(Path(tempfile.mkdtemp(prefix="noetic-tests-")) / "noetic_state.json")
os.environ["NOETIC_CONFIG"] = ""
for _name in ("NOETIC_PROVIDER", *(f"NOETIC_{kind}_{r}" for kind in ("MODEL", "MAX_TOKENS") for r in "CNVS")):
    os.environ.pop(_name, None)

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE.parent), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import provider                     # noqa: E402  (imports must follow the sys.path setup)
import state as state_module        # noqa: E402
from fakes import FakeProvider      # noqa: E402

_PROVIDER_ENV = ("NOETIC_PROVIDER", *(f"NOETIC_{kind}_{r}" for kind in ("MODEL", "MAX_TOKENS") for r in "CNVS"))


@pytest.fixture(autouse=True)
def fake(tmp_path, monkeypatch):
    """A private state file, a clean provider environment and a scripted provider for every test."""
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NOETIC_CONFIG", "")
    monkeypatch.setattr(state_module, "STATE_FILE", tmp_path / "noetic_state.json")
    double = FakeProvider()
    monkeypatch.setattr(provider, "complete", double.complete)
    monkeypatch.setattr(provider, "_anthropic_client", None)
    monkeypatch.setattr(provider, "_openrouter_client", None)
    return double


@pytest.fixture
def orch(fake):
    from orchestrator import NoeticOrchestrator
    return NoeticOrchestrator()


@pytest.fixture
def client(fake, monkeypatch):
    """A FastAPI test client with a fresh orchestrator and an empty session table."""
    from collections import OrderedDict

    from fastapi.testclient import TestClient

    import api
    monkeypatch.setattr(api, "_orchestrator", None)
    monkeypatch.setattr(api, "_sessions", OrderedDict())
    return TestClient(api.app)

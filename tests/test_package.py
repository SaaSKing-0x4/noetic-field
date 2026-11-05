"""
The package as it is installed: the modules inside a noetic_field folder, importing one another by relative imports.

Everything else in this suite runs the modules side by side, as in a source checkout. These tests lay the same files out
the way the wheel does (a noetic_field folder in an otherwise empty directory) and run Python in a subprocess there, so
that a module still reaching for a plain `import agents` would fail. They need no installation and no network.
"""

import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
DATA_FILES = {"config.example.json", "PROVIDERS_2026-11-04.md"}
TOOLING = {"hatch_build.py"}                      # root files that are build tooling, not part of the package


def wheel_files() -> dict:
    """{source file: path inside the wheel}, from the force-include table in pyproject.toml."""
    section = re.search(r"^\[tool\.hatch\.build\.targets\.wheel\.force-include\]\s*\n(.*?)(?=^\[|\Z)", PYPROJECT, re.S | re.M)
    assert section, "pyproject.toml has no wheel force-include table"
    return dict(re.findall(r'^"([^"]+)"\s*=\s*"([^"]+)"', section.group(1), re.M))


def project_version() -> str:
    return re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT, re.M).group(1)


# ── the build description ─────────────────────────────────────────────────────

def test_the_wheel_lists_every_module_and_the_two_data_files_and_nothing_else():
    files = wheel_files()
    modules = {p.name for p in ROOT.glob("*.py")} - TOOLING
    assert set(files) == modules | DATA_FILES, "a module was added to (or removed from) the folder; update pyproject.toml"
    assert all(dst == "noetic_field/" + src for src, dst in files.items())
    assert all((ROOT / src).is_file() for src in files)


def test_the_version_is_the_same_in_the_package_and_in_pyproject():
    declared = re.search(r'^__version__\s*=\s*"([^"]+)"', (ROOT / "__init__.py").read_text(encoding="utf-8"), re.M).group(1)
    assert declared == project_version()


def test_the_command_line_entry_point_names_a_function_that_exists():
    target = re.search(r'^noetic-field\s*=\s*"([\w.]+):(\w+)"', PYPROJECT, re.M)
    assert target and target.group(1) == "noetic_field.cli"
    assert re.search(rf"^def {target.group(2)}\(", (ROOT / "cli.py").read_text(encoding="utf-8"), re.M)


def test_the_package_needs_only_anthropic_and_the_rest_is_optional():
    deps = re.search(r"^dependencies\s*=\s*\[(.*?)\]", PYPROJECT, re.S | re.M).group(1)
    assert re.findall(r'"([^"]+)"', deps) == ["anthropic>=0.40"]
    extras = re.search(r"^openrouter\s*=\s*\[(.*?)\]", PYPROJECT, re.M).group(1)
    assert re.findall(r'"([^"]+)"', extras) == ["openai>=1.0"]


# ── running as the installed package ──────────────────────────────────────────

@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """An empty directory holding a copy of the package laid out the way the wheel lays it out."""
    root = tmp_path_factory.mktemp("site")
    (root / "noetic_field").mkdir()
    for src, dst in wheel_files().items():
        shutil.copy2(ROOT / src, root / dst)
    return root


def run(site, code=None, args=None, flags=(), **env_extra):
    """Run Python in `site` with no NOETIC_* settings, so nothing of the test process (or the user's) leaks in."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("NOETIC_") and k != "PYTHONPATH"}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", **env_extra)
    cmd = [sys.executable, *flags, "-c", textwrap.dedent(code)] if code is not None else [sys.executable, *flags, *args]
    return subprocess.run(cmd, cwd=site, env=env, capture_output=True, text=True, timeout=180)


def json_out(done):
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_importing_the_package_loads_nothing_else_and_the_names_resolve_on_demand(site):
    out = json_out(run(site, """
        import importlib.util, json, sys
        import noetic_field
        before = sorted(m for m in ("anthropic", "openai", "noetic_field.pipeline", "noetic_field.provider") if m in sys.modules)
        from noetic_field import Pipeline, ChatResult, AgentError, ConfigError, InputError, ProviderError, __version__
        import noetic_field.pipeline as p
        namespace = {}
        exec("from noetic_field import *", namespace)
        print(json.dumps({
            "file": noetic_field.__file__, "version": __version__, "loaded_by_import": before,
            "flat_agents_importable": importlib.util.find_spec("agents") is not None,
            "same_objects": Pipeline is p.Pipeline and ChatResult is p.ChatResult and AgentError is p._agents.AgentError,
            "in_dir": all(n in dir(noetic_field) for n in noetic_field.__all__),
            "star": sorted(n for n in namespace if not n.startswith("__")),
        }))
    """))
    assert Path(out["file"]).parent == site / "noetic_field" and out["version"] == project_version()
    assert out["loaded_by_import"] == [] and out["same_objects"] and out["in_dir"]
    assert out["flat_agents_importable"] is False                   # nothing of the folder is importable by its plain name
    assert out["star"] == ["AgentError", "ChatResult", "ConfigError", "InputError", "Pipeline", "ProviderError"]


def test_an_unknown_name_is_an_attribute_error(site):
    done = run(site, "import noetic_field\ntry:\n    noetic_field.nothing_here\nexcept AttributeError as e:\n    print('AttributeError', e)")
    assert done.returncode == 0 and "AttributeError" in done.stdout and "nothing_here" in done.stdout


def test_every_module_imports_with_relative_imports_only(site):
    stems = sorted(p.stem for p in (site / "noetic_field").glob("*.py") if p.stem != "__init__")
    if importlib.util.find_spec("fastapi") is None:
        stems.remove("api")                                         # the HTTP interface is an optional extra
    out = json_out(run(site, f"""
        import importlib, json
        done = {{}}
        for stem in {stems!r}:
            module = importlib.import_module("noetic_field." + stem)
            done[stem] = module.__name__
        print(json.dumps(done))
    """))
    assert out == {stem: f"noetic_field.{stem}" for stem in stems}


FAKE_REPLIES = '''
    import json
    C = json.dumps({"reasoning_steps": ["looked at the question", "recalled the fact"], "response": "The answer is 42.",
                    "confidence": 0.8, "domain": "math", "uncertainty_flags": [], "default_state_active": False})
    N = json.dumps({"explanation_matches_reasoning": True, "confabulation_risk": "low", "divergence_notes": [],
                    "overall_assessment": "The response follows from the steps."})
    V = json.dumps({"status": "pass", "values_notes": [], "sycophancy_detected": False,
                    "hidden_uncertainty_detected": False, "recommendation": "pass as-is"})
    S = json.dumps({"patterns_found": [], "new_insights": [], "domains_to_flag": [], "values_tension_pattern": "",
                    "dark_matter_signals": [], "overall_health": "healthy - nothing alarming in a small sample."})
'''


def test_chat_works_end_to_end_as_the_installed_package_and_keeps_its_state_at_home(site, tmp_path):
    home = tmp_path / "home"
    code = FAKE_REPLIES + '''
    import json, os
    import noetic_field.provider as provider
    from noetic_field import Pipeline

    seen = []

    def fake(role, system, user, *, max_tokens, timeout=120.0):
        seen.append([role, provider.model_for(role)])
        return provider.Completion(text={"C": C, "N": N, "V": V, "S": S}[role], stop_reason="end_turn", model="fake", provider="fake")

    provider.complete = fake
    pipeline = Pipeline(api_key="sk-ant-local-test-not-a-real-key", agents={"V": {"model": "claude-v-test"}})
    result = pipeline.chat("hello")
    print(json.dumps({"result": result.to_dict(), "seen": sorted(seen), "state": os.path.join(os.environ["NOETIC_HOME"], "noetic_state.json")}))
    '''
    out = json_out(run(site, code, NOETIC_HOME=str(home)))
    result = out["result"]
    assert result["answer"] == "The answer is 42." and result["divergence_score"] == 0.0 and result["values_conflict"] is False
    assert result["trace"] == "1. looked at the question\n2. recalled the fact"
    assert out["seen"] == [["C", "claude-sonnet-4-6"], ["N", "claude-sonnet-4-6"], ["V", "claude-v-test"]]      # N and V threads included
    assert Path(out["state"]) == home / "noetic_state.json" and Path(out["state"]).is_file()
    assert not list((site / "noetic_field").glob("*.json.*")) and not (site / "noetic_field" / "noetic_state.json").exists()


def test_the_command_line_runs_as_a_module_of_the_package(site, tmp_path):
    home = tmp_path / "home"
    done = run(site, args=["-m", "noetic_field.cli", "--help"], NOETIC_HOME=str(home))
    assert done.returncode == 0 and "Usage:" in done.stdout and "noetic-field --insight" in done.stdout
    assert "python cli.py" not in done.stdout                       # installed, the command is noetic-field
    assert not home.exists()                                        # --help touches nothing
    done = run(site, args=["-m", "noetic_field.cli", "--self"], NOETIC_HOME=str(home))
    assert done.returncode == 0 and "Noetic Self-Model (Session 0)" in done.stdout       # not a session, so not counted
    assert (home / "noetic_state.json").is_file()                   # a missing file starts a blank state, here: in NOETIC_HOME
    done = run(site, args=["-m", "noetic_field.audit", "confabulation"], NOETIC_HOME=str(home))
    assert done.returncode == 0 and "No interaction history yet." in done.stdout


def test_run_from_a_source_folder_the_usage_text_is_unchanged():
    import cli                                                      # flat: the modules sit side by side in this process
    assert "python cli.py --self" in cli.USAGE and "noetic-field" not in cli.USAGE


def test_the_http_module_names_itself_correctly_when_run_as_the_package():
    source = (ROOT / "api.py").read_text(encoding="utf-8")
    assert 'f"{__package__}.api:app" if __package__ else "api:app"' in source


# ── the editable-install hook ─────────────────────────────────────────────────

def finder_source(root: Path) -> str:
    """The module hatch_build.py writes into an editable install, for `root`, read without importing hatchling."""
    for node in ast.parse((ROOT / "hatch_build.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "FINDER":
            return ast.literal_eval(node.value).format(root=str(root), name="noetic_field")
    raise AssertionError("hatch_build.py has no FINDER")


def test_the_editable_finder_maps_the_checkout_folder_to_the_package_and_nothing_more(tmp_path):
    checkout = tmp_path / "any-folder-name"                         # a clone is not necessarily called noetic_field
    checkout.mkdir()
    for src in wheel_files():
        shutil.copy2(ROOT / src, checkout / src)
    env_site = tmp_path / "env"
    env_site.mkdir()
    (env_site / "_noetic_field_editable.py").write_text(finder_source(checkout), encoding="utf-8")
    # -S: no site-packages and no .pth files, so an editable install of this very package (CI has one, and it registers a
    # finder under the same module name) cannot stand in for the finder that is being tested.
    out = json_out(run(env_site, f"""
        import importlib.util, json, sys
        import _noetic_field_editable
        import noetic_field
        import noetic_field.state as state
        print(json.dumps({{"package": noetic_field.__file__, "state": state.__file__, "name": state.__name__,
                          "search": list(noetic_field.__path__), "flat": importlib.util.find_spec("state") is not None,
                          "other": importlib.util.find_spec("noetic_fieldx") is not None}}))
    """, flags=("-S",)))
    assert Path(out["package"]) == checkout / "__init__.py" and Path(out["state"]) == checkout / "state.py"
    assert out["name"] == "noetic_field.state" and [Path(p) for p in out["search"]] == [checkout]
    assert out["flat"] is False and out["other"] is False

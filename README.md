# noetic-field

A four-agent LLM pipeline for confabulation detection and values isolation.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23965032.svg)](https://doi.org/10.5281/zenodo.23965032)

One model answers and writes out its reasoning. A second reads that reasoning and the answer and says whether the one accounts for the other. A third judges the answer against a list of values without seeing the reasoning. A fourth reviews the whole record every tenth interaction. You get the answer back with two values a program can act on: how far the explanation drifts from the answer, and whether the values check objected.

## Install

```bash
pip install noetic-field                  # Anthropic
pip install "noetic-field[openrouter]"    # adds the openai package, to reach any model through OpenRouter
pip install "noetic-field[api]"           # adds FastAPI, for the HTTP interface
```

Python 3.10 or newer.

## Quickstart

### One model (Anthropic, the default)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
from noetic_field import Pipeline

pipeline = Pipeline()                          # or Pipeline(api_key="sk-ant-...")
result = pipeline.chat("Why is the sky blue?")

print(result.answer)             # the final answer, as the pipeline presents it
print(result.divergence_score)   # 0.0 to 1.0: how far the explanation drifts from the answer
print(result.values_conflict)    # True if the values check flagged, vetoed or could not check
print(result.trace)              # the reasoning steps behind the answer, or None
```

All four agents use `claude-sonnet-4-6` unless you say otherwise. `chat()` is multi-turn: the pipeline remembers the earlier turns until you call `pipeline.new_conversation()`.

### A different model for each agent (OpenRouter)

```python
import os
from noetic_field import Pipeline

pipeline = Pipeline(
    provider="openrouter",
    api_key=os.environ["OPENROUTER_API_KEY"],          # sk-or-...
    agents={
        "C": {"model": "anthropic/claude-sonnet-4.5"},
        "N": {"model": "google/gemini-2.5-pro"},
        "V": {"model": "deepseek/deepseek-r1-0528", "max_tokens": 4000},
        "S": {"model": "anthropic/claude-sonnet-4.5"},
    },
)
result = pipeline.chat("your prompt here")
```

* OpenRouter ids are `vendor/model` and use dots in the version (`anthropic/claude-sonnet-4.5`, not `-4-5`). Check the id on [openrouter.ai/models](https://openrouter.ai/models); the model list changes.
* An agent you leave out uses its default (`anthropic/claude-sonnet-4.6` on OpenRouter).
* `max_tokens` is the reply limit of one agent (defaults: C 2000, N 800, V 600, S 1200). A reasoning model spends part of it on reasoning, so give it more.
* The key and the models belong to this `Pipeline` alone. Two pipelines in one process can use different providers and keys; nothing is written to `os.environ`.

## What `chat()` returns

A frozen `ChatResult`. The first four fields are the interface:

| Field | Type | Meaning |
|---|---|---|
| `answer` | `str` | The final answer exactly as the pipeline presents it: the answer, with a `[VALUES CIRCUIT VETO]` prefix if the values check vetoed it and a `*[Noetic: ...]*` note if the checks had something to say. The command line prints this and the HTTP API returns it as `response`. |
| `divergence_score` | `float`, 0.0 to 1.0 | How far the answer's explanation drifts from its own reasoning, from the narration check: `low` is 0.0, `medium` 0.5, `high` 1.0. |
| `values_conflict` | `bool` | `True` when the values check flagged or vetoed the answer, **or could not check it**. `False` only when it looked and passed. |
| `trace` | `str` or `None` | The reasoning steps behind the answer, numbered, one per line. `None` when there are none. |

An answer that was not checked is never reported as a clean one. If the narration check failed, was cut off or could not be read, `divergence_score` is `1.0` and `divergence_checked` is `False`; if the values check could not run, `values_conflict` is `True` and `values_checked` is `False`. Test those two flags before you trust a low score.

More detail on the same object:

* `body` — the answer on its own, without the veto prefix or the note.
* `confabulation_risk` (`low` / `medium` / `high` / `unknown`), `divergence_notes` — what the narration check saw.
* `values_status` (`pass` / `flag` / `veto` / `unchecked`), `vetoed`, `values_notes`, `sycophancy_detected` — what the values check saw. Sycophancy is reported on its own; it does not by itself set `values_conflict`.
* `confidence` — what the answering model reported about its own answer (0.0 to 1.0), or `None` if it reported none. A self-report, not a measurement.
* `domain`, `truncated` (the answer may be cut off by the length limit), `parse_issues` (an agent's reply was malformed or missing).
* `reasoning_steps` — `trace` as a tuple.
* `review_triggered`, `review_error` — whether the self-review ran after this message, and why it failed or stored nothing. The answer is never affected.
* `to_dict()` — everything as plain JSON-ready data.

## The four agents

```
message ──► C  computes the answer and writes its reasoning steps, separately from the answer
              │
              ├──► N  reads the steps and the answer: does the explanation account for the answer?
              └──► V  reads the message and the answer only: does the answer fit the values?
              │
              ▼
        answer + notes ──► every 10th interaction: S reviews the record
```

* **C, computation.** Answers the message. It is given the system's self-model (its values, notes from earlier reviews and how confident it has been in each domain) and the recent turns of the conversation.
* **N, narration.** Compares C's written reasoning with C's written answer and rates the confabulation risk: whether the stated explanation is what produced the answer. It sees text only; its rating is a judgment of text, not a measurement of the model's internals.
* **V, values.** Judges the answer against a list of values and watches for sycophancy, hidden uncertainty and approval-seeking. It is kept apart from C: it sees the user's message and the answer, never the reasoning steps or the confidence. `pass`, `flag` or `veto`; a veto marks the answer, it does not block it.
* **S, self-review.** Every tenth interaction it reads exact statistics computed by code, the recent log and the stored notes, and reports patterns across time. What it concludes is stored and shown to C afterwards.

N and V run in parallel. If one of them fails or returns something unreadable, the result says "unchecked" or "unknown"; it is never a pass.

## Configuration

**Keys.** `ANTHROPIC_API_KEY` for Anthropic, `OPENROUTER_API_KEY` for OpenRouter, or `Pipeline(api_key=...)`. A key that plainly belongs to the other provider (`sk-ant-...` for OpenRouter, `sk-or-...` for Anthropic) is refused before anything is sent. Do not commit keys; read them from the environment.

**Order of precedence** (first one set wins): the `Pipeline(...)` arguments, the environment, the config file, the defaults.

| Environment variable | Effect |
|---|---|
| `NOETIC_PROVIDER` | `anthropic` (default) or `openrouter` |
| `NOETIC_MODEL_C`, `_N`, `_V`, `_S` | the model one agent uses |
| `NOETIC_MAX_TOKENS_C`, `_N`, `_V`, `_S` | the reply limit of one agent |
| `NOETIC_CONFIG` | path of a JSON config file; copy [`config.example.json`](https://github.com/SaaSKing-0x4/noetic-field/blob/main/config.example.json) (it ships inside the package, as `noetic_field/config.example.json`) |
| `NOETIC_HOME` | the folder an installed package keeps its state in (default `~/.noetic_field`) |
| `NOETIC_STATE_FILE` | the exact state file to use |

A model id that does not suit the provider (a `vendor/model` id on Anthropic, a bare id on OpenRouter) is refused when the `Pipeline` is created. [`PROVIDERS_2026-11-04.md`](https://github.com/SaaSKing-0x4/noetic-field/blob/main/PROVIDERS_2026-11-04.md) describes the provider layer in full.

**State.** The pipeline keeps a self-model on disk (values, notes, per-domain confidence, a bounded log). An installed package writes `~/.noetic_field/noetic_state.json`; run from a source checkout it writes next to the code. `Pipeline(state_file="path.json")` gives one pipeline a file of its own. Creating a `Pipeline` reads and writes nothing; the first `chat()` does, and counts as one session. The file is shared with the command line and the HTTP interface.

## Command line and HTTP

```bash
noetic-field                      # interactive conversation
noetic-field --verbose            # also show the reasoning trace and both checks
noetic-field --self               # show the self-model
noetic-field --insight "text"     # add a note
noetic-field --value "text"       # add a value
```

```bash
pip install "noetic-field[api]"
uvicorn noetic_field.api:app --host 127.0.0.1
```

The HTTP server has no authentication and its `/value` and `/insight` endpoints change what every later session is told. Bind it to localhost only.

## Errors

```python
from noetic_field import AgentError, ConfigError, InputError
```

* `ConfigError` — raised by `Pipeline(...)` for an unknown provider or agent, a model id that does not suit the provider, a bad config file or a key for the other provider.
* `InputError` (a `ValueError`) — the prompt is empty or longer than 8000 characters.
* `AgentError` — the answering model could not be reached, declined, or returned nothing. Nothing was recorded.
* `OSError` — the state file could not be read or written.

`chat()` blocks until the answer has been checked, and may be called from several threads.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
```

The suite runs on a scripted provider and a local stub server; it makes no call to a real provider. The repository root is the package folder: the modules sit beside `pyproject.toml`, so a checkout also runs as it always has (`python cli.py`, `uvicorn api:app`), while the installed package imports them as `noetic_field.*`. `hatch_build.py` is what lets `pip install -e .` expose the folder as `noetic_field`.

## Status and limits

Version 0.2, alpha. The automated tests do not call a real provider: run one message against your provider before you depend on it. N and V are language models reading text. They can be wrong, and agreeing with each other is not proof that an answer is true. The design is motivated by observations in Anthropic's "On the Biology of a Large Language Model" (2025), where a model's written account of its method sometimes differed from how it produced the answer; the checks here are text-level and are not measurements of a model's internals.

## License

MIT. See [LICENSE](https://github.com/SaaSKing-0x4/noetic-field/blob/main/LICENSE).

The underlying theoretical framework (Multi-Convergence Hypothesis, Noetic Field Model) is separately documented and reserved. See https://doi.org/10.5281/zenodo.23965032.

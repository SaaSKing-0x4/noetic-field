# Noetic Field (Python): the provider layer, 2026-11-04

Any agent (C, N, V, S) can now run on any model OpenRouter routes (Anthropic, OpenAI, Google, DeepSeek, xAI and more), each on a different one, or on Anthropic directly. **This is a transport change only.** What each agent is asked, how its answer is read and every check, fallback and the pipeline are the same whichever model answers; the code that does it is `provider.py` and `agents.call_model` calls it.

- **Backup of the code before this change:** `_backup_before_provider_2026-11-04/` (every `.py`, `requirements*.txt`, the change log and `tests/`).
- **Run the tests:** `pip install -r requirements-dev.txt`, then `python -m pytest tests -q`. 715 tests, all offline. Without the `openai` package 11 of them are skipped (the ones that run the real `openai` SDK against a local stub); `pip install openai` runs them.
- **Not done:** no call has ever been made to a live provider, Anthropic or OpenRouter.

## How to use it

Nothing changes unless you set something: no environment variable and no `config.json` means Anthropic, `claude-sonnet-4-6` for every agent, `ANTHROPIC_API_KEY`, the same request as before.

| Setting | Meaning |
|---|---|
| `NOETIC_PROVIDER` | `anthropic` (default) or `openrouter` |
| `OPENROUTER_API_KEY` | the key for `openrouter` (`ANTHROPIC_API_KEY` for `anthropic`); keys are never read from a file |
| `NOETIC_MODEL_C`, `_N`, `_V`, `_S` | the model one agent uses |
| `NOETIC_MAX_TOKENS_C`, `_N`, `_V`, `_S` | an agent's reply limit (defaults: C 2000, N 800, V 600, S 1200) |
| `NOETIC_CONFIG` | path of a JSON file; unset: `config.json` next to `provider.py` if it exists; empty: no file at all |

The file (`config.example.json` is a working example; copy it to `config.json`):

```json
{
  "provider": "openrouter",
  "agents": {
    "C": { "model": "anthropic/claude-sonnet-4.6" },
    "N": { "model": "google/gemini-2.5-pro" },
    "V": { "model": "deepseek/deepseek-r1-0528", "max_tokens": 4000 },
    "S": { "model": "anthropic/claude-sonnet-4.6" }
  }
}
```

A top-level `"model"` is the default for every agent that has none of its own. The environment wins over the file, and the file is read on every call, so a change takes effect without a restart. A mistake (an unknown provider, agent or field, a model id that does not suit the provider, a bad limit, a file that is not JSON) is an error that says what is wrong, and the request fails as an `AgentError` before anything is sent: a typo cannot pass silently. Anthropic ids have no slash (`claude-sonnet-4-6`); OpenRouter ids are `vendor/model` with dots in versions, and a wrong pairing is refused.

- **Anthropic backend:** the Anthropic SDK, `api.anthropic.com`, the request exactly as before.
- **OpenRouter backend:** the `openai` SDK with `base_url="https://openrouter.ai/api/v1"` and the headers `HTTP-Referer: noetic-field` and `X-Title: Noetic Field`; the system prompt goes as the system message. The `openai` package is imported only when this backend is first used.
- **Replies** become one shape (`provider.Completion`): the text, a stop reason (`stop`/`length`/`content_filter` become `end_turn`/`max_tokens`/`refusal`, the names the agents already use) and the model that answered. Refusals, empty replies, truncation and SDK errors reach the agents exactly as before.

## Where this differs from the request, and why

1. **The default model stays `claude-sonnet-4-6`, not `claude-sonnet-4-5`.** The code, and a test, use 4-6, and "exactly as before" with no settings is the stricter requirement. The matching OpenRouter default is `anthropic/claude-sonnet-4.6`.
2. **Several of the example ids do not exist on OpenRouter.** Checked against its public model list on 2026-11-04: `anthropic/claude-sonnet-4-5` (hyphens; the id is `anthropic/claude-sonnet-4.5`), `google/gemini-pro-1.5`, `x-ai/grok-3` and `deepseek/deepseek-r1` were not there; `anthropic/claude-sonnet-4.5`, `anthropic/claude-sonnet-4.6`, `google/gemini-2.5-pro`, `x-ai/grok-4.3`, `deepseek/deepseek-r1-0528` and `openai/gpt-5.1` were. The example file uses ids that exist.
3. **There was no hardcoded self-identification in the S prompt to replace.** `self_review.build_s_prompt` never named a model. What did name one was a leftover constant, `self_review.MODEL`, which nothing read; it is removed, and a test asserts that no prompt of S or V names a model or vendor and that the constant is gone. The strings "Claude 3.5 Haiku" and "Claude Opus 4.1" in `agents.py` and `state.py` are citations of what the papers studied, not claims about the running model, and stay. No `GET /models` call is made: the system does not assume which model is answering.
4. **One sentence of the self-model was true only for one configuration, and is now made true for all.** The awareness text told C that the values agent "is the same underlying model". When V is configured on a different model than C that is false, so `state.get_context_summary` swaps in a wording that says it is not (`state._OTHER_MODEL`). With one model, or with no configuration, the text is exactly as before; the stored text and `AWARENESS_REV` are unchanged.
5. **There is no "existing phi-interval schedule" for S.** S runs when the interaction counter is a multiple of `N_REVIEW` (10), as before; the golden-ratio numbers (1, 1.618, 2.618, 4.236 s) are only the pulse timings in the browser page's CSS. S has one `model` like the other agents; it does not rotate. Rotating S through models would be a new behaviour, not transport, and was not added.
6. **Added, because the example configuration needs it:** an optional reply limit per agent (`max_tokens`, `NOETIC_MAX_TOKENS_<agent>`). A reasoning model such as DeepSeek R1 spends part of the limit on its reasoning, and V's default limit is 600 tokens, so without a larger one V on such a model would often return nothing and every answer would be marked unchecked. The defaults are unchanged.
7. **One provider for all agents.** `NOETIC_PROVIDER` is global; a mixed Anthropic-direct and OpenRouter setup is not supported.

## What changed, by file

- **`provider.py` (new).** Configuration (`load_config`, `model_for`, `models`, `provider_name`, `max_tokens_for`), the two backends, `complete(role, system, user, max_tokens, timeout)` and the `Completion`, `ProviderError` and `ConfigError` types.
- **`agents.py`.** `call_model` calls `provider.complete` and keeps everything else it did (cleaning, the refusal and empty-reply errors, the truncation flag). The `anthropic` import, the `client` global and `_get_client` are gone. `MODEL`, `MAX_TOKENS` and `REQUEST_TIMEOUT` stay as the defaults.
- **`self_review.py`.** The unused `MODEL` constant is removed; a sentence in the docstring says S's prompt names no model.
- **`state.py`.** The awareness sentence about the values agent is a named constant (`_SAME_MODEL`, byte-identical text) and `get_context_summary` replaces it when V's model differs from C's (item 4).
- **`requirements.txt`.** Adds `openai>=1.0.0` (used only for OpenRouter).
- **`config.example.json` (new).** The example above, with the ids checked on 2026-11-04.
- **`PROVIDERS_2026-11-04.md` (new)** and a pointer at the top of `CHANGES_2026-11-03.md`.
- **Tests.** `tests/fakes.py` and `tests/conftest.py` now stand in for `provider.complete` instead of the Anthropic client, and give every test a clean provider environment (no setting, no `config.json`). Six test files (`test_agents`, `test_orchestrator`, `test_api_cli`, `test_self_review`, `test_review_fixes`, `test_audit_round2`) lost their imports of `anthropic` and `httpx` (a failed call is now a `ProviderError`, as the provider layer raises it) or their assignment of `agents.client`; `test_sdk_integration.py` keeps the real Anthropic SDK against a local stub and now injects its client into the provider. The 640 tests are 640 still; **`tests/test_provider.py` adds 75**: configuration and its refusals, per-agent models and limits reaching every call, the request each backend sends (the Anthropic one unchanged, the OpenRouter one to `chat/completions` with the two headers, the bearer key and each agent's model), reply and error reading for both, the missing key and missing package messages, the self-model sentence, no model name in S's or V's prompt, the whole pipeline through the OpenRouter backend with fake SDK objects (and, with `openai` installed, through the real SDK over HTTP to a local stub).

## What was checked

- **Backward compatibility, on the wire.** The code before this change and the code now were each run, with no provider setting, through one interaction and one manual review against the same local stand-in for the Anthropic API: 4 requests each, identical in path, every header and every body field (timestamps masked), and identical results.
- **Mutation testing of the new lines.** 23 single-line changes (the base URL, each header, the system message, the finish-reason mapping, precedence of the environment over the file, the slash checks, client caching, the configured model and limit, the same-model test): the tests stopped all 23 after two gaps found by the first run were closed.
- **Not checked:** any live provider call; models other than Claude (the prompts were written and tested for Claude, and another model may follow the JSON format less closely, which shows as "unchecked"); OpenRouter's behaviour for reasoning budgets beyond what `max_tokens` controls; data handling by OpenRouter and the model's own provider.

## Added for the pip package: the settings of one `Pipeline`

`noetic_field.Pipeline(provider=..., api_key=..., agents={...})` carries its own provider, key and models (see the README). In this layer that is a `provider.Settings`, built and checked by `provider.make_settings` and made current for the length of one call by `provider.scoped(...)`:

- **Precedence:** the `Pipeline` arguments, then the environment, then the file, then the defaults. `load_config` takes the current settings into account, so `model_for`, `models`, `max_tokens_for` and `complete` all follow them.
- **Where they live:** in a context variable, not in `os.environ` and not in a module global, so another thread or another `Pipeline` never sees them. The N and V checks run in pool threads; the orchestrator therefore starts each of them inside a copy of the caller's context (`contextvars.copy_context().run`). The state file of a `Pipeline` is scoped the same way (`state.use_state_file`).
- **The API key** given to a `Pipeline` gets a client of its own, made once per key (`provider._keyed_clients`, indexed by a hash of the key), and is never read from or written to the environment. A key that plainly belongs to the other provider (`sk-ant-` for OpenRouter, `sk-or-` for Anthropic) is refused when the `Pipeline` is made and again when a call is made.
- **Not changed:** the prompts, the parsers, the checks, the fallbacks and every default. With no `Pipeline` involved nothing in this document is different.
- **Tests:** the suite is now 783 tests (772 without the `openai` package, which skips 11): `tests/test_pipeline.py` (56) covers the interface and the properties above, `tests/test_package.py` (12) runs the modules laid out as the wheel lays them out. `tests/test_sdk_integration.py` was made independent of the SDK's HTTP library and header names, which changed in `anthropic` 2 (`httpx2`, a bearer token).

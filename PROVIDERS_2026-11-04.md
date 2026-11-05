# Noetic Field: providers and models

Any agent (C, N, V, S) can run on any model OpenRouter routes (Anthropic, OpenAI, Google, DeepSeek, xAI and more), each on a different one, or on Anthropic directly. **This is a transport layer only.** What each agent is asked, how its answer is read and every check, fallback and the pipeline are the same whichever model answers; the code is `provider.py`, which `agents.call_model` calls.

## How to use it

Without any setting the system uses Anthropic, `claude-sonnet-4-6` for every agent, and `ANTHROPIC_API_KEY`.

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

- **Anthropic backend:** the Anthropic SDK, `api.anthropic.com`.
- **OpenRouter backend:** the `openai` SDK with `base_url="https://openrouter.ai/api/v1"` and the headers `HTTP-Referer: noetic-field` and `X-Title: Noetic Field`; the system prompt goes as the system message. The `openai` package is imported only when this backend is first used.
- **Replies** become one shape (`provider.Completion`): the text, a stop reason (`stop`/`length`/`content_filter` become `end_turn`/`max_tokens`/`refusal`, the names the agents already use) and the model that answered. Refusals, empty replies, truncation and SDK errors reach the agents the same way whichever backend answered.

## Model ids

Anthropic ids have no slash (`claude-sonnet-4-6`); OpenRouter ids are `vendor/model` with dots in versions (`anthropic/claude-sonnet-4.5`, not `anthropic/claude-sonnet-4-5`). Checked against OpenRouter's public model list on 2026-11-04: `anthropic/claude-sonnet-4.5`, `anthropic/claude-sonnet-4.6`, `google/gemini-2.5-pro`, `x-ai/grok-4.3`, `deepseek/deepseek-r1-0528` and `openai/gpt-5.1` were there; `anthropic/claude-sonnet-4-5` (hyphens), `google/gemini-pro-1.5`, `x-ai/grok-3` and `deepseek/deepseek-r1` were not. The example file uses ids that exist. The list changes: check [openrouter.ai/models](https://openrouter.ai/models).

## Behavior worth knowing

- **One provider for all agents.** `NOETIC_PROVIDER` is global; a mixed Anthropic-direct and OpenRouter setup is not supported.
- **A reply limit per agent** (`max_tokens`, `NOETIC_MAX_TOKENS_<agent>`). A reasoning model such as DeepSeek R1 spends part of the limit on its reasoning, and V's default limit is 600 tokens, so without a larger one V on such a model would often return nothing and every answer would be marked unchecked.
- **The self-model sentence about the values agent.** The self-model tells C that the values agent is the same underlying model. When V is configured on a different model than C that would be false, so `state.get_context_summary` swaps in a wording that says it is not (`state._OTHER_MODEL`). With one model, or with no configuration, the text is unchanged.
- **The system does not assume which model is answering.** No prompt of S or V names a model or vendor, and no `GET /models` call is made. The strings "Claude 3.5 Haiku" and "Claude Opus 4.1" in `agents.py` and `state.py` are citations of what the papers studied, not claims about the running model.

## Limits

- Models other than Claude: the prompts were written and tested for Claude, and another model may follow the JSON format less closely, which shows as "unchecked".
- OpenRouter's behavior for reasoning budgets beyond what `max_tokens` controls was not checked.
- Data handling by OpenRouter and by the model's own provider is theirs: read their terms.
- The automated tests do not call a live provider (see the README, "Status and limits").

## The settings of one `Pipeline`

`noetic_field.Pipeline(provider=..., api_key=..., agents={...})` carries its own provider, key and models (see the README). In this layer that is a `provider.Settings`, built and checked by `provider.make_settings` and made current for the length of one call by `provider.scoped(...)`:

- **Precedence:** the `Pipeline` arguments, then the environment, then the file, then the defaults. `load_config` takes the current settings into account, so `model_for`, `models`, `max_tokens_for` and `complete` all follow them.
- **Where they live:** in a context variable, not in `os.environ` and not in a module global, so another thread or another `Pipeline` never sees them. The N and V checks run in pool threads; the orchestrator therefore starts each of them inside a copy of the caller's context (`contextvars.copy_context().run`). The state file of a `Pipeline` is scoped the same way (`state.use_state_file`).
- **The API key** given to a `Pipeline` gets a client of its own, made once per key (`provider._keyed_clients`, indexed by a hash of the key), and is never read from or written to the environment. A key that plainly belongs to the other provider (`sk-ant-` for OpenRouter, `sk-or-` for Anthropic) is refused when the `Pipeline` is made and again when a call is made.

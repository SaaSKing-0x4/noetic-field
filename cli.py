"""
CLI for the Noetic Field system.

Commands:
  (blank)          interactive conversation mode
  --verbose        show reasoning trace, narration check, values check
  --self           show current self-model state
  --insight TEXT   add a manual insight to persistent memory
  --value TEXT     add a new value to the values circuit

Inside the conversation: exit | --self | --verbose | --review | --new | --history

Usage:
  python cli.py
  python cli.py --verbose
  python cli.py --self
  python cli.py --insight "LLMs confabulate by architecture"
  python cli.py --value "Always acknowledge what you don't know"
"""

import sys

if __package__:                 # installed (pip install noetic-field): the package's own modules
    from . import agents
    from . import llm_json as lj
    from . import state as state_module
    from .consoleutil import use_utf8
    from .orchestrator import NoeticOrchestrator
else:                           # run from a source folder (python cli.py): modules side by side
    import agents
    import llm_json as lj
    import state as state_module
    from consoleutil import use_utf8
    from orchestrator import NoeticOrchestrator


def _print_review(review: dict) -> None:
    print(f"\nHealth: {review.get('overall_health') or 'n/a'}")
    for p in review.get("patterns_found", []):
        print(f"  Pattern: {p}")
    for i in review.get("new_insights", []):
        print(f"  Insight: {i}")
    applied = review.get("applied") or {}
    if applied.get("skipped"):
        print(f"  [Not stored: {applied['skipped']}]")
    else:
        flagged = applied.get("flagged") or []
        print(f"  [Stored: {applied.get('insights', 0)} insight(s), {applied.get('signals', 0)} signal(s)"
              + (f", flagged: {', '.join(flagged)}" if flagged else "") + "]")


USAGE = """Usage:
  python cli.py [--verbose]          interactive conversation
  python cli.py --self               show the self-model
  python cli.py --insight TEXT       add an insight (at most {ins} characters)
  python cli.py --value TEXT         add a value (at most {val} characters)
  python cli.py --help               this text
Inside the conversation: exit | --self | --verbose | --review | --new | --history""".format(
    ins=state_module.MAX_INSIGHT_CHARS, val=state_module.MAX_VALUE_CHARS)
if __package__:                     # installed: the command is noetic-field (or python -m noetic_field.cli)
    USAGE = USAGE.replace("python cli.py", "noetic-field")


def _fail(message: str) -> None:
    print(f"{message}\n\n{USAGE}", file=sys.stderr)
    sys.exit(2)


def _text_argument(args: list, flag: str, limit: int) -> str:
    """The text after a flag, or an error exit. The text is whatever was typed after the flag, word for word (a word
    like --verbose or -h inside it is text), cleaned and measured exactly as the API does, so both interfaces accept
    the same input."""
    text = lj.clean_line(" ".join(args[1:]))
    if not text:
        _fail(f"{flag} needs text.")
    if len(text) > limit:
        _fail(f"{flag} text is too long ({len(text)} characters; the limit is {limit}).")
    return text


def _one_shot(command: str, args: list) -> None:
    """--self, --insight and --value: arguments are checked before anything is read or written, and they are not a
    session. A state file that cannot be read or written is reported in one line, not as a traceback."""
    if command == "--self":
        print(NoeticOrchestrator(count_session=False).show_self_model())
    elif command == "--insight":
        insight = _text_argument(args, "--insight", state_module.MAX_INSIGHT_CHARS)
        if NoeticOrchestrator(count_session=False).add_insight(insight):
            print(f"Insight saved: {insight}")
        else:
            print("Insight not saved: one like it is already kept (that one was refreshed).")
    else:
        value = _text_argument(args, "--value", state_module.MAX_VALUE_CHARS)
        if NoeticOrchestrator(count_session=False).add_value(value):
            print(f"Value added: {value}")
        else:
            print(f"Value not added: it is a placeholder, a duplicate, or the list already holds {state_module.MAX_VALUES} values.")


def main():
    use_utf8()                      # a model's answer can hold characters a Windows pipe or file cannot encode
    args = sys.argv[1:]
    verbose = False
    while args and args[0] in ("--verbose", "--help", "-h"):       # flags come before the command
        if args[0] != "--verbose":
            print(USAGE)
            return
        verbose = True
        args = args[1:]
    if args and args[0] not in ("--self", "--insight", "--value"):
        _fail(f"Unknown argument: {args[0]}")
    if args and args[0] == "--self" and ("--help" in args[1:] or "-h" in args[1:]):
        print(USAGE)
        return
    if args:                        # after --insight or --value everything is the text, word for word
        try:
            _one_shot(args[0], args)
        except OSError as e:
            print(f"[{e}]", file=sys.stderr)
            sys.exit(1)
        return

    if sys.stdin is None or getattr(sys.stdin, "closed", False):    # nothing can be read: do not count a session
        print("[No input is available, so there is nothing to do. Use --self, --insight or --value for one-shot commands.]",
              file=sys.stderr)
        sys.exit(1)
    try:
        orchestrator = NoeticOrchestrator()
    except OSError as e:
        print(f"[{e}]", file=sys.stderr)
        sys.exit(1)

    # Interactive conversation
    print("Noetic Field System — v1.0")
    print("Grounded in: On the Biology of a Large Language Model (Anthropic, 2025)")
    print("Four-agent pipeline: Computation | Narration | Values | Self-Review")
    print("Commands: 'exit' quit | '--self' self-model | '--verbose' toggle trace")
    print("          '--review' manual S-agent | '--new' new conversation | '--history' show turns")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").lstrip(chr(0xFEFF)).strip()     # a redirected file can begin with a byte-order mark
        except (EOFError, KeyboardInterrupt, RuntimeError, ValueError, OSError):    # RuntimeError, ValueError: stdin or stdout is gone
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Exiting.")
            break

        if user_input == "--self":
            try:
                print(orchestrator.show_self_model())
            except OSError as e:
                print(f"[The state file could not be read: {e}]")
            continue

        if user_input == "--verbose":
            verbose = not verbose
            print(f"Verbose mode: {'on' if verbose else 'off'}")
            continue

        if user_input == "--review":
            print("[Running S-agent self-review...]")
            try:
                _print_review(orchestrator.run_manual_review())
            except agents.AgentError as e:
                print(f"[Self-review failed: {e}]")
            except OSError as e:
                print(f"[Self-review failed: the result could not be saved: {e}]")
            continue

        if user_input == "--new":
            orchestrator.new_conversation()
            print("[New conversation started. Persistent state kept.]")
            continue

        if user_input == "--history":
            summary = orchestrator.conversation.summary()
            print(f"Conversation: {summary}")
            continue

        try:
            result = orchestrator.run(user_input, verbose=verbose)
        except ValueError as e:
            print(f"[{e}]")
            continue
        except agents.AgentError as e:
            print(f"[The request failed and nothing was saved: {e}]")
            continue
        except OSError as e:                      # the state file could not be written: the answer is discarded
            print(f"[The request failed and nothing was saved: {e}]")
            continue

        print(f"\nNoetic: {result['response']}")

        if result.get("review_error"):
            print(f"\n[Self-review: {result['review_error']}]")

        if verbose:
            print("\n--- Transparency Report ---")
            if result.get("confidence_defaulted"):
                if (result.get("parse_status") or {}).get("C") == "ok":
                    print("Confidence: not reported (the reply held no usable number)")
                else:
                    print("Confidence: not reported (the reply was cut off or could not be read as structured output)")
            else:
                print(f"Confidence: {result.get('confidence', 0):.0%} (self-reported)")
            print(f"Domain: {result.get('domain', 'n/a')}")
            print(f"Confabulation risk: {result.get('confabulation_risk', 'n/a')}")
            print(f"Values status: {result.get('values_status', 'n/a')}")
            status = result.get("parse_status")
            if status:
                print("Parse status: " + ", ".join(f"{k}={v}" for k, v in status.items()))

            steps = result.get("reasoning_steps", [])
            if steps:
                print("\nReasoning trace:")
                for i, s in enumerate(steps, 1):
                    print(f"  {i}. {s}")

            flags = result.get("uncertainty_flags", [])
            if flags:
                print("\nUncertainty flags:")
                for f in flags:
                    print(f"  - {f}")

            divs = result.get("divergence_notes", [])
            if divs:
                print("\nNarration divergences:")
                for d in divs:
                    print(f"  ! {d}")

            v_notes = result.get("v_notes", [])
            if v_notes:
                print("\nValues notes:")
                for n in v_notes:
                    print(f"  * {n}")

            if result.get("review_triggered"):
                print("\n[S-agent self-review triggered]")
                if result.get("review_summary"):
                    print(f"  Health: {result['review_summary']}")


if __name__ == "__main__":
    main()

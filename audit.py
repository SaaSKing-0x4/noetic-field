"""
Audit tools — surface patterns from the interaction log.

The interaction log is the system's own history. These tools make that history
readable: where did the narration check rate its own reasoning as risky? Where
did the values check flag or veto? What domains has the system reported
confidence in, and where has the self-review agent noted unexplained
divergences?

What the numbers are: counts of other agents' judgments of text (risk ratings,
values statuses) and of the confidence the model reported. They are heuristics
and self-reports, not measurements of what happens inside the model. The "dark
matter" signals are hypotheses recorded by the self-review agent, not
measurements; the research this system draws on does not give a share of the
computation that is unexplained.

Every report accepts an optional `state`, so a caller that already holds one,
such as full_health_report or the API (which passes its live state), reads one
consistent snapshot instead of loading the file again for each section.

A report only reads. It never creates, replaces or moves the state file: with no
state given it reads the file with state.try_load(). A missing file and an
unreadable one both report an empty history with a notice on stderr saying which
it was, and the file is left exactly as it was.
"""

import sys
from collections import Counter
from typing import Optional

if __package__:                 # installed (pip install noetic-field): the package's own modules
    from . import state as state_module
else:                           # run from a source folder: modules side by side
    import state as state_module


def _state(state: Optional[dict]) -> dict:
    if state is not None:
        return state
    loaded = state_module.try_load()
    if loaded is not None:
        return loaded
    path = state_module.state_path()
    if path.exists():
        print(f"[audit] {path} is unreadable: showing an empty report; the file was not touched",
              file=sys.stderr)
    else:                           # a wrong path or folder must not look like a brand-new system
        print(f"[audit] no state file at {path}: showing an empty report; nothing was created",
              file=sys.stderr)
    return state_module.fresh_state()


def _share(count: int, total: int) -> str:
    return f"{count:4d} ({100 * count // total:3d}%)" if total else f"{count:4d}"


def confabulation_report(state: Optional[dict] = None) -> str:
    state = _state(state)
    log = state.get("interaction_log", [])
    if not log:
        return "No interaction history yet."

    counts = Counter(entry.get("confabulation_risk", "unknown") for entry in log)
    total = len(log)

    lines = [
        f"=== Confabulation Report (last {total} interactions; ratings are the narration agent's judgments) ===",
        f"Low risk:     {_share(counts.get('low', 0), total)}",
        f"Medium risk:  {_share(counts.get('medium', 0), total)}",
        f"High risk:    {_share(counts.get('high', 0), total)}",
        f"Unknown:      {_share(counts.get('unknown', 0), total)}   (not checked, or the check could not be read)",
        "",
        "Recent high-risk interactions:",
    ]

    high_risk = [e for e in log if e.get("confabulation_risk") == "high"]
    for entry in high_risk[-5:]:
        lines.append(f"  [{str(entry.get('timestamp', ''))[:16]}] {entry.get('user_msg_preview', '')}")

    return "\n".join(lines)


def values_report(state: Optional[dict] = None) -> str:
    state = _state(state)
    log = state.get("interaction_log", [])
    if not log:
        return "No interaction history yet."

    counts = Counter(entry.get("values_status", "unchecked") for entry in log)
    total = len(log)

    lines = [
        f"=== Values Circuit Report (last {total} interactions) ===",
        f"Pass:      {_share(counts.get('pass', 0), total)}",
        f"Flag:      {_share(counts.get('flag', 0), total)}",
        f"Veto:      {_share(counts.get('veto', 0), total)}",
        f"Unchecked: {_share(counts.get('unchecked', 0), total)}   (the values check did not run, or could not be read)",
        "",
        "Current values:",
    ]
    for v in state.get("values", []):
        lines.append(f"  - {v}")

    lines.append("\nRecent flags/vetoes:")
    flagged = [e for e in log if e.get("values_status") in ("flag", "veto")]
    for entry in flagged[-5:]:
        lines.append(
            f"  [{str(entry.get('values_status', '?')).upper()}] [{str(entry.get('timestamp', ''))[:16]}] "
            f"{entry.get('user_msg_preview', '')}"
        )

    return "\n".join(lines)


def domain_report(state: Optional[dict] = None) -> str:
    state = _state(state)
    domains = state.get("domain_confidence", {})
    if not domains:
        return "No domain confidence data yet."

    lines = ["=== Domain Confidence Map (average confidence the model reported, smoothed) ==="]
    sorted_domains = sorted(domains.items(), key=lambda x: x[1].get("confidence", 0), reverse=True)
    for domain, info in sorted_domains:
        conf = info.get("confidence", 0)
        bar = "█" * int(conf * 20)
        flag = "  (flagged by self-review)" if info.get("flagged_at") else ""
        lines.append(f"  {domain:<30} {conf:.0%} {bar}  n={info.get('n', 1)}{flag}")

    return "\n".join(lines)


def dark_matter_report(state: Optional[dict] = None) -> str:
    state = _state(state)
    dm = state.get("uncertainty_map", {})
    insights = state.get("insights", [])
    sm = state["self_model"]

    health = sm.get("last_review_health") or (
        "reviewed, but the review gave no health text" if sm.get("last_review_at") else "not yet reviewed")
    lines = [
        "=== Dark Matter Map (hypotheses from the self-review agent, not measurements) ===",
        f"Known unknown signals: {len(dm)}",
        f"Accumulated insights:  {len(insights)}",
        f"Last review health:    {health}",
        *([f"Last review tension:   {sm['last_review_tension']}"] if sm.get("last_review_tension") else []),
        "",
    ]

    if sm.get("last_review_patterns"):
        lines.append("Last review patterns found:")
        for p in sm["last_review_patterns"]:
            lines.append(f"  - {p}")
        lines.append("")

    if dm:
        lines.append("Uncertainty signals:")
        for key, val in list(dm.items())[-10:]:
            seen = f" (seen {val.get('count', 1)}x)" if val.get("count", 1) > 1 else ""
            lines.append(f"  [{val.get('source', '?')}] {val.get('signal', key)}{seen}")

    if insights:
        lines.append("\nRecent insights (last 5):")
        for ins in insights[-5:]:
            lines.append(f"  [{str(ins.get('timestamp', ''))[:10]}] ({ins.get('source', '?')}) {ins.get('insight', '')}")

    return "\n".join(lines)


def full_health_report(state: Optional[dict] = None) -> str:
    state = _state(state)
    sm = state["self_model"]

    sections = [
        "=== Noetic Field Health Report ===",
        f"Session count: {sm.get('session_count', 0)}",
        f"Last session:  {str(sm['last_session'])[:16] if sm.get('last_session') else 'never'}",
        f"Interactions:  {sm.get('total_interactions', 0)} in total ({len(state.get('interaction_log', []))} in the log)",
        f"Insights:      {len(state.get('insights', []))}",
        f"Values:        {len(state.get('values', []))}",
        "",
        confabulation_report(state),
        "",
        values_report(state),
        "",
        domain_report(state),
        "",
        dark_matter_report(state),
    ]

    return "\n".join(sections)


if __name__ == "__main__":
    import sys

    if __package__:
        from .consoleutil import use_utf8
    else:
        from consoleutil import use_utf8
    use_utf8()                      # the bars use a block character that a Windows pipe or file cannot encode
    cmd = sys.argv[1] if len(sys.argv) > 1 else "full"
    reports = {
        "confabulation": confabulation_report,
        "values": values_report,
        "domains": domain_report,
        "dark": dark_matter_report,
        "full": full_health_report,
    }
    fn = reports.get(cmd)
    if fn is None:
        print(f"unknown report '{cmd}'. Valid: {', '.join(reports)}", file=sys.stderr)
        sys.exit(2)
    print(fn())

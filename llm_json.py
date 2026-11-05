"""
Tolerant JSON extraction and coercion for LLM output.

Everything in this module is total: no function raises on bad model output. A
reply that cannot be parsed yields None (or a safe default), never an
exception, so a malformed agent reply degrades one field instead of crashing a
request. Each rule below closes a bug class found when this system was
reviewed:

  * Only TOP-LEVEL {...} spans are considered, left to right. After a span that
    fails to parse and contains a colon (a failed attempt at JSON, whatever
    quote style it used), nothing inside or after it is promoted; otherwise a
    quoted '{"status": "pass"}' inside a malformed reply could replace the real
    verdict.
  * Output cut off by max_tokens is repaired by closing strings and brackets
    (a cut inside a \\uXXXX escape drops the fragment), and the result is
    flagged 'repaired' so callers can refuse to persist it.
  * Stray backslashes (\\d, \\(, \\$) and raw control characters inside strings
    are escaped before parsing; trailing commas are dropped.
  * NaN / Infinity are rejected, and arrays, primitives and null are rejected.
  * Integers too large for a float (more than 308 digits) read as infinite, so
    they are "not a usable number" instead of an OverflowError.
  * Work stays linear in the input size, however deeply the text nests; no
    regular expression here can backtrack quadratically.
  * Lone surrogates (invalid in UTF-8) are replaced wherever model text becomes
    a string, so a reply can never make saving or prompting fail. Text with
    nothing left but replacement characters is empty (clean_line, to_str_list).
  * A short answer to a yes/no question is read by its first word, so near
    spellings ('False.', 'No.', 'does not follow', ['yes']) mean what the exact
    ones mean; anything else is "unknown", never a guess.
  * Untrusted text placed in a prompt goes through fence(), which defangs every
    opener of the tag it sits in, whatever follows the tag name.
"""

import json
import math
import re
from typing import Any, Callable, Iterable, Iterator, Optional

MAX_CHARS = 200_000          # model output is bounded by max_tokens; this is a hard backstop
MAX_BRACE_SPANS = 40
MAX_REPAIR_COMMAS = 30

_HEX = frozenset("0123456789abcdefABCDEF")
_PLACEHOLDER = re.compile(r"^(none|n/a|na|null|nil|nothing|no|-+|\[\]|\{\})\.?$", re.IGNORECASE)
_NUMBER = re.compile(r"^(-?\d+(?:\.\d+)?|-?\.\d+)\s*(%?)$")
_FENCE_HEAD = re.compile(r"[\w-]{0,30}[ \t]{0,8}\r?\n?")       # the optional language tag after an opening ```
# Characters that never belong in text shown to a person or sent to a model: C0 controls except tab, line feed and
# carriage return; DEL; C1 controls; and lone surrogates (invalid in UTF-8). Built with chr(): no escapes in this source.
_BAD_RANGES = ((0x00, 0x08), (0x0B, 0x0C), (0x0E, 0x1F), (0x7F, 0x9F), (0xD800, 0xDFFF))
_BAD_CHARS = re.compile("[" + "".join(chr(a) + "-" + chr(b) for a, b in _BAD_RANGES) + "]")
_REPLACEMENT = chr(0xFFFD)


def scrub(s: str) -> str:
    """Replace control characters (the escape character included, so model text cannot drive a terminal) and lone
    surrogates with U+FFFD. Tab, line feed, carriage return, emoji pairs and everything else are untouched."""
    return _BAD_CHARS.sub(_REPLACEMENT, s)


# ── scanning ──────────────────────────────────────────────────────────────────

def match_brace(s: str, start: int) -> int:
    """Index of the '}' matching the '{' at s[start] (string and escape aware), or -1."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def lenient_fix(s: str) -> str:
    """Repair what models commonly get wrong inside JSON: stray backslashes,
    raw control characters in strings, and trailing commas."""
    out: list[str] = []
    in_str = False
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if in_str:
            if c == "\\":
                nxt = s[i + 1] if i + 1 < n else None
                if nxt is not None and nxt in '"\\/bfnrt':
                    out.append(c + nxt)
                    i += 2
                    continue
                if nxt == "u" and i + 6 <= n and all(ch in _HEX for ch in s[i + 2:i + 6]):
                    out.append(s[i:i + 6])
                    i += 6
                    continue
                out.append("\\\\")           # a stray backslash: keep it as a literal backslash
                i += 1
                continue
            if c == '"':
                out.append(c)
                in_str = False
                i += 1
                continue
            code = ord(c)
            if code < 0x20:
                out.append("\\n" if code == 10 else "\\r" if code == 13 else "\\t" if code == 9 else "\\u%04x" % code)
                i += 1
                continue
            out.append(c)
            i += 1
            continue
        if c == '"':
            in_str = True
        if c == ",":
            j = i + 1
            while j < n and s[j].isspace():
                j += 1
            if j < n and s[j] in "}]":
                i += 1
                continue
        out.append(c)
        i += 1
    return "".join(out)


def _reject_constant(name: str) -> Any:
    raise ValueError("non-finite JSON constant: " + name)


def safe_parse_json(text: Any) -> Optional[dict]:
    """A JSON object, or None. Arrays, primitives, null and NaN/Infinity are rejected."""
    if not isinstance(text, str) or not text:
        return None
    tries = [text]
    fixed = lenient_fix(text)
    if fixed != text:
        tries.append(fixed)
    for t in tries:
        try:
            value = json.loads(t, parse_constant=_reject_constant)
        except (ValueError, RecursionError):
            continue
        return value if isinstance(value, dict) else None
    return None


def _scan_state(s: str, start: int, end: int) -> tuple[str, bool, bool]:
    """(closers for the brackets still open at `end`, inside-a-string?, dangling backslash?)"""
    stack: list[str] = []
    in_str = False
    esc = False
    for i in range(start, end):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            stack.append("}")
        elif c == "[":
            stack.append("]")
        elif (c == "}" or c == "]") and stack:
            stack.pop()
    return "".join(reversed(stack)), in_str, esc


def _trim_tail(s: str) -> str:
    return s.rstrip(" \t\r\n,")


def _drop_partial_unicode_escape(body: str) -> str:
    """If a cut-off string ends inside a \\uXXXX escape (a backslash, 'u' and fewer than four hex digits),
    drop the fragment. An escaped backslash followed by 'u12' is ordinary text and is kept."""
    j = len(body)
    while j > 0 and len(body) - j < 3 and body[j - 1] in _HEX:
        j -= 1
    if j > 0 and body[j - 1] == "u":
        k = j - 1
        run = 0
        while k > 0 and body[k - 1] == "\\":
            k -= 1
            run += 1
        if run % 2 == 1:
            return body[:j - 2]               # the backslash that opens the escape, the 'u' and the digits
    return body


def _drop_dangling_high_surrogate(body: str) -> str:
    """A cut-off string that ends in the escape for the first half of a surrogate pair (an emoji written as two
    \\uXXXX escapes) has lost the second half; drop the first so the repaired text is a clean prefix."""
    tail = body[-6:]
    if (len(tail) == 6 and tail[0] == "\\" and tail[1] == "u" and tail[2] in "dD" and tail[3] in "89abAB"
            and all(ch in _HEX for ch in tail[4:])):
        k = len(body) - 6
        run = 0
        while k > 0 and body[k - 1] == "\\":
            k -= 1
            run += 1
        if run % 2 == 0:                       # the backslash that opens the escape is not itself escaped
            return body[:-6]
    return body


def repair_candidates(s: str, start: int) -> Iterator[str]:
    """Candidate repairs for output that was cut off, or is malformed, from `start` on."""
    end = len(s)
    closers, in_str, esc = _scan_state(s, start, end)
    body = s[start:]
    if in_str:
        # A cut after the backslash that opens the second half of an escaped pair leaves the first half dangling too.
        body = _drop_dangling_high_surrogate(body[:-1] if esc else _drop_partial_unicode_escape(body))
        body += '"'
    yield _trim_tail(body) + closers
    commas: list[int] = []          # only the last few commas are ever tried, so cost stays linear
    in_str = False
    esc = False
    for i in range(start, end):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == ",":
            commas.append(i)
            if len(commas) > MAX_REPAIR_COMMAS:
                commas.pop(0)
    for pos in reversed(commas):
        yield s[start:pos] + _scan_state(s, start, pos)[0]


def _fenced_body(t: str) -> Optional[str]:
    """The text inside a reply that is one ``` fenced block, or None. Plain string operations plus one anchored,
    bounded match: nothing here can backtrack quadratically."""
    if len(t) < 6 or not t.startswith("```") or not t.endswith("```"):
        return None
    inner = t[3:-3]
    head = _FENCE_HEAD.match(inner)
    return (inner[head.end():] if head else inner).strip()


def extract_json(text: Any, expect_keys: Iterable[str] = ()) -> Optional[dict]:
    """
    Find the JSON object in a model reply.

    Returns {"value": dict, "str": str, "how": "fenced"|"direct"|"brace"|"repaired", "weak": bool}
    or None. `weak` means an object parsed but none of `expect_keys` is in it.
    """
    expect = tuple(expect_keys)
    if text is None:
        return None
    t = (text if isinstance(text, str) else str(text)).lstrip(chr(0xFEFF)).strip()
    if not t:
        return None
    t = t[:MAX_CHARS]
    weak: Optional[dict] = None

    def take(candidate: str, how: str):
        """A result dict, False (did not parse) or None (parsed, but not the expected shape)."""
        nonlocal weak
        value = safe_parse_json(candidate)
        if value is None:
            return False
        if not expect or any(k in value for k in expect):
            return {"value": value, "str": candidate, "how": how, "weak": False}
        if weak is None:
            weak = {"value": value, "str": candidate, "how": how, "weak": True}
        return None

    body = _fenced_body(t)                                      # the whole reply is one fenced block
    if body is not None:
        r = take(body, "fenced")
        if r:
            return r
    r = take(t, "direct")
    if r:
        return r
    start_from = 0
    repair_at = -1
    for _ in range(MAX_BRACE_SPANS):                            # top-level {...} spans, left to right
        i = t.find("{", start_from)
        if i < 0:
            break
        j = match_brace(t, i)
        if j < 0:
            repair_at = i                                       # unbalanced: truncated or malformed from here on
            break
        span = t[i:j + 1]
        r = take(span, "brace")
        if r:
            return r
        if r is False and ":" in span:                          # a failed JSON attempt: promote nothing inside or after it
            repair_at = i
            break
        start_from = j + 1
    if repair_at >= 0:
        for candidate in repair_candidates(t, repair_at):
            r = take(candidate, "repaired")
            if r:
                return r
    return weak


def parse_with(text: Any, expect_keys: Iterable[str], normalize: Callable[[dict], dict],
               fallback: Callable[[str], dict], truncated: bool = False) -> dict:
    """
    extract_json -> normalize, falling back to `fallback(raw)`. Never raises.
    The result always carries parse_status ('ok' | 'repaired' | 'fallback') and parse_how.
    """
    raw = "" if text is None else (text if isinstance(text, str) else str(text))
    hit = None
    try:
        hit = extract_json(raw, expect_keys)
    except Exception:                                           # extraction is best effort by design
        hit = None
    data = None
    status = "fallback"
    if hit and not hit["weak"]:
        try:
            data = normalize(hit["value"])
            status = "repaired" if hit["how"] == "repaired" else "ok"
        except Exception:
            data = None
    if data is None:
        data = fallback(raw)
        status = "fallback"
    data["parse_status"] = status
    data["parse_how"] = hit["how"] if hit and not hit["weak"] and status != "fallback" else "none"
    if truncated:
        data["truncated"] = True
    return data


def looks_broken(raw: str) -> bool:
    """True if a reply that failed to parse still looks like structured output."""
    return bool(re.match(r"\s*(\{|```|\[)", raw or ""))


# ── coercion ──────────────────────────────────────────────────────────────────

def to_num(x: Any) -> float:
    if isinstance(x, bool):
        return math.nan
    if isinstance(x, int):
        try:
            return float(x)
        except OverflowError:                 # more than ~308 digits: not a usable number, never an exception
            return math.inf if x > 0 else -math.inf
    if isinstance(x, float):
        return x
    if isinstance(x, str):
        m = _NUMBER.match(x.strip())
        if m:
            v = float(m.group(1))
            return v / 100 if m.group(2) else v
    return math.nan


def to_confidence(x: Any) -> Optional[float]:
    """A float in [0, 1], or None when x is not a usable number (NaN and inf included)."""
    v = to_num(x)
    if not math.isfinite(v):
        return None
    if 1 < v < 2:
        v = 1.0                          # just over 1: an over-range probability, not 1.5 percent
    elif 2 <= v <= 100:
        v = v / 100                      # 85 -> 0.85
    return max(0.0, min(1.0, v))


def to_int(x: Any, default: int = 0, lo: int = 0, hi: int = 10**9) -> int:
    v = to_num(x)
    if not math.isfinite(v):
        return default
    return max(lo, min(hi, int(v)))


def to_str(x: Any, max_len: int = 2000) -> str:
    if isinstance(x, str):
        return scrub(x).strip()[:max_len].rstrip()          # rstrip: a cut on a space must not break idempotence
    if x is None:
        return ""
    if isinstance(x, (bool, int, float)):
        return str(x)[:max_len]
    try:
        return scrub(json.dumps(x, ensure_ascii=False))[:max_len]
    except (TypeError, ValueError, RecursionError):
        return ""


# How a short free-text answer to a yes/no question is read: by its first word (or first two, for "does not").
# Anything outside these lists is "unknown", never a guess.
_NO_FIRST = frozenset((
    "false", "no", "n", "0", "none", "not", "never", "nothing", "neither", "nil", "null", "na", "n/a", "absent",
    "undetected", "cannot", "can't", "doesn't", "don't", "isn't", "aren't", "wasn't", "weren't", "didn't"))
_NO_AUX = frozenset(("does", "do", "did", "is", "are", "was", "were", "can", "could", "would", "will", "has", "have"))
_YES_FIRST = frozenset((
    "true", "yes", "y", "1", "yep", "yeah", "detected", "present", "partial", "partially", "possible", "possibly",
    "likely", "probable", "probably", "some", "mild", "mildly", "slight", "slightly", "moderate", "moderately",
    "significant", "significantly", "strong", "strongly", "follows", "matches", "consistent"))
_WORD = re.compile(r"[a-z0-9/]+(?:'[a-z]+)?")


def _reading(x: Any) -> Optional[bool]:
    """True or False if a short text (or a one-item list holding one) clearly says yes or no, else None."""
    if isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    if isinstance(x, bool):
        return x
    if not isinstance(x, str):
        return None
    words = _WORD.findall(x[:120].lower())
    if not words:
        return None
    first = words[0]
    if first in _NO_FIRST or (first in _NO_AUX and len(words) > 1 and words[1] in ("not", "n't")):
        return False
    if first in _YES_FIRST:
        return True
    return None


def to_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str) and not x.strip():
        return False
    if isinstance(x, (str, list, tuple)):
        r = _reading(x)
        return default if r is None else r
    if isinstance(x, int):
        return x != 0
    if isinstance(x, float):
        return x != 0 if math.isfinite(x) else default
    return default


def to_tri(x: Any) -> Optional[bool]:
    """True, False, or None (unknown). Near spellings ('False.', 'No.', 'does not follow', ['false']) read like the
    exact ones, so a check cannot contradict itself in a spelling that slips past."""
    if isinstance(x, bool):
        return x
    return _reading(x)


def is_placeholder(s: str) -> bool:
    return bool(_PLACEHOLDER.match(s))


def to_str_list(x: Any, max_items: int = 12, max_len: int = 500) -> list[str]:
    """A list of non-empty strings. A bare string becomes lines, never characters."""
    if isinstance(x, list):
        items = x
    elif isinstance(x, str):
        items = x.split("\n") if x.strip() else []
    elif isinstance(x, dict):
        items = list(x.values())
    else:
        items = []
    out: list[str] = []
    for it in items:
        if isinstance(it, str):
            s = it
        elif isinstance(it, dict):
            v = None
            for k in ("step", "text", "note", "description", "content", "value"):
                if it.get(k) is not None:
                    v = it[k]
                    break
            s = to_str(v if v is not None else it, max_len)
        elif isinstance(it, list):
            s = " ".join(p if isinstance(p, str) else to_str(p, max_len) for p in it)
        elif it is None:
            continue
        else:
            s = to_str(it, max_len)
        s = " ".join(scrub(s).split())[:max_len].rstrip()
        if not s or is_placeholder(s) or not s.replace(_REPLACEMENT, "").strip():
            continue                                            # nothing but replacement characters is nothing
        out.append(s)
        if len(out) >= max_items:
            break
    return out


def clean_line(x: Any, max_len: Optional[int] = None) -> str:
    """One line of text for storage: control characters replaced, whitespace collapsed, cut to `max_len` if given.
    Empty if all that is left is replacement characters (what control characters become): that is no text at all."""
    s = x if isinstance(x, str) else to_str(x, 10_000)
    t = " ".join(scrub(s).split())
    if max_len is not None:
        t = t[:max_len].rstrip()
    return t if t.replace(_REPLACEMENT, "").strip() else ""


def one_of(x: Any, allowed: Iterable[str], default: str, aliases: Optional[dict] = None) -> str:
    allowed = tuple(allowed)
    if isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    s = ("" if x is None else str(x))[:80].strip().lower().rstrip(".!?\"'`:; \t\r\n")
    if s in allowed:
        return s
    if aliases and aliases.get(s) in allowed:
        return aliases[s]
    return default


def clip(s: Any, n: int) -> str:
    t = " ".join(scrub(str("" if s is None else s)).split())
    return t if len(t) <= n else t[: max(0, n - 1)] + "…"


def norm_domain(x: Any) -> str:
    """Lowercase, trimmed, at most 40 chars. Only strings and numbers can be domains."""
    if isinstance(x, bool):
        return ""
    if isinstance(x, (int, float)):
        raw = str(x)
    elif isinstance(x, str):
        raw = x
    else:
        return ""
    d = scrub(raw[:80]).lower().lstrip("\"'` \t\r\n").rstrip("\"'`. \t\r\n")
    return " ".join(d.split())[:40].rstrip()


def slug(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str("" if s is None else s).lower()).strip("_")


def fence(tag: str, text: Any) -> str:
    """
    Wrap untrusted text in tags it cannot close. Structural rule, not a list of variants: every '<' (and the
    full-width '＜' and the small '﹤', which Unicode compatibility normalization turns into '<') inside the text
    becomes '&lt;', so no text can contain any tag at all, whatever invisible characters, attributes, case or
    spacing it uses. Only the fence's own two tags carry a real '<'.
    """
    raw = scrub("" if text is None else str(text))
    safe = raw.replace("<", "&lt;").replace(chr(0xFF1C), "&lt;").replace(chr(0xFE64), "&lt;")
    return f"<{tag}>\n{safe}\n</{tag}>"

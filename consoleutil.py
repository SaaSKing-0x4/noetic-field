"""
Console encoding.

On Windows a redirected or piped stdout uses the legacy code page (cp1252), which
cannot encode characters such as the block used in audit bars or an "≈" in a
model's answer; printing one raises UnicodeEncodeError. use_utf8() makes output
UTF-8 and replaces anything still unencodable, so printing never fails.
"""

import sys


def use_utf8() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):          # stdin too: piped UTF-8 text is decoded as cp1252 otherwise
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass            # not a text stream that can be reconfigured: leave it as it is

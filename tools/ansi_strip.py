"""Strip ANSI escape sequences from subprocess output.

Used by terminal_tool, code_execution_tool, and process_registry to clean
command output before returning it to the model.  This prevents ANSI codes
from entering the model's context — which is the root cause of models
copying escape sequences into file writes.

Covers the full ECMA-48 spec: CSI (including private-mode ``?`` prefix,
colon-separated params, intermediate bytes), OSC (BEL and ST terminators),
DCS/SOS/PM/APC string sequences, nF multi-byte escapes, Fp/Fe/Fs
single-byte escapes, and 8-bit C1 control characters.
"""

import re

# Fast-path check — skip the scanner when no escape-like bytes are present.
_HAS_ESCAPE = re.compile(r"[\x1b\x80-\x9f]")

# C0 control characters (minus tab/newline/carriage-return, handled
# separately) plus DEL. Bare controls can survive strip_ansi() even though
# complete and incomplete escape sequences are removed. They are still
# dangerous or garbled when echoed back to a terminal (BEL rings,
# backspace/DEL overwrite, NUL truncates in some terminals).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Fast-path check for sanitize_display_text — any C0 control (except
# tab/newline), CR, DEL, ESC, or C1 byte triggers the slow path.
_HAS_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

# Unicode TAG characters (U+E0000–U+E007F).  Deprecated as language tags,
# these render as nothing in every terminal and chat UI but are perfectly
# visible to an LLM tokenizer — the classic "ASCII smuggling" prompt-injection
# channel (hide `\u{E0069}\u{E0067}\u{E006E}...` = invisible instructions
# inside otherwise benign tool output).  Ported from block/goose#10746.
#
# The ONLY legitimate modern use is emoji tag sequences (Unicode TR51):
# a U+1F3F4 black-flag base followed by tag spec characters and the
# U+E007F CANCEL TAG terminator (e.g. the flags of Scotland/Wales/England).
# goose strips those too; we preserve them — same rationale as keeping ZWJ
# inside emoji sequences.
_UNICODE_TAG_SUB_RE = re.compile(
    r"(\U0001F3F4[\U000E0020-\U000E007E]+\U000E007F)"  # valid emoji tag seq (kept)
    r"|[\U000E0000-\U000E007F]"                        # any other tag char (stripped)
)

# Fast-path check — plane-14 tag chars only.
_HAS_UNICODE_TAG = re.compile(r"[\U000E0000-\U000E007F]")


def _skip_csi(text: str, index: int) -> int:
    """Return the first index after one CSI sequence, complete or partial."""
    length = len(text)
    in_intermediates = False
    while index < length:
        codepoint = ord(text[index])
        if not in_intermediates and 0x30 <= codepoint <= 0x3F:
            index += 1
            continue
        if 0x20 <= codepoint <= 0x2F:
            in_intermediates = True
            index += 1
            continue
        if 0x40 <= codepoint <= 0x7E:
            return index + 1
        # Invalid input terminates the control sequence without consuming the
        # first ordinary/control character that follows it.
        return index
    # A partial CSI at end-of-input is control data, not visible text.
    return length


def _skip_control_string(
    text: str,
    index: int,
    *,
    bell_terminates: bool,
) -> int:
    """Return the index after an OSC/DCS/SOS/PM/APC string.

    The scan is monotonic. An incomplete string consumes the remainder rather
    than leaking attacker-controlled terminal payload into visible output.
    """
    length = len(text)
    while index < length:
        character = text[index]
        if bell_terminates and character == "\x07":
            return index + 1
        if character == "\x9c":  # 8-bit ST
            return index + 1
        if character == "\x1b" and index + 1 < length and text[index + 1] == "\\":
            return index + 2
        index += 1
    return length


def _skip_escape(text: str, index: int) -> int:
    """Return the index after a 7-bit ESC sequence starting at ``index``."""
    length = len(text)
    next_index = index + 1
    if next_index >= length:
        return length
    introducer = text[next_index]
    if introducer == "[":
        return _skip_csi(text, next_index + 1)
    if introducer == "]":
        return _skip_control_string(
            text,
            next_index + 1,
            bell_terminates=True,
        )
    if introducer in "PX^_":
        return _skip_control_string(
            text,
            next_index + 1,
            bell_terminates=False,
        )

    codepoint = ord(introducer)
    if 0x20 <= codepoint <= 0x2F:
        next_index += 1
        while next_index < length and 0x20 <= ord(text[next_index]) <= 0x2F:
            next_index += 1
        if next_index < length and 0x30 <= ord(text[next_index]) <= 0x7E:
            return next_index + 1
        return next_index
    if 0x30 <= codepoint <= 0x7E:
        return next_index + 1
    # A bare ESC before an unrelated byte removes only ESC.
    return next_index


def _strip_ansi_single_pass(text: str) -> str:
    """Strip ECMA-48 controls with a monotonic finite-state scan."""
    length = len(text)
    parts: list[str] = []
    literal_start = 0
    index = 0
    while index < length:
        character = text[index]
        codepoint = ord(character)
        if character == "\x1b":
            if literal_start < index:
                parts.append(text[literal_start:index])
            index = _skip_escape(text, index)
            literal_start = index
            continue
        if 0x80 <= codepoint <= 0x9F:
            if literal_start < index:
                parts.append(text[literal_start:index])
            if character == "\x9b":
                index = _skip_csi(text, index + 1)
            elif character == "\x9d":
                index = _skip_control_string(
                    text,
                    index + 1,
                    bell_terminates=True,
                )
            elif character in {"\x90", "\x98", "\x9e", "\x9f"}:
                index = _skip_control_string(
                    text,
                    index + 1,
                    bell_terminates=False,
                )
            else:
                index += 1
            literal_start = index
            continue
        index += 1
    if literal_start < length:
        parts.append(text[literal_start:])
    return "".join(parts)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text.

    Returns the input unchanged (fast path) when no ESC or C1 bytes are
    present.  Safe to call on any string — clean text passes through
    with negligible overhead.
    """
    if not text or not _HAS_ESCAPE.search(text):
        return text
    return _strip_ansi_single_pass(text)


def sanitize_display_text(text: str) -> str:
    """Sanitize stored/untrusted text before echoing it to a terminal.

    Removes ANSI/ECMA-48 escape sequences AND bare control characters,
    preserving only newlines and tabs (carriage returns are normalized
    to newlines so ``\\r``-overwrite spoofing can't hide content).

    Use this when re-rendering conversation history or other persisted
    text in a terminal UI (e.g. the ``/resume`` recap): a message that
    arrived with embedded escapes — pasted content, gateway-origin
    text, or model output echoing injected tool results — must not be
    able to clear the screen, retitle the window, move the cursor, or
    restyle adjacent UI when replayed. Rich's ``Text()`` does NOT
    neutralize raw escape bytes, so sanitization has to happen before
    display. Mirrors openai/codex#31494 (``sanitize_user_text``).
    """
    if not text or not _HAS_CONTROL.search(text):
        return text
    text = strip_ansi(text)
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_CHARS_RE.sub("", text)


def strip_unicode_tags(text: str) -> str:
    """Remove invisible Unicode TAG characters (U+E0000–U+E007F) from text.

    Tag characters are invisible in terminals and chat UIs but fully visible
    to LLM tokenizers, making them a prompt-injection smuggling channel for
    untrusted tool output (MCP servers, web content).  Valid emoji tag
    sequences (U+1F3F4 base + tag spec + U+E007F CANCEL TAG — regional
    flags like Scotland/Wales) are preserved.

    Returns the input unchanged (fast path) when no plane-14 tag characters
    are present.  Ported from block/goose#10746.
    """
    if not text or not _HAS_UNICODE_TAG.search(text):
        return text
    return _UNICODE_TAG_SUB_RE.sub(lambda m: m.group(1) or "", text)

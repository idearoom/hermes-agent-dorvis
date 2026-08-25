"""Comprehensive tests for ANSI escape sequence stripping (ECMA-48).

The strip_ansi function in tools/ansi_strip.py is the source-level fix for
ANSI codes leaking into the model's context via terminal/execute_code output.
It must strip ALL terminal escape sequences while preserving legitimate text.
"""

from tools.ansi_strip import sanitize_display_text, strip_ansi


class TestStripAnsiBasicSGR:
    """Select Graphic Rendition — the most common ANSI sequences."""

    def test_reset(self):
        assert strip_ansi("\x1b[0m") == ""


    def test_truecolor_colon_separated(self):
        """Modern terminals use colon-separated SGR params."""
        assert strip_ansi("\x1b[38:2:255:0:0m") == ""
        assert strip_ansi("\x1b[48:2:0:255:0m") == ""


class TestStripAnsiCSIPrivateMode:
    """CSI sequences with ? prefix (DEC private modes)."""

    def test_cursor_show_hide(self):
        assert strip_ansi("\x1b[?25h") == ""
        assert strip_ansi("\x1b[?25l") == ""


    def test_bracketed_paste(self):
        assert strip_ansi("\x1b[?2004h") == ""


class TestStripAnsiCSIIntermediate:
    """CSI sequences with intermediate bytes (space, etc.)."""

    def test_cursor_shape(self):
        assert strip_ansi("\x1b[0 q") == ""
        assert strip_ansi("\x1b[2 q") == ""
        assert strip_ansi("\x1b[6 q") == ""


class TestStripAnsiOSC:
    """Operating System Command sequences."""

    def test_bel_terminator(self):
        assert strip_ansi("\x1b]0;title\x07") == ""


    def test_hyperlink_preserves_text(self):
        assert strip_ansi(
            "\x1b]8;;https://example.com\x1b\\click\x1b]8;;\x1b\\"
        ) == "click"

    def test_unterminated_osc_is_dropped_instead_of_rescanned_or_leaked(self):
        assert strip_ansi("visible\x1b]" + ("untrusted" * 1_000)) == "visible"

    def test_repeated_unterminated_osc_has_a_single_pass_work_bound(self):
        class CountingText(str):
            inspections = 0

            def __getitem__(self, key):
                if isinstance(key, int):
                    self.inspections += 1
                return super().__getitem__(key)

        raw = CountingText("\x1b]" * 50_000)

        assert strip_ansi(raw) == ""
        # A stateful scanner inspects characters monotonically. This bound is
        # deterministic and catches a fallback to suffix-rescanning regexes.
        assert 0 < raw.inspections <= 2 * len(raw) + 8


class TestStripAnsiDECPrivate:
    """DEC private / Fp escape sequences."""

    def test_save_restore_cursor(self):
        assert strip_ansi("\x1b7") == ""
        assert strip_ansi("\x1b8") == ""

    def test_keypad_modes(self):
        assert strip_ansi("\x1b=") == ""
        assert strip_ansi("\x1b>") == ""


class TestStripAnsiFe:
    """Fe (C1 as 7-bit) escape sequences."""

    def test_reverse_index(self):
        assert strip_ansi("\x1bM") == ""


    def test_index_and_newline(self):
        assert strip_ansi("\x1bD") == ""
        assert strip_ansi("\x1bE") == ""


class TestStripAnsiNF:
    """nF (character set selection) sequences."""

    def test_charset_selection(self):
        assert strip_ansi("\x1b(A") == ""
        assert strip_ansi("\x1b(B") == ""
        assert strip_ansi("\x1b(0") == ""


class TestStripAnsiDCS:
    """Device Control String sequences."""

    def test_dcs(self):
        assert strip_ansi("\x1bP+q\x1b\\") == ""

    def test_all_control_string_families_and_partial_sequences(self):
        for introducer in "PX^_":  # DCS, SOS, PM, APC
            assert strip_ansi(f"left\x1b{introducer}payload\x1b\\right") == "leftright"
            assert strip_ansi(f"left\x1b{introducer}unterminated") == "left"

    def test_partial_csi_and_nf_sequences_are_not_leaked(self):
        assert strip_ansi("left\x1b[38;2") == "left"
        assert strip_ansi("left\x1b(") == "left"

    def test_bare_escape_removal_preserves_unrelated_following_control(self):
        assert strip_ansi("left\x1b\nright") == "left\nright"


class TestStripAnsi8BitC1:
    """8-bit C1 control characters."""

    def test_8bit_csi(self):
        assert strip_ansi("\x9b31m") == ""
        assert strip_ansi("\x9b38;2;255;0;0m") == ""

    def test_8bit_standalone(self):
        assert strip_ansi("\x9c") == ""
        assert strip_ansi("\x9d") == ""
        assert strip_ansi("\x90") == ""

    def test_8bit_string_controls_complete_and_partial(self):
        for introducer in ("\x90", "\x98", "\x9e", "\x9f"):
            assert strip_ansi(f"left{introducer}payload\x9cright") == "leftright"
            assert strip_ansi(f"left{introducer}unterminated") == "left"
        assert strip_ansi("left\x9dtitle\x07right") == "leftright"
        assert strip_ansi("left\x9dunterminated") == "left"


class TestStripAnsiRealWorld:
    """Real-world contamination scenarios from bug reports."""

    def test_colored_shebang(self):
        """The original reported bug: shebang corrupted by color codes."""
        assert strip_ansi(
            "\x1b[32m#!/usr/bin/env python3\x1b[0m\nprint('hello')"
        ) == "#!/usr/bin/env python3\nprint('hello')"


    def test_ansi_mid_code(self):
        assert strip_ansi(
            "def foo(\x1b[33m):\x1b[0m\n    return 42"
        ) == "def foo():\n    return 42"


class TestStripAnsiPassthrough:
    """Clean content must pass through unmodified."""

    def test_plain_text(self):
        assert strip_ansi("normal text") == "normal text"

    def test_empty(self):
        assert strip_ansi("") == ""


    def test_square_brackets_in_code(self):
        """Array indexing must not be confused with CSI."""
        code = "arr[0] = arr[31]"
        assert strip_ansi(code) == code


class TestSanitizeDisplayText:
    """sanitize_display_text — escape sequences AND bare control chars.

    Port of the openai/codex#31494 bug class: stored/untrusted text
    replayed into a terminal UI (e.g. the /resume recap) must not be able
    to clear the screen, retitle the window, or corrupt adjacent output.
    """

    def test_csi_removed(self):
        assert sanitize_display_text("a\x1b[2Jb") == "ab"

    def test_osc_title_removed(self):
        assert sanitize_display_text("x\x1b]0;pwned\x07y") == "xy"


    def test_empty(self):
        assert sanitize_display_text("") == ""

    def test_codex_31494_fixture(self):
        """The exact input shape from openai/codex#31494's test."""
        raw = "_count_r\x1b[13;2:3uows\tindent\n\x00two\x7f"
        assert sanitize_display_text(raw) == "_count_rows\tindent\ntwo"

    def test_mixed_escape_and_controls(self):
        raw = "hello \x1b[2J\x1b]0;pwned\x07 world \x9b31m red\x07"
        assert sanitize_display_text(raw) == "hello  world  red"

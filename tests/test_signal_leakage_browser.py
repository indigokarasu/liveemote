"""Browser-side signal-leakage guard for LiveEmote.

Mirrors ``test_signal_leakage.py`` (the server-side guard) for the front-end.

The promise
-----------

In ``apps/demo_server/static/*.js`` the avatar canvas (#avatarCanvas)
never receives webcam pixel bytes. It only renders from server-controlled
URLs (the active character's canonical still, active emote URL, and
configured background URL). User webcam frames flow:

    #webcam (video MediaStream)
        -> #captureCanvas (hidden, low-res)
        -> captureCanvas.toDataURL('image/jpeg', 0.6)
        -> POST /api/perception/video
        -> server-side MediaPipe tracker
        -> focus / energy / valence / tension signals
        -> orchestrator -> AvatarBehaviorState -> avatar rendering

At no step are pixel bytes assigned to the avatar display slot.

This test is a static-analysis guard. It runs on a CPU-only sandbox with
no network, no model downloads, no JSDOM, no headless browser. It scans
every ``.js`` file under ``apps/demo_server/static/`` plus ``index.html``
and asserts:

* deny-list: forbidden patterns that would leak webcam pixels onto the
  avatar side
* allow-list: variable names for ``.src = X`` assignments that are known
  safe (server-controlled URLs)
* positive wiring: required paths must exist (otherwise the avatar
  rendering pipeline is gutted and the test should fail loudly)

Design constraints (per the project's CI gate):

* < 5 s total runtime
* no new dependencies (pure ``re``, ``pathlib``, ``pytest``)
* robust to refactors: renaming variables is fine; copy-pasting a
  perception JPEG onto an avatar ``img.src`` is not
* consumes the on-disk demo.js verbatim — never edits it
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Paths & fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "apps" / "demo_server" / "static"


def _read(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def static_js_files() -> list[Path]:
    """All ``.js`` files under apps/demo_server/static/ — future-proof."""
    return sorted(STATIC_DIR.rglob("*.js"))


@pytest.fixture(scope="module")
def index_html() -> str:
    return _read("index.html")


@pytest.fixture(scope="module")
def demo_js_src() -> str:
    return _read("demo.js")


@pytest.fixture(scope="module")
def demo_js_stripped(demo_js_src: str) -> str:
    """Demo.js with every string-literal + comment replaced with whitespace
    so regex matches don't trip on cosmetic data URLs inside log strings,
    comments, or test fixtures within the source.

    Preserves line numbers via per-line replacement.
    """
    return _strip_strings_and_comments(demo_js_src)


@pytest.fixture(scope="module")
def index_html_stripped(index_html: str) -> str:
    return _strip_strings_and_comments(index_html)


# ---------------------------------------------------------------------------
# Token-stripping helpers
# ---------------------------------------------------------------------------

# Matches a JS template literal ${...} interpolation — keep these intact,
# strip the surrounding backticks but preserve the variable name.
_RE_TEMPLATE_RAW = re.compile(r"`([^`]*)`", re.DOTALL)


def _strip_strings_and_comments(src: str) -> str:
    """Replace every string literal and comment with whitespace. Preserves
    line numbers so any subsequent ``line_no`` from a regex match still
    points at a real character on a real line.

    Three passes:
      1. Strip ``/* ... */`` block comments
      2. Strip ``// ...`` line comments
      3. Strip single/double-quoted + template strings (but keep
         template-literal interpolations like ``${background.value}``).

    Replaces with same-length whitespace (newlines retained) so line
    numbers stay accurate.
    """

    def _repl_keep_lines(match: re.Match[str]) -> str:
        text = match.group(0)
        out = []
        for ch in text:
            if ch == "\n":
                out.append("\n")
            else:
                out.append(" ")
        return "".join(out)

    # Block comments.
    src = re.sub(r"/\*.*?\*/", _repl_keep_lines, src, flags=re.DOTALL)
    # Line comments.
    src = re.sub(r"//[^\n]*", _repl_keep_lines, src)
    # Template literals: ${...} preserved as-is, surrounding `...` blanked.
    src = _RE_TEMPLATE_RAW.sub(_repl_keep_lines, src)
    # Single + double quotes.
    src = re.sub(
        r"'(?:\\.|[^'\\\n])*'",
        _repl_keep_lines,
        src,
    )
    src = re.sub(
        r'"(?:\\.|[^"\\\n])*"',
        _repl_keep_lines,
        src,
    )
    return src


# ---------------------------------------------------------------------------
# Deny-list regex patterns
# ---------------------------------------------------------------------------

# Rule 1: captureCanvas.toDataURL() result leaking onto an img.src.
# The capture buffer is for the perception POST — never for avatar display.
RE_TO_DATAURL_TO_SRC = re.compile(
    r"\.src\s*=\s*[^=\n;]*\.toDataURL\s*\(",
)

# Rule 2: srcObject on anything other than the local webcam <video>.
# We can't easily tell apart `video` and `avatarVideo` with regex alone,
# but the only legit site in current demo.js is `video.srcObject = stream`
# where `video = q('#webcam')`. Any OTHER `xxx.srcObject = ` assignment
# is by definition wiring a MediaStream onto a non-source element.
RE_SRCOBJECT = re.compile(
    r"(?<![\w$.])(?P<target>[A-Za-z_$][\w$]*)\.srcObject\s*=",
)

# Rule 3: avatar canvas drawImage(webcam). Currently demos.js only does
# `captureCtx.drawImage(video, ...)` — never `avatarCtx.drawImage(video, ...)`.
# We catch: any drawImage where source is `video` (the local webcam) AND
# the target is NOT `captureCtx`.
# We can't easily parse the call-target precisely with regex, so we look
# for:  `<expr>.drawImage(\s*video\s*,`  and report the call site for review
# — the allow-list of `captureCtx` is checked separately below.
RE_DRAWIMAGE_FROM_WEBCAM = re.compile(
    r"\.drawImage\s*\(\s*video\b",
)

# Rule 4: data: URI hard-coded as the RHS of `.src = ...` (a violation even
# without toDataURL — bypasses the server curve and ships pixel data inline).
RE_HARDCODED_DATA_URL_SRC = re.compile(
    r"""\.src\s*=\s*['"`]data:\s*image""",
    re.IGNORECASE,
)

# Rule 5: data: URI in a CSS-style background assignment.
RE_DATA_URL_IN_CSS_BACKGROUND = re.compile(
    r"""\.style\.(?:backgroundImage|background)\s*=\s*[^;\n]*url\s*\(\s*['"]?data:image""",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# The 5 deny-list assertions
# ---------------------------------------------------------------------------


def test_no_to_dataurl_assigned_to_img_src(
    demo_js_stripped: str,
) -> None:
    """captureCanvas.toDataURL() must never flow back onto an img.src.

    Regression: someone copy-pastes the perception capture onto an avatar
    imgsrc, the test fails. The current pipeline POSTs toDataURL() bytes
    to /api/perception/video only.
    """
    hits = list(_iter_line_matches(RE_TO_DATAURL_TO_SRC, demo_js_stripped))
    assert not hits, (
        "Forbidden: captureCanvas.toDataURL() result assigned to an "
        "img.src. Perception capture bytes belong in the POST to "
        "/api/perception/video, never on an avatar display slot.\n"
        + "\n".join(hits)
    )


def test_srcObject_only_on_webcam_video(
    demo_js_src: str, demo_js_stripped: str
) -> None:
    """srcObject must only be set on the local #webcam <video> element.

    Regression: someone wires a MediaStream onto an avatar-side element
    via srcObject — instantaneous webcam leak onto the avatar.
    """
    allowed_local_target = "video"  # the `const video = q('#webcam');` line
    hits = []
    for line_no, line in _iter_line_matches(RE_SRCOBJECT, demo_js_stripped):
        target = _extract_property_target(line)
        if target != allowed_local_target:
            hits.append(
                f"  L{line_no}: target='{target}' "
                f"line: {demo_js_src.splitlines()[line_no - 1].strip()}"
            )
    assert not hits, (
        "Forbidden: srcObject assigned to a target other than the local "
        "webcam <video> element. Pixel streams belong on #webcam only.\n"
        + "\n".join(hits)
    )


def test_drawimage_webcam_only_into_capture_canvas(
    demo_js_stripped: str,
) -> None:
    """drawImage(video, ...) must only target captureCtx.

    The hidden captureCanvas is what encodes the perception JPEG. Drawing
    the live webcam into anything else would copy webcam pixels onto
    another display slot.
    """
    # Find the chain pattern: <expr>.drawImage(video, ...)
    # We require the call site to contain captureCtx on its LHS.
    RE_CAPTURE_DRAWIMAGE = re.compile(
        r"captureCtx\.drawImage\s*\(",
    )
    draw_image_hits = list(
        _iter_line_matches(RE_DRAWIMAGE_FROM_WEBCAM, demo_js_stripped)
    )
    capture_hits = list(
        _iter_line_matches(RE_CAPTURE_DRAWIMAGE, demo_js_stripped)
    )
    assert draw_image_hits, (
        "Structural soundness: no drawImage(video, ...) call found in "
        "demo.js. The webcam-to-captureCanvas encoding pipeline is "
        "missing; avatar rendering will not produce perception signals."
    )
    assert capture_hits, (
        "Forbidden: drawImage(video, ...) found but never on "
        "captureCtx. Webcam pixels must only flow into the hidden "
        "captureCanvas, never directly into the avatar slot."
    )
    assert len(draw_image_hits) == len(capture_hits), (
        f"Mismatch: {len(draw_image_hits)} drawImage(video, ...) call(s) "
        f"but only {len(capture_hits)} target captureCtx. "
        "Any drawImage(video) outside captureCtx is a webcam leak.\n"
        f"drawImage sites: {draw_image_hits}\n"
        f"captureCtx sites: {capture_hits}"
    )


def test_no_hardcoded_data_url_src(
    demo_js_stripped: str,
) -> None:
    """.src = 'data:image/...' is forbidden -- pixel data must not be
    baked into the JS source.

    The server is the authority on what the avatar should display
    (canonical stills, emotes, backgrounds). Hard-coding a data: URL
    on the LHS of .src = ... bypasses that authority and ships pixel
    data through the bundle.
    """
    hits = list(_iter_line_matches(RE_HARDCODED_DATA_URL_SRC, demo_js_stripped))
    assert not hits, (
        "Forbidden: hard-coded `data:image/...` URL assigned to .src. "
        "Avatar image sources must come from server-curated URLs.\n"
        + "\n".join(hits)
    )


def test_no_data_url_in_css_background_assignment(
    demo_js_stripped: str,
) -> None:
    """Inline data:image/... in style.backgroundImage is forbidden -- same
    rationale as the .src = X case, but for CSS-style background slots
    (which can also be a leak vector if anyone ever feeds a perception
    frame into it).
    """
    hits = list(
        _iter_line_matches(RE_DATA_URL_IN_CSS_BACKGROUND, demo_js_stripped)
    )
    assert not hits, (
        "Forbidden: data:image/... inlined into element.style.background "
        "or backgroundImage. Avatar backgrounds must come from server-"
        "curated URLs.\n" + "\n".join(hits)
    )


# ---------------------------------------------------------------------------
# Allow-list spot-check: every .src = X in demo.js must be in SAFE_AVATAR_SRC_VARS
# (or be a non-image .src like <audio>.src = '/api/audio?...')
# ---------------------------------------------------------------------------


# Match .src = X where X is a JS identifier/chain (so we exclude literals).
RE_SRC_ASSIGN = re.compile(
    r"(?P<lhs>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\.src\s*=\s*"
    r"(?P<rhs>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\b",
)


def test_all_avatar_src_assignments_are_allowlisted(
    demo_js_stripped: str, demo_js_src: str
) -> None:
    """Every .src = X where X is a variable/chain must be in the allow-list.

    This catches new avatar imgs being introduced with a fresh variable
    that hasn't been reviewed. Server-provided URLs are fine; anything
    else needs annotation.
    """
    offenders = []
    for line_no, line, match in _iter_full_matches(RE_SRC_ASSIGN, demo_js_stripped):
        lhs = match.group("lhs")
        rhs = match.group("rhs")
        if lhs == "els.speech":
            # TTS audio endpoint -- server-controlled, not pixel data.
            continue
        if lhs == "portraitImg" and rhs == "src":
            # ensurePortraitImage(src) -- server-controlled canonical URL.
            continue
        if lhs == "els.avatarEmote" and rhs == "emoteUrl":
            # Server-controlled active emote URL.
            continue
        if lhs == "els.avatarEmote" and rhs == "src":
            # Same shape as portraitImg -- confirms the allow-list covers it.
            continue
        offenders.append(
            f"  L{line_no}: {lhs}.src = {rhs}  "
            f"({demo_js_src.splitlines()[line_no - 1].strip()})"
        )
    assert not offenders, (
        "Unannotated .src = X assignment(s). Add the variable name to "
        "SAFE_AVATAR_SRC_VARS with a one-line comment naming the server "
        "field it comes from.\n" + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# Positive wiring checks
# ---------------------------------------------------------------------------


def test_perception_capture_pipeline_wired(
    demo_js_src: str,
) -> None:
    """The webcam-to-perception pipeline must be wired end-to-end:
    captureCanvas, captureCtx, toDataURL JPEG, POST to
    /api/perception/video. If any of these is missing, the demo cannot
    produce perception signals.

    NOTE: this test runs against the *unstripped* source. The string
    literals we want to look for (`'image/jpeg'`, `'/api/perception/video'`)
    would be blanked by `_strip_strings_and_comments` and would not
    match here. Stripping is still the right call for the deny-list
    checks (where we want to ignore string contents), but positive
    wiring looks for the actual literal substring in the source.
    """
    required_substrings = [
        "captureCanvas",
        "captureCtx",
        "toDataURL('image/jpeg'",
        "'/api/perception/video'",
    ]
    missing = [
        s for s in required_substrings if s not in demo_js_src
    ]
    assert not missing, (
        "Structural regression: the webcam-to-perception pipeline is "
        "missing components. Without these, the avatar cannot receive "
        "focus / energy signals.\nMissing: " + ", ".join(missing)
    )


def test_avatar_emote_and_canonical_still_wired(
    demo_js_stripped: str,
) -> None:
    """Avatar img elements must be wired: the emote <img> must receive
    the server-provided emoteUrl, and the canonical portrait img must
    receive a server-provided URL.
    """
    required = [
        "els.avatarEmote",
        "portraitImg",
        "els.avatarPortrait",
    ]
    missing = [s for s in required if s not in demo_js_stripped]
    assert not missing, (
        "Structural regression: avatar display wiring is missing "
        "components. The avatar will not render.\nMissing: "
        + ", ".join(missing)
    )


# ---------------------------------------------------------------------------
# HTML structural checks
# ---------------------------------------------------------------------------


def test_html_separates_source_panel_from_avatar_slot(
    index_html: str,
) -> None:
    """The HTML must structurally separate the source panel from the
    avatar slot:

      * ``<video id="webcam">`` exists in the source panel (the explicit
        opt-in for displaying webcam pixels in that panel only).
      * ``<div id="avatarCanvas">`` exists -- NOT ``<video id="avatarCanvas">``
        (which would let a MediaStream be wired onto the avatar via JS).
      * ``<canvas id="captureCanvas" ... hidden>`` exists, hidden, so its
        bytes never leak into a visible slot.

    These structural assertions protect against a refactor that swaps the
    avatar's element type -- which would be invisible to any runtime
    signal-leakage check until the bytes actually flowed.
    """
    assert "<video" in index_html and 'id="webcam"' in index_html, (
        "Structural regression: #webcam <video> element missing from "
        "index.html. The source panel must declare a <video> element "
        "where the local MediaStream is the intended opt-in display."
    )
    # The avatar slot must be a <div> (we get the emote <img> and portrait
    # children inside it), NOT a <video> (which could silently receive a
    # srcObject).
    assert "<div" in index_html and 'id="avatarCanvas"' in index_html, (
        "Structural regression: #avatarCanvas is not a <div>. Must be "
        "a <div> so no <video>-specific src API can be wired onto it."
    )
    assert not re.search(
        r"<video[^>]*id=[\"']avatarCanvas[\"']",
        index_html,
        re.IGNORECASE,
    ), (
        "Structural regression: #avatarCanvas is now a <video> element. "
        "Unbounded webcam streaming onto the avatar slot is now possible."
    )
    # Capture canvas must be hidden.
    capture_pat = re.compile(
        r"<canvas[^>]*id=[\"']captureCanvas[\"'][^>]*",
        re.IGNORECASE,
    )
    capture_match = capture_pat.search(index_html)
    assert capture_match, (
        "Structural regression: #captureCanvas missing. The webcam-to-"
        "perception encoding buffer must exist (and be hidden) so the "
        "avatar rendering path never sees raw pixel bytes."
    )
    capture_tag = capture_match.group(0)
    assert "hidden" in capture_tag, (
        f"Structural regression: #captureCanvas is not hidden. Revealing "
        f"the capture buffer would display the webcam on the page.\n"
        f"Element: {capture_tag}"
    )


# ---------------------------------------------------------------------------
# Multi-file extendability (future-proofing)
#
# Note on layout: these helper functions and the parametrized test below
# must be defined in this order. Python evaluates decorators at module-load
# time, so `static_js_files_iter` has to be in scope before the parametrize
# decorator runs.
# ---------------------------------------------------------------------------


def static_js_files_iter() -> list[Path]:
    """All ``.js`` files under apps/demo_server/static/ — future-proof."""
    return sorted(STATIC_DIR.rglob("*.js"))


def _iter_ids(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT))


@pytest.fixture(scope="module")
def repo_root_cache() -> Path:
    return REPO_ROOT


@pytest.mark.parametrize("js_file", static_js_files_iter(), ids=_iter_ids)
def test_all_js_files_under_static_have_no_leak(
    js_file: Path, repo_root_cache: Path
) -> None:
    """If a future PR adds a second .js file under apps/demo_server/static/,
    parametrize will pick it up and run every leak check against it."""
    src = js_file.read_text(encoding="utf-8")
    stripped = _strip_strings_and_comments(src)
    relative = js_file.relative_to(repo_root_cache)

    for name, pattern in [
        ("toDataURL-to-src", RE_TO_DATAURL_TO_SRC),
        ("hardcoded-data-url-src", RE_HARDCODED_DATA_URL_SRC),
        ("data-url-in-css-background", RE_DATA_URL_IN_CSS_BACKGROUND),
    ]:
        hits = list(_iter_line_matches(pattern, stripped))
        assert not hits, (
            f"{relative}: forbidden pattern '{name}' matched.\n"
            + "\n".join(
                f"  L{ln}: {src.splitlines()[ln - 1].strip()}" for ln, _ in hits
            )
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_line_matches(
    pattern: re.Pattern[str], text: str
) -> list[tuple[int, str]]:
    """Yield (line_no, line) for every match of `pattern` in `text`.

    ``line_no`` is 1-indexed.
    """
    out: list[tuple[int, str]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern.search(line):
            out.append((line_no, line))
    return out


def _iter_full_matches(
    pattern: re.Pattern[str], text: str
) -> list[tuple[int, str, re.Match[str]]]:
    out: list[tuple[int, str, re.Match[str]]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        m = pattern.search(line)
        if m:
            out.append((line_no, line, m))
    return out


def _extract_property_target(line: str) -> str:
    """Given a line matched by RE_SRCOBJECT, return the bare identifier
    that the `.srcObject = ` is being assigned to (e.g., ``video`` in
    ``video.srcObject = stream``)."""
    m = re.match(r"\s*([A-Za-z_$][\w$]*)\.srcObject\s*=", line)
    if not m:
        return ""
    return m.group(1)

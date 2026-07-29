"""Runtime-variant browser signal-leakage guard for LiveEmote.

Companion test to ``tests/test_signal_leakage_browser.py`` (the static
guard). The static guard runs regex over the JS source and is good
enough for blatant direct leaks, but **cannot** detect dynamic
variable-aliasing regressions like::

    const av = avatarVideo;            // alias through a const
    av.srcObject = stream;             // webcam leak via the alias

    const { m: myImg } = { m: els.avatarEmote };   // destructuring alias
    myImg.src = 'data:image/jpeg;base64,...';

    const elements = { a: document.createElement('video'), b: video };
    elements.a.srcObject = elements.b.srcObject;   // property alias

    const myCam = video;
    aliasCanvas.getContext('2d').drawImage(myCam, 0, 0);  // chain alias

This test loads the real ``apps/demo_server/static/index.html`` and the
real ``apps/demo_server/static/demo.js`` into a headless Chromium and
executes them. Before any app script runs, three prototype-level hooks
intercept the DOM-side sinks:

1. ``HTMLMediaElement.prototype.srcObject`` setter — flags any
   element with ``id != 'webcam'`` that receives a MediaStream.
2. ``HTMLImageElement.prototype.src`` setter — flags hard-coded
   ``data:image/...`` assignments.
3. ``CanvasRenderingContext2D.prototype.drawImage`` — flags draws of
   the ``#webcam`` element into any canvas whose ``id !=
   'captureCanvas'``.

The introspection is at the *sink*, not the *source*, so any aliasing
chain that ultimately writes a webcam stream / inline data URL / a
drawImage from the webcam node to the avatar side is intercepted
regardless of how many intermediate variables it traverses.

Two tests:

* **Positive** — load real demo.js, run for 2 seconds, assert zero
  leaks. Must be green on the current code.
* **Regression-fixture** — load real demo.js augmented with the four
  aliasing shapes above, assert the hooks detect at least 3 distinct
  leak categories. Proves the detector actually catches real
  regressions.

Auto-skip semantics
-------------------

The module auto-skips on module import if either:

1. The ``playwright`` Python package is not installed, or
2. ``hermes_avatar.demo.meeting_join.find_chromium_binary()`` returns
   ``None`` (no system Chromium / Chrome / Edge on PATH).

This matches the existing ``gpu`` marker pattern in pyproject.toml:
thin CI images stay green, dev machines + GPU CI lanes run the test
for real. The whole file imports cleanly without either dependency —
no ImportError crashes other tests on a thin CI image.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "apps" / "demo_server" / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"
DEMO_JS_PATH = STATIC_DIR / "demo.js"
DEMO_CSS_PATH = STATIC_DIR / "demo.css"


# ---------------------------------------------------------------------------
# Prereq detection — auto-skip on thin CI images
# ---------------------------------------------------------------------------

try:
    from playwright.sync_api import Browser, Page, sync_playwright  # noqa: F401
    _PLAYWRIGHT_IMPORT_OK = True
except ImportError:
    sync_playwright = None  # type: ignore[assignment]
    _PLAYWRIGHT_IMPORT_OK = False

try:
    from hermes_avatar.demo.meeting_join import find_chromium_binary
    _CHROMIUM_BINARY = find_chromium_binary()
except Exception:  # noqa: BLE001 — any resolution failure counts as "not present"
    _CHROMIUM_BINARY = None


_SKIP_REASON: Optional[str] = None
if not _PLAYWRIGHT_IMPORT_OK:
    _SKIP_REASON = (
        "playwright Python package not installed; skipping runtime browser "
        "guard tests. Install with `pip install playwright>=1.40`."
    )
elif _CHROMIUM_BINARY is None:
    _SKIP_REASON = (
        "Chromium binary not present on this system; skipping runtime "
        "browser guard tests. `make setup` then `playwright install "
        "chromium` will enable them on dev hosts."
    )

if _SKIP_REASON is not None:
    pytest.skip(_SKIP_REASON, allow_module_level=True)


# ---------------------------------------------------------------------------
# Mark all the tests in this module with the `browser` marker so
# `pytest -m 'not browser'` keeps thin CI green by exclusion.
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.browser


# ---------------------------------------------------------------------------
# Init-script — three prototype hooks installed BEFORE any app JS runs.
#
# Idempotent: ``__liveemoteLeakHooks__`` flag prevents double-installation
# if the page navigates and the init script re-runs.
# ---------------------------------------------------------------------------

LEAK_HOOKS_INIT_SCRIPT = r"""
(() => {
  if (window.__liveemoteLeakHooks__) return;
  window.__liveemoteLeakHooks__ = true;
  window.__leaks__ = [];

  // 1. HTMLMediaElement.prototype.srcObject — flag any non-#webcam
  //    element receiving a MediaStream.
  try {
    const d = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, 'srcObject');
    Object.defineProperty(HTMLMediaElement.prototype, 'srcObject', {
      configurable: true,
      set(v) {
        try {
          const id = this && this.id;
          if (id !== 'webcam' && v) {
            const tag = id || (this.tagName || '?').toLowerCase();
            window.__leaks__.push('srcObject on non-webcam: ' + tag);
          }
        } catch (e) { /* never throw out of the setter */ }
        if (d && d.set) d.set.call(this, v);
      },
      get() { return d && d.get ? d.get.call(this) : undefined; },
    });
  } catch (e) { /* hook failed, but don't break the page */ }

  // 2. CanvasRenderingContext2D.prototype.drawImage — flag webcam draws
  //    that bypass the hidden #captureCanvas buffer.
  try {
    const orig = CanvasRenderingContext2D.prototype.drawImage;
    CanvasRenderingContext2D.prototype.drawImage = function(img, ...rest) {
      try {
        if (img && img.id === 'webcam') {
          const c = this.canvas;
          const cid = c && c.id;
          if (cid !== 'captureCanvas') {
            const name = cid || (c && c.tagName && c.tagName.toLowerCase()) || '?';
            window.__leaks__.push('drawImage(#webcam) into non-capture canvas: ' + name);
          }
        }
      } catch (e) { /* never throw out of the override */ }
      return orig.apply(this, [img, ...rest]);
    };
  } catch (e) { /* hook failed, but don't break the page */ }

  // 3. HTMLImageElement.prototype.src — flag any data:image/... assignment
  //    to an image, regardless of how it was produced.
  try {
    const d = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, 'src');
    Object.defineProperty(HTMLImageElement.prototype, 'src', {
      configurable: true,
      set(v) {
        try {
          if (typeof v === 'string' && v.startsWith('data:image/')) {
            window.__leaks__.push('image src set to data: URL');
          }
        } catch (e) { /* never throw out of the setter */ }
        if (d && d.set) d.set.call(this, v);
      },
      get() { return d && d.get ? d.get.call(this) : undefined; },
    });
  } catch (e) { /* hook failed, but don't break the page */ }
})();
"""


# ---------------------------------------------------------------------------
# Stub bodies for /api/* so demo.js's setInterval(poll, 1500) and the
# /api/perception/video POST do not CORS-fail inside the route-handled
# origin. The shape mirrors the real /api/status contract enough for
# renderAvatar() / update() not to throw.
# ---------------------------------------------------------------------------

_FAKE_STATUS_JSON = json.dumps({
    "avatar": {
        "mode": "idle",
        "affect": "neutral",
        "emote_id": None,
        "intensity": 0.5,
        "gaze_target": "soft_forward",
    },
    "user": {},
    "character_id": "test_char",
    "characters": [],
    "styles": [],
    "backgrounds": [],
    "active_style_id": None,
    "active_background_id": None,
    "capabilities": {
        "renderer": {
            "online": True,
            "backend": "fake",
            "avatar_visual": {
                "portrait_kind": "svg_fallback",
                "canonical_url": None,
                "active_emote_url": None,
            },
        },
        "perception": {
            "backend": "fake",
            "available": False,
        },
        "voice": {
            "backend": "fake",
            "last_engine": "fake",
            "last_latency_ms": 0,
        },
    },
})


# ---------------------------------------------------------------------------
# Session fixtures — one Chromium per test session, fresh page per test.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def chromium_executable_path() -> str:
    assert _CHROMIUM_BINARY is not None  # module-level skip guards this
    return _CHROMIUM_BINARY


@pytest.fixture(scope="session")
def pw_browser(chromium_executable_path: str):
    """A single Chromium process across the whole module — launch is
    the slowest step (~3-5s) so we amortize it.

    Two flags matter::

      * ``--use-fake-ui-for-media-stream`` — auto-accepts the camera
        permission prompt so demo.js's ``getUserMedia`` succeeds
        without a real camera.
      * ``--use-fake-device-for-media-stream`` — replaces the camera
        with Chromium's synthetic green/colored bar, so the
        regression-fixture test can actually feed a real
        ``MediaStream`` to the aliasing target.
    """
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=chromium_executable_path,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture()
def pw_page(pw_browser: Browser) -> Page:
    """A fresh BrowserContext + Page per test — keeps leak arrays
    isolated between the positive and the regression-fixture test.
    """
    ctx = pw_browser.new_context(permissions=["camera", "microphone"])
    page = ctx.new_page()
    try:
        yield page
    finally:
        ctx.close()


def _install_routes(page: Page, leaky_js: Optional[str] = None) -> None:
    """Install init script + route stubs. ``leaky_js`` is the body to
    serve in place of real ``demo.js`` for the regression-fixture test.
    Pass ``None`` to serve the real, unmodified ``demo.js``.
    """
    # Init hooks BEFORE any document script runs.
    page.add_init_script(LEAK_HOOKS_INIT_SCRIPT)

    # Root: real index.html.
    page.route(
        "http://localhost/",
        lambda r: r.fulfill(
            body=INDEX_HTML_PATH.read_text(encoding="utf-8"),
            content_type="text/html; charset=utf-8",
        ),
    )

    # demo.js — real file for the positive test, leaky for the regression
    # test.
    if leaky_js is None:
        demo_body = DEMO_JS_PATH.read_text(encoding="utf-8")
    else:
        demo_body = leaky_js
    page.route(
        "http://localhost/static/demo.js",
        lambda r: r.fulfill(
            body=demo_body,
            content_type="application/javascript; charset=utf-8",
        ),
    )

    # Other static assets: serve if present, empty CSS body otherwise.
    page.route(
        "http://localhost/static/demo.css",
        lambda r: r.fulfill(
            body=DEMO_CSS_PATH.read_text(encoding="utf-8")
            if DEMO_CSS_PATH.exists() else "/* dummy */",
            content_type="text/css; charset=utf-8",
        ),
    )
    page.route(
        "http://localhost/static/**",
        lambda r: r.fulfill(body="/* dummy */", content_type="text/css"),
    )

    # Stub the /api/* surface demo.js talks to. /api/status is the
    # most-shaped response — every other /api call accepts "{}".
    page.route(
        "http://localhost/api/status",
        lambda r: r.fulfill(
            body=_FAKE_STATUS_JSON,
            content_type="application/json",
            headers={"content-type": "application/json"},
        ),
    )
    page.route(
        "http://localhost/api/**",
        lambda r: r.fulfill(
            body="{}",
            content_type="application/json",
            headers={"content-type": "application/json"},
        ),
    )


def _drain_leaks(page: Page) -> list[str]:
    """Read the leaks buffer synchronously and return it as a fresh list."""
    return list(page.evaluate("() => (window.__leaks__ || [])"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_real_demo_js_produces_zero_leaks(pw_page: Page) -> None:
    """**Positive** — the current ``demo.js`` must not leak webcam pixels
    onto the avatar side, even after ``captureCtx`` is initialized and
    the setInterval-driven stream has been running for ~2 seconds.

    This test passes if and only if:

    * every ``HTMLMediaElement.srcObject`` write targets an element
      whose ``id == 'webcam'`` (the only legit MediaStream sink), AND
    * every ``HTMLImageElement.src`` write is a string URL, never
      ``data:image/...``, AND
    * every ``CanvasRenderingContext2D.drawImage`` from the ``#webcam``
      element writes into the ``#captureCanvas`` buffer (perception
      encoder) and nothing else.

    Regression coverage the static guard cannot provide.
    """
    _install_routes(pw_page)
    pw_page.goto("http://localhost/", wait_until="load")
    # Allow webcam init (~250ms with --use-fake-device), one poll
    # tick (1500ms), and ~6 streamPerceptionFrame ticks (320ms × 6).
    pw_page.wait_for_timeout(2000)

    leaks = _drain_leaks(pw_page)
    assert not leaks, (
        "Runtime signal-leak detected in real demo.js — the static guard "
        "did not catch it but the prototype hooks did:\n"
        + "\n".join(f"  - {leak!r}" for leak in leaks)
    )


def test_regression_fixture_catches_aliasing(pw_page: Page) -> None:
    """**Regression** — synthesize four aliasing shapes at the tail of
    ``demo.js`` and prove each one trips at least one prototype hook.

    The four shapes cover the dynamic-pattern space we care about:

    1. const declaration alias (an element created and assigned a
       stream via a const indirection).
    2. Variable alias of a known img element + hard-coded ``data:`` URL
       (proves the img.src hook fires regardless of how the LHS was
       reached).
    3. const-to-canvas chain alias — a const pointing at ``#webcam``
       drawn into a *new* canvas attached to ``#avatarPortrait``
       (proves the drawImage hook fires off-stream of camera capture).
    4. Destructuring + property alias — the ``#webcam`` element
       reached via ``const {cam: aliasSrc} = {cam: wc};`` and assigned
       as ``srcObject`` to a hidden avatar-side ``<video>`` (proves
       destructuring patterns don't bypass the hook).

    Together they exercise every shape the user explicitly called out
    (``const av = avatar; avatarVideo = av;`` plus the three
    structurally-aware variants).
    """
    real_demo = DEMO_JS_PATH.read_text(encoding="utf-8")

    # Append after real demo.js so the global ``els``, ``q`` and
    # document state are settled.
    leaky_tail = r"""

;(() => {
  // Single deferred burst — runs 500ms after page load, after webcam()
  // has resolved its getUserMedia promise and captureCtx is set up.
  setTimeout(() => {
    try {
      const wc = document.querySelector('#webcam');

      // Shape 1: const alias → srcObject on a fresh <video> in #avatarCanvas
      const av = document.createElement('video');
      av.id = 'leakVideo1';
      document.querySelector('#avatarCanvas').appendChild(av);
      const streamAlias1 = wc;            // alias
      av.srcObject = streamAlias1.srcObject;  // <-- should trip srcObject hook

      // Shape 2: variable alias → data: URL on a known img
      const myImg = els.avatarEmote;       // alias
      myImg.src = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAf4kKAAA=';  // trips img.src hook

      // Shape 3: const chain → drawImage(webcam) into a new avatar-side canvas
      const myCam = wc;                                                    // alias
      const leakCanvas3 = document.createElement('canvas');
      leakCanvas3.id = 'leakCanvas3';
      els.avatarPortrait.appendChild(leakCanvas3);
      const drawCtx3 = leakCanvas3.getContext('2d');
      drawCtx3.drawImage(myCam, 0, 0);   // <-- trips drawImage hook

      // Shape 4: destructuring + property assignment → srcObject via alias
      const sinkEl4 = document.createElement('video');
      sinkEl4.id = 'avatarStream';
      document.querySelector('#avatarCanvas').appendChild(sinkEl4);
      const { cam: aliasSrc4 } = { cam: wc };     // destructuring alias
      sinkEl4.srcObject = aliasSrc4.srcObject;    // <-- trips srcObject hook
    } catch (e) {
      window.__leaks__.push('regression-fixture threw: ' + (e && e.message));
    }
  }, 500);
})();
"""

    _install_routes(pw_page, leaky_js=real_demo + leaky_tail)
    pw_page.goto("http://localhost/", wait_until="load")
    pw_page.wait_for_timeout(2000)

    leaks = _drain_leaks(pw_page)

    # We injected 4 distinct leak triggers (3 distinct categories — the
    # sinkEl4 alias and the av alias both hit the srcObject hook, so
    # the actual category count is 3). The bare-minimum the detector
    # must produce is 3 leaks (one per category), so we assert >= 3.
    assert len(leaks) >= 3, (
        f"Regression-fixture produced only {len(leaks)} leak signal(s); "
        f"the detector failed to catch one or more aliasing shapes.\n"
        f"Leaks observed: {leaks}"
    )

    # And each category must be represented at least once.
    text = "\n".join(leaks)
    assert "srcObject" in text, (
        f"Missing srcObject leak signal in detector output: {leaks}"
    )
    assert "data:" in text, (
        f"Missing data: URL leak signal in detector output: {leaks}"
    )
    assert "drawImage" in text, (
        f"Missing drawImage leak signal in detector output: {leaks}"
    )

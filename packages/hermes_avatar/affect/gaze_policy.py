"""
Conversational gaze triangle (3-point model).

Real humans don't stare fixedly at a single point during conversation.
The eyes cycle naturally between three zones:

    eyes  ──→  mouth  ──→  away  ──→  eyes   (repeat)

Dwell times vary by cognitive mode and attention level.  This policy
produces the next gaze_point for each tick, consumed by the browser
to adjust the avatar's iris / eye animation.

Design (per architecture review):
  - The server sets the broad ``cognitive_mode`` ("listening" / "speaking"
    / "thinking") based on context.
  - The client-side JS animation loop runs the actual timer and advances
    the gaze point, using the dwell ranges exported here as guidance.
  - The server *also* emits a suggested ``gaze_point`` for clients that
    don't run their own saccade timer (graceful degradation).

Coverage: realism improvement #5 (gaze triangle).
"""

from dataclasses import dataclass
from typing import Optional
import random


# ---------------------------------------------------------------------------
# Gaze-point dwell ranges (ms) per cognitive mode
# ---------------------------------------------------------------------------

@dataclass
class DwellRange:
    """min / max dwell in milliseconds for a gaze point."""
    min_ms: int
    max_ms: int


# Dwell curves: during *listening* the avatar looks at the user's eyes
# more (engagement signal); during *speaking* it glances away more often
# (reducing intensity); during *thinking* it stays away / soft-forward
# (cognitive load).

_GAZE_DWELLS: dict[str, dict[str, DwellRange]] = {
    "listening": {
        "eyes":  DwellRange(1200, 2800),   # long eye contact = engagement
        "mouth": DwellRange(400,  1200),   # occasional lip reading
        "away":  DwellRange(600,  1500),   # brief breaks
    },
    "speaking": {
        "eyes":  DwellRange(600,  1600),   # shorter — "I'm talking"
        "mouth": DwellRange(200,  800),    # rare (speaker doesn't lip-read)
        "away":  DwellRange(800,  2200),   # longer breaks while speaking
    },
    "thinking": {
        "eyes":  DwellRange(300,  1000),   # minimal eye contact when processing
        "mouth": DwellRange(100,  500),
        "away":  DwellRange(1500, 3500),   # long "thinking" gazes
    },
}

# Default dwell when no mode is active
_DEFAULT_DWELLS = _GAZE_DWELLS["listening"]


# ---------------------------------------------------------------------------
# Gaze point transition probabilities (Markov-like)
# ---------------------------------------------------------------------------

# Each entry maps current_point → {next_point: weight}
_GAZE_TRANSITIONS: dict[str, dict[str, float]] = {
    "eyes":  {"mouth": 0.35, "away": 0.45, "eyes": 0.20},
    "mouth": {"eyes": 0.50, "away": 0.30, "mouth": 0.20},
    "away":  {"eyes": 0.55, "mouth": 0.25, "away": 0.20},
    "soft_forward": {"eyes": 0.30, "mouth": 0.15, "away": 0.35, "soft_forward": 0.20},
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class GazeResult:
    """Emitted by GazePolicy.tick()."""

    gaze_point: str              # "eyes" | "mouth" | "away" | "soft_forward"
    dwell_ms: int                # recommended dwell for this point
    cognitive_mode: str          # current mode


class GazePolicy:
    """Produces natural gaze-point cycling for conversational realism.

    Usage::

        gaze = GazePolicy()
        for tick_ms in range(0, 30000, 50):
            result = gaze.tick(cognitive_mode="listening", dt_ms=50)
            # result.gaze_point → send to browser
    """

    def __init__(self) -> None:
        self._current_point: str = "eyes"
        self._elapsed_ms: float = 0.0
        self._dwell_ms: int = self._random_dwell("eyes", _DEFAULT_DWELLS)
        self._cognitive_mode: str = "listening"

    # -- public API --------------------------------------------------------

    def tick(self, cognitive_mode: str, dt_ms: float = 0.0) -> GazeResult:
        """Advance the gaze timer and return the current (possibly new) gaze point.

        *cognitive_mode* should be one of  "listening" | "speaking" | "thinking".
        """
        if cognitive_mode != self._cognitive_mode:
            self._cognitive_mode = cognitive_mode
            # Mode change: re-randomise dwell to avoid stale long dwells
            self._dwell_ms = self._random_dwell(
                self._current_point, self._dwells_for_mode(cognitive_mode)
            )

        self._elapsed_ms += dt_ms

        if self._elapsed_ms >= self._dwell_ms:
            # Time to move to the next point
            self._current_point = self._next_point(self._current_point)
            self._dwell_ms = self._random_dwell(
                self._current_point, self._dwells_for_mode(cognitive_mode)
            )
            self._elapsed_ms = 0.0

        return GazeResult(
            gaze_point=self._current_point,
            dwell_ms=self._dwell_ms,
            cognitive_mode=cognitive_mode,
        )

    def reset(self, point: str = "eyes", mode: str = "listening") -> None:
        """Hard-reset the gaze policy."""
        self._current_point = point
        self._cognitive_mode = mode
        self._dwell_ms = self._random_dwell(point, self._dwells_for_mode(mode))
        self._elapsed_ms = 0.0

    # -- internal ----------------------------------------------------------

    @staticmethod
    def _dwells_for_mode(mode: str) -> dict[str, DwellRange]:
        return _GAZE_DWELLS.get(mode, _DEFAULT_DWELLS)

    @staticmethod
    def _random_dwell(point: str, dwells: dict[str, DwellRange]) -> int:
        dr = dwells.get(point)
        if dr is None:
            return 1500
        return random.randint(dr.min_ms, dr.max_ms)

    @staticmethod
    def _next_point(current: str) -> str:
        """Pick the next gaze point given the current one."""
        weights = _GAZE_TRANSITIONS.get(current, _GAZE_TRANSITIONS["soft_forward"])
        points = list(weights.keys())
        probs = list(weights.values())
        return random.choices(points, weights=probs, k=1)[0]

    # -- property accessors ------------------------------------------------

    @property
    def current_point(self) -> str:
        return self._current_point

    @property
    def cognitive_mode(self) -> str:
        return self._cognitive_mode

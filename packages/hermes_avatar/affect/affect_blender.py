"""
Emotional inertia: crossfade between affect states over 400–800 ms.

Real emotions have momentum.  If a user was frustrated two seconds ago,
the avatar cannot instantly snap to "warm_acknowledging" — even if the
policy demands it.  The AffectBlender produces intermediate blended
states during the transition window.

Design (per architecture review):
  - The server computes the *target* affect and the recommended *fade
    duration*.  The client (demo.js + demo.css) performs the actual CSS
    transition for buttery-smooth rendering.
  - This module runs server-side inside AffectRuntime.tick() and emits
    AvatarBehaviorState with transition metadata so the client knows
    what to animate.

Coverage: realism improvement #1 (emotional inertia).
"""

from dataclasses import dataclass
from typing import Optional

from .state import AvatarBehaviorState


# ---------------------------------------------------------------------------
# Per-affect crossfade presets (calibrated for psychological plausibility)
# ---------------------------------------------------------------------------

# Longer fades for negative → positive (the avatar shouldn't jerk into
# warmth), shorter fades for neutral → anything.
_AFFECT_FADE_MS: dict[str, int] = {
    # Default for most transitions
    "default": 600,

    # Validating transitions (negative → warm): slow, deliberate
    "angry→warm": 800,
    "frustrated→warm": 800,
    "sad→warm": 750,
    "anxious→calm": 700,

    # Mirror transitions (joy → joy): relatively quick
    "neutral→happy": 400,
    "neutral→excited": 350,
    "happy→excited": 300,

    # Downgrade (excited → neutral): gentle deceleration
    "excited→neutral": 650,
    "happy→neutral": 500,
}


@dataclass
class BlendResult:
    """Output of AffectBlender.tick() — merged into AvatarBehaviorState."""

    target_affect: str
    prev_affect: str
    intensity: float
    prev_intensity: float
    transition_progress: float         # 0.0 → 1.0
    affect_fade_ms: int                # recommended client crossfade duration
    emote_id: str = "neutral"


class AffectBlender:
    """Tracks affect transitions and emits crossfade metadata.

    Usage inside AffectRuntime.tick()::

        blender = AffectBlender()
        ...
        result = blender.tick(target_affect="happy", target_intensity=0.7, dt_ms=16)
        # result.affect_fade_ms tells the client how long to crossfade.
    """

    def __init__(self, default_fade_ms: int = 600):
        self._default_fade_ms = default_fade_ms
        self._current_affect: str = "neutral"
        self._current_intensity: float = 0.5
        self._previous_affect: str = "neutral"
        self._previous_intensity: float = 0.5
        self._elapsed_transition_ms: float = 0.0
        self._fade_duration_ms: int = default_fade_ms
        self._transitioning: bool = False

    # -- public API --------------------------------------------------------

    def tick(
        self,
        target_affect: str,
        target_intensity: float,
        dt_ms: float = 0.0,
    ) -> BlendResult:
        """Advance the blender by *dt_ms* and return blend metadata.

        When *target_affect* changes, the blender begins a crossfade.
        The returned ``transition_progress`` (0→1) tells the caller how
        far along the fade we are; the client uses this to interpolate
        between the old and new CSS expression classes.
        """
        # Detect affect change
        if target_affect != self._current_affect:
            self._previous_affect = self._current_affect
            self._previous_intensity = self._current_intensity
            self._current_affect = target_affect
            self._current_intensity = target_intensity
            self._fade_duration_ms = self._lookup_fade_ms(
                self._previous_affect, self._current_affect
            )
            self._elapsed_transition_ms = 0.0
            self._transitioning = True

        # Detect intensity-only change (no affect label shift)
        elif target_intensity != self._current_intensity:
            self._previous_intensity = self._current_intensity
            self._current_intensity = target_intensity
            if not self._transitioning:
                # Small intensity tweak: short crossfade
                self._fade_duration_ms = 200
                self._elapsed_transition_ms = 0.0
                self._transitioning = True

        # Advance timer
        if self._transitioning:
            self._elapsed_transition_ms += dt_ms
            if self._elapsed_transition_ms >= self._fade_duration_ms:
                self._elapsed_transition_ms = float(self._fade_duration_ms)
                self._transitioning = False

        progress = (
            self._elapsed_transition_ms / self._fade_duration_ms
            if self._fade_duration_ms > 0
            else 1.0
        )

        if not self._transitioning:
            progress = 1.0
            self._previous_affect = self._current_affect
            self._previous_intensity = self._current_intensity

        # Blend intensity linearly during transition
        blended_intensity = self._previous_intensity + (
            self._current_intensity - self._previous_intensity
        ) * progress

        return BlendResult(
            target_affect=self._current_affect,
            prev_affect=self._previous_affect,
            intensity=blended_intensity,
            prev_intensity=self._previous_intensity,
            transition_progress=progress,
            affect_fade_ms=self._fade_duration_ms,
            emote_id=self._current_affect,
        )

    def reset(self, affect: str = "neutral", intensity: float = 0.5) -> None:
        """Hard-reset the blender (e.g. on mode switch or startup)."""
        self._current_affect = affect
        self._previous_affect = affect
        self._current_intensity = intensity
        self._previous_intensity = intensity
        self._elapsed_transition_ms = 0.0
        self._transitioning = False

    # -- internal ----------------------------------------------------------

    def _lookup_fade_ms(self, prev: str, current: str) -> int:
        """Look up the recommended crossfade duration for a transition."""
        key = f"{prev}→{current}"
        return _AFFECT_FADE_MS.get(key, self._default_fade_ms)

    # -- property accessors ------------------------------------------------

    @property
    def current_affect(self) -> str:
        return self._current_affect

    @property
    def is_transitioning(self) -> bool:
        return self._transitioning

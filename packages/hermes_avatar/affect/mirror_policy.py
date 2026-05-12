from __future__ import annotations
from .smoothing import clamp
from .state import UserAffectState


def _social_presence(user: UserAffectState) -> float:
    """Estimate whether mimicry would be legible instead of uncanny."""
    if not user.face_detected:
        return 0.35
    attention = max(user.attention, 0.25 if user.gaze_direction == "toward_user" else 0.0)
    return clamp(attention, 0.0, 1.0)


def mirrored_affect(user: UserAffectState) -> tuple[str, float]:
    """Return a subtle, delayed congruent response.

    Psychological model: emotional mimicry is not a literal copy. It is strongest
    for affiliative/positive signals, moderated by attention and social context,
    and damped for high-threat negative affect to avoid emotional contagion.
    """
    presence = _social_presence(user)
    arousal = clamp(user.arousal, 0.0, 1.0)
    tension = clamp(user.tension, 0.0, 1.0)

    if user.dominant_expression == "happy" and user.valence >= 0:
        return "small_delayed_smile", clamp(
            0.18 + (0.18 * presence) + (0.08 * arousal), 0.16, 0.42
        )
    if user.dominant_expression in {"angry", "frustrated"} or tension > 0.58:
        return "grounded_concern_soft_brow", clamp(
            0.08 + (0.08 * presence) - (0.05 * tension), 0.06, 0.16
        )
    if user.dominant_expression == "sad" or user.valence < -0.35:
        return "soft_concern", clamp(
            0.12 + (0.08 * presence) - (0.04 * arousal), 0.10, 0.22
        )
    if user.dominant_expression == "tired" or arousal < 0.18:
        return "calm_attentive", clamp(0.10 + (0.06 * presence), 0.10, 0.18)
    return "neutral_attentive", clamp(0.10 + (0.05 * presence), 0.10, 0.18)

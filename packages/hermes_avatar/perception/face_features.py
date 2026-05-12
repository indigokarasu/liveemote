from __future__ import annotations


def expression_proxy(blendshapes: dict[str, float] | None = None) -> dict[str, float]:
    b = blendshapes or {}
    smile = max(b.get("mouthSmileLeft", 0), b.get("mouthSmileRight", 0), b.get("smile", 0))
    frown = max(b.get("mouthFrownLeft", 0), b.get("mouthFrownRight", 0), b.get("frown", 0))
    brow = max(b.get("browOuterUpLeft", 0), b.get("browOuterUpRight", 0), b.get("brow_raise", 0))
    eye = max(b.get("eyeWideLeft", 0.5), b.get("eyeWideRight", 0.5), b.get("eye_open", 0.5))
    confidence = max(smile, frown, brow, abs(eye - 0.5) * 2, b.get("confidence", 0.0))
    return {"smile": smile, "frown": frown, "brow_raise": brow, "eye_open": eye, "emotion_confidence": min(1.0, confidence)}


def classify_emotion(features: dict[str, float]) -> dict[str, float | str]:
    smile = features.get("smile", 0.0)
    frown = features.get("frown", 0.0)
    brow = features.get("brow_raise", 0.0)
    eye = features.get("eye_open", 0.5)
    if frown > 0.55 and brow > 0.25:
        label, confidence = "frustrated", max(frown, brow)
    elif smile > 0.45:
        label, confidence = "happy", smile
    elif frown > 0.42:
        label, confidence = "sad", frown
    elif eye < 0.25:
        label, confidence = "tired", 1 - eye
    else:
        label, confidence = "neutral", max(0.35, 1 - max(smile, frown, brow))
    return {"emotion": label, "confidence": min(1.0, confidence)}

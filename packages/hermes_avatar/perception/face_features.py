from __future__ import annotations

def expression_proxy(blendshapes: dict[str, float] | None = None) -> dict[str, float]:
    b = blendshapes or {}
    return {"smile": max(b.get("mouthSmileLeft", 0), b.get("mouthSmileRight", 0)), "frown": max(b.get("mouthFrownLeft", 0), b.get("mouthFrownRight", 0)), "brow_raise": max(b.get("browOuterUpLeft", 0), b.get("browOuterUpRight", 0)), "eye_open": max(b.get("eyeWideLeft", 0.5), b.get("eyeWideRight", 0.5))}

from __future__ import annotations
from pathlib import Path

TAG_BY_STATE = {
    "listening": ["blink", "soft_nod", "attentive"],
    "thinking": ["glance_down", "pause"],
    "happy": ["smile", "warm"],
    "concerned": ["brow", "grounded"],
    "apologetic": ["soft_face", "lower_energy"],
    "neutral": ["idle", "blink"],
    "amused": ["small_smile"],
    "sad": ["consoling", "steady"],
    "error_recovery": ["reset", "calm"],
}

def tags_for(path: Path, state: str) -> list[str]:
    stem_tags = [part for part in path.stem.replace("-", "_").split("_") if part and not part.isdigit()]
    return sorted(set(TAG_BY_STATE.get(state, []) + stem_tags))

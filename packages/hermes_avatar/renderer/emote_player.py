from __future__ import annotations
from hermes_avatar.character.asset_index import CharacterIndex

class EmotePlayer:
    def __init__(self, index: CharacterIndex) -> None:
        self.index = index
        self.active_emote_id: str | None = None
    def choose(self, state: str) -> str | None:
        emote = self.index.find_emote(state)
        self.active_emote_id = emote.id if emote else None
        return self.active_emote_id

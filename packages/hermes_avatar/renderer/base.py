from __future__ import annotations
from abc import ABC, abstractmethod
from hermes_avatar.character.asset_index import CharacterIndex
from hermes_avatar.affect.state import AvatarBehaviorState

class Renderer(ABC):
    @abstractmethod
    def load_character(self, character_index: CharacterIndex) -> None: ...
    @abstractmethod
    def set_behavior(self, behavior: AvatarBehaviorState) -> None: ...
    @abstractmethod
    def speak(self, audio_path: str, text: str, behavior: AvatarBehaviorState) -> None: ...
    @abstractmethod
    def interrupt(self) -> None: ...

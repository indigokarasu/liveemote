from __future__ import annotations
from pydantic import BaseModel

class DemoControlMessage(BaseModel):
    action: str
    text: str | None = None
    mode: str | None = None

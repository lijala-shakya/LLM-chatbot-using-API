
from __future__ import annotations
from typing import Literal, Any
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str
    provider: Literal["groq", "gemini"]
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    extra: dict[str, Any] = Field(default_factory=dict)


class UsageStats(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    content: str
    model: str
    provider: str
    usage: UsageStats | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        return self.content
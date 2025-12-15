from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False

    # Accept extra fields from clients (temperature, max_tokens, etc.).
    model_config = ConfigDict(extra="allow")


class ErrorResponse(BaseModel):
    error: dict[str, Any] = Field(default_factory=dict)


def normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    if isinstance(content, dict):
        if content.get("type") == "text" and isinstance(content.get("text"), str):
            return content["text"]
    return str(content)


def messages_to_prompt(messages: list[ChatMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message.role.upper()
        text = normalize_message_content(message.content)
        parts.append(f"{role}: {text}")
    return "\n\n".join(parts).strip()

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
    max_tokens: int | None = None

    # Accept extra fields from clients (temperature, etc.).
    model_config = ConfigDict(extra="allow")


class ErrorResponse(BaseModel):
    error: dict[str, Any] = Field(default_factory=dict)


class ChatCompletionRequestCompat(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] | None = None
    input: Any = None
    instructions: str | None = None
    stream: bool = False
    max_tokens: int | None = None
    max_output_tokens: int | None = None

    # Accept extra fields from clients (temperature, etc.).
    model_config = ConfigDict(extra="allow")


class ResponsesRequest(BaseModel):
    model: str | None = None
    input: Any = None
    stream: bool = False
    max_output_tokens: int | None = None
    instructions: str | None = None

    # Accept extra fields from clients (temperature, etc.).
    model_config = ConfigDict(extra="allow")


def _coerce_responses_part(part: dict[str, Any]) -> dict[str, Any] | None:
    part_type = part.get("type")
    if part_type in {"input_text", "output_text", "text"} and isinstance(part.get("text"), str):
        return {"type": "text", "text": part["text"]}
    if part_type in {"image_url", "input_image"}:
        return part
    return None


def _coerce_responses_content(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        coerced = _coerce_responses_part(content)
        if coerced is not None:
            return [coerced] if coerced.get("type") in {"image_url", "input_image"} else coerced["text"]
        return str(content)
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            coerced = _coerce_responses_part(part)
            if coerced is not None:
                parts.append(coerced)
        if parts:
            return parts
        texts = [
            part.get("text")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        if texts:
            return "".join(texts)
        return ""
    return str(content)


def responses_input_to_messages(input_obj: Any) -> list[ChatMessage]:
    messages: list[ChatMessage] = []

    def _add(role: str, content: Any) -> None:
        messages.append(ChatMessage(role=role, content=content))

    def _coerce_item(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            _add("user", item)
            return
        if not isinstance(item, dict):
            _add("user", str(item))
            return
        role = item.get("role")
        item_type = item.get("type")
        if item_type == "message" or isinstance(role, str):
            role = role if isinstance(role, str) else "user"
            _add(role, _coerce_responses_content(item.get("content")))
            return
        if item_type in {"input_text", "output_text", "text"} and isinstance(item.get("text"), str):
            _add("user", item["text"])
            return
        if item_type in {"image_url", "input_image"}:
            _add("user", _coerce_responses_content(item))
            return

    if input_obj is None:
        return messages
    if isinstance(input_obj, list):
        for item in input_obj:
            _coerce_item(item)
        return messages

    _coerce_item(input_obj)
    return messages


def responses_request_to_chat_request(req: ResponsesRequest) -> ChatCompletionRequest:
    messages: list[ChatMessage] = []
    if isinstance(req.instructions, str) and req.instructions.strip():
        messages.append(ChatMessage(role="system", content=req.instructions))
    messages.extend(responses_input_to_messages(req.input))

    extra = dict(getattr(req, "model_extra", None) or {})
    max_tokens = req.max_output_tokens
    if max_tokens is None:
        fallback = extra.get("max_tokens")
        if isinstance(fallback, int):
            max_tokens = fallback
    for key in ("model", "input", "stream", "max_output_tokens", "max_tokens", "instructions"):
        extra.pop(key, None)

    return ChatCompletionRequest(
        model=req.model,
        messages=messages,
        stream=req.stream,
        max_tokens=max_tokens,
        **extra,
    )


def compat_chat_request_to_chat_request(req: ChatCompletionRequest | ChatCompletionRequestCompat) -> ChatCompletionRequest:
    if isinstance(req, ChatCompletionRequest):
        return req

    messages = req.messages
    if not messages:
        messages = []
        if isinstance(req.instructions, str) and req.instructions.strip():
            messages.append(ChatMessage(role="system", content=req.instructions))
        messages.extend(responses_input_to_messages(req.input))

    if not messages:
        raise ValueError("Missing messages or input")

    extra = dict(getattr(req, "model_extra", None) or {})
    max_tokens = req.max_tokens
    if max_tokens is None and req.max_output_tokens is not None:
        max_tokens = req.max_output_tokens
    if max_tokens is None:
        fallback = extra.get("max_tokens")
        if isinstance(fallback, int):
            max_tokens = fallback
    for key in ("model", "messages", "input", "instructions", "stream", "max_tokens", "max_output_tokens"):
        extra.pop(key, None)

    return ChatCompletionRequest(
        model=req.model,
        messages=messages,
        stream=req.stream,
        max_tokens=max_tokens,
        **extra,
    )


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


def extract_image_urls_from_content(content: Any) -> list[str]:
    urls: list[str] = []
    if content is None:
        return urls

    # Accept single-part formats in addition to the OpenAI list-of-parts format.
    if isinstance(content, dict):
        part_type = content.get("type")
        if part_type in {"image_url", "input_image"}:
            image = content.get("image_url")
            if isinstance(image, dict):
                url = image.get("url")
                if isinstance(url, str) and url:
                    urls.append(url)
            elif isinstance(image, str) and image:
                urls.append(image)
        return urls

    if not isinstance(content, list):
        return urls

    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type not in {"image_url", "input_image"}:
            continue
        image = part.get("image_url")
        if isinstance(image, dict):
            url = image.get("url")
            if isinstance(url, str) and url:
                urls.append(url)
        elif isinstance(image, str) and image:
            urls.append(image)

    return urls


def extract_image_urls(messages: list[ChatMessage]) -> list[str]:
    urls: list[str] = []
    for message in messages:
        urls.extend(extract_image_urls_from_content(message.content))
    return urls

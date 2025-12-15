from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

SandboxMode = Literal["read-only", "workspace-write", "danger-full-access"]
ApprovalPolicy = Literal["untrusted", "on-failure", "on-request", "never"]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return raw in {"1", "true", "t", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw


def _env_csv(name: str) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return []
    items: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            items.append(part)
    return items


@dataclass(frozen=True)
class Settings:
    host: str = os.environ.get("CODEX_GATEWAY_HOST", "127.0.0.1")
    port: int = _env_int("CODEX_GATEWAY_PORT", 8000)

    # If set, requests must include `Authorization: Bearer <token>`.
    bearer_token: str | None = os.environ.get("CODEX_GATEWAY_TOKEN")

    # Working directory for `codex exec --cd ...`.
    workspace: str = os.environ.get("CODEX_WORKSPACE", os.getcwd())

    # Optional HOME override for the Codex CLI subprocess. Use this to point at a minimal
    # `~/.codex/config.toml` (e.g. without MCP servers) for much lower latency.
    codex_cli_home: str | None = os.environ.get("CODEX_CLI_HOME")

    # Codex CLI options.
    default_model: str = os.environ.get("CODEX_MODEL", "gpt-5-codex")
    # Some local Codex configs default to xhigh, which is not accepted by all models.
    model_reasoning_effort: str | None = (
        _env_str("CODEX_MODEL_REASONING_EFFORT", "high").strip() or None
    )
    sandbox: SandboxMode = os.environ.get("CODEX_SANDBOX", "read-only")  # type: ignore[assignment]
    approval_policy: ApprovalPolicy = os.environ.get("CODEX_APPROVAL_POLICY", "never")  # type: ignore[assignment]
    skip_git_repo_check: bool = _env_bool("CODEX_SKIP_GIT_REPO_CHECK", True)
    enable_search: bool = _env_bool("CODEX_ENABLE_SEARCH", False)
    add_dirs: list[str] = field(default_factory=lambda: _env_csv("CODEX_ADD_DIRS"))

    # Hard safety caps.
    max_prompt_chars: int = _env_int("CODEX_MAX_PROMPT_CHARS", 200_000)
    timeout_seconds: int = _env_int("CODEX_TIMEOUT_SECONDS", 600)
    max_concurrency: int = _env_int("CODEX_MAX_CONCURRENCY", 2)

    # CORS (comma-separated origins). Empty disables CORS.
    cors_origins: str = os.environ.get("CODEX_CORS_ORIGINS", "")


settings = Settings()

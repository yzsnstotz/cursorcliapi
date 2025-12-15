from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

from .config import ApprovalPolicy, SandboxMode


@dataclass(frozen=True)
class CodexResult:
    text: str
    usage: dict[str, int] | None
    raw_events: list[dict] | None = None


def _build_env(codex_cli_home: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if codex_cli_home:
        env["HOME"] = codex_cli_home
    return env


def _build_codex_exec_cmd(
    *,
    prompt: str,
    model: str,
    cd: str,
    sandbox: SandboxMode,
    approval_policy: ApprovalPolicy,
    enable_search: bool,
    add_dirs: list[str],
    json_events: bool,
    skip_git_repo_check: bool,
    model_reasoning_effort: str | None,
) -> list[str]:
    # Note: some flags (e.g. `-a/--ask-for-approval`, `--search`) are global and must appear
    # before the `exec` subcommand.
    cmd: list[str] = ["codex", "-a", approval_policy]
    if enable_search:
        cmd.append("--search")

    cmd.extend(
        [
            "exec",
            # Ensure consistent behavior across machines/configs.
            *(
                ["-c", f'model_reasoning_effort="{model_reasoning_effort}"']
                if model_reasoning_effort
                else []
            ),
            "--color",
            "never",
            "--sandbox",
            sandbox,
            "--model",
            model,
            "--cd",
            cd,
        ]
    )
    if skip_git_repo_check:
        cmd.append("--skip-git-repo-check")
    if enable_search:
        # already added as a global flag above
        pass
    for d in add_dirs:
        cmd.extend(["--add-dir", d])
    if json_events:
        cmd.append("--json")
    cmd.append(prompt)
    return cmd


async def run_codex_final(
    *,
    prompt: str,
    model: str,
    cd: str,
    sandbox: SandboxMode,
    skip_git_repo_check: bool,
    model_reasoning_effort: str | None,
    approval_policy: ApprovalPolicy,
    enable_search: bool,
    add_dirs: list[str],
    codex_cli_home: str | None,
    timeout_seconds: int,
) -> CodexResult:
    cmd = _build_codex_exec_cmd(
        prompt=prompt,
        model=model,
        cd=cd,
        sandbox=sandbox,
        approval_policy=approval_policy,
        enable_search=enable_search,
        add_dirs=add_dirs,
        json_events=False,
        skip_git_repo_check=skip_git_repo_check,
        model_reasoning_effort=model_reasoning_effort,
    )

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_build_env(codex_cli_home),
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise

    if proc.returncode != 0:
        raise RuntimeError(err.decode(errors="ignore").strip() or f"codex exec failed: {proc.returncode}")

    return CodexResult(text=out.decode(errors="ignore").strip(), usage=None)


async def iter_codex_events(
    *,
    prompt: str,
    model: str,
    cd: str,
    sandbox: SandboxMode,
    skip_git_repo_check: bool,
    model_reasoning_effort: str | None,
    approval_policy: ApprovalPolicy,
    enable_search: bool,
    add_dirs: list[str],
    codex_cli_home: str | None,
    timeout_seconds: int,
    capture_events: bool = False,
) -> AsyncIterator[dict]:
    cmd = _build_codex_exec_cmd(
        prompt=prompt,
        model=model,
        cd=cd,
        sandbox=sandbox,
        approval_policy=approval_policy,
        enable_search=enable_search,
        add_dirs=add_dirs,
        json_events=True,
        skip_git_repo_check=skip_git_repo_check,
        model_reasoning_effort=model_reasoning_effort,
    )

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_build_env(codex_cli_home),
    )

    stderr_buf: bytearray = bytearray()
    last_error: str | None = None

    async def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        while True:
            chunk = await proc.stderr.read(4096)
            if not chunk:
                return
            stderr_buf.extend(chunk)
            if len(stderr_buf) > 64_000:
                del stderr_buf[:-64_000]

    drain_task = asyncio.create_task(_drain_stderr())

    try:
        if proc.stdout is None:
            raise RuntimeError("codex exec stdout not available")

        while True:
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout_seconds)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                raise

            if not line:
                break

            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line.decode(errors="ignore"))
            except Exception:
                continue

            if evt.get("type") in {"error", "turn.failed"}:
                msg = evt.get("message")
                if isinstance(msg, str) and msg.strip():
                    last_error = msg.strip()
                err_obj = evt.get("error")
                if isinstance(err_obj, dict) and isinstance(err_obj.get("message"), str):
                    last_error = err_obj["message"].strip() or last_error

            if capture_events:
                yield {"_event": evt}
            else:
                yield evt

        rc = await proc.wait()
        await drain_task
        if rc != 0:
            msg = bytes(stderr_buf).decode(errors="ignore").strip()
            raise RuntimeError(msg or last_error or f"codex exec failed: {rc}")
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        if not drain_task.done():
            drain_task.cancel()


async def collect_codex_text_and_usage_from_events(
    events: AsyncIterator[dict],
) -> CodexResult:
    text_parts: list[str] = []
    usage: dict[str, int] | None = None

    async for evt in events:
        if evt.get("type") == "item.completed":
            item = evt.get("item") or {}
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        if evt.get("type") == "turn.completed":
            raw_usage = evt.get("usage") or {}
            if isinstance(raw_usage, dict):
                # codex usage fields: input_tokens, cached_input_tokens, output_tokens
                in_tokens = int(raw_usage.get("input_tokens") or 0)
                out_tokens = int(raw_usage.get("output_tokens") or 0)
                usage = {
                    "prompt_tokens": in_tokens,
                    "completion_tokens": out_tokens,
                    "total_tokens": in_tokens + out_tokens,
                }

    return CodexResult(text="".join(text_parts).strip(), usage=usage)

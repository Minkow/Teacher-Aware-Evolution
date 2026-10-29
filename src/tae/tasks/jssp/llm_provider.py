from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class LLMProvider(Protocol):
    def complete(self, prompt: str) -> str:
        ...


@dataclass
class MockProvider:
    """Deterministic local provider for testing the LLM loop."""

    variant: int = 0
    calls: int = 0

    def complete(self, prompt: str) -> str:
        variants = [
            """
def score(feature, state):
    urgency = feature.remaining_work - 1.5 * feature.earliest_finish
    congestion = 0.5 * feature.machine_total_work + 0.25 * feature.machine_queue_len
    return urgency + congestion - 0.2 * feature.processing_time
""",
            """
def score(feature, state):
    start_pressure = -feature.earliest_start - feature.earliest_finish
    bottleneck = feature.machine_total_work + feature.machine_ready_time
    progress_penalty = -0.75 * feature.job_progress - 0.3 * feature.remaining_ops
    return start_pressure + 0.6 * bottleneck + 0.7 * feature.remaining_work + progress_penalty
""",
            """
def score(feature, state):
    if feature.remaining_ops <= 2:
        tail_bonus = 0.4 * feature.remaining_work
    else:
        tail_bonus = 0.0
    return (
        0.8 * feature.remaining_work
        + 0.9 * feature.machine_total_work
        - 1.2 * feature.earliest_finish
        - 0.8 * feature.earliest_start
        - 0.4 * feature.processing_time
        + tail_bonus
    )
""",
        ]
        choice = variants[(self.variant + self.calls) % len(variants)]
        self.calls += 1
        return choice


@dataclass(frozen=True)
class CommandProvider:
    """Provider that sends the prompt to an external command on stdin."""

    command: str
    timeout_s: int = 120

    def complete(self, prompt: str) -> str:
        result = subprocess.run(
            self.command,
            input=prompt,
            shell=True,
            text=True,
            capture_output=True,
            timeout=self.timeout_s,
            check=True,
        )
        return result.stdout


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    """Minimal OpenAI-compatible chat completions client using stdlib HTTP."""

    model: str
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    timeout_s: int = 120
    stream: bool = False
    retries: int = 5

    def complete(self, prompt: str) -> str:
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for openai-compatible provider")
        base_url = self.base_url
        if not base_url.startswith(("http://", "https://")):
            base_url = "https://" + base_url
        url = base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "temperature": 0.8,
            "stream": self.stream,
            "messages": [
                {
                    "role": "system",
                    "content": "You design deterministic Python priority rules for JSSP. Return only code.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    if self.stream:
                        return _read_sse_chat_completion(resp)
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as exc:
                last_error = exc
                if isinstance(exc, urllib.error.HTTPError) and exc.code < 500:
                    raise
                if attempt == self.retries:
                    break
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"LLM request failed after {self.retries} attempts: {last_error}") from last_error


def _read_sse_chat_completion(resp) -> str:
    chunks: list[str] = []
    for raw_line in resp:
        line = raw_line.decode("utf-8").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        row = json.loads(data)
        choices = row.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content")
        if content:
            chunks.append(content)
    return "".join(chunks)


def make_provider(
    name: str,
    *,
    model: str = "gpt-4.1-mini",
    command: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    mock_variant: int = 0,
    stream: bool = False,
) -> LLMProvider:
    if name == "mock":
        return MockProvider(variant=mock_variant)
    if name == "command":
        if not command:
            raise ValueError("--command is required for command provider")
        timeout_s = int(os.environ.get("LLM_TIMEOUT_S", "120"))
        return CommandProvider(command=command, timeout_s=timeout_s)
    if name in {"openai", "openai-compatible"}:
        timeout_s = int(os.environ.get("LLM_TIMEOUT_S", "120"))
        retries = int(os.environ.get("LLM_RETRIES", "5"))
        return OpenAICompatibleProvider(
            model=model,
            base_url=base_url,
            timeout_s=timeout_s,
            stream=stream,
            retries=retries,
        )
    raise ValueError(f"unknown provider: {name}")

import json
import os
import random
import socket
import threading
import time
import http.client
import urllib.error
import urllib.request


RETRYABLE_HTTP_STATUS = {409, 425, 429}


class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.timeout_s = int(os.environ.get("LLM_TIMEOUT_S", "120"))
        self.n_trial = int(os.environ.get("LLM_RETRIES", "3"))
        self.backoff_cap_s = int(os.environ.get("LLM_RETRY_BACKOFF_CAP_S", "60"))
        self.temperature = float(os.environ.get("LLM_TEMPERATURE", "0"))
        self.top_p = float(os.environ.get("LLM_TOP_P", "1"))
        self.reasoning_effort = os.environ.get("LLM_REASONING_EFFORT", "").strip()
        self._url = self._resolve_endpoint(api_endpoint)
        self._deadline_local = threading.local()

    def set_deadline(self, deadline):
        self._deadline_local.value = deadline

    def clear_deadline(self):
        self._deadline_local.value = None

    def _remaining_timeout(self):
        deadline = getattr(self._deadline_local, "value", None)
        if deadline is None:
            return float(self.timeout_s)
        remaining = float(deadline) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("per-FE deadline exhausted before LLM request")
        return min(float(self.timeout_s), remaining)

    def get_response(self, prompt_content):
        request_deadline = time.monotonic() + self._remaining_timeout()
        payload = {
            "model": self.model_LLM,
            "stream": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": [
                {"role": "user", "content": prompt_content},
            ],
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.api_key,
                "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
                "Content-Type": "application/json",
                "x-api2d-no-cache": "1",
            },
            method="POST",
        )

        last_error = None
        for attempt in range(1, self.n_trial + 1):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=min(self._remaining_timeout(), self._request_remaining(request_deadline)),
                ) as response:
                    return self._read_with_deadline(response, request_deadline)
            except (
                TimeoutError,
                urllib.error.URLError,
                urllib.error.HTTPError,
                RuntimeError,
                http.client.HTTPException,
                OSError,
            ) as exc:
                if isinstance(exc, (TimeoutError, socket.timeout)) or isinstance(getattr(exc, "reason", None), (TimeoutError, socket.timeout)):
                    raise TimeoutError("LLM timeout is terminal; no retry") from exc
                if isinstance(exc, urllib.error.HTTPError) and exc.code in {408, 504}:
                    raise TimeoutError("Upstream timeout is terminal; no retry") from exc
                self._request_remaining(request_deadline)
                last_error = exc
                if (
                    isinstance(exc, urllib.error.HTTPError)
                    and exc.code < 500
                    and exc.code not in RETRYABLE_HTTP_STATUS
                ):
                    raise
                if attempt == self.n_trial:
                    break
                if self.debug_mode:
                    print(
                        f"LLM API retry {attempt}/{self.n_trial} after "
                        f"{type(exc).__name__}: {exc}"
                    )
                delay = min(self.backoff_cap_s, 2 ** attempt) + random.uniform(0.0, 1.0)
                deadline = getattr(self._deadline_local, "value", None)
                if deadline is not None:
                    remaining = float(deadline) - time.monotonic()
                    if remaining <= 0:
                        break
                    delay = min(delay, remaining)
                time.sleep(delay)
        if last_error is None:
            return None
        raise RuntimeError(
            f"LLM request failed after {self.n_trial} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _request_remaining(deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("LLM total deadline exhausted; no retry")
        return remaining

    def _read_with_deadline(self, response, deadline):
        expired = threading.Event()
        raw = getattr(getattr(response, "fp", None), "raw", None)
        sock = getattr(raw, "_sock", None)

        def interrupt():
            expired.set()
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        timer = threading.Timer(self._request_remaining(deadline), interrupt)
        timer.daemon = True
        timer.start()
        try:
            return self._read_streaming_content(response)
        finally:
            timer.cancel()
            if expired.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("LLM streaming total deadline exhausted; no retry")

    def _read_streaming_content(self, res):
        content_parts = []
        for raw_line in res:
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if piece:
                content_parts.append(piece)
                continue
            message = choices[0].get("message") or {}
            piece = message.get("content")
            if piece:
                content_parts.append(piece)
        response = "".join(content_parts).strip()
        if not response:
            raise RuntimeError("empty streaming response")
        return response

    def _resolve_endpoint(self, api_endpoint):
        endpoint = (api_endpoint or "").strip()
        if "://" not in endpoint:
            endpoint = "https://" + endpoint
        if endpoint.rstrip("/").endswith("/v1"):
            return endpoint.rstrip("/") + "/chat/completions"
        return endpoint.rstrip("/") + "/v1/chat/completions"

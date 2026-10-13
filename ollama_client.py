"""Small, retryable HTTP client for a local Ollama model."""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from config import OLLAMA_RETRIES, OLLAMA_TIMEOUT


class OllamaConnectionError(RuntimeError):
    """Raised when Ollama or the configured model cannot be reached."""


class OllamaChatClient:
    def __init__(self, base_url: str, model: str, timeout: int = OLLAMA_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def check_ready(self) -> None:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaConnectionError(
                f"无法连接 Ollama：{self.base_url}。请确认本机 Ollama 服务已启动。"
            ) from exc

        names = {item.get("name") for item in response.json().get("models", [])}
        if self.model not in names:
            raise OllamaConnectionError(f"Ollama 未找到模型 {self.model}，请执行：ollama pull {self.model}")

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.1) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        last_error: requests.RequestException | None = None
        for attempt in range(OLLAMA_RETRIES + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                content = str(response.json().get("message", {}).get("content", "")).strip()
                if not content:
                    raise OllamaConnectionError(f"Ollama 模型 {self.model} 返回了空内容。")
                return content
            except requests.RequestException as exc:
                last_error = exc
                if attempt < OLLAMA_RETRIES:
                    time.sleep(0.5 * (2**attempt))
        raise OllamaConnectionError(f"调用 Ollama 模型 {self.model} 失败，请检查服务状态。") from last_error


def extract_json(text: str) -> dict[str, Any]:
    """Extract one balanced JSON object from a model response."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError(f"模型没有返回 JSON：{text}")

    depth = 0
    in_string = False
    escaped = False
    end = -1
    for index in range(start, len(cleaned)):
        character = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end < start:
        raise ValueError(f"模型没有返回完整 JSON：{text}")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("模型 JSON 顶层必须是对象")
    return value

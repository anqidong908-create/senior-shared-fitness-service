"""Configuration for the shared-fitness multi-agent service."""

from __future__ import annotations

import os
from pathlib import Path


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", "2"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "1000"))
DEVICE_DATA_MODE = os.getenv("DEVICE_DATA_MODE", "simulator")
APP_ENV = os.getenv("APP_ENV", "demo")

_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
CORS_ORIGINS = [item.strip() for item in _origins.split(",") if item.strip()]
AUTH_SECRET = os.getenv("AUTH_SECRET", "local-demo-secret-change-me")
DEMO_LOGIN_KEY = os.getenv("DEMO_LOGIN_KEY", "local-demo-login")
SESSION_DB_PATH = os.getenv("SESSION_DB_PATH", str(Path(__file__).with_name("shared_fitness.sqlite3")))

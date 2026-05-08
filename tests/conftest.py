"""
pytest configuration — sets required environment variables before any app
module is imported so that pydantic-settings can validate them at import time.

Uses setdefault() so values already present in the environment (e.g. from a
real .env file) are not overwritten.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-sk-00000000000000000000000000000000")
os.environ.setdefault("API_KEYS", "test-api-key-valid")
os.environ.setdefault("CORS_ALLOWED_ORIGINS", "http://localhost:3000")
os.environ.setdefault("MODEL_NAME", "gpt-4o")
os.environ.setdefault("DATA_DIR", "/tmp/test-agent-data")

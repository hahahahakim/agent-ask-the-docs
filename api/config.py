"""
Centralised configuration for the 0G Docs Agent API.

All settings are read from environment variables (or a .env file).
pydantic-settings validates every required value at import time, so a
missing or malformed variable fails the process immediately with a clear
error rather than a KeyError at request time.
"""

from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM backend (re-exported so build_agent() picks them up via os.environ)
    openai_api_key: str
    openai_base_url: Optional[str] = None
    model_name: str = "gpt-4o"

    # Agent behaviour
    data_dir: str = "./data"
    verbose: bool = False

    # API security — required, validated below
    api_keys: str           # comma-separated; e.g. "key-a,key-b"
    cors_allowed_origins: str  # comma-separated; e.g. "https://mysite.com"

    # Rate limiting (requests per minute per API key)
    rate_limit_per_minute: int = 20

    # Lazy TTL expiry — threads inactive longer than this are wiped on next access
    thread_ttl_hours: int = 24

    @field_validator("api_keys")
    @classmethod
    def _api_keys_not_empty(cls, v: str) -> str:
        keys = [k.strip() for k in v.split(",") if k.strip()]
        if not keys:
            raise ValueError("API_KEYS must contain at least one non-empty key")
        return v

    @field_validator("cors_allowed_origins")
    @classmethod
    def _origins_not_empty(cls, v: str) -> str:
        origins = [o.strip() for o in v.split(",") if o.strip()]
        if not origins:
            raise ValueError(
                "CORS_ALLOWED_ORIGINS must contain at least one origin. "
                "Wildcard CORS (*) is not supported."
            )
        return v

    def get_api_keys(self) -> set:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    def get_allowed_origins(self) -> list:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]


settings = Settings()

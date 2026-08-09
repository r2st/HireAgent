"""Application settings, loaded from environment / .env."""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- App ---
    app_name: str = "HireAgent"
    environment: str = "development"
    debug: bool = True
    api_v1_prefix: str = "/api/v1"

    # --- Database ---
    database_url: str = "postgresql+asyncpg://hireagent:hireagent@localhost:5436/hireagent"
    db_echo: bool = False
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # --- Redis / queue ---
    redis_url: str = "redis://localhost:6380/0"

    # --- Security ---
    # Override in every non-development environment.
    secret_key: str = "dev-only-insecure-secret-change-me"
    # Fernet key for column-level encryption of resume/PII data. Generate with
    # `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
    encryption_key: str = "ZmFrZS1kZXYta2V5LWZvci1sb2NhbC10ZXN0aW5nLTEyMw="
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30

    # --- LLM (OpenRouter) ---
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Free-tier models per the design doc's cost-efficiency requirement.
    llm_model_parsing: str = "meta-llama/llama-3.3-70b-instruct:free"
    llm_model_scoring: str = "meta-llama/llama-3.3-70b-instruct:free"
    llm_model_writing: str = "meta-llama/llama-3.3-70b-instruct:free"
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 3
    # When false (default in tests), LLM calls short-circuit to deterministic
    # heuristics instead of hitting the network.
    llm_enabled: bool = True

    # --- Calendar sync (design §4.3) ---
    # When false, slot proposals fall back to configured working hours and no
    # external calendar events are written.
    calendar_sync_enabled: bool = True
    google_client_id: str = ""
    google_client_secret: str = ""
    microsoft_client_id: str = ""
    microsoft_client_secret: str = ""
    microsoft_tenant_id: str = "common"

    # --- Public URLs ---
    # Where the candidate-facing frontend lives. Booking and unsubscribe links
    # that a human clicks point here; the page then calls the API.
    public_base_url: str = "http://localhost:3000"
    # Where this API is reachable from outside. Distinct from public_base_url
    # because a mail client's one-click unsubscribe POSTs straight at the API
    # and never loads a page — see outreach_service.one_click_unsubscribe_url.
    api_base_url: str = "http://localhost:8000"

    # --- Interview scheduling ---
    # How long a proposed-slot booking link stays valid.
    booking_token_ttl_hours: int = 168
    # Slots are never offered closer than this to now.
    interview_min_notice_hours: int = 12
    # How far ahead slot search looks by default.
    interview_horizon_days: int = 14

    # --- Outreach (design §4.2) ---
    # Master switch for outbound sending. False keeps the whole pipeline —
    # enrolment, scheduling, rendering — working while no mail leaves the box,
    # which is what staging and the test suite want.
    outreach_sending_enabled: bool = True
    # Warm-up ramp: a fresh mailbox starts here and climbs by the daily step
    # until it reaches the cap, at which point it is READY.
    warmup_initial_daily_limit: int = 10
    warmup_daily_increment: int = 5
    warmup_target_daily_limit: int = 200
    # A sender whose reputation falls below this is pulled from rotation.
    sender_min_reputation: float = 50.0
    # Retries before a queued message is abandoned.
    outreach_max_send_attempts: int = 3

    # --- Storage ---
    storage_dir: str = "./var/storage"
    max_upload_bytes: int = 15 * 1024 * 1024

    # --- Rate limiting (design §6.1: 200 req/min per org) ---
    rate_limit_per_minute: int = 200

    # --- CORS ---
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # --- Compliance ---
    default_retention_months: int = 24

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

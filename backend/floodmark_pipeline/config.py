from __future__ import annotations

import os
from dataclasses import dataclass

# 511GA source camera IDs that call dry scenes flooded. From the ML evaluation
# (11372 hazy lens, 17397 bridge pier, 13750 gated booth) and from live
# daytime false alarms (13417 GDOT-0272, 17356 GDOT-1342, 13536 GDOT-0676,
# 14256 GDOT-0675).
DEFAULT_ALERT_BLOCKLIST = "11372,17397,13750,13417,17356,13536,14256"


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    redis_url: str
    s3_endpoint_url: str
    s3_region: str
    s3_access_key_id: str
    s3_secret_access_key: str
    s3_bucket: str
    resend_api_key: str
    resend_alerts_from: str
    public_app_url: str
    notification_unsubscribe_secret: str
    camera_base_url: str
    download_concurrency: int
    s3_upload_concurrency: int
    inference_concurrency: int
    database_concurrency: int
    database_pool_size: int
    max_response_bytes: int
    max_inflight_bytes: int
    request_timeout_seconds: float
    arq_max_tries: int
    arq_retry_delay_seconds: int
    arq_result_ttl_seconds: int
    debug: bool
    scheduler_run_at_startup: bool
    alert_streak_frames: int
    alert_blocklist: frozenset[str]
    alert_baseline_days: int
    alert_baseline_min_frames: int
    alert_baseline_margin: float
    alert_require_rain: bool
    alert_min_rain_mm: float
    alert_rain_window_hours: int
    alert_storm_streak_frames: int
    alert_fast_poll_seconds: int
    alert_fast_poll_max: int
    weather_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        required = (
            "DATABASE_URL", "S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_BUCKET"
        )
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

        def number(name: str, default: str, cast: type[int] | type[float] = int) -> int | float:
            value = cast(os.getenv(name, default))
            if value <= 0:
                raise RuntimeError(f"{name} must be greater than zero")
            return value

        def count(name: str, default: str) -> int:
            value = int(os.getenv(name, default))
            if value < 0:
                raise RuntimeError(f"{name} must not be negative")
            return value

        def boolean(name: str, default: bool = False) -> bool:
            value = os.getenv(name, str(default)).lower()
            if value not in {"true", "false"}:
                raise RuntimeError(f"{name} must be true or false")
            return value == "true"

        return cls(
            database_url=os.environ["DATABASE_URL"],
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            s3_endpoint_url=os.environ["S3_ENDPOINT_URL"].rstrip("/"),
            s3_region=os.getenv("S3_REGION", "us-east-1"),
            s3_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
            s3_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
            s3_bucket=os.environ["S3_BUCKET"],
            resend_api_key=os.getenv("RESEND_API_KEY", ""),
            resend_alerts_from=os.getenv("RESEND_ALERTS_FROM", "alerts@floodmark.tech"),
            public_app_url=os.getenv("PUBLIC_APP_URL", "http://localhost:5173").rstrip("/"),
            notification_unsubscribe_secret=os.getenv("NOTIFICATION_UNSUBSCRIBE_SECRET", ""),
            camera_base_url=os.getenv("CAMERA_BASE_URL", "https://511ga.org/map/Cctv").rstrip("/"),
            download_concurrency=int(number("DOWNLOAD_CONCURRENCY", "300")),
            s3_upload_concurrency=int(number("S3_UPLOAD_CONCURRENCY", "32")),
            inference_concurrency=int(number("INFERENCE_CONCURRENCY", "4")),
            database_concurrency=int(number("DATABASE_CONCURRENCY", "16")),
            database_pool_size=int(number("DATABASE_POOL_SIZE", "16")),
            max_response_bytes=int(number("MAX_RESPONSE_BYTES", "8388608")),
            max_inflight_bytes=int(number("MAX_INFLIGHT_BYTES", "536870912")),
            request_timeout_seconds=float(number("REQUEST_TIMEOUT_SECONDS", "20", float)),
            arq_max_tries=int(number("ARQ_MAX_TRIES", "3")),
            arq_retry_delay_seconds=int(number("ARQ_RETRY_DELAY_SECONDS", "15")),
            arq_result_ttl_seconds=int(number("ARQ_RESULT_TTL_SECONDS", "900")),
            debug=boolean("DEBUG"),
            scheduler_run_at_startup=boolean("SCHEDULER_RUN_AT_STARTUP"),
            alert_streak_frames=int(number("ALERT_STREAK_FRAMES", "3")),
            alert_blocklist=frozenset(
                item.strip() for item in os.getenv("ALERT_BLOCKLIST", DEFAULT_ALERT_BLOCKLIST).split(",") if item.strip()
            ),
            alert_baseline_days=int(number("ALERT_BASELINE_DAYS", "7")),
            alert_baseline_min_frames=int(number("ALERT_BASELINE_MIN_FRAMES", "24")),
            alert_baseline_margin=float(number("ALERT_BASELINE_MARGIN", "0.10", float)),
            alert_require_rain=boolean("ALERT_REQUIRE_RAIN", True),
            alert_min_rain_mm=float(number("ALERT_MIN_RAIN_MM", "1.0", float)),
            alert_rain_window_hours=int(number("ALERT_RAIN_WINDOW_HOURS", "6")),
            alert_storm_streak_frames=int(number("ALERT_STORM_STREAK_FRAMES", "2")),
            alert_fast_poll_seconds=int(number("ALERT_FAST_POLL_SECONDS", "60")),
            alert_fast_poll_max=count("ALERT_FAST_POLL_MAX", "0"),
            weather_timeout_seconds=float(number("WEATHER_TIMEOUT_SECONDS", "5", float)),
        )

import pytest

from floodmark_pipeline.config import Settings


def test_settings_read_s3_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/floodmark")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.us-east-1.wasabisys.com/")
    monkeypatch.setenv("S3_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("S3_BUCKET", "images")

    settings = Settings.from_env()

    assert settings.s3_endpoint_url == "https://s3.us-east-1.wasabisys.com"
    assert settings.s3_region == "us-east-1"
    assert settings.s3_access_key_id == "access"
    assert settings.s3_secret_access_key == "secret"
    assert settings.s3_bucket == "images"
    assert settings.s3_upload_concurrency == 32


def test_alert_settings_have_defaults_and_read_blocklist(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/floodmark")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.us-east-1.wasabisys.com/")
    monkeypatch.setenv("S3_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("S3_BUCKET", "images")

    settings = Settings.from_env()
    assert settings.alert_streak_frames == 3
    assert {"11372", "17397", "13750", "13417"} <= settings.alert_blocklist
    assert settings.alert_require_rain is True
    assert settings.alert_min_rain_mm == 1.0
    assert settings.alert_rain_window_hours == 6
    assert settings.alert_baseline_margin == 0.10
    assert settings.alert_storm_streak_frames == 2
    assert settings.alert_fast_poll_seconds == 60
    assert settings.alert_fast_poll_max == 4

    monkeypatch.setenv("ALERT_BLOCKLIST", " 1 ,2,,3 ")
    monkeypatch.setenv("ALERT_REQUIRE_RAIN", "false")
    monkeypatch.setenv("ALERT_FAST_POLL_MAX", "0")
    settings = Settings.from_env()
    assert settings.alert_fast_poll_max == 0
    assert settings.alert_blocklist == frozenset({"1", "2", "3"})
    assert settings.alert_require_rain is False


def test_settings_reject_old_r2_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/floodmark")
    monkeypatch.setenv("R2_ACCOUNT_ID", "account")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("R2_BUCKET", "images")

    with pytest.raises(RuntimeError, match="S3_ENDPOINT_URL"):
        Settings.from_env()

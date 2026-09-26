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


def test_settings_reject_old_r2_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/floodmark")
    monkeypatch.setenv("R2_ACCOUNT_ID", "account")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("R2_BUCKET", "images")

    with pytest.raises(RuntimeError, match="S3_ENDPOINT_URL"):
        Settings.from_env()

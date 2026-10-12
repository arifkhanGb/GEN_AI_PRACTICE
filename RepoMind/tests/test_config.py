from app.config import Settings


def test_settings_load_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "RepoMind"
    assert settings.app_version == "0.1.0"
    assert settings.environment == "development"
    assert settings.debug is False
    assert settings.api_v1_prefix == "/api/v1"


def test_settings_read_environment_variables(
    monkeypatch,
) -> None:
    monkeypatch.setenv("REPOMIND_APP_NAME", "TestRepoMind")
    monkeypatch.setenv("REPOMIND_DEBUG", "true")

    settings = Settings(_env_file=None)

    assert settings.app_name == "TestRepoMind"
    assert settings.debug is True

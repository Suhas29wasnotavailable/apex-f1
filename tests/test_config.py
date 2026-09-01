"""Configuration behaviour that results depend on.

The config hash is stamped onto generated artefacts, so its two properties -- sensitive to
anything that changes a number, insensitive to where files happen to live -- are load-bearing
for reproducibility claims, not cosmetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from apex.config import AppSettings, DataSettings, PathSettings, get_settings


def test_defaults_load_from_yaml() -> None:
    settings = AppSettings()
    assert settings.data.train_seasons == (2022, 2023, 2024)
    assert settings.data.holdout_seasons == (2025,)
    assert settings.calibration.exclusions.excluded_track_status == ("2", "4", "5", "6", "7")


def test_holdout_seasons_are_never_trained_on() -> None:
    train = set(AppSettings().data.train_seasons)
    holdout = set(AppSettings().data.holdout_seasons)
    assert not train & holdout


def test_environment_overrides_yaml() -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("APEX_CALIBRATION__EXCLUSIONS__MAX_LAP_TIME_RATIO", "1.05")
        patch.setenv("APEX_SEED", "1234")
        settings = AppSettings()
    assert settings.calibration.exclusions.max_lap_time_ratio == 1.05
    assert settings.seed == 1234


def test_overlapping_split_is_rejected() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        DataSettings(train_seasons=(2022, 2023), holdout_seasons=(2023,))


def test_seasons_before_telemetry_coverage_are_rejected() -> None:
    with pytest.raises(ValidationError, match="predate FastF1 telemetry"):
        DataSettings(train_seasons=(2016,), holdout_seasons=(2025,))


def test_empty_training_set_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        DataSettings(train_seasons=(), holdout_seasons=(2025,))


def test_unknown_keys_are_rejected() -> None:
    """A typo in a config file must fail loudly, not be silently ignored."""
    with pytest.raises(ValidationError):
        DataSettings(trian_seasons=(2022,))  # type: ignore[call-arg]


def test_settings_are_immutable() -> None:
    settings = AppSettings()
    with pytest.raises(ValidationError):
        settings.seed = 1  # type: ignore[misc]


def test_config_hash_ignores_machine_specific_paths(tmp_path: Path) -> None:
    baseline = AppSettings()
    relocated = AppSettings(paths=PathSettings(processed=tmp_path / "elsewhere"))
    assert relocated.paths.processed != baseline.paths.processed
    assert relocated.config_hash() == baseline.config_hash()


def test_config_hash_tracks_anything_that_changes_a_result() -> None:
    baseline = AppSettings()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("APEX_CALIBRATION__EXCLUSIONS__MAX_LAP_TIME_RATIO", "1.05")
        altered = AppSettings()
    assert altered.config_hash() != baseline.config_hash()


def test_relative_paths_resolve_against_the_project_root() -> None:
    assert AppSettings().paths.params.is_absolute()


def test_ensure_directories_creates_every_configured_location(tmp_path: Path) -> None:
    paths = PathSettings(
        fastf1_cache=tmp_path / "cache",
        processed=tmp_path / "processed",
        params=tmp_path / "params",
        figures=tmp_path / "figures",
        mlruns=tmp_path / "mlruns",
    )
    paths.ensure_directories()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "cache",
        "figures",
        "mlruns",
        "params",
        "processed",
    ]


def test_seed_bundle_follows_the_configured_seed() -> None:
    assert AppSettings(seed=99).seeds().master_seed == 99


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()

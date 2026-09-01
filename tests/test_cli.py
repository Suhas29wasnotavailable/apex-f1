"""The CLI is the documented entry point, so its contract is tested."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from apex import __version__
from apex.cli import app
from apex.config import get_settings
from apex.data.ingest import summarise, write_race
from apex.data.sources import to_lap_frame
from tests.factories import raw_lap, raw_race

runner = CliRunner()


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the process-wide settings at an empty temporary processed store."""
    monkeypatch.setenv("APEX_PATHS__PROCESSED", str(tmp_path / "processed"))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def write_a_race(rows: int = 12, season: int = 2023) -> None:
    settings = get_settings()
    raw = raw_race([raw_lap(LapNumber=float(n)) for n in range(2, 2 + rows)], season=season)
    laps = to_lap_frame(raw)
    write_race(settings, summarise(settings, raw, laps), laps)


def test_version_matches_the_package() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_config_emits_valid_json_with_a_provenance_hash() -> None:
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert len(payload["config_hash"]) == 16
    assert payload["data"]["holdout_seasons"] == [2025]


def test_config_omits_machine_specific_paths_by_default() -> None:
    assert "paths" not in json.loads(runner.invoke(app, ["config"]).stdout)
    assert "paths" in json.loads(runner.invoke(app, ["config", "--paths"]).stdout)


def test_seed_reports_the_run_identifier() -> None:
    result = runner.invoke(app, ["seed"])
    assert result.exit_code == 0
    assert "master_seed=" in result.stdout
    assert "run_id=" in result.stdout


def test_races_explains_itself_when_nothing_is_ingested(store: Path) -> None:
    """An empty store is a normal state on a fresh clone, so the message must say what to run."""
    result = runner.invoke(app, ["races"])
    assert result.exit_code == 1
    assert "apex ingest" in result.stdout
    assert store.exists()


def test_races_lists_what_is_on_disk(store: Path) -> None:
    write_a_race()
    result = runner.invoke(app, ["races"])
    assert result.exit_code == 0
    assert "Test Grand Prix" in result.stdout
    assert "1 races ingested, 1 dry" in result.stdout
    assert store.exists()


def test_laps_reports_the_exclusion_diagnostics(store: Path) -> None:
    write_a_race(season=2023)
    result = runner.invoke(app, ["laps", "--season", "2023"])
    assert result.exit_code == 0
    assert "input" in result.stdout
    assert "short_stint" in result.stdout
    assert "retained" in result.stdout
    assert store.exists()


def test_laps_refuses_to_report_on_nothing(store: Path) -> None:
    write_a_race(season=2023)
    result = runner.invoke(app, ["laps", "--season", "2024"])
    assert result.exit_code == 1
    assert "apex ingest" in result.stdout
    assert store.exists()

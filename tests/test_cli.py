"""The CLI is the documented entry point, so its contract is tested."""

from __future__ import annotations

import json

from apex import __version__
from apex.cli import app
from typer.testing import CliRunner

runner = CliRunner()


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

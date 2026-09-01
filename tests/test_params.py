"""The parameter file's contract with everything downstream.

`params/params.json` is the only channel between calibration and the simulator, so the
guarantees it makes are the ones the project's honesty claims rest on: it says which seasons it
saw, every number in it carries an interval, and it round-trips exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apex.calibrate.dataset import CalibrationData
from apex.calibrate.estimates import Estimate
from apex.calibrate.exclusions import clean_laps
from apex.calibrate.features import add_features
from apex.calibrate.params import PARAMS_SCHEMA_VERSION, ParameterSet, calibrate, write_schema
from apex.calibrate.report import render_report
from apex.config import AppSettings
from tests.factories import lap_frame, settings_in
from tests.test_pace import LINEAR, simulated_race


def calibration_data(
    *, circuits: tuple[str, ...] = ("Alpha", "Beta"), races: int = 3
) -> CalibrationData:
    """Enough simulated racing for every parameter to be estimable."""
    rng = np.random.default_rng(4)
    rows: list[dict[str, object]] = []
    round_number = 0
    for circuit in circuits:
        for index in range(races):
            round_number += 1
            race = simulated_race(2023, round_number, circuit, rng=rng)
            if index == 0:
                # One neutralisation per circuit, so the hazard has something to find.
                for row in race:
                    if row["lap"] in (11, 12):
                        row["track_status"] = "4"
            rows.extend(race)
    featured = add_features(lap_frame(rows), LINEAR)
    report = clean_laps(featured, LINEAR)
    return CalibrationData(
        laps=report.laps,
        all_laps=featured,
        diagnostics=report.diagnostics,
        seasons=(2023,),
    )


@pytest.fixture(scope="module")
def params() -> ParameterSet:
    return calibrate(LINEAR, calibration_data())


def test_the_file_records_which_seasons_it_was_allowed_to_see(params: ParameterSet) -> None:
    """ADR-0002's split is auditable from the file alone, without the configuration."""
    assert params.provenance.seasons == (2023,)
    assert params.provenance.holdout_seasons == LINEAR.data.holdout_seasons
    assert not set(params.provenance.seasons) & set(params.provenance.holdout_seasons)


def test_the_file_records_the_code_and_configuration_that_produced_it(
    params: ParameterSet,
) -> None:
    assert params.provenance.config_hash == LINEAR.config_hash()
    assert params.provenance.seed == LINEAR.seed
    assert params.provenance.schema_version == PARAMS_SCHEMA_VERSION
    assert params.provenance.generated_at.tzinfo is not None


def test_the_exclusion_audit_travels_with_the_parameters(params: ParameterSet) -> None:
    """The laps thrown away must be as visible as the ones that were kept."""
    rules = [entry.rule for entry in params.exclusions]
    assert rules[0] == "input"
    assert "short_stint" in rules
    assert params.provenance.n_laps_used == params.exclusions[-1].remaining
    assert params.provenance.n_laps_available == params.exclusions[0].remaining


def test_every_published_number_carries_an_interval(params: ParameterSet) -> None:
    """The claim ADR-0002 makes, checked over the whole file rather than asserted in prose."""
    for estimate in _every_estimate(params):
        assert estimate.level == LINEAR.calibration.confidence_level
        assert estimate.interval is not None, f"no interval on {estimate}"
        low, high = estimate.interval
        assert low <= estimate.value <= high, estimate


def _every_estimate(params: ParameterSet) -> list[Estimate]:
    found: list[Estimate] = []

    def walk(value: object) -> None:
        if isinstance(value, Estimate):
            found.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list | tuple):
            for item in value:
                walk(item)
        elif hasattr(value, "__pydantic_fields__"):
            for name in type(value).model_fields:  # type: ignore[attr-defined]
                walk(getattr(value, name))

    walk(params)
    return found


def test_the_file_round_trips_exactly(params: ParameterSet, tmp_path: Path) -> None:
    path = tmp_path / "params.json"
    params.write(path)
    assert ParameterSet.read(path) == params


def test_every_circuit_in_the_data_appears_in_the_file(params: ParameterSet) -> None:
    circuits = {model.circuit for model in params.circuit_pace}
    assert circuits == {"Alpha", "Beta"}
    assert params.circuit_pace_at("Alpha").circuit == "Alpha"


def test_asking_for_a_circuit_that_was_never_raced_is_an_error(params: ParameterSet) -> None:
    with pytest.raises(KeyError, match="Nürburgring"):
        params.circuit_pace_at("Nürburgring")


def test_the_schema_is_written_for_consumers_to_validate_against(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    path = write_schema(settings)
    assert path.is_file()
    assert "provenance" in path.read_text(encoding="utf-8")


def test_calibrating_without_data_fails_rather_than_writing_an_empty_result(
    tmp_path: Path,
) -> None:
    """An empty parameter file would be worse than none, because it would look like a result."""
    empty = CalibrationData(
        laps=lap_frame([]),
        all_laps=lap_frame([]),
        diagnostics=lap_frame([]),
        seasons=(2023,),
    )
    with pytest.raises(ValueError, match="apex ingest"):
        calibrate(settings_in(tmp_path), empty)


def test_the_report_states_the_split_and_the_exclusions(params: ParameterSet) -> None:
    text = render_report(params)
    assert "Held out" in text
    assert str(params.provenance.config_hash) in text
    assert "short_stint" in text
    assert "Pit loss" in text
    assert "Safety Car" in text


def test_the_report_is_rendered_from_the_file_alone(params: ParameterSet, tmp_path: Path) -> None:
    """Report and parameters cannot drift apart if one is a pure function of the other."""
    path = tmp_path / "params.json"
    params.write(path)
    assert render_report(ParameterSet.read(path)) == render_report(params)


def test_settings_used_here_keep_the_split_disjoint() -> None:
    settings: AppSettings = LINEAR
    assert not set(settings.data.train_seasons) & set(settings.data.holdout_seasons)

"""The hazard the whole project turns on.

If safety cars were predictable there would be nothing for reinforcement learning to earn its
place against, so an error in this rate does not just move a number -- it decides the answer to
the question the repository exists to ask. The tests below construct races with a known number
of neutralisations and check that exactly those are found and counted.
"""

from __future__ import annotations

import polars as pl
import pytest

from apex.calibrate.features import add_features
from apex.calibrate.safety_car import fit_safety_car, neutralisation_laps
from apex.config import AppSettings, CalibrationSettings
from tests.factories import lap, lap_frame

SETTINGS = AppSettings(calibration=CalibrationSettings(min_races_per_circuit=2))
SAFETY_CAR = ("4",)


def race(
    statuses: dict[int, str],
    *,
    season: int = 2023,
    round_number: int = 1,
    circuit: str = "Alpha",
    total_laps: int = 20,
    drivers: tuple[str, ...] = ("VER", "HAM"),
) -> list[dict[str, object]]:
    """A race whose track status is stated lap by lap; unlisted laps run green."""
    return [
        lap(
            season=season,
            round=round_number,
            circuit=circuit,
            driver=driver,
            lap=number,
            track_status=statuses.get(number, "1"),
        )
        for number in range(1, total_laps + 1)
        for driver in drivers
    ]


def featured(rows: list[dict[str, object]]) -> pl.DataFrame:
    return add_features(lap_frame(rows), SETTINGS)


def test_consecutive_neutralised_laps_are_one_deployment_not_several() -> None:
    """Counting laps would inflate the hazard exactly where safety cars last longest."""
    laps = neutralisation_laps(featured(race({5: "4", 6: "4", 7: "4"})), SAFETY_CAR)
    assert int(laps["starts"].sum()) == 1
    assert int(laps["active"].sum()) == 3


def test_separate_incidents_are_separate_deployments() -> None:
    laps = neutralisation_laps(featured(race({5: "4", 6: "4", 12: "4"})), SAFETY_CAR)
    assert int(laps["starts"].sum()) == 2


def test_a_first_lap_safety_car_counts() -> None:
    """A race starts green, so lap one is a lap at risk and an incident on it is a deployment."""
    laps = neutralisation_laps(featured(race({1: "4", 2: "4"})), SAFETY_CAR)
    assert int(laps["starts"].sum()) == 1
    assert bool(laps.filter(laps["lap"] == 1)["starts"].item())


def test_laps_already_under_a_safety_car_are_not_at_risk_of_starting_one() -> None:
    """The three neutralised laps take the lap after them out of the denominator too.

    A deployment beginning on the very lap another ends is therefore counted as one event rather
    than two, which is the same convention the numerator uses. Consistent, and it errs toward the
    lower hazard rather than double-counting a single incident.
    """
    laps = neutralisation_laps(featured(race({5: "4", 6: "4", 7: "4"})), SAFETY_CAR)
    assert int(laps["at_risk"].sum()) == 20 - 3


def test_the_status_is_a_property_of_the_track_not_of_one_car() -> None:
    """One car's timing gap must not hide a neutralisation the rest of the field saw."""
    rows = [lap(driver="VER", lap=n, track_status="4" if n == 5 else "1") for n in range(1, 11)] + [
        lap(driver="HAM", lap=n, track_status="1") for n in range(1, 11)
    ]
    laps = neutralisation_laps(featured(rows), SAFETY_CAR)
    assert int(laps["starts"].sum()) == 1


def test_concatenated_statuses_within_a_lap_are_matched() -> None:
    """FastF1 reports every status seen during a lap, so '671' contains a virtual safety car."""
    laps = neutralisation_laps(featured(race({5: "671"})), ("6", "7"))
    assert int(laps["starts"].sum()) == 1
    assert int(neutralisation_laps(featured(race({5: "671"})), SAFETY_CAR)["starts"].sum()) == 0


def test_full_and_virtual_safety_cars_are_counted_apart() -> None:
    """They are strategically opposite, so one rate covering both would describe no real event."""
    model = fit_safety_car(featured(race({4: "4", 5: "4", 12: "6", 13: "7"})), SETTINGS)
    assert model.safety_car.pooled.n_deployments == 1
    assert model.virtual_safety_car.pooled.n_deployments == 1
    for duration in (model.safety_car.duration_laps, model.virtual_safety_car.duration_laps):
        assert duration is not None
        assert duration.value == pytest.approx(2.0)


def test_the_hazard_matches_the_rate_the_races_actually_show() -> None:
    rows: list[dict[str, object]] = []
    for round_number in range(1, 11):
        statuses = {5: "4", 6: "4"} if round_number <= 5 else {}
        rows.extend(race(statuses, round_number=round_number))
    model = fit_safety_car(featured(rows), SETTINGS)

    hazard = model.safety_car.pooled
    assert hazard.n_deployments == 5
    assert model.safety_car.races_with_any == 5
    assert model.safety_car.share_of_races == 0.5
    # The Jeffreys half-deployment nudges the rate up, and by design it can only nudge: the
    # adjustment is worth less than one extra deployment however small the sample.
    raw = hazard.n_deployments / hazard.laps_at_risk
    assert raw < hazard.per_lap_probability.value < raw + 1.0 / hazard.laps_at_risk


def test_a_circuit_that_never_neutralises_is_not_declared_impossible() -> None:
    """Zero in three races is evidence, not proof; the interval has to stay open."""
    rows: list[dict[str, object]] = []
    for round_number in range(1, 7):
        statuses = {5: "4"} if round_number > 3 else {}
        circuit = "Quiet" if round_number <= 3 else "Busy"
        rows.extend(race(statuses, round_number=round_number, circuit=circuit))

    model = fit_safety_car(featured(rows), SETTINGS)
    quiet = model.safety_car.circuit("Quiet").per_lap_probability
    assert quiet.value > 0.0
    assert quiet.interval is not None
    assert 0.0 < quiet.interval[0] < quiet.interval[1] < 1.0


def test_a_circuit_with_no_signal_of_its_own_reports_the_pooled_rate_exactly() -> None:
    """Two pooled numbers that disagree slightly would be worse than one."""
    rows: list[dict[str, object]] = []
    for round_number in range(1, 5):
        rows.extend(race({5: "4"}, round_number=round_number, circuit="Common"))
    rows.extend(race({}, round_number=9, circuit="Rare"))

    model = fit_safety_car(featured(rows), SETTINGS)
    rare = model.safety_car.circuit("Rare").per_lap_probability
    assert rare.shrinkage == 0.0
    assert rare.value == pytest.approx(model.safety_car.pooled.per_lap_probability.value)


def test_a_race_with_no_neutralisation_at_all_still_reports_a_rate() -> None:
    model = fit_safety_car(featured(race({})), SETTINGS)
    assert model.safety_car.pooled.n_deployments == 0
    assert 0.0 < model.safety_car.pooled.per_lap_probability.value < 0.05
    assert model.safety_car.duration_laps is None


def test_a_kind_with_no_codes_is_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="at least one track status code"):
        neutralisation_laps(featured(race({})), ())

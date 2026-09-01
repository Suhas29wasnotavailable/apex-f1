"""Pit loss, measured on races where the true cost of a stop is known by construction."""

from __future__ import annotations

import polars as pl
import pytest

from apex.calibrate.features import add_features
from apex.calibrate.pit import fit_pit_loss, pit_stops
from apex.config import AppSettings, CalibrationSettings
from tests.factories import lap, lap_frame, stint

GREEN_LAP = 90.0
TRUE_LOSS = 21.0
SETTINGS = AppSettings(calibration=CalibrationSettings(min_races_per_circuit=2))


def race_with_a_stop(
    *,
    season: int = 2023,
    round_number: int = 1,
    circuit: str = "Alpha",
    driver: str = "VER",
    loss: float = TRUE_LOSS,
    stop_lap: int = 20,
    track_status: str = "1",
) -> list[dict[str, object]]:
    """One car, one stop, with the cost split between the in-lap and the out-lap.

    Half the loss on each lap is arbitrary but harmless: the estimator adds the two laps together
    and subtracts the two neighbours, so only the total matters.
    """
    common = {"season": season, "round": round_number, "circuit": circuit, "driver": driver}
    rows = [
        *stint(6, first_lap=stop_lap - 6, stint=1, **common),
        lap(
            lap=stop_lap,
            stint=1,
            is_pit_in=True,
            pit_in_s=1000.0,
            lap_time_s=GREEN_LAP + loss / 2,
            track_status=track_status,
            **common,
        ),
        lap(
            lap=stop_lap + 1,
            stint=2,
            is_pit_out=True,
            pit_out_s=1030.0,
            lap_time_s=GREEN_LAP + loss / 2,
            track_status=track_status,
            **common,
        ),
        *stint(6, first_lap=stop_lap + 2, stint=2, **common),
    ]
    return rows


def featured(rows: list[dict[str, object]]) -> pl.DataFrame:
    return add_features(lap_frame(rows), SETTINGS)


def test_the_cost_of_a_stop_is_measured_against_the_neighbouring_laps() -> None:
    stops = pit_stops(featured(race_with_a_stop()), SETTINGS)
    assert stops.height == 1
    assert stops["loss_s"].item() == pytest.approx(TRUE_LOSS)
    assert stops["in_lap"].item() == 20


def test_a_stop_under_a_safety_car_is_found_but_flagged() -> None:
    """Its cost is real and much lower; using it as a green-flag pit loss would be a bad error."""
    stops = pit_stops(featured(race_with_a_stop(track_status="4")), SETTINGS)
    assert stops["under_safety_car"].to_list() == [True]


def test_safety_car_stops_are_excluded_from_the_fitted_loss() -> None:
    rows = [
        *race_with_a_stop(round_number=1),
        *race_with_a_stop(round_number=2),
        *race_with_a_stop(round_number=3, loss=4.0, track_status="4"),
    ]
    model = fit_pit_loss(featured(rows), SETTINGS)
    assert model.n_stops == 2
    assert model.excluded_under_safety_car == 1
    assert model.pooled.seconds.value == pytest.approx(TRUE_LOSS)


def test_the_true_loss_is_recovered_across_many_stops() -> None:
    rows: list[dict[str, object]] = []
    for round_number in range(1, 7):
        for index, driver in enumerate(("VER", "HAM", "LEC", "NOR")):
            rows.extend(
                race_with_a_stop(
                    round_number=round_number,
                    driver=driver,
                    loss=TRUE_LOSS + 0.4 * (index - 1.5),
                    stop_lap=18 + index,
                )
            )
    model = fit_pit_loss(featured(rows), SETTINGS)
    assert model.pooled.seconds.value == pytest.approx(TRUE_LOSS, abs=0.3)
    interval = model.pooled.seconds.interval
    assert interval is not None
    assert interval[0] <= TRUE_LOSS <= interval[1]


def test_the_median_ignores_a_stop_that_went_wrong() -> None:
    """A stuck wheel nut says nothing about the circuit, and a mean would let it say a lot."""
    rows: list[dict[str, object]] = []
    for round_number in range(1, 5):
        for driver in ("VER", "HAM", "LEC"):
            rows.extend(race_with_a_stop(round_number=round_number, driver=driver))
    rows.extend(race_with_a_stop(round_number=5, driver="STR", loss=90.0))

    model = fit_pit_loss(featured(rows), SETTINGS)
    assert model.pooled.seconds.value == pytest.approx(TRUE_LOSS, abs=0.1)


def test_circuits_that_differ_keep_their_own_pit_loss() -> None:
    """Monza's pit lane is genuinely longer than Montreal's; pooling must not hide that."""
    rows: list[dict[str, object]] = []
    round_number = 0
    for circuit, loss in (("Long", 26.0), ("Short", 19.0)):
        for _ in range(4):
            round_number += 1
            for index, driver in enumerate(("VER", "HAM", "LEC", "NOR", "PIA")):
                rows.extend(
                    race_with_a_stop(
                        round_number=round_number,
                        circuit=circuit,
                        driver=driver,
                        loss=loss + 0.3 * (index - 2),
                        stop_lap=18 + index,
                    )
                )
    model = fit_pit_loss(featured(rows), SETTINGS)
    assert model.circuit("Long").seconds.value > model.circuit("Short").seconds.value + 4.0
    assert model.circuit("Long").seconds.value == pytest.approx(26.0, abs=1.0)


def test_a_circuit_with_one_race_takes_the_pooled_value() -> None:
    rows: list[dict[str, object]] = []
    for round_number in range(1, 4):
        for driver in ("VER", "HAM", "LEC"):
            rows.extend(race_with_a_stop(round_number=round_number, driver=driver))
    rows.extend(race_with_a_stop(round_number=9, circuit="Rare", driver="STR", loss=40.0))

    model = fit_pit_loss(featured(rows), SETTINGS)
    rare = model.circuit("Rare")
    assert rare.seconds.shrinkage == 0.0
    assert rare.seconds.value == pytest.approx(model.pooled.seconds.value)
    assert rare.n_stops == 1


def test_a_stop_without_a_comparable_neighbour_is_not_guessed_at() -> None:
    """No counterfactual, no measurement: an unmeasurable stop is dropped, not imputed."""
    rows = [
        lap(lap=1, stint=1),
        lap(lap=2, stint=1, is_pit_in=True, pit_in_s=100.0),
        lap(lap=3, stint=2, is_pit_out=True, pit_out_s=130.0),
    ]
    assert pit_stops(featured(rows), SETTINGS).height == 0


def test_a_race_with_no_measurable_stop_fails_loudly() -> None:
    with pytest.raises(ValueError, match="no green-flag pit stop"):
        fit_pit_loss(featured(list(stint(10))), SETTINGS)


def test_the_bootstrap_interval_is_reproducible_from_the_seed() -> None:
    """Every interval this project publishes has to come back the same on the next run."""
    rows: list[dict[str, object]] = []
    for round_number in range(1, 5):
        for index, driver in enumerate(("VER", "HAM", "LEC")):
            rows.extend(
                race_with_a_stop(
                    round_number=round_number, driver=driver, loss=TRUE_LOSS + 0.5 * index
                )
            )
    laps = featured(rows)
    first = fit_pit_loss(laps, SETTINGS).pooled.seconds
    second = fit_pit_loss(laps, SETTINGS).pooled.seconds
    assert (first.interval, first.std_error) == (second.interval, second.std_error)

    # A different master seed draws a different resample. The interval endpoints here fall on the
    # same few discrete loss values either way, so the check is on the resampling spread, which is
    # continuous and would be identical if the stream were not actually seeded from the config.
    other_seed = SETTINGS.model_copy(update={"seed": SETTINGS.seed + 1})
    assert fit_pit_loss(laps, other_seed).pooled.seconds.std_error != first.std_error

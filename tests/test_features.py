"""The derived quantities the pace model regresses on.

Each of these is a modelling choice with a defensible alternative, so each is pinned by a test
that states the choice rather than merely exercising the code.
"""

from __future__ import annotations

import polars as pl
import pytest

from apex.calibrate.features import FEATURE_COLUMNS, add_features
from apex.config import AppSettings, CalibrationSettings
from tests.factories import lap, lap_frame, stint


def test_every_declared_feature_is_produced() -> None:
    frame = add_features(lap_frame(stint(5)))
    assert set(FEATURE_COLUMNS) <= set(frame.columns)


def test_identifiers_separate_cars_races_and_stints() -> None:
    frame = add_features(
        lap_frame(
            [
                lap(lap=5, driver="VER", stint=1),
                lap(lap=6, driver="VER", stint=2),
                lap(lap=5, driver="HAM", stint=1),
                lap(lap=5, driver="VER", stint=1, round=2),
            ]
        )
    )
    assert frame["race_id"].n_unique() == 2
    assert frame["car_id"].n_unique() == 3
    assert frame["stint_id"].n_unique() == 4


def test_race_progress_runs_from_the_first_lap_to_the_last_completed_one() -> None:
    """Measured against the distance actually run, so a shortened race is not treated as full."""
    frame = add_features(lap_frame(stint(11, first_lap=1))).sort("lap")
    assert frame["race_progress"].to_list()[0] == 0.0
    assert frame["race_progress"].to_list()[-1] == 1.0


def test_a_one_lap_race_has_no_progress_rather_than_a_division_by_zero() -> None:
    assert add_features(lap_frame([lap(lap=1)]))["race_progress"].item() == 0.0


def test_the_gap_is_to_the_car_that_last_crossed_the_line() -> None:
    """Ordering by crossing time, not by lap number, is what makes this right for lapped cars."""
    frame = add_features(
        lap_frame(
            [
                lap(driver="VER", lap=10, lap_end_s=1000.0),
                lap(driver="HAM", lap=10, lap_end_s=1001.5),
                # A lapped car, on lap 9, crossing between the two leaders.
                lap(driver="STR", lap=9, lap_end_s=1000.4),
            ]
        )
    )
    gaps = dict(frame.select("driver", "gap_ahead_s").rows())
    assert gaps["VER"] is None, "the first car across the line has nobody ahead"
    assert gaps["STR"] == pytest.approx(0.4)
    assert gaps["HAM"] == pytest.approx(1.1), "the car ahead of HAM is the lapped car, not VER"


def test_the_leader_runs_in_clean_air_rather_than_missing_data() -> None:
    frame = add_features(lap_frame([lap(driver="VER", lap=10, lap_end_s=1000.0)]))
    assert frame["gap_ahead_s"].item() is None
    assert frame["dirty_air"].item() == 0.0


def test_dirty_air_decays_with_the_gap() -> None:
    frame = add_features(
        lap_frame(
            [
                lap(driver="A", lap=10, lap_end_s=1000.0),
                lap(driver="B", lap=10, lap_end_s=1000.0 + 2.0),
                lap(driver="C", lap=10, lap_end_s=1000.0 + 2.0 + 20.0),
            ]
        )
    )
    values = dict(frame.select("driver", "dirty_air").rows())
    assert values["A"] == 0.0
    assert values["B"] == pytest.approx(0.3679, abs=1e-4), "one time constant behind"
    assert values["C"] < 0.001


def test_the_dirty_air_time_constant_comes_from_configuration() -> None:
    laps = lap_frame(
        [lap(driver="A", lap=10, lap_end_s=1000.0), lap(driver="B", lap=10, lap_end_s=1004.0)]
    )
    wide = AppSettings(calibration=CalibrationSettings(dirty_air_saturation_s=4.0))
    narrow = AppSettings(calibration=CalibrationSettings(dirty_air_saturation_s=1.0))
    behind = pl.col("driver") == "B"
    assert (
        add_features(laps, wide).filter(behind)["dirty_air"].item()
        > add_features(laps, narrow).filter(behind)["dirty_air"].item()
    )


def test_stint_laps_are_numbered_from_one_within_each_stint() -> None:
    frame = add_features(
        lap_frame([*stint(3, first_lap=2, stint=1), *stint(3, first_lap=20, stint=2)])
    ).sort("lap")
    assert frame["stint_lap"].to_list() == [1, 2, 3, 1, 2, 3]


def test_tyre_age_is_the_life_of_the_tyre_not_the_length_of_the_stint() -> None:
    """A stint started on used tyres begins part-way up the degradation curve, not at zero."""
    frame = add_features(lap_frame([lap(lap=5, tyre_life=12, fresh_tyre=False)]))
    assert frame["tyre_age"].item() == 12.0
    assert frame["stint_lap"].item() == 1

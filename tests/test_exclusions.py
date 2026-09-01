"""What the exclusion rules remove, and what they promise about the count.

These rules decide which laps are allowed to move a fitted parameter, so a silent change in
their behaviour is a silent change in every number this project publishes. The properties tested
here are the ones the calibration diagnostics claim: rules are attributed in order, the counts
add up, and each rule removes exactly the laps it says it does.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from apex.calibrate.exclusions import clean_laps
from apex.config import AppSettings, CalibrationSettings, ExclusionSettings

from tests.factories import lap, lap_frame, settings_in, stint


def removed(report: pl.DataFrame, rule: str) -> int:
    return int(report.filter(pl.col("rule") == rule)["removed"].item())


def test_clean_laps_survive_every_rule() -> None:
    report = clean_laps(lap_frame(stint(10)))
    assert report.laps.height == 10
    assert report.diagnostics["removed"].sum() == 0
    assert report.retained_fraction == 1.0


def test_diagnostics_account_for_every_input_lap() -> None:
    """The report is only honest if nothing vanishes between rules."""
    laps = lap_frame(
        [
            *stint(10),
            *stint(6, first_lap=2, driver="HAM", track_status="4"),
            *stint(3, first_lap=20, driver="LEC"),
        ]
    )
    report = clean_laps(laps)
    assert int(report.diagnostics["remaining"][0]) == laps.height
    assert report.diagnostics["removed"].sum() + report.laps.height == laps.height


def test_track_status_matches_codes_anywhere_in_the_string() -> None:
    """FastF1 concatenates every status seen during a lap: '671' contains a VSC period."""
    laps = lap_frame([*stint(9), lap(lap=11, track_status="671")])
    report = clean_laps(laps)
    assert removed(report.diagnostics, "track_status") == 1
    assert 11 not in report.laps["lap"].to_list()


def test_all_clear_laps_are_kept() -> None:
    assert clean_laps(lap_frame(stint(6, track_status="1"))).laps.height == 6


def test_lap_after_a_safety_car_is_dropped() -> None:
    laps = lap_frame([*stint(10), lap(lap=12, track_status="4"), lap(lap=13)])
    report = clean_laps(laps)
    assert removed(report.diagnostics, "lap_after_safety_car") == 1
    assert 13 not in report.laps["lap"].to_list()


def test_the_lap_after_a_safety_car_is_found_before_other_rules_remove_it() -> None:
    """The neighbour lookup runs on the unfiltered frame.

    Lap 12 here is both a safety-car lap and an in-lap, so an earlier rule removes it. Lap 13 is
    still the lap after a safety car and must still go; computing the flag on the surviving laps
    would have made lap 13's predecessor lap 11, and let a distorted lap into the fit.
    """
    laps = lap_frame(
        [*stint(10), lap(lap=12, track_status="4", is_pit_in=True, pit_in_s=500.0), lap(lap=13)]
    )
    report = clean_laps(laps)
    assert removed(report.diagnostics, "lap_after_safety_car") == 1
    assert 13 not in report.laps["lap"].to_list()


def test_the_safety_car_neighbour_is_per_car() -> None:
    """One driver's safety-car lap must not invalidate another driver's next lap."""
    laps = lap_frame(
        [
            *stint(10, driver="VER"),
            *stint(10, driver="HAM"),
            lap(lap=12, driver="VER", track_status="4"),
            lap(lap=13, driver="VER"),
            lap(lap=13, driver="HAM"),
        ]
    )
    report = clean_laps(laps)
    survivors = report.laps.filter(pl.col("lap") == 13)["driver"].to_list()
    assert survivors == ["HAM"]


def test_opening_laps_are_dropped() -> None:
    laps = lap_frame([lap(lap=1), *stint(9, first_lap=2)])
    report = clean_laps(laps)
    assert removed(report.diagnostics, "opening_lap") == 1


def test_pit_laps_are_dropped() -> None:
    laps = lap_frame(
        [
            *stint(9),
            lap(lap=12, is_pit_in=True, pit_in_s=1000.0),
            lap(lap=13, is_pit_out=True, pit_out_s=1030.0),
        ]
    )
    report = clean_laps(laps)
    assert removed(report.diagnostics, "pit_in_or_out_lap") == 2


def test_wet_and_unknown_compounds_are_dropped() -> None:
    laps = lap_frame([*stint(8), *stint(4, first_lap=20, compound="INTERMEDIATE", stint=2)])
    report = clean_laps(laps)
    assert removed(report.diagnostics, "wet_or_unknown_compound") == 4
    assert report.laps["compound"].unique().to_list() == ["MEDIUM"]


def test_laps_without_a_time_are_dropped() -> None:
    laps = lap_frame([*stint(9), lap(lap=12, lap_time_s=None)])
    assert removed(clean_laps(laps).diagnostics, "missing_lap_time") == 1


def test_inaccurate_deleted_and_generated_laps_are_dropped() -> None:
    laps = lap_frame(
        [
            *stint(9),
            lap(lap=12, is_accurate=False),
            lap(lap=13, deleted=True),
            lap(lap=14, fastf1_generated=True),
        ]
    )
    report = clean_laps(laps)
    assert removed(report.diagnostics, "not_accurate") == 1
    assert removed(report.diagnostics, "deleted_by_stewards") == 1
    assert removed(report.diagnostics, "fastf1_generated") == 1


def test_slow_outliers_are_dropped_relative_to_the_cars_own_median() -> None:
    """A spin costs a driver seconds; the threshold is a ratio, so it travels between circuits."""
    laps = lap_frame([*stint(9, lap_time_s=90.0), lap(lap=12, lap_time_s=120.0)])
    report = clean_laps(laps)
    assert removed(report.diagnostics, "slow_outlier") == 1
    assert report.laps["lap_time_s"].max() == 90.0


def test_a_lap_just_inside_the_ratio_is_kept() -> None:
    laps = lap_frame([*stint(9, lap_time_s=90.0), lap(lap=12, lap_time_s=98.9)])
    report = clean_laps(laps)
    assert removed(report.diagnostics, "slow_outlier") == 0


def test_short_stints_cannot_identify_a_slope() -> None:
    laps = lap_frame([*stint(8, stint=1), *stint(3, first_lap=30, stint=2)])
    report = clean_laps(laps)
    assert removed(report.diagnostics, "short_stint") == 3
    assert report.laps["stint"].unique().to_list() == [1]


def test_the_stint_rule_counts_only_surviving_laps() -> None:
    """A six-lap stint with three laps behind a safety car is not a usable four-lap stint."""
    laps = lap_frame(
        [
            *stint(8, stint=1),
            *stint(3, first_lap=30, stint=2, track_status="4"),
            *stint(3, first_lap=33, stint=2),
        ]
    )
    report = clean_laps(laps)
    assert report.laps["stint"].unique().to_list() == [1]


def test_rules_can_be_disabled_from_configuration() -> None:
    settings = AppSettings(
        calibration=CalibrationSettings(
            exclusions=ExclusionSettings(drop_opening_laps=0, drop_lap_after_safety_car=False)
        )
    )
    laps = lap_frame(
        [lap(lap=1), *stint(9, first_lap=2), lap(lap=12, track_status="4"), lap(lap=13)]
    )
    report = clean_laps(laps, settings)
    rules = report.diagnostics["rule"].to_list()
    assert "opening_lap" not in rules
    assert "lap_after_safety_car" not in rules
    assert 1 in report.laps["lap"].to_list()


def test_an_empty_race_produces_an_empty_report() -> None:
    report = clean_laps(lap_frame([]))
    assert report.laps.height == 0
    assert report.retained_fraction == 0.0


@pytest.mark.parametrize("count", [4, 5, 12])
def test_a_stint_at_the_minimum_length_is_kept(count: int) -> None:
    assert clean_laps(lap_frame(stint(count))).laps.height == count


def test_settings_factory_does_not_touch_the_repository(tmp_path: Path) -> None:
    """The factory the ingestion tests rely on must never write into the working tree."""
    assert str(settings_in(tmp_path).paths.processed).startswith(str(tmp_path))

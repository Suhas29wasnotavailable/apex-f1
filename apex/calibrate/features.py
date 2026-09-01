"""Per-lap quantities the pace model needs but the timing feed does not report.

Two of these deserve their reasoning written down.

**Traffic.** The gap to the car ahead is measured at the start/finish line: within a race, laps
are ordered by the time they ended, and each lap's gap is the interval since the previous car
crossed. Ordering by crossing time rather than by lap number is what makes the measure correct
for lapped traffic -- the car directly ahead on track is the one that most recently crossed the
line, whatever lap it is on. It is imperfect for a car that has just entered the pit lane or
retired, which stays in the sequence for one crossing after it stops being on track.

**Race progress.** Fuel load falls monotonically through a race, and its effect on lap time is
not separable from tyre age within a single stint -- both grow lap by lap. It is separable across
stints, because tyre age resets at a pit stop and fuel does not. Progress is therefore measured
against the race distance actually completed rather than the scheduled distance, so a race
stopped early is not treated as though its final laps were run on a full tank.

Features are derived before the exclusion rules run, for the same reason the safety-car
neighbour is: the car ahead may itself be on a lap the rules will remove, and its gap is still
the gap that was actually driven.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from apex.config import AppSettings, get_settings

__all__ = ["FEATURE_COLUMNS", "add_features"]

FEATURE_COLUMNS: Final = (
    "race_id",
    "car_id",
    "stint_id",
    "tyre_age",
    "stint_lap",
    "race_progress",
    "gap_ahead_s",
    "dirty_air",
)
"""Columns :func:`add_features` appends to a canonical lap table."""


def add_features(laps: pl.DataFrame, settings: AppSettings | None = None) -> pl.DataFrame:
    """Append the modelling features to a canonical lap table.

    Args:
        laps: Canonical laps, unfiltered. Passing already-cleaned laps would silently corrupt the
            traffic measure, since the car ahead would be missing from the crossing order.
        settings: Configuration; the process-wide settings by default.
    """
    settings = settings or get_settings()
    saturation = settings.calibration.dirty_air_saturation_s

    identified = laps.with_columns(
        pl.format("{}-{}", "season", "round").alias("race_id"),
        pl.format("{}-{}-{}", "season", "round", "driver").alias("car_id"),
        pl.format("{}-{}-{}-{}", "season", "round", "driver", "stint").alias("stint_id"),
        pl.col("tyre_life").cast(pl.Float64).alias("tyre_age"),
    )

    with_traffic = identified.sort("race_id", "lap_end_s", nulls_last=True).with_columns(
        (pl.col("lap_end_s") - pl.col("lap_end_s").shift(1))
        .over("race_id")
        .clip(lower_bound=0.0)
        .alias("gap_ahead_s")
    )

    laps_completed = pl.col("lap").max().over("race_id")
    return with_traffic.with_columns(
        pl.when(laps_completed > 1)
        .then((pl.col("lap") - 1) / (laps_completed - 1))
        .otherwise(0.0)
        .cast(pl.Float64)
        .alias("race_progress"),
        pl.col("lap").rank("ordinal").over("stint_id").cast(pl.Int16).alias("stint_lap"),
        # No car ahead means clean air, which is the absence of the penalty rather than a
        # missing measurement, so the leader's lap is kept with a zero rather than dropped.
        (-pl.col("gap_ahead_s").fill_null(float("inf")) / saturation)
        .exp()
        .fill_nan(0.0)
        .cast(pl.Float64)
        .alias("dirty_air"),
    ).sort("season", "round", "driver", "lap")

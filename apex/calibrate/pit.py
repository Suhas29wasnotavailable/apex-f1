"""Pit loss: what a stop actually costs, measured from the laps around it.

A stop costs the difference between what the in- and out-laps took and what those two laps would
have taken at racing speed. The counterfactual comes from the driver's own neighbouring clean
laps: the last one before the stop and the first one after it.

Using the two neighbours rather than a race median is deliberate, and it is what makes the
estimate close to unbiased. The laps either side of a stop are one or two laps apart, so fuel
load and track state are effectively identical. Tyre age is not -- the lap before a stop is the
oldest tyre of the stint and the lap after the out-lap is nearly the freshest -- but those two
errors have opposite signs and largely cancel when both neighbours are used together.

Stops made under a safety car are excluded. The field is slow, so the stop costs far less, and a
simulator that used a green-flag pit loss for a safety-car stop would get the single most
important strategic decision in a race backwards. That parameter is worth fitting separately
once there are enough neutralised stops to fit it from.

The circuit estimate is the median across stops, not the mean: a stop can go wrong for reasons
that have nothing to do with the circuit -- a stuck wheel nut, a penalty served, a driver waved
out into traffic -- and a mean lets one such stop move a circuit's number by a second. The
uncertainty comes from a seeded bootstrap, so the interval is reproducible from the config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from apex.calibrate.estimates import Estimate
from apex.calibrate.exclusions import CAR_KEYS, safety_car_laps
from apex.calibrate.pooling import partial_pool
from apex.config import AppSettings, get_settings

__all__ = ["PitLoss", "PitLossModel", "fit_pit_loss", "pit_stops"]

BOOTSTRAP_RESAMPLES = 2000
"""Enough that the interval is stable to about a hundredth of a second, and cheap at this size."""

_MIN_STOPS_FOR_BOOTSTRAP = 2


class PitLoss(BaseModel):
    """Time lost to one green-flag pit stop at one circuit, or pooled across all of them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    circuit: str | None = None
    seconds: Estimate
    n_stops: int = Field(ge=0)


class PitLossModel(BaseModel):
    """Pit loss everywhere it could be measured."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pooled: PitLoss
    circuits: tuple[PitLoss, ...]
    n_stops: int
    n_races: int
    excluded_under_safety_car: int
    """Stops dropped because they were made during a neutralisation, reported rather than
    silently discarded: it is a large share of all stops at some circuits."""

    def circuit(self, name: str) -> PitLoss:
        for entry in self.circuits:
            if entry.circuit == name:
                return entry
        raise KeyError(f"no pit loss fitted for circuit {name!r}")


def pit_stops(all_laps: pl.DataFrame, settings: AppSettings | None = None) -> pl.DataFrame:
    """Find every pit stop and what it cost.

    Args:
        all_laps: Featured but *unfiltered* laps -- ``CalibrationData.all_laps``. The pace filter
            removes exactly the laps this measurement is made from.
        settings: Configuration; the process-wide settings by default.

    Returns:
        One row per stop, with the circuit, the in- and out-lap times, the neighbouring clean lap
        times used as the counterfactual, the loss in seconds, and whether it happened under a
        safety car. Stops whose neighbours are missing -- a stop on the last lap, or one either
        side of another excluded lap -- are absent, because there is nothing to compare them to.
    """
    settings = settings or get_settings()
    ordered = all_laps.sort(*CAR_KEYS, "lap")

    # A clean lap here is only about being comparable: green flag, timed, not itself a pit lap.
    # It is a looser test than the pace filter's, because the neighbours are being used as a
    # local reference rather than as evidence about degradation.
    comparable = (
        pl.col("lap_time_s").is_not_null()
        & ~pl.col("is_pit_in")
        & ~pl.col("is_pit_out")
        & (pl.col("track_status") == "1")
    )
    reference = pl.when(comparable).then(pl.col("lap_time_s")).otherwise(None)

    with_context = ordered.with_columns(
        reference.forward_fill().shift(1).over(list(CAR_KEYS)).alias("_before"),
        reference.backward_fill().shift(-1).over(list(CAR_KEYS)).alias("_after"),
        pl.col("lap_time_s").shift(-1).over(list(CAR_KEYS)).alias("_out_lap_s"),
        pl.col("is_pit_out").shift(-1).over(list(CAR_KEYS)).alias("_next_is_out"),
        safety_car_laps(ordered, settings).alias("_neutralised"),
        safety_car_laps(ordered, settings)
        .shift(-1)
        .over(list(CAR_KEYS))
        .alias("_next_neutralised"),
    )

    stops = with_context.filter(
        pl.col("is_pit_in")
        & pl.col("_next_is_out").fill_null(value=False)
        & pl.col("lap_time_s").is_not_null()
        & pl.col("_out_lap_s").is_not_null()
        & pl.col("_before").is_not_null()
        & pl.col("_after").is_not_null()
    )

    return stops.select(
        "season",
        "round",
        "race_id",
        "circuit",
        "driver",
        pl.col("lap").alias("in_lap"),
        pl.col("lap_time_s").alias("in_lap_s"),
        pl.col("_out_lap_s").alias("out_lap_s"),
        pl.col("_before").alias("before_s"),
        pl.col("_after").alias("after_s"),
        (pl.col("lap_time_s") + pl.col("_out_lap_s") - pl.col("_before") - pl.col("_after")).alias(
            "loss_s"
        ),
        (pl.col("_neutralised") | pl.col("_next_neutralised").fill_null(value=False)).alias(
            "under_safety_car"
        ),
    ).sort("season", "round", "driver", "in_lap")


def _bootstrap_median(values: np.ndarray, rng: np.random.Generator, level: float) -> Estimate:
    """Median with a percentile interval from a seeded resample of the stops."""
    if values.size < _MIN_STOPS_FOR_BOOTSTRAP:
        # A single stop is a measurement, not a distribution: report it with no claim of
        # precision rather than a bootstrap interval of zero width.
        return Estimate(value=float(values[0]), level=level, n_obs=int(values.size))
    draws = rng.integers(0, values.size, size=(BOOTSTRAP_RESAMPLES, values.size))
    medians = np.median(values[draws], axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(medians, [tail, 1.0 - tail])
    return Estimate(
        value=float(np.median(values)),
        std_error=float(np.std(medians, ddof=1)),
        interval=(float(low), float(high)),
        level=level,
        n_obs=int(values.size),
    )


@dataclass(frozen=True, slots=True)
class _Stops:
    green: pl.DataFrame
    neutralised: int


def _green_flag_stops(all_laps: pl.DataFrame, settings: AppSettings) -> _Stops:
    stops = pit_stops(all_laps, settings)
    green = stops.filter(~pl.col("under_safety_car"))
    return _Stops(green=green, neutralised=stops.height - green.height)


def fit_pit_loss(all_laps: pl.DataFrame, settings: AppSettings | None = None) -> PitLossModel:
    """Estimate pit loss overall and per circuit, pooling the sparse circuits.

    Args:
        all_laps: Featured, unfiltered laps -- ``CalibrationData.all_laps``.
        settings: Configuration; the process-wide settings by default.

    Raises:
        ValueError: if no measurable green-flag stop is present.
    """
    settings = settings or get_settings()
    level = settings.calibration.confidence_level
    stops = _green_flag_stops(all_laps, settings)
    if stops.green.height == 0:
        raise ValueError("no green-flag pit stop has the neighbouring laps needed to measure it")

    rng = settings.seeds().derive("calibration").stream("pit_loss")
    pooled_estimate = _bootstrap_median(stops.green["loss_s"].to_numpy(), rng, level)

    minimum = settings.calibration.min_races_per_circuit
    per_circuit: dict[str, Estimate] = {}
    counts: dict[str, int] = {}
    for (circuit,), group in stops.green.group_by(["circuit"], maintain_order=True):
        name = str(circuit)
        counts[name] = group.height
        if group.select("race_id").n_unique() >= minimum:
            per_circuit[name] = _bootstrap_median(group["loss_s"].to_numpy(), rng, level)

    shrunk = (
        partial_pool(per_circuit, level=level, toward=pooled_estimate).estimates
        if per_circuit
        else {}
    )
    circuits = tuple(
        PitLoss(
            circuit=name,
            seconds=shrunk.get(
                name,
                pooled_estimate.model_copy(update={"shrinkage": 0.0, "n_obs": counts[name]}),
            ),
            n_stops=counts[name],
        )
        for name in sorted(counts)
    )

    return PitLossModel(
        pooled=PitLoss(circuit=None, seconds=pooled_estimate, n_stops=stops.green.height),
        circuits=circuits,
        n_stops=stops.green.height,
        n_races=stops.green.select("race_id").n_unique(),
        excluded_under_safety_car=stops.neutralised,
    )

"""Pit loss: what a stop actually costs, measured from the laps around it.

A stop costs the difference between what the in- and out-laps took and what those two laps would
have taken at racing speed. The counterfactual comes from the driver's own neighbouring clean
laps: the last one before the stop and the first one after it.

Using the two neighbours rather than a race median is deliberate, and it is what makes the
estimate close to unbiased. The laps either side of a stop are one or two laps apart, so fuel
load and track state are effectively identical. Tyre age is not -- the lap before a stop is the
oldest tyre of the stint and the lap after the out-lap is nearly the freshest -- but those two
errors have opposite signs and largely cancel when both neighbours are used together.

Stops made under a safety car are excluded from the green-flag estimate and measured separately,
**by a different method**, because the method above breaks down under a neutralisation and does so
in a direction that inverts the answer.

The green-flag estimate uses the driver's own neighbouring clean laps as the counterfactual. Under
a safety car the in- and out-laps are run at neutralised speed while those neighbours are
green-flag laps, so the difference charges the entire safety-car slowdown to the pit stop. Measured
that way the median neutralised stop appears to cost 65 seconds against 23 under green -- which
would tell a simulator that stopping under a safety car is catastrophic, when it is very nearly
free and is the reason safety cars decide races.

The quantity that actually matters strategically is not what the clock says the stop took; it is
how much time the stop lost *against the cars that stayed out*. So a neutralised stop is measured
against the field: the pitting car's in- plus out-lap, minus the median of the same two laps taken
over every car that did not pit. Both sides are then under the same safety car at the same moment,
the slowdown cancels, and what remains is the pit lane. This closes the gap P1 left open.

The circuit estimate is the median across stops, not the mean: a stop can go wrong for reasons
that have nothing to do with the circuit -- a stuck wheel nut, a penalty served, a driver waved
out into traffic -- and a mean lets one such stop move a circuit's number by a second. The
uncertainty comes from a seeded bootstrap, so the interval is reproducible from the config.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from apex.calibrate.estimates import Estimate, bootstrap_median
from apex.calibrate.exclusions import CAR_KEYS, is_safety_car_lap
from apex.calibrate.pooling import partial_pool
from apex.config import AppSettings, get_settings

__all__ = ["PitLoss", "PitLossModel", "fit_pit_loss", "neutralised_stop_loss", "pit_stops"]


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
    """Stops dropped from the green-flag estimate because they were made during a neutralisation,
    reported rather than silently discarded: it is a large share of all stops at some circuits.
    They are measured separately in :attr:`under_safety_car`."""

    under_safety_car: PitLoss | None = None
    """What a stop costs when the field is neutralised, measured against the cars that stayed out.

    This is the discount that makes a safety car the most strategically decisive event in a race,
    and it is measured on a different footing from the green-flag number for a reason given at
    length in the module docstring. ``None`` when no neutralised stop had enough cars staying out
    to compare it against."""

    n_stops_under_safety_car: int = Field(default=0, ge=0)

    @property
    def safety_car_discount_s(self) -> float | None:
        """Seconds a stop saves by being made under a neutralisation rather than under green."""
        if self.under_safety_car is None:
            return None
        return self.pooled.seconds.value - self.under_safety_car.seconds.value

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
        is_safety_car_lap(settings).alias("_neutralised"),
        is_safety_car_lap(settings).shift(-1).over(list(CAR_KEYS)).alias("_next_neutralised"),
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


def _require(estimate: Estimate | None) -> Estimate:
    """Assert an estimate exists. The caller has already checked the sample is non-empty."""
    if estimate is None:  # pragma: no cover - guarded by an emptiness check at every call site
        raise ValueError("no observations to estimate from")
    return estimate


@dataclass(frozen=True, slots=True)
class _Stops:
    green: pl.DataFrame
    neutralised: int


def _green_flag_stops(all_laps: pl.DataFrame, settings: AppSettings) -> _Stops:
    stops = pit_stops(all_laps, settings)
    green = stops.filter(~pl.col("under_safety_car"))
    return _Stops(green=green, neutralised=stops.height - green.height)


def neutralised_stop_loss(
    all_laps: pl.DataFrame, settings: AppSettings | None = None
) -> pl.DataFrame:
    """Measure stops made under a neutralisation against the cars that stayed out.

    Args:
        all_laps: Featured, unfiltered laps -- ``CalibrationData.all_laps``.
        settings: Configuration; the process-wide settings by default.

    Returns:
        One row per neutralised stop, with the two-lap time the pitting car took, the median the
        non-pitting cars took over the same two laps, and the difference. Stops where no car stayed
        out are absent: there is no reference to measure them against.
    """
    settings = settings or get_settings()
    keys = list(CAR_KEYS)
    ordered = all_laps.sort(*keys, "lap")

    # A car's two-lap block starting at this lap, and whether it pitted anywhere inside it. Both
    # laps must be present and consecutive, or the block spans a retirement rather than a stop.
    pitting = pl.col("is_pit_in") | pl.col("is_pit_out")
    blocks = ordered.with_columns(
        pl.col("lap_time_s").shift(-1).over(keys).alias("_next_s"),
        pl.col("lap").shift(-1).over(keys).alias("_next_lap"),
        pitting.alias("_pit"),
        pitting.shift(-1).over(keys).alias("_next_pit"),
        is_safety_car_lap(settings).alias("_neutralised"),
        is_safety_car_lap(settings).shift(-1).over(keys).alias("_next_neutralised"),
    ).filter(
        pl.col("lap_time_s").is_not_null()
        & pl.col("_next_s").is_not_null()
        & (pl.col("_next_lap") == pl.col("lap") + 1)
    )

    blocks = blocks.with_columns(
        (pl.col("lap_time_s") + pl.col("_next_s")).alias("block_s"),
        (pl.col("_pit") | pl.col("_next_pit").fill_null(value=False)).alias("block_has_stop"),
        (pl.col("_neutralised") | pl.col("_next_neutralised").fill_null(value=False)).alias(
            "block_neutralised"
        ),
    )

    # The reference is the median block among cars that stayed out, on exactly the same two laps of
    # the same race, so the neutralisation itself is common to both sides and cancels.
    stayed_out = (
        blocks.filter(~pl.col("block_has_stop"))
        .group_by("race_id", "lap")
        .agg(
            pl.col("block_s").median().alias("reference_s"),
            pl.len().alias("n_stayed_out"),
        )
    )

    return (
        blocks.filter(pl.col("block_has_stop") & pl.col("_pit") & pl.col("block_neutralised"))
        .join(stayed_out, on=["race_id", "lap"], how="inner")
        .select(
            "season",
            "round",
            "race_id",
            "circuit",
            "driver",
            pl.col("lap").alias("in_lap"),
            pl.col("block_s").alias("stop_block_s"),
            "reference_s",
            "n_stayed_out",
            (pl.col("block_s") - pl.col("reference_s")).alias("loss_s"),
        )
        .sort("season", "round", "driver", "in_lap")
    )


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
    pooled_estimate = _require(bootstrap_median(stops.green["loss_s"].to_numpy(), rng, level))

    minimum = settings.calibration.min_races_per_circuit
    per_circuit: dict[str, Estimate] = {}
    counts: dict[str, int] = {}
    for (circuit,), group in stops.green.group_by(["circuit"], maintain_order=True):
        name = str(circuit)
        counts[name] = group.height
        if group.select("race_id").n_unique() >= minimum:
            estimate = bootstrap_median(group["loss_s"].to_numpy(), rng, level)
            if estimate is not None:
                per_circuit[name] = estimate

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

    neutralised = neutralised_stop_loss(all_laps, settings)
    under_safety_car = (
        PitLoss(
            circuit=None,
            seconds=_require(bootstrap_median(neutralised["loss_s"].to_numpy(), rng, level)),
            n_stops=neutralised.height,
        )
        if neutralised.height > 0
        else None
    )

    return PitLossModel(
        pooled=PitLoss(circuit=None, seconds=pooled_estimate, n_stops=stops.green.height),
        circuits=circuits,
        n_stops=stops.green.height,
        n_races=stops.green.select("race_id").n_unique(),
        excluded_under_safety_car=stops.neutralised,
        under_safety_car=under_safety_car,
        n_stops_under_safety_car=neutralised.height,
    )

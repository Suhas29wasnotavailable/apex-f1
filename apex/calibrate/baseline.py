"""The pace level the fixed effects threw away, recovered so a race can be simulated.

ADR-0005 fits pace with one dummy per car-race, which is what makes the tyre and fuel
coefficients trustworthy: everything constant about a car in a race -- its engine, its driver, its
downforce level, the circuit it is at -- is absorbed rather than left to contaminate the slopes.
That is the right choice for estimating a slope and a disaster for running a simulation, because
the absorbed quantity *is* the lap time. A simulator handed only the fitted deltas knows how much
slower a twenty-lap-old hard tyre is than a fresh soft, and has no idea whether a lap at this
circuit takes 70 seconds or 105.

This module recovers what was swept away, by subtracting the fitted model from each real lap:

    corrected = lap_time_s - (compound offset + tyre degradation + progress + dirty air)

What remains is that lap stripped of every effect the pace model explains -- the time the car
would have set on a fresh reference-compound tyre, in clean air, on the first lap of the race. Its
level is the circuit. Its spread within a race is the competitive order. Both are published here.

Two-way decomposition, because both halves move. A circuit-and-season median mixes a property of
the place with a property of the era: 2024 cars are faster everywhere than 2022 cars, so a circuit
raced only in 2022 would otherwise look permanently slow. Circuit and season effects are therefore
fitted jointly by least squares over the circuit-season medians, which is identified because most
circuits appear in several seasons and every season spans many circuits. The season trend that
falls out is a real quantity a reader can check against the sport, and it is what an unanchored
forecast of a future season has to extrapolate.

The corrected lap time is deliberately a *median* over each cell rather than a mean. It is
computed on laps that survived the pace filter, but that filter keeps a lap at 1.10x the driver's
median, which still admits a lift for a yellow flag or a lock-up; a median at this level of
aggregation is unmoved by a handful of them and a mean is not.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from apex.calibrate.estimates import Estimate, bootstrap_median
from apex.calibrate.pace import PaceModel
from apex.config import AppSettings, get_settings

__all__ = [
    "BaselineModel",
    "CircuitBaseline",
    "SeasonPace",
    "car_pace_frame",
    "corrected_lap_time",
    "fit_baseline",
]

_MIN_LAPS_PER_CAR_RACE = 5
"""Below this a car-race median is an accident of which laps survived rather than its pace."""

_MIN_CELLS_FOR_TREND = 2
"""Seasons needed before a season effect is identified at all."""

_FLYING_REFERENCE_LAPS = (3, 6)
"""Laps the standing start is measured against: past the opening scramble, still on fresh tyres and
a full tank, so the difference is the start rather than the stint."""


def tyre_delta(pace: PaceModel) -> pl.Expr:
    """Lap time a car's current tyre costs it, against a fresh reference-compound tyre.

    The compound's own pace offset plus its degradation at the current age. This is the one
    quantity that must be computed identically during calibration and during simulation -- the
    overtaking model is fitted on the pace difference it produces and the simulator evaluates that
    same difference every lap -- so it lives in one place and both read it.

    Null for a compound the model never fitted, which cannot be corrected with another compound's
    curve and is dropped instead.
    """
    age = pl.col("tyre_age")
    result = pl.lit(None, dtype=pl.Float64)
    for entry in pace.degradation:
        effect = pl.lit(0.0 if entry.offset_s is None else entry.offset_s.value)
        for degree, coefficient in enumerate(entry.coefficients):
            effect = effect + coefficient.value * age.pow(degree + 1)
        result = pl.when(pl.col("compound") == entry.compound).then(effect).otherwise(result)
    return result


def corrected_lap_time(pace: PaceModel) -> pl.Expr:
    """Lap time with every effect the pace model explains removed.

    Args:
        pace: The fitted model whose effects are being subtracted.

    Returns:
        An expression for the corrected lap time in seconds -- the lap the car would have set on a
        fresh tyre of the reference compound, in clean air, at the start of the race. It reads
        ``lap_time_s``, ``compound``, ``tyre_age``, ``race_progress`` and ``dirty_air``, so it must
        be evaluated against a featured frame; normally the cleaned one, since the correction is
        only as good as the model and the model was fitted on green-flag laps.
    """
    correction = (
        pl.col("race_progress") * pace.fuel_effect_s.value
        + pl.col("dirty_air") * pace.dirty_air_s.value
    )
    return (pl.col("lap_time_s") - correction - tyre_delta(pace)).alias("corrected_lap_s")


class CircuitBaseline(BaseModel):
    """The pace level of one circuit, in the reference season."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    circuit: str
    reference_lap_s: Estimate
    """Corrected lap time of a field-median car, adjusted to the reference season."""

    n_laps: int = Field(ge=0)
    n_races: int = Field(ge=0)
    seasons: tuple[int, ...]


class SeasonPace(BaseModel):
    """How much faster or slower a season was than the reference, averaged over circuits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    season: int
    offset_s: Estimate
    n_races: int = Field(ge=0)


class BaselineModel(BaseModel):
    """Absolute pace: where each circuit sits, how the seasons moved, how far a field spreads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    circuits: tuple[CircuitBaseline, ...]
    seasons: tuple[SeasonPace, ...]
    reference_season: int
    """The season every circuit baseline is quoted in, and the zero of the season offsets."""

    season_trend_s: Estimate | None
    """Seconds per lap gained per season, fitted across the season offsets. This is the number an
    unanchored simulation of a future season extrapolates, and it is published separately from the
    offsets so that extrapolating it is a visible decision rather than a hidden one. ``None`` when
    fewer than two seasons were fitted, which cannot identify a trend."""

    field_spread_s: Estimate
    """Standard deviation of car pace within a race. The competitive spread of the era, and the
    scale any constructed field has to reproduce."""

    standing_start_s: Estimate | None
    """What the first lap of a race costs over a flying lap.

    The exclusion rules drop opening laps, correctly: a standing start is not representative
    green-flag pace, and leaving it in would bias every degradation slope. But a simulator that
    then runs lap one as though it were lap forty finishes every race several seconds early, and
    that error is systematic rather than noise -- it showed up as a uniformly optimistic race time
    across every held-out race. So the penalty the rules exclude is measured here rather than
    ignored: the median lap-one time against the median of laps three to six under green, taken per
    race and pooled. ``None`` when no race had both.

    It covers the whole of a standing start -- the launch, the run to the first corner, and the
    first-lap scramble -- because those are not separable in this data and a simulator needs their
    sum."""

    n_car_races: int = Field(ge=0)

    def circuit(self, name: str) -> CircuitBaseline:
        for entry in self.circuits:
            if entry.circuit == name:
                return entry
        raise KeyError(f"no baseline fitted for circuit {name!r}")

    def season_offset(self, season: int) -> float:
        """Pace offset for a season: fitted where it was observed, extrapolated where it was not.

        A season outside the training window is reached along :attr:`season_trend_s` from the
        nearest fitted season. That is an extrapolation and the caller is entitled to know, so the
        trend is published; there is no way to run an unanchored simulation of an unseen season
        without one.
        """
        for entry in self.seasons:
            if entry.season == season:
                return entry.offset_s.value
        if self.season_trend_s is None:
            return 0.0
        fitted = [entry.season for entry in self.seasons]
        nearest = min(fitted, key=lambda year: abs(year - season))
        anchor = next(entry.offset_s.value for entry in self.seasons if entry.season == nearest)
        return anchor + self.season_trend_s.value * (season - nearest)

    def reference_lap_s(self, circuit: str, season: int) -> float:
        """Expected corrected lap time for a field-median car at a circuit in a season."""
        return self.circuit(circuit).reference_lap_s.value + self.season_offset(season)


def car_pace_frame(
    laps: pl.DataFrame,
    pace: PaceModel | Mapping[str, PaceModel],
    settings: AppSettings | None = None,
) -> pl.DataFrame:
    """Per-car corrected pace for every car-race, relative to its own race's field median.

    Args:
        laps: Cleaned, featured laps.
        pace: The pooled model, or a mapping from circuit to model. A mapping is preferred: the
            correction is more accurate when it uses the degradation actually fitted at that
            circuit, and the residual pace is then less contaminated by the pooled curve's error.
        settings: Configuration; the process-wide settings by default.

    Returns:
        One row per car-race with ``pace_s`` -- seconds per lap relative to the field median, so
        that the level is the circuit's and the spread is the competitive order's. Car-races with
        too few surviving laps to median are absent.
    """
    settings = settings or get_settings()
    if laps.height == 0:
        return pl.DataFrame(schema=_CAR_PACE_SCHEMA)

    corrected = _with_corrected(laps, pace)
    per_car = (
        corrected.group_by("race_id", "car_id")
        .agg(
            pl.col("season").first(),
            pl.col("circuit").first(),
            pl.col("driver").first(),
            pl.col("team").first(),
            pl.col("grid_position").first(),
            pl.col("corrected_lap_s").median().alias("_median_s"),
            pl.len().alias("n_laps"),
        )
        .filter(pl.col("n_laps") >= _MIN_LAPS_PER_CAR_RACE)
    )
    return (
        per_car.with_columns(
            (pl.col("_median_s") - pl.col("_median_s").median().over("race_id")).alias("pace_s")
        )
        .drop("_median_s", "car_id")
        .sort("race_id", "pace_s")
    )


_CAR_PACE_SCHEMA: dict[str, pl.DataType] = {
    "race_id": pl.String(),
    "season": pl.Int16(),
    "circuit": pl.String(),
    "driver": pl.String(),
    "team": pl.String(),
    "grid_position": pl.Int16(),
    "n_laps": pl.UInt32(),
    "pace_s": pl.Float64(),
}


def _with_corrected(laps: pl.DataFrame, pace: PaceModel | Mapping[str, PaceModel]) -> pl.DataFrame:
    """Attach the corrected lap time, per circuit where a per-circuit model is available."""
    if isinstance(pace, PaceModel):
        return laps.with_columns(corrected_lap_time(pace)).drop_nulls("corrected_lap_s")

    frames = [
        part.with_columns(corrected_lap_time(pace[str(circuit)]))
        for (circuit,), part in laps.group_by(["circuit"], maintain_order=True)
        if str(circuit) in pace
    ]
    if not frames:
        raise ValueError("no circuit in the laps has a fitted pace model")
    return pl.concat(frames).drop_nulls("corrected_lap_s")


@dataclass(frozen=True, slots=True)
class _TwoWayFit:
    """Circuit and season levels fitted jointly, each with the standard error behind it.

    Attributes:
        circuit_level: Corrected lap time per circuit, in the reference season.
        circuit_error: Standard error of each circuit level.
        season_offset: Pace offset per season; the reference season's is exactly zero.
        season_error: Standard error of each season offset; the reference season's is zero
            because it is the definition of the zero rather than an estimate of it.
        reference_season: The season the circuit levels are quoted in.
        residual_dof: Residual degrees of freedom of the joint fit.
    """

    circuit_level: dict[str, float]
    circuit_error: dict[str, float]
    season_offset: dict[int, float]
    season_error: dict[int, float]
    reference_season: int
    residual_dof: int


def _two_way_fit(cells: pl.DataFrame) -> _TwoWayFit:
    """Fit ``median = circuit + season`` by least squares over the circuit-season medians.

        The reference season is the latest observed, so a circuit baseline is quoted in the most
        recent
        era rather than the oldest, and simulating the season after the training window extrapolates
    from the nearest data rather than across the whole window.
    """
    circuits = sorted(str(name) for name in cells["circuit"].unique().to_list())
    seasons = sorted(int(year) for year in cells["season"].unique().to_list())
    reference = seasons[-1]
    others = [year for year in seasons if year != reference]

    circuit_index = {name: position for position, name in enumerate(circuits)}
    season_index = {year: len(circuits) + position for position, year in enumerate(others)}

    design = np.zeros((cells.height, len(circuits) + len(others)), dtype=np.float64)
    target = cells["median_s"].to_numpy().astype(np.float64)
    weights = cells["n_laps"].to_numpy().astype(np.float64)
    for row, (circuit, season) in enumerate(
        zip(cells["circuit"].to_list(), cells["season"].to_list(), strict=True)
    ):
        design[row, circuit_index[str(circuit)]] = 1.0
        if int(season) in season_index:
            design[row, season_index[int(season)]] = 1.0

    # Weighted by the laps behind each median: a circuit-season with four surviving cars should
    # not move the season trend as far as one with twenty.
    root = np.sqrt(weights)
    solution, *_ = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)
    residuals = target - design @ solution
    dof = max(cells.height - solution.size, 1)

    # Standard errors from the weighted normal equations, so a baseline resting on one race
    # reports a wider interval than one resting on three.
    scale = float(weights @ residuals**2) / dof
    covariance = np.linalg.pinv((design * weights[:, None]).T @ design) * scale
    errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    return _TwoWayFit(
        circuit_level={name: float(solution[circuit_index[name]]) for name in circuits},
        circuit_error={name: float(errors[circuit_index[name]]) for name in circuits},
        season_offset={reference: 0.0, **{y: float(solution[season_index[y]]) for y in others}},
        season_error={reference: 0.0, **{y: float(errors[season_index[y]]) for y in others}},
        reference_season=reference,
        residual_dof=dof,
    )


def _standing_start(
    all_laps: pl.DataFrame | None, level: float, rng: np.random.Generator
) -> Estimate | None:
    """What the opening lap costs over a flying lap, pooled across races.

    Measured per race and then pooled, rather than pooled over laps, because the quantity is a
    property of a race start and a circuit with a long run to the first corner should count once.

    **Races that did not start green are excluded.** A first-lap incident brings out the safety car
    or red-flags the race, and lap one then takes several minutes -- but that is a neutralisation,
    which the simulator already models from its own fitted parameters. Leaving those races in would
    count the safety car twice and put the standing-start penalty somewhere in the minutes.

    The interval comes from a seeded bootstrap of the median, not from a normal standard error. Even
    among green starts the sample has a long right tail, and treating it as normal reported an
    interval a hundred seconds wide around an eight-second effect.
    """
    if all_laps is None or all_laps.height == 0:
        return None
    first, last = _FLYING_REFERENCE_LAPS

    green_start = (
        all_laps.filter(pl.col("lap") == 1)
        .group_by("race_id")
        .agg((pl.col("track_status").fill_null("") == "1").all().alias("started_green"))
        .filter(pl.col("started_green"))
        .select("race_id")
    )
    per_race = (
        all_laps.join(green_start, on="race_id", how="inner")
        .filter(pl.col("lap_time_s").is_not_null())
        .group_by("race_id")
        .agg(
            pl.col("lap_time_s").filter(pl.col("lap") == 1).median().alias("opening_s"),
            pl.col("lap_time_s")
            .filter(
                pl.col("lap").is_between(first, last)
                & (pl.col("track_status").fill_null("") == "1")
                & ~pl.col("is_pit_in")
                & ~pl.col("is_pit_out")
            )
            .median()
            .alias("flying_s"),
        )
        .filter(pl.col("opening_s").is_not_null() & pl.col("flying_s").is_not_null())
    )
    if per_race.height == 0:
        return None

    penalties = (per_race["opening_s"] - per_race["flying_s"]).to_numpy().astype(np.float64)
    return bootstrap_median(penalties, rng, level)


def fit_baseline(
    laps: pl.DataFrame,
    pace: PaceModel | Mapping[str, PaceModel],
    settings: AppSettings | None = None,
    *,
    all_laps: pl.DataFrame | None = None,
) -> BaselineModel:
    """Recover absolute pace from the laps the pace model was fitted on.

    Args:
        laps: Cleaned, featured training laps.
        pace: The fitted pace model, pooled or per circuit.
        settings: Configuration; the process-wide settings by default.
        all_laps: Featured but unfiltered laps. Required to measure the standing start, which the
            exclusion rules remove from ``laps`` by design; omitted, that parameter is reported as
            unmeasured rather than guessed.

    Raises:
        ValueError: if no car-race has enough surviving laps to median. A baseline of nothing
            would be a number with no evidence behind it, which this project does not publish.
    """
    settings = settings or get_settings()
    level = settings.calibration.confidence_level

    cars = car_pace_frame(laps, pace, settings)
    if cars.height == 0:
        raise ValueError("no car-race has enough clean laps to establish a pace baseline")

    corrected = _with_corrected(laps, pace)
    cells = (
        corrected.group_by("circuit", "season")
        .agg(
            pl.col("corrected_lap_s").median().alias("median_s"),
            pl.len().alias("n_laps"),
            pl.n_unique("race_id").alias("n_races"),
        )
        .sort("circuit", "season")
    )

    fit = _two_way_fit(cells)

    per_circuit = (
        cells.group_by("circuit")
        .agg(
            pl.col("n_laps").sum().alias("n_laps"),
            pl.col("n_races").sum().alias("n_races"),
            pl.col("season").sort().alias("seasons"),
        )
        .sort("circuit")
    )
    circuits = tuple(
        CircuitBaseline(
            circuit=str(row["circuit"]),
            reference_lap_s=Estimate.from_standard_error(
                fit.circuit_level[str(row["circuit"])],
                fit.circuit_error[str(row["circuit"])],
                n_obs=int(row["n_laps"]),
                level=level,
                degrees_of_freedom=float(max(int(row["n_races"]) - 1, 1)),
            ),
            n_laps=int(row["n_laps"]),
            n_races=int(row["n_races"]),
            seasons=tuple(int(year) for year in row["seasons"]),
        )
        for row in per_circuit.iter_rows(named=True)
    )

    races_per_season = {
        int(season): int(count)
        for season, count in cells.group_by("season").agg(pl.col("n_races").sum()).iter_rows()
    }
    seasons = tuple(
        SeasonPace(
            season=year,
            offset_s=Estimate.from_standard_error(
                offset,
                fit.season_error[year],
                n_obs=races_per_season.get(year, 0),
                level=level,
                degrees_of_freedom=float(fit.residual_dof),
            ),
            n_races=races_per_season.get(year, 0),
        )
        for year, offset in sorted(fit.season_offset.items())
    )

    return BaselineModel(
        circuits=circuits,
        seasons=seasons,
        reference_season=fit.reference_season,
        season_trend_s=_season_trend(seasons, level),
        field_spread_s=_field_spread(cars, level),
        standing_start_s=_standing_start(
            all_laps, level, settings.seeds().derive("calibration").stream("standing_start")
        ),
        n_car_races=cars.height,
    )


def _season_trend(seasons: tuple[SeasonPace, ...], level: float) -> Estimate | None:
    """Least-squares slope of the season offsets against the year."""
    if len(seasons) < _MIN_CELLS_FOR_TREND:
        return None
    years = np.array([entry.season for entry in seasons], dtype=np.float64)
    offsets = np.array([entry.offset_s.value for entry in seasons], dtype=np.float64)
    centred = years - years.mean()
    denominator = float(centred @ centred)
    if denominator <= 0.0:  # pragma: no cover - guarded by the length check above
        return None
    slope = float(centred @ offsets / denominator)
    residuals = offsets - (offsets.mean() + slope * centred)
    dof = max(len(seasons) - 2, 1)
    std_error = float(np.sqrt(float(residuals @ residuals) / dof / denominator))
    return Estimate.from_standard_error(
        slope, std_error, n_obs=len(seasons), level=level, degrees_of_freedom=float(dof)
    )


def _field_spread(cars: pl.DataFrame, level: float) -> Estimate:
    """Within-race standard deviation of car pace, pooled across races."""
    per_race = cars.group_by("race_id").agg(
        pl.col("pace_s").std().alias("spread"), pl.len().alias("n_cars")
    )
    usable = per_race.filter(pl.col("spread").is_not_null() & (pl.col("n_cars") > 1))
    if usable.height == 0:  # pragma: no cover - a race with one car cannot reach here
        return Estimate(value=0.0, std_error=None, interval=None, level=level, n_obs=0)
    spreads = usable["spread"].to_numpy().astype(np.float64)
    return Estimate.from_standard_error(
        float(spreads.mean()),
        float(spreads.std(ddof=1) / np.sqrt(spreads.size)) if spreads.size > 1 else 0.0,
        n_obs=int(usable["n_cars"].sum()),
        level=level,
        degrees_of_freedom=float(max(spreads.size - 1, 1)),
    )

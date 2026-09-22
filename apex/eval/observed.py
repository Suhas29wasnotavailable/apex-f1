"""What actually happened in a real race, in the simulator's own vocabulary.

Validation compares two races, and they have to be described the same way or the comparison measures
the translation instead of the model. So everything here reduces an ingested lap table to exactly
the
quantities a :class:`~apex.sim.trace.RaceTrace` reports: a finishing order, a lap-by-lap table, the
stints each car ran, and the laps the race was neutralised for.

Three of these are also *inputs* to a validation run rather than only outputs, which is the point.
To ask whether the twin reproduces a race, the twin has to be given that race's decisions -- its
grid, its pit stops, its safety cars -- and left to produce everything else. :func:`observed_stops`
becomes a replay strategy and :func:`observed_schedule` becomes an imposed flag schedule, so what
remains for the simulator to get right is the racing: lap times, tyre degradation, traffic, who was
where at the end.

**Classified finishers only.** A car that retired on lap nine has no simulated counterpart, because
the simulator models no retirements, and including it would score the twin on an event it was never
given the parameters to predict. Retirements are therefore excluded from the comparison and
counted, so the cost of that exclusion is visible rather than buried.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from apex.calibrate.baseline import car_pace_frame
from apex.calibrate.exclusions import clean_laps
from apex.calibrate.pace import PaceModel
from apex.calibrate.params import ParameterSet
from apex.config import AppSettings, get_settings
from apex.sim.strategy import CarView, PitCall, TrackPhase

__all__ = [
    "ObservedRace",
    "ReplayStops",
    "observed_pace",
    "observed_race",
    "observed_schedule",
    "observed_stops",
]

_CLASSIFIED_FRACTION = 0.9
"""Share of the leader's laps a car must complete to count as a finisher. The regulations use 90% of
the winner's distance, and a car below it is not a result the simulator should be scored against."""


@dataclass(frozen=True, slots=True)
class ObservedRace:
    """One real race, reduced to what a simulated race also reports.

    Attributes:
        event: Event name.
        season: Season.
        circuit: Circuit name.
        total_laps: Laps the winner completed.
        laps: Per-driver per-lap table for classified finishers.
        finishing_order: Drivers in finishing order.
        total_time_s: Cumulative race time, for classified finishers whose every lap was timed.
            A race with untimed laps yields fewer entries here than in ``finishing_order``, and the
            time-based metrics report how many cars they covered.
        stops: Pit laps and fitted compounds per driver.
        schedule: Flags by lap.
        retired: Drivers who started and were not classified.
    """

    event: str
    season: int
    circuit: str
    total_laps: int
    laps: pl.DataFrame
    finishing_order: tuple[str, ...]
    total_time_s: dict[str, float]
    stops: dict[str, tuple[tuple[int, str], ...]]
    schedule: dict[int, TrackPhase]
    retired: tuple[str, ...]

    @property
    def n_classified(self) -> int:
        return len(self.finishing_order)

    @property
    def safety_car_laps(self) -> tuple[int, ...]:
        return tuple(
            lap for lap, phase in sorted(self.schedule.items()) if phase is TrackPhase.SAFETY_CAR
        )

    @property
    def virtual_safety_car_laps(self) -> tuple[int, ...]:
        return tuple(
            lap
            for lap, phase in sorted(self.schedule.items())
            if phase is TrackPhase.VIRTUAL_SAFETY_CAR
        )


def observed_schedule(
    laps: pl.DataFrame, settings: AppSettings | None = None
) -> dict[int, TrackPhase]:
    """The flags lap by lap, reduced from the field's per-driver track status.

    Track status is a property of the track, so a lap counts as neutralised if any car on it saw
    the status. A full safety car takes precedence over a virtual one on a lap that shows both,
    matching the simulator's own rule and reflecting that the full deployment is the strategically
    decisive one.
    """
    settings = settings or get_settings()
    calibration = settings.calibration

    def matches(codes: tuple[str, ...]) -> pl.Expr:
        return pl.any_horizontal(
            [
                pl.col("track_status").fill_null("").str.contains(code, literal=True)
                for code in codes
            ]
        )

    per_lap = (
        laps.group_by("lap")
        .agg(
            matches(calibration.full_safety_car_status).any().alias("full"),
            matches(calibration.virtual_safety_car_status).any().alias("virtual"),
        )
        .sort("lap")
    )
    schedule: dict[int, TrackPhase] = {}
    for lap, full, virtual in per_lap.iter_rows():
        if full:
            schedule[int(lap)] = TrackPhase.SAFETY_CAR
        elif virtual:
            schedule[int(lap)] = TrackPhase.VIRTUAL_SAFETY_CAR
    return schedule


def observed_pace(
    laps: pl.DataFrame, params: ParameterSet, settings: AppSettings | None = None
) -> dict[str, float]:
    """Each car's actual race pace, measured from the race being validated.

    **This is leakage, and it is only ever used as an oracle.** It reads the answer out of the race
    the twin is being asked to predict, so no result computed with it is a prediction. It exists to
    decompose the validation error, which is otherwise uninterpretable: when a twin whose car pace
    comes from grid position gets the finishing order wrong, nothing says whether the race model is
    wrong or whether qualifying simply failed to predict race pace. Supplying the true pace and
    re-running answers that, and the gap between the two runs is the cost of the forecast.

    Measured exactly as the fitted car pace was -- corrected lap times, median per car, relative to
    the race's field median -- so the number substitutes cleanly for the grid model's estimate.
    """
    settings = settings or get_settings()
    circuit = str(laps["circuit"][0])
    try:
        pace: PaceModel | dict[str, PaceModel] = params.circuit_pace_at(circuit)
    except KeyError:
        pace = params.pace

    cleaned = clean_laps(laps, settings).laps
    if cleaned.height == 0:
        return {}
    cars = car_pace_frame(cleaned, pace, settings)
    return {str(row["driver"]): float(row["pace_s"]) for row in cars.iter_rows(named=True)}


def observed_stops(laps: pl.DataFrame) -> dict[str, tuple[tuple[int, str], ...]]:
    """Every car's real pit laps and what it fitted at each.

    A stop is read from the stint boundary rather than from the pit-in flag, so that a drive-through
    or a pit entry that did not change tyres is not counted as a strategic stop. The lap recorded is
    the in-lap, which is the lap the simulator charges the pit loss to.

    A car's *first* stint is never a stop, and it is identified by being that car's lowest stint
    number rather than by starting on lap one. The difference matters: a car whose opening laps are
    missing from the timing feed begins its first stint on lap two or three, and a rule keyed on lap
    one reads that as a pit stop the car never made.
    """
    stints = (
        laps.filter(pl.col("stint").is_not_null())
        .group_by("driver", "stint")
        .agg(
            pl.col("lap").min().alias("first_lap"),
            pl.col("compound").sort_by("lap").first().alias("compound"),
        )
        .sort("driver", "stint")
    )
    opening = dict(stints.group_by("driver").agg(pl.col("stint").min()).iter_rows())

    stops: dict[str, list[tuple[int, str]]] = {}
    for row in stints.iter_rows(named=True):
        driver = str(row["driver"])
        if row["stint"] == opening.get(driver):
            continue
        stops.setdefault(driver, []).append((int(row["first_lap"]) - 1, str(row["compound"])))
    return {driver: tuple(sorted(entries)) for driver, entries in stops.items()}


def observed_race(laps: pl.DataFrame, settings: AppSettings | None = None) -> ObservedRace:
    """Reduce one real race to the quantities a simulated race also reports.

    Args:
        laps: Featured laps of exactly one race, unfiltered. The unfiltered frame is required:
            neutralised laps and pit laps are part of what is being described.
        settings: Configuration; the process-wide settings by default.

    Raises:
        ValueError: if the frame does not hold exactly one race.
    """
    settings = settings or get_settings()
    races = laps.select("season", "round").unique()
    if races.height != 1:
        raise ValueError(f"expected exactly one race, got {races.height}")

    total_laps = int(laps["lap"].max() or 0)  # type: ignore[arg-type]
    threshold = total_laps * _CLASSIFIED_FRACTION

    per_driver = (
        laps.group_by("driver")
        .agg(
            pl.col("lap").max().alias("last_lap"),
            pl.col("lap_time_s").sum().alias("summed_time_s"),
            pl.col("lap_time_s").null_count().alias("missing_laps"),
        )
        .sort("driver")
    )

    # Classification is on distance alone. Requiring a complete set of lap times as well looks
    # tempting and is wrong: at Jeddah in 2025 an opening-lap incident left laps two and three
    # untimed for eighteen cars, and a rule that wanted every lap timed threw out the entire field
    # and with it the whole race. Whether a car finished is a question about distance.
    classified = per_driver.filter(pl.col("last_lap") >= threshold)
    retired = tuple(
        str(name) for name in per_driver.filter(pl.col("last_lap") < threshold)["driver"]
    )

    finishers = set(classified["driver"].to_list())
    kept = laps.filter(pl.col("driver").is_in(list(finishers)))

    # A total race time is only quoted for a car that finished on the lead lap with every lap timed.
    # Both conditions matter and each was got wrong once. Summing across a gap in the timing reports
    # a car as minutes faster than it was. And summing across a *different number of laps* is not a
    # race time at all: a car one lap down has fifty-seven lap times where the winner has
    # fifty-eight,
    # so differencing the two sums silently mixes "slower" with "shorter" and made the simulator's
    # finishing gaps look tens of seconds wrong when they were not. Lapped cars keep their place in
    # the finishing order, which is the thing the position metrics measure; they are simply not
    # quoted a gap, exactly as the sport does not quote them one.
    times = {
        str(row["driver"]): float(row["summed_time_s"])
        for row in classified.iter_rows(named=True)
        if int(row["missing_laps"]) == 0
        and row["summed_time_s"] is not None
        and int(row["last_lap"]) == total_laps
    }

    # Finishing order is the real classification: position on the final lap each finisher ran.
    final = (
        kept.filter(pl.col("lap") == total_laps)
        .select("driver", "position")
        .drop_nulls()
        .sort("position")
    )
    order = tuple(str(name) for name in final["driver"])
    if len(order) < len(finishers):
        # A finisher with no final-lap position -- a lapped car, which never crossed the line on the
        # leader's last lap -- falls in behind, ordered the way the sport orders them: distance
        # completed first, then elapsed time, then the driver's name.
        #
        # That last tie-break is not cosmetic. Lapped cars are the cars with no quoted race time,
        # so ordering them by time alone compares several `inf` values and leaves a stable sort
        # preserving the order of a *set* -- which Python hashes differently in every process. The
        # validation numbers were reproducible within a run and moved between runs, which is the
        # failure ADR-0004 exists to prevent.
        distance = {
            str(row["driver"]): int(row["last_lap"]) for row in classified.iter_rows(named=True)
        }
        missing = sorted(
            finishers - set(order),
            key=lambda name: (-distance.get(name, 0), times.get(name, float("inf")), name),
        )
        order = (*order, *missing)

    return ObservedRace(
        event=str(laps["event"][0]),
        season=int(laps["season"][0]),
        circuit=str(laps["circuit"][0]),
        total_laps=total_laps,
        laps=kept,
        finishing_order=order,
        total_time_s=times,
        stops=observed_stops(kept),
        schedule=observed_schedule(laps, settings),
        retired=retired,
    )


@dataclass(frozen=True)
class ReplayStops:
    """A strategy that reproduces one real race's pit stops exactly.

    This is what makes the strongest validation mode possible. Given the real stops, the real grid
    and the real safety cars, everything left for the simulator to produce is the racing itself, so
    a discrepancy in the finishing order is attributable to the pace, degradation and traffic
    models rather than to having guessed a different strategy.

    Stateless, like every strategy here: the stop to take next is looked up by the car's own stop
    count, so one instance serves the whole field.
    """

    stops: Mapping[str, tuple[tuple[int, str], ...]]
    """Pit laps and compounds per driver, read and never written."""

    fallback_compound: str = "HARD"
    """Fitted when a real stop names a compound the simulator is not configured to run -- a wet tyre
    in an otherwise dry race, most often. Rare, and better than dropping the stop."""

    def reset(self) -> None:
        """Nothing to reset: the plan is read from the view's stop count."""

    def decide(self, view: CarView) -> PitCall | None:
        planned = self.stops.get(view.driver, ())
        if view.stops >= len(planned):
            return None
        lap, compound = planned[view.stops]
        if view.lap != lap:
            return None
        if compound not in view.available_compounds:
            compound = self.fallback_compound
        return PitCall(compound=compound)

"""What defines a race, and how to build one from information that exists before the start.

A scenario is the complete pre-race specification: the circuit, its length, and who is lined up
where in what car. It deliberately contains no lap times and no outcome. Everything that happens
after the lights go out is produced by the simulator from the fitted parameters, which is the only
arrangement under which a validation result means anything.

Two ways to build one, for the two things this project has to do.

:func:`from_grid` takes a starting order and nothing else. Pace comes from the fitted grid model,
so a lineup somebody types into the front end becomes a field of cars with plausible relative
performance. This is also the mode a held-out race is validated in: grid position is known before
the race, so a prediction built from it is a prediction.

:func:`from_race` takes a real ingested race and rebuilds its grid. It is the same thing as
:func:`from_grid` with the grid read off a real event rather than typed, and it exists so that
validation does not have to hand-transcribe twenty drivers per race.

The reference lap time -- the absolute pace level of the weekend -- is the one quantity a scenario
may take from outside the fitted parameters, and it is separated out under
:attr:`RaceScenario.anchor` precisely so that it can be counted. Two modes:

*   **Fitted.** The circuit's baseline plus the season trend, extrapolated if the season was never
    calibrated. Entirely predictive and entirely uncertain: the 2025 trend is an extrapolation from
    three seasons.
*   **Anchored.** The real median lap time of the race being reproduced. One number, describing how
    fast the cars were that weekend, which no strategy model can be expected to predict and every
    race simulator in the industry is given. Anchoring lets the validation test race *dynamics*
    rather than a season-ahead pace forecast.

Both are supported, validation reports both, and the scenario records which one it used, so no
reader has to wonder whether a good result came from the twin or from the anchor.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from apex.calibrate.baseline import corrected_lap_time
from apex.calibrate.grid import PIT_LANE_GRID
from apex.calibrate.params import ParameterSet
from apex.config import AppSettings, get_settings

__all__ = ["Anchor", "AnchorMode", "Entrant", "RaceScenario", "from_grid", "from_race"]

_MIN_STARTERS = 2
"""A race needs two cars before anything about it is a race."""


class AnchorMode(enum.StrEnum):
    """Where a scenario's absolute pace level came from."""

    FITTED = "fitted"
    """From the calibrated circuit baseline and season trend. Fully predictive."""

    ANCHORED = "anchored"
    """From the real race's own median lap time. One observed number, recorded as such."""


class Anchor(BaseModel):
    """The absolute pace level of a race, and the honest label for where it came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reference_lap_s: float = Field(gt=0.0)
    """Lap time of a field-median car on a fresh reference-compound tyre, full fuel, in clean
    air."""

    mode: AnchorMode


class Entrant(BaseModel):
    """One car on the grid."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    driver: str
    team: str = ""
    grid_position: int = Field(ge=0)
    """Starting slot. Zero is a pit-lane start, which lines up last."""

    pace_s: float
    """Seconds per lap relative to the field median, before tyres, fuel or traffic. Negative is
    faster. This is the car and the driver together: the fitted parameters cannot separate them and
    this project does not claim to."""

    starting_compound: str

    @property
    def starting_slot(self) -> int:
        """Grid position for ordering purposes, with a pit-lane start placed at the back."""
        return 1_000 if self.grid_position == PIT_LANE_GRID else self.grid_position


class RaceScenario(BaseModel):
    """Everything needed to simulate a race, and nothing that reveals its outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    circuit: str
    season: int
    total_laps: int = Field(ge=2)
    entrants: tuple[Entrant, ...] = Field(min_length=2)
    anchor: Anchor

    event: str = ""
    """Event name, carried for display and for matching a simulated race back to a real one."""

    @property
    def n_cars(self) -> int:
        return len(self.entrants)

    def starting_order(self) -> tuple[Entrant, ...]:
        """Entrants in grid order, pit-lane starts last and ties broken by driver for
        determinism."""
        return tuple(sorted(self.entrants, key=lambda entry: (entry.starting_slot, entry.driver)))

    @model_validator(mode="after")
    def _check_unique_drivers(self) -> RaceScenario:
        drivers = [entry.driver for entry in self.entrants]
        duplicates = sorted({name for name in drivers if drivers.count(name) > 1})
        if duplicates:
            raise ValueError(f"a driver cannot appear twice in one race: {duplicates}")
        return self


def _fitted_anchor(params: ParameterSet, circuit: str, season: int) -> Anchor:
    return Anchor(
        reference_lap_s=params.baseline.reference_lap_s(circuit, season), mode=AnchorMode.FITTED
    )


def from_grid(
    params: ParameterSet,
    *,
    circuit: str,
    season: int,
    total_laps: int,
    grid: Sequence[tuple[str, str]] | Sequence[str],
    starting_compound: str = "MEDIUM",
    anchor: Anchor | None = None,
    pace_overrides: dict[str, float] | None = None,
    settings: AppSettings | None = None,
) -> RaceScenario:
    """Build a scenario from a starting order alone.

    Args:
        params: Fitted parameters. The grid model supplies each car's pace.
        circuit: Circuit name, which must be one the baseline was fitted at.
        season: Season, used only to place the pace level along the fitted season trend.
        total_laps: Race distance.
        grid: Drivers in starting order, either as codes or as ``(driver, team)`` pairs.
        starting_compound: Compound every car starts on, unless overridden per car later.
        anchor: Pace level. Fitted from the circuit baseline when omitted.
        pace_overrides: Per-driver pace in seconds relative to the field median, replacing the
            grid model's estimate. This is how a caller who knows more than the grid -- a real
            qualifying gap, a known engine deficit -- says so.
        settings: Configuration; the process-wide settings by default.

    Raises:
        ValueError: if the starting compound is not one the simulator is configured to run.
    """
    settings = settings or get_settings()
    if starting_compound not in settings.simulation.compounds:
        raise ValueError(
            f"starting compound {starting_compound!r} is not available; "
            f"configured compounds are {list(settings.simulation.compounds)}"
        )

    pairs: list[tuple[str, str]] = [
        (entry, "") if isinstance(entry, str) else (entry[0], entry[1]) for entry in grid
    ]
    overrides = pace_overrides or {}
    entrants = tuple(
        Entrant(
            driver=driver,
            team=team,
            grid_position=position,
            pace_s=overrides.get(driver, params.grid_pace.pace_s(position)),
            starting_compound=starting_compound,
        )
        for position, (driver, team) in enumerate(pairs, start=1)
    )
    return RaceScenario(
        circuit=circuit,
        season=season,
        total_laps=total_laps,
        entrants=entrants,
        anchor=anchor or _fitted_anchor(params, circuit, season),
    )


def from_race(
    params: ParameterSet,
    laps: pl.DataFrame,
    *,
    anchor_mode: AnchorMode = AnchorMode.ANCHORED,
    settings: AppSettings | None = None,
    pace_overrides: dict[str, float] | None = None,
) -> RaceScenario:
    """Rebuild the starting grid of one real race as a scenario.

        Only pre-race facts are read: who started where, in which car, on which compound, and how
        many
        laps the race ran to. Lap times are read for one purpose only -- the anchor -- and only when
        :attr:`AnchorMode.ANCHORED` is asked for.

        Args:
            params: Fitted parameters.
            laps: The featured laps of exactly one race.
            anchor_mode: Whether to take the pace level from this race's own median lap time or
            from the
                fitted circuit baseline.
            settings: Configuration; the process-wide settings by default.
            pace_overrides: Per-driver pace replacing the grid model's forecast. Supplying pace
            measured
    from this race makes the scenario an oracle rather than a prediction, so a caller that
                passes this must label the result accordingly.

        Raises:
            ValueError: if the frame holds more or fewer than one race, or no usable grid.
    """
    settings = settings or get_settings()
    races = laps.select("season", "round").unique()
    if races.height != 1:
        raise ValueError(f"expected exactly one race, got {races.height}")

    season = int(laps["season"][0])
    circuit = str(laps["circuit"][0])
    event = str(laps["event"][0])
    total_laps = int(laps["lap"].max() or 0)  # type: ignore[arg-type]

    starters = (
        laps.group_by("driver")
        .agg(
            pl.col("team").first(),
            pl.col("grid_position").min(),
            pl.col("compound").sort_by("lap").first().alias("starting_compound"),
        )
        .sort("grid_position", "driver")
    )
    available = set(settings.simulation.compounds)
    overrides = pace_overrides or {}
    entrants = tuple(
        Entrant(
            driver=str(row["driver"]),
            team=str(row["team"] or ""),
            grid_position=int(row["grid_position"]),
            pace_s=overrides.get(
                str(row["driver"]), params.grid_pace.pace_s(int(row["grid_position"]))
            ),
            starting_compound=(
                str(row["starting_compound"])
                if str(row["starting_compound"]) in available
                else settings.simulation.compounds[0]
            ),
        )
        for row in starters.iter_rows(named=True)
    )
    if len(entrants) < _MIN_STARTERS:
        raise ValueError(f"{event} has fewer than two classified starters")

    if anchor_mode is AnchorMode.ANCHORED:
        anchor = Anchor(
            reference_lap_s=_observed_reference_lap(laps, params, circuit),
            mode=AnchorMode.ANCHORED,
        )
    else:
        anchor = _fitted_anchor(params, circuit, season)

    return RaceScenario(
        circuit=circuit,
        season=season,
        total_laps=total_laps,
        entrants=entrants,
        anchor=anchor,
        event=event,
    )


def _observed_reference_lap(laps: pl.DataFrame, params: ParameterSet, circuit: str) -> float:
    """The race's own pace level, on the same footing as the fitted baselines.

    The median of the race's corrected lap times, which is exactly how the circuit baselines were
    measured, so an anchored scenario and a fitted one differ in the value of this number and not
    in its meaning.
    """
    try:
        pace = params.circuit_pace_at(circuit)
    except KeyError:
        pace = params.pace

    corrected = (
        laps.filter(
            pl.col("lap_time_s").is_not_null()
            & (pl.col("track_status").fill_null("") == "1")
            & ~pl.col("is_pit_in")
            & ~pl.col("is_pit_out")
        )
        .with_columns(corrected_lap_time(pace))
        .drop_nulls("corrected_lap_s")
    )
    if corrected.height == 0:
        raise ValueError(f"no clean laps to anchor the pace level of {circuit}")
    return float(corrected["corrected_lap_s"].median())  # type: ignore[arg-type]

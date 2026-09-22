"""The conditions a race was run in, and the car performance behind the grid.

Everything in this module is **observed, not simulated**, and that distinction is the whole reason
it is a separate module rather than part of the replay payload's own construction.

ADR-0002 scoped this project to dry races. There is no weather model, no track-temperature effect
on the pace model, and no engine model — so a replay that invented a rain shower, or a track
temperature that changed the lap times, would be showing the viewer a number the simulator does not
have and cannot act on. That would be worse than showing nothing.

What can be shown honestly is what the timing feed actually recorded at that circuit: air and track
temperature lap by lap, whether it rained, and how fast the cars were through the speed trap. Those
are real measurements from a real session. They are displayed as **context** — the conditions this
circuit's races are run in — and the page says so. A viewer learns that Singapore runs at fifty
degrees of track temperature and Silverstone at half that, which is true and worth knowing, without
being told that the simulator did anything with it.

The speed trap is the closest thing to an engine reading this project has. It is a direct
measurement of straight-line speed, which is mostly power and drag, and it is reported per team
because that is the level at which a power unit is shared. It is emphatically not a fitted
parameter: the simulator knows a car's pace as one number per car-race and has no notion of where on
the lap that pace comes from.
"""

from __future__ import annotations

import math

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from apex.config import AppSettings, get_settings
from apex.data import load_laps
from apex.data.schema import canonical_circuit

__all__ = ["CircuitConditions", "TeamPerformance", "circuit_conditions"]

_CONDITION_SAMPLES = 40
"""Points in the published temperature series. Finer than a viewer can read, coarser than a lap."""


class TeamPerformance(BaseModel):
    """What one team's cars actually did, measured rather than fitted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    team: str
    speed_trap_kph: float
    """Median speed through the trap. Mostly power and drag, which is as close to an engine reading
    as this project's data goes."""

    drivers: tuple[str, ...]
    n_laps: int = Field(ge=0)


class CircuitConditions(BaseModel):
    """Conditions recorded at one circuit, in one real season.

    Attributes:
        circuit: Circuit name.
        season: The season these readings come from.
        event: Event name of the session they were read from.
        air_temp_c: Air temperature through the race, evenly sampled across its distance.
        track_temp_c: Track temperature through the race, sampled the same way.
        rainfall_fraction: Share of laps on which rain was reported. Zero for every race the
            simulator is calibrated on, since those are dry by construction.
        compounds: Compounds actually run in that race, most-used first, with their lap counts.
        teams: Per-team speed trap and drivers.
        was_dry: Whether the session these readings come from was itself dry. A circuit whose most
            recent running was wet still gets conditions, from its most recent *dry* running, and
            this records which it was -- the simulator represents dry races and showing it a wet
            one's readings would imply a model it does not have.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    circuit: str
    season: int
    event: str
    air_temp_c: tuple[float, ...]
    track_temp_c: tuple[float, ...]
    rainfall_fraction: float = Field(ge=0.0, le=1.0)
    compounds: tuple[tuple[str, int], ...]
    teams: tuple[TeamPerformance, ...]
    was_dry: bool = True

    @property
    def track_temp_range(self) -> tuple[float, float]:
        """Coolest and hottest track reading of the race, for a one-line summary."""
        values = [value for value in self.track_temp_c if not math.isnan(value)]
        return (min(values), max(values)) if values else (0.0, 0.0)


def _preferred_season(laps: pl.DataFrame, settings: AppSettings) -> int:
    """The most recent season at this circuit that was run dry, else the most recent of any kind.

    A circuit's latest running is not always the right one to describe it by: Silverstone in 2025
    spent six hundred laps on intermediates, and quoting that race's conditions beside a dry
    simulation would suggest the simulator did something with the rain. It did not -- it has no
    model for it.
    """
    wet = pl.col("compound").is_in(settings.data.wet_compounds)
    per_season = (
        laps.group_by("season").agg(wet.sum().alias("wet_laps")).sort("season", descending=True)
    )
    for row in per_season.iter_rows(named=True):
        if int(row["wet_laps"]) == 0:
            return int(row["season"])
    return int(per_season["season"][0])


def _series(frame: pl.DataFrame, column: str, total_laps: int) -> tuple[float, ...]:
    """A per-lap reading reduced to an evenly spaced series across the race distance.

    Sampled by race fraction rather than by lap so that the series lines up with a simulated race of
    a different length: a fifty-three lap simulation and a fifty-three lap reference race are the
    same race, but a scenario run to a different distance still gets conditions that span it.
    """
    per_lap = (
        frame.group_by("lap")
        .agg(pl.col(column).mean().alias("value"))
        .drop_nulls("value")
        .sort("lap")
    )
    if per_lap.height == 0 or total_laps < 1:
        return ()
    laps = per_lap["lap"].to_list()
    values = per_lap["value"].to_list()

    out: list[float] = []
    for index in range(_CONDITION_SAMPLES):
        fraction = index / (_CONDITION_SAMPLES - 1) if _CONDITION_SAMPLES > 1 else 0.0
        target = 1 + fraction * (max(laps) - 1)
        # Nearest recorded lap. Interpolating would invent a reading between two measurements.
        nearest = min(range(len(laps)), key=lambda i: abs(laps[i] - target))
        out.append(round(float(values[nearest]), 1))
    return tuple(out)


def circuit_conditions(
    circuit: str, season: int | None = None, settings: AppSettings | None = None
) -> CircuitConditions | None:
    """Read the conditions and car performance recorded at one circuit.

    Args:
        circuit: Circuit name as the parameter file spells it.
        season: Season to read. The most recent ingested season at that circuit by default.
        settings: Configuration; the process-wide settings by default.

    Returns:
        The recorded conditions, or ``None`` when no race at that circuit is ingested — which is not
        an error, because a replay without a conditions panel is still a replay.
    """
    settings = settings or get_settings()
    wanted = canonical_circuit(circuit)
    laps = load_laps(settings, dry_only=False).filter(pl.col("circuit") == wanted)
    if laps.height == 0:
        return None

    chosen = int(season) if season is not None else _preferred_season(laps, settings)
    race = laps.filter(pl.col("season") == chosen)
    if race.height == 0:
        chosen = int(str(laps["season"].max()))
        race = laps.filter(pl.col("season") == chosen)

    total_laps = int(race["lap"].max() or 0)  # type: ignore[arg-type]

    compounds = (
        race.drop_nulls("compound")
        .filter(pl.col("compound") != "")
        .group_by("compound")
        .agg(pl.len().alias("laps"))
        .sort("laps", descending=True)
    )
    teams = (
        race.drop_nulls(["team", "speed_trap_kph"])
        .group_by("team")
        .agg(
            pl.col("speed_trap_kph").median().alias("trap"),
            pl.col("driver").unique().sort().alias("drivers"),
            pl.len().alias("n_laps"),
        )
        .sort("trap", descending=True)
    )

    rain = race["rainfall"].drop_nulls()
    wet_laps = int(race.filter(pl.col("compound").is_in(settings.data.wet_compounds)).height)
    return CircuitConditions(
        circuit=wanted,
        season=chosen,
        event=str(race["event"][0]),
        air_temp_c=_series(race, "air_temp_c", total_laps),
        track_temp_c=_series(race, "track_temp_c", total_laps),
        rainfall_fraction=round(float(str(rain.mean() or 0.0)), 4) if rain.len() else 0.0,
        was_dry=wet_laps == 0,
        compounds=tuple(
            (str(row["compound"]), int(row["laps"])) for row in compounds.iter_rows(named=True)
        ),
        teams=tuple(
            TeamPerformance(
                team=str(row["team"]),
                speed_trap_kph=round(float(row["trap"]), 1),
                drivers=tuple(str(name) for name in row["drivers"]),
                n_laps=int(row["n_laps"]),
            )
            for row in teams.iter_rows(named=True)
        ),
    )

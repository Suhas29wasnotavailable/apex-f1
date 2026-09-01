"""The only module that talks to FastF1.

Everything that knows about `fastf1` -- its column names, its pandas frames, its cache, its
network calls -- is confined here, and it exports Polars frames in APEX's own vocabulary. Two
things follow. The rest of the codebase is testable without the network, because the pure
translation step :func:`to_lap_frame` takes a frame rather than a session. And if FastF1 renames
a column or changes a dtype, exactly one file needs editing.

`fastf1` is imported lazily, inside the functions that need it, so that the core package (and CI)
does not require the optional ``data`` extra to be installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import polars as pl

from apex.data.schema import EventMeta, ScheduledRace, conform_lap_frame

__all__ = [
    "RateLimitError",
    "RawRace",
    "fastf1_version",
    "fetch_race",
    "race_calendar",
    "scheduled_race",
    "to_lap_frame",
]

RAW_LAP_COLUMNS: Final = (
    "Driver",
    "DriverNumber",
    "Team",
    "LapNumber",
    "Stint",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "LapTime",
    "LapStartTime",
    "Time",
    "Position",
    "TrackStatus",
    "IsAccurate",
    "PitInTime",
    "PitOutTime",
    "Deleted",
    "FastF1Generated",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "SpeedST",
    "AirTemp",
    "TrackTemp",
    "Rainfall",
)
"""FastF1 lap columns APEX reads, plus the three weather columns joined onto each lap."""

_INSTALL_HINT: Final = (
    "fastf1 is not installed. It is an optional dependency: `uv sync --extra data`."
)


class RateLimitError(RuntimeError):
    """FastF1's timing API quota is exhausted.

    Raised in APEX's own vocabulary rather than FastF1's, so that callers can stop a long pull
    cleanly without importing the optional dependency to name its exception. Everything already
    written stays on disk, and re-running the same command resumes where it stopped.
    """


def _fastf1() -> ModuleType:
    """Import FastF1 on demand, with an actionable message if the extra is absent."""
    try:
        import fastf1  # noqa: PLC0415 - deliberately lazy: the `data` extra is optional
    except ModuleNotFoundError as error:  # pragma: no cover - exercised only without the extra
        raise ModuleNotFoundError(_INSTALL_HINT) from error
    module: ModuleType = fastf1
    return module


def fastf1_version() -> str:
    """Version of the FastF1 that produced a file, recorded in every session sidecar."""
    return str(_fastf1().__version__)


@dataclass(frozen=True, slots=True)
class RawRace:
    """One loaded race session, translated into Polars but not yet into APEX's schema.

    Attributes:
        event: Identity of the race, including its scheduled distance.
        laps: FastF1's lap table with its original column names, plus the per-lap weather
            columns ``AirTemp``, ``TrackTemp`` and ``Rainfall``.
        results: Classification, carrying at minimum ``Abbreviation`` and ``GridPosition``.
        rainfall_fraction: Share of the session's weather samples that reported rain.
    """

    event: EventMeta
    laps: pl.DataFrame
    results: pl.DataFrame
    rainfall_fraction: float


def _seconds(frame: pl.DataFrame, column: str) -> pl.Expr:
    """Convert a duration column to float seconds, tolerating an already-numeric column."""
    expression = pl.col(column)
    if isinstance(frame.schema[column], pl.Duration):
        expression = expression.dt.total_nanoseconds() / 1e9
    return expression.cast(pl.Float64, strict=False)


def to_lap_frame(raw: RawRace) -> pl.DataFrame:
    """Translate a raw session into the canonical lap table.

    Pure: no network, no filtering, no imputation. Every row of ``raw.laps`` survives, including
    laps that later exclusion rules will discard, because the record of what happened and the
    decision about what to model are different things.

    Raises:
        ValueError: if FastF1 did not supply a column APEX depends on.
    """
    missing = [name for name in RAW_LAP_COLUMNS if name not in raw.laps.columns]
    if missing:
        raise ValueError(f"session {raw.event.key} is missing FastF1 columns: {missing}")

    grid = raw.results.select(
        pl.col("Abbreviation").cast(pl.String).alias("driver"),
        pl.col("GridPosition").cast(pl.Int16, strict=False).alias("grid_position"),
    ).unique(subset="driver", keep="first")

    laps = raw.laps.select(
        pl.lit(raw.event.season, dtype=pl.Int16).alias("season"),
        pl.lit(raw.event.round, dtype=pl.Int16).alias("round"),
        pl.lit(raw.event.event).alias("event"),
        pl.lit(raw.event.circuit).alias("circuit"),
        pl.lit(raw.event.country).alias("country"),
        pl.col("Driver").alias("driver"),
        pl.col("DriverNumber").alias("driver_number"),
        pl.col("Team").alias("team"),
        pl.col("LapNumber").alias("lap"),
        pl.col("Stint").alias("stint"),
        pl.col("Compound").alias("compound"),
        pl.col("TyreLife").alias("tyre_life"),
        pl.col("FreshTyre").alias("fresh_tyre"),
        _seconds(raw.laps, "LapTime").alias("lap_time_s"),
        _seconds(raw.laps, "LapStartTime").alias("lap_start_s"),
        _seconds(raw.laps, "Time").alias("lap_end_s"),
        _seconds(raw.laps, "Sector1Time").alias("sector1_s"),
        _seconds(raw.laps, "Sector2Time").alias("sector2_s"),
        _seconds(raw.laps, "Sector3Time").alias("sector3_s"),
        pl.col("SpeedST").alias("speed_trap_kph"),
        pl.col("Position").alias("position"),
        pl.col("TrackStatus").alias("track_status"),
        pl.col("IsAccurate").alias("is_accurate"),
        _seconds(raw.laps, "PitInTime").alias("pit_in_s"),
        _seconds(raw.laps, "PitOutTime").alias("pit_out_s"),
        pl.col("Deleted").alias("deleted"),
        pl.col("FastF1Generated").alias("fastf1_generated"),
        pl.col("AirTemp").alias("air_temp_c"),
        pl.col("TrackTemp").alias("track_temp_c"),
        pl.col("Rainfall").alias("rainfall"),
    ).with_columns(
        pl.col("pit_in_s").is_not_null().alias("is_pit_in"),
        pl.col("pit_out_s").is_not_null().alias("is_pit_out"),
    )

    joined = laps.join(grid, on="driver", how="left")
    return conform_lap_frame(joined).sort("driver", "lap")


def _configure(cache_dir: Path) -> ModuleType:
    """Point FastF1 at the on-disk cache and quieten its progress logging."""
    fastf1 = _fastf1()
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    fastf1.set_log_level("WARNING")
    return fastf1


def _scalar(row: Any, key: str) -> Any:
    """Read one field from a pandas Series-like schedule row."""
    return row[key]


def race_calendar(season: int, cache_dir: Path) -> tuple[ScheduledRace, ...]:
    """Return the season's championship races, in round order, excluding testing."""
    fastf1 = _configure(cache_dir)
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    races: list[ScheduledRace] = []
    for _, row in schedule.iterrows():
        round_number = int(_scalar(row, "RoundNumber"))
        if round_number < 1:
            continue
        races.append(_scheduled_from_row(season, row))
    return tuple(sorted(races, key=lambda race: race.round))


def scheduled_race(season: int, event: str | int, cache_dir: Path) -> ScheduledRace:
    """Identify a race from the schedule alone, without loading its timing data.

    Cheap -- the schedule is a single cached request for the whole season -- which is what lets
    ingestion decide whether a race is already on disk before paying for the session download.
    Event names are matched the way FastF1 matches them, so ``"Bahrain"`` resolves.
    """
    fastf1 = _configure(cache_dir)
    row = fastf1.get_event(season, event)
    return _scheduled_from_row(season, row)


def _scheduled_from_row(season: int, row: Any) -> ScheduledRace:
    return ScheduledRace(
        season=season,
        round=int(_scalar(row, "RoundNumber")),
        event=str(_scalar(row, "EventName")),
        circuit=str(_scalar(row, "Location")),
        country=str(_scalar(row, "Country")),
        event_date=_scalar(row, "EventDate").date(),
    )


def fetch_race(season: int, event: str | int, cache_dir: Path) -> RawRace:
    """Load one race session from FastF1's cache, downloading it if absent.

    Args:
        season: Championship year.
        event: Event name (``"Bahrain Grand Prix"``) or round number.
        cache_dir: Directory FastF1 caches raw API responses in. Never cleaned by ``make clean``:
            re-downloading a full season is slow and puts avoidable load on the timing API.
    """
    fastf1 = _configure(cache_dir)
    from fastf1.req import (  # noqa: PLC0415 - lazy, as above
        RateLimitExceededError,
    )

    try:
        session = fastf1.get_session(season, event, "R")
        session.load(telemetry=False, weather=True, messages=False)
    except RateLimitExceededError as error:
        raise RateLimitError(str(error)) from error

    laps_pandas = session.laps.reset_index(drop=True)
    weather = laps_pandas.get_weather_data().reset_index(drop=True)
    for column in ("AirTemp", "TrackTemp", "Rainfall"):
        laps_pandas[column] = weather[column].to_numpy()

    schedule_row = session.event
    scheduled_laps = session.total_laps
    total_laps = (
        int(scheduled_laps) if scheduled_laps is not None else int(laps_pandas["LapNumber"].max())
    )

    rainfall = session.weather_data["Rainfall"]
    rainfall_fraction = float(rainfall.mean()) if len(rainfall) else 0.0
    if math.isnan(rainfall_fraction):
        rainfall_fraction = 0.0

    scheduled = _scheduled_from_row(season, schedule_row)
    return RawRace(
        event=EventMeta(**scheduled.model_dump(), total_laps=total_laps),
        laps=pl.from_pandas(laps_pandas[list(RAW_LAP_COLUMNS)]),
        results=pl.from_pandas(
            session.results[["Abbreviation", "GridPosition"]].reset_index(drop=True)
        ),
        rainfall_fraction=rainfall_fraction,
    )

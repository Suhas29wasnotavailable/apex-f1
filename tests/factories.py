"""Builders for the frames the data layer passes around.

Tests construct races rather than download them. A factory keeps each test's setup down to the
one or two fields it is actually about, so a test named ``test_lap_after_safety_car`` shows only
the safety car, and a new schema column does not require editing thirty test bodies.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from apex.config import AppSettings, PathSettings
from apex.data.schema import LAP_SCHEMA, EventMeta, conform_lap_frame
from apex.data.sources import RawRace

Row = Mapping[str, Any]

CLEAN_LAP: dict[str, Any] = {
    "season": 2023,
    "round": 1,
    "event": "Test Grand Prix",
    "circuit": "Testville",
    "country": "Testland",
    "driver": "VER",
    "driver_number": "1",
    "team": "Test Team",
    "lap": 5,
    "stint": 1,
    "compound": "MEDIUM",
    "tyre_life": 5,
    "fresh_tyre": True,
    "lap_time_s": 90.0,
    "lap_start_s": 400.0,
    "lap_end_s": 490.0,
    "sector1_s": 30.0,
    "sector2_s": 30.0,
    "sector3_s": 30.0,
    "speed_trap_kph": 300.0,
    "position": 1,
    "grid_position": 1,
    "track_status": "1",
    "is_accurate": True,
    "is_pit_in": False,
    "is_pit_out": False,
    "pit_in_s": None,
    "pit_out_s": None,
    "deleted": False,
    "fastf1_generated": False,
    "air_temp_c": 25.0,
    "track_temp_c": 35.0,
    "rainfall": False,
}
"""A lap that survives every exclusion rule. Tests override only what they are testing."""


def lap(**overrides: Any) -> dict[str, Any]:
    """One canonical lap, clean unless overridden."""
    unknown = set(overrides) - set(CLEAN_LAP)
    if unknown:
        raise KeyError(f"not lap columns: {sorted(unknown)}")
    return {**CLEAN_LAP, **overrides}


def stint(count: int, *, first_lap: int = 2, **overrides: Any) -> list[dict[str, Any]]:
    """A run of consecutive laps on one set of tyres, all clean unless overridden."""
    return [
        lap(lap=first_lap + offset, tyre_life=1 + offset, **overrides) for offset in range(count)
    ]


def lap_frame(rows: Sequence[Row]) -> pl.DataFrame:
    """Canonical lap table from row dictionaries; correctly typed and empty for no rows."""
    if not rows:
        return pl.DataFrame(schema=LAP_SCHEMA)
    # Scan every row before settling on a type. Columns like ``pit_in_s`` are null for the first
    # few hundred laps of a race and then suddenly are not, which the default sample would miss.
    return conform_lap_frame(pl.DataFrame([dict(row) for row in rows], infer_schema_length=None))


RAW_LAP: dict[str, Any] = {
    "Driver": "VER",
    "DriverNumber": "1",
    "Team": "Test Team",
    "LapNumber": 5.0,
    "Stint": 1.0,
    "Compound": "MEDIUM",
    "TyreLife": 5.0,
    "FreshTyre": True,
    "LapTime": dt.timedelta(seconds=90.5),
    "LapStartTime": dt.timedelta(seconds=400),
    "Time": dt.timedelta(seconds=490.5),
    "Position": 1.0,
    "TrackStatus": "1",
    "IsAccurate": True,
    "PitInTime": None,
    "PitOutTime": None,
    "Deleted": False,
    "FastF1Generated": False,
    "Sector1Time": dt.timedelta(seconds=30),
    "Sector2Time": dt.timedelta(seconds=30),
    "Sector3Time": dt.timedelta(seconds=30.5),
    "SpeedST": 300.0,
    "AirTemp": 25.0,
    "TrackTemp": 35.0,
    "Rainfall": False,
}
"""One lap in FastF1's own vocabulary, as :func:`apex.data.sources.to_lap_frame` receives it."""


def raw_lap(**overrides: Any) -> dict[str, Any]:
    unknown = set(overrides) - set(RAW_LAP)
    if unknown:
        raise KeyError(f"not FastF1 lap columns: {sorted(unknown)}")
    return {**RAW_LAP, **overrides}


def raw_race(
    rows: Sequence[Row] | None = None,
    *,
    rainfall_fraction: float = 0.0,
    grid: Mapping[str, int] | None = None,
    **event_overrides: Any,
) -> RawRace:
    """A loaded race session in FastF1's vocabulary, ready for translation."""
    laps = [dict(row) for row in (rows if rows is not None else [raw_lap()])]
    drivers = grid if grid is not None else {"VER": 1}
    event = EventMeta(
        **{
            "season": 2023,
            "round": 1,
            "event": "Test Grand Prix",
            "circuit": "Testville",
            "country": "Testland",
            "event_date": dt.date(2023, 3, 5),
            "total_laps": 57,
            **event_overrides,
        }
    )
    return RawRace(
        event=event,
        laps=pl.DataFrame(laps),
        results=pl.DataFrame(
            {
                "Abbreviation": list(drivers),
                "GridPosition": [float(value) for value in drivers.values()],
            }
        ),
        rainfall_fraction=rainfall_fraction,
    )


def settings_in(root: Any, **overrides: Any) -> AppSettings:
    """Settings whose generated artefacts live under a temporary directory."""
    return AppSettings(
        paths=PathSettings(
            fastf1_cache=root / "fastf1_cache",
            processed=root / "processed",
            params=root / "params",
            figures=root / "figures",
            mlruns=root / "mlruns",
        ),
        **overrides,
    )

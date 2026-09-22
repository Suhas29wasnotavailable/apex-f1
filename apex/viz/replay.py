"""Turning a simulated race into something a browser can animate.

The simulator reports one row per car per lap: the time it crossed the line, where it was running,
what tyre it was on. A replay needs a car's position at *any* instant, which is a different
question, and this module packages the answer rather than computing it -- the interpolation happens
in the page, sixty times a second, where it belongs.

What that needs is each car's crossing times as a cumulative array. Given a wall-clock instant, the
page finds which lap a car is on by walking that array, takes its fractional progress through the
lap, converts time-fraction to distance-fraction through the circuit's motion profile, and reads a
point off the centreline. Nothing is precomputed per frame, so the payload stays small -- a 20-car,
60-lap race is about 1,200 records -- and scrubbing backwards costs what playing forwards costs.

**Teams and colours come from the ingested data, not from the simulator.** A scenario built from a
typed-in grid carries no team, because a team is not a fitted parameter and the simulator has no use
for one. But a replay drawing twenty identical dots is unreadable, so the driver-to-team mapping is
looked up from the most recent season on disk and each team gets a colour. Those colours are
presentation and nothing else: no number in the simulation reads them.
"""

from __future__ import annotations

from typing import Any, Final

import polars as pl

from apex.config import AppSettings, get_settings
from apex.data import load_laps
from apex.sim.trace import RaceTrace
from apex.viz.track import TrackGeometry, track_payload

__all__ = ["COMPOUND_COLOURS", "TEAM_COLOURS", "replay_payload"]

TEAM_COLOURS: Final[dict[str, str]] = {
    "Red Bull Racing": "#3671C6",
    "Ferrari": "#E8002D",
    "Mercedes": "#27F4D2",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine": "#FF87BC",
    "Williams": "#64C4FF",
    "RB": "#6692FF",
    "Racing Bulls": "#6692FF",
    "Kick Sauber": "#52E252",
    "Haas F1 Team": "#B6BABD",
    "AlphaTauri": "#5E8FAA",
    "Alfa Romeo": "#C92D4B",
}
"""Team colours, for telling twenty dots apart. Presentation only: nothing fitted reads this."""

_FALLBACK_COLOURS: Final[tuple[str, ...]] = (
    "#3671C6",
    "#E8002D",
    "#27F4D2",
    "#FF8000",
    "#229971",
    "#FF87BC",
    "#64C4FF",
    "#6692FF",
    "#52E252",
    "#B6BABD",
)
"""Assigned in grid order to any car whose team is unknown, so a made-up field is still readable."""

COMPOUND_COLOURS: Final[dict[str, str]] = {
    "SOFT": "#F5323C",
    "MEDIUM": "#F5D02C",
    "HARD": "#EDEDED",
    "INTERMEDIATE": "#41D26E",
    "WET": "#3B73DE",
}
"""Pirelli's own sidewall colours, which is what a viewer expects to see."""


def _teams_from_disk(settings: AppSettings, season: int | None = None) -> dict[str, str]:
    """Driver-to-team mapping from the most recent ingested season.

    Returns an empty mapping when nothing is ingested, which is not an error: a replay of a
    hypothetical grid is still worth watching, it just gets fallback colours.
    """
    laps = load_laps(settings, dry_only=False)
    if laps.height == 0:
        return {}
    latest = season if season is not None else int(laps["season"].max())  # type: ignore[arg-type]
    recent = laps.filter(pl.col("season") == latest)
    if recent.height == 0:
        recent = laps
    pairs = (
        recent.group_by("driver")
        .agg(pl.col("team").drop_nulls().mode().first().alias("team"))
        .drop_nulls("team")
    )
    return {str(row["driver"]): str(row["team"]) for row in pairs.iter_rows(named=True)}


def _lap_entries(records: list[Any]) -> list[dict[str, Any]]:
    """One entry per lap, with the crossing times made exactly self-consistent.

    ``end`` is cumulative race time at the line and is the only quantity the animation strictly
    needs; the page recovers where a lap began by subtracting ``time`` from it. Rounding the two
    independently breaks that subtraction by a millisecond, which is invisible on screen and is
    still a payload whose own arithmetic does not close. So the ends are rounded first and each
    duration is derived as the gap between them, which makes the identity exact by construction.
    """
    entries: list[dict[str, Any]] = []
    previous = round(records[0].elapsed_s - records[0].lap_time_s, 3) if records else 0.0
    for record in records:
        end = round(record.elapsed_s, 3)
        entries.append(
            {
                "lap": record.lap,
                "end": end,
                "time": round(end - previous, 3),
                "pos": record.position,
                "compound": record.compound,
                "age": record.tyre_age,
                "gapAhead": round(record.gap_ahead_s, 3),
                "gapLeader": round(record.gap_to_leader_s, 3),
                "pit": record.is_pit_lap,
                "held": record.held_up,
                "passed": record.overtook,
            }
        )
        previous = end
    return entries


def _colour_for(team: str, slot: int) -> str:
    if team in TEAM_COLOURS:
        return TEAM_COLOURS[team]
    return _FALLBACK_COLOURS[slot % len(_FALLBACK_COLOURS)]


def replay_payload(
    trace: RaceTrace,
    geometry: TrackGeometry,
    settings: AppSettings | None = None,
    *,
    title: str = "",
) -> dict[str, Any]:
    """Package one simulated race for the front end.

    Args:
        trace: The simulated race.
        geometry: The circuit's geometry, from :func:`~apex.viz.track.load_track`.
        settings: Configuration; the process-wide settings by default.
        title: Heading for the page. Derived from the scenario when empty.

    Returns:
        A JSON-serialisable payload holding the track, the flags, and every car's crossing times.
    """
    settings = settings or get_settings()
    scenario = trace.scenario
    teams = _teams_from_disk(settings)

    # Laps grouped per driver, ordered, so the page can binary-search crossing times.
    by_driver: dict[str, list[Any]] = {}
    for record in trace.laps:
        by_driver.setdefault(record.driver, []).append(record)

    cars: list[dict[str, Any]] = []
    for slot, entrant in enumerate(scenario.starting_order()):
        records = sorted(by_driver.get(entrant.driver, []), key=lambda r: r.lap)
        team = entrant.team or teams.get(entrant.driver, "")
        result = trace.result(entrant.driver)
        laps = _lap_entries(records)
        cars.append(
            {
                "driver": entrant.driver,
                "team": team,
                "colour": _colour_for(team, slot),
                "grid": entrant.grid_position,
                "finish": result.finish_position,
                "stops": result.stops,
                "overtakes": result.overtakes,
                "lapsHeldUp": result.laps_held_up,
                "paceS": round(entrant.pace_s, 3),
                "laps": laps,
            }
        )

    flags = {str(lap): "sc" for lap in trace.safety_car_laps}
    flags.update({str(lap): "vsc" for lap in trace.virtual_safety_car_laps})

    finish = max((car["laps"][-1]["end"] for car in cars if car["laps"]), default=0.0)
    return {
        "title": title or f"{scenario.event or scenario.circuit} {scenario.season}",
        "circuit": scenario.circuit,
        "season": scenario.season,
        "totalLaps": scenario.total_laps,
        "referenceLapS": round(scenario.anchor.reference_lap_s, 3),
        "anchorMode": str(scenario.anchor.mode),
        "runId": trace.seed_run_id,
        "raceDurationS": round(finish, 3),
        "track": track_payload(geometry),
        "flags": flags,
        "compoundColours": COMPOUND_COLOURS,
        "cars": cars,
    }

"""Real track geometry, and how a car moves around it.

The simulator produces lap times. A replay needs positions, and the conversion between them is the
one piece of geometry this project cannot fit from a lap table: where the track actually goes.

Both halves come from the same source: the position telemetry of one fast lap, which FastF1 exposes
as X/Y coordinates alongside the distance and time at each sample.

**The centreline** is resampled to a fixed number of points evenly spaced by distance. That choice
makes the front end's job trivial -- a car 40% of the way around the lap is at index 0.4 of the way
along the array, with no arc-length search. Coordinates are normalised into a unit box with the
aspect ratio preserved, so the page never needs to know that Monza is measured in metres and centred
on nothing in particular.

**The motion profile** is the part that would be easy to get wrong. A car does not travel at a
constant speed around a lap, so mapping elapsed time linearly onto distance would slide cars through
corners at full speed and bunch them absurdly on the straights. The profile records, for evenly
spaced fractions of a lap's *time*, the fraction of its *distance* covered -- read off a real fast
lap. Every simulated car follows that shape scaled by its own lap time, so a slow lap looks like a
slow version of a real lap rather than a uniform crawl.

Geometry is cached to disk per circuit after the first fetch, because it is a property of the place
rather than of any race, and because fetching it is the only thing in the visualisation layer that
touches the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from apex.config import AppSettings, get_settings
from apex.data import load_laps
from apex.data.schema import canonical_circuit, event_slug

__all__ = [
    "CENTRELINE_POINTS",
    "PROFILE_SAMPLES",
    "TrackGeometry",
    "load_track",
    "track_path",
]

CENTRELINE_POINTS = 480
"""Points on the resampled centreline. Enough that a 5 km lap is described every ten metres, which
is finer than a car moves between animation frames."""

PROFILE_SAMPLES = 200
"""Samples in the time-to-distance profile. At a 90-second lap that is one sample every 0.45 s."""

_MIN_TELEMETRY_SAMPLES = 8
"""Below this there is no lap to trace, only a handful of points."""


class TrackGeometry(BaseModel):
    """One circuit's shape, and the shape of a lap around it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    circuit: str
    season: int
    """Season the geometry was taken from. Circuits are occasionally re-profiled, so which year this
    describes is part of what it is."""

    lap_distance_m: float = Field(gt=0.0)

    centreline: tuple[tuple[float, float], ...] = Field(min_length=8)
    """Points evenly spaced by distance, normalised into a unit box with the aspect ratio kept. The
    first point is the start/finish line and the loop is closed implicitly."""

    aspect: float = Field(gt=0.0)
    """Width divided by height of the original coordinates, so the page can size its canvas."""

    distance_profile: tuple[float, ...] = Field(min_length=8)
    """Fraction of lap distance covered, at evenly spaced fractions of lap time. Starts at 0.0 and
    ends at 1.0. This is what makes a car slow down for a corner."""

    reference_lap_s: float = Field(gt=0.0)
    """Lap time the profile was measured from, recorded so the sampling interval is recoverable."""

    def point_at(self, distance_fraction: float) -> tuple[float, float]:
        """The centreline point at a fraction of the way around the lap."""
        wrapped = distance_fraction % 1.0
        index = int(wrapped * len(self.centreline)) % len(self.centreline)
        return self.centreline[index]

    def distance_at(self, time_fraction: float) -> float:
        """Fraction of the lap's distance covered after a fraction of its time."""
        clamped = min(max(time_fraction, 0.0), 1.0)
        position = clamped * (len(self.distance_profile) - 1)
        low = int(position)
        if low >= len(self.distance_profile) - 1:
            return self.distance_profile[-1]
        weight = position - low
        return self.distance_profile[low] * (1.0 - weight) + self.distance_profile[low + 1] * weight

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json() + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> TrackGeometry:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def track_path(settings: AppSettings, circuit: str) -> Path:
    return settings.paths.tracks / f"{event_slug(circuit)}.json"


def _resample_centreline(
    x: np.ndarray, y: np.ndarray, fraction: np.ndarray
) -> tuple[tuple[tuple[float, float], ...], float]:
    """Resample a traced lap onto points evenly spaced around it, normalised to a unit box.

    Interpolating against the *distance* fraction rather than the sample index is what makes the
    spacing even: telemetry samples arrive on a clock, so they crowd together in slow corners and
    spread out on the straights.
    """
    targets = np.linspace(0.0, 1.0, CENTRELINE_POINTS, endpoint=False)
    resampled_x = np.interp(targets, fraction, x)
    resampled_y = np.interp(targets, fraction, y)

    width = float(resampled_x.max() - resampled_x.min())
    height = float(resampled_y.max() - resampled_y.min())
    span = max(width, height)
    if span <= 0.0:  # pragma: no cover - a circuit with no extent is not a circuit
        raise ValueError("track coordinates have no extent")

    # Centre on the origin, then scale the larger axis to 1. Scaling each axis independently would
    # stretch the circuit into whatever box the page happens to be.
    centred_x = (resampled_x - (resampled_x.max() + resampled_x.min()) / 2.0) / span
    centred_y = (resampled_y - (resampled_y.max() + resampled_y.min()) / 2.0) / span
    points = tuple(
        (round(float(px), 5), round(float(py), 5))
        for px, py in zip(centred_x, centred_y, strict=True)
    )
    return points, (width / height if height > 0 else 1.0)


def _distance_profile(
    time_fraction: np.ndarray, distance_fraction: np.ndarray
) -> tuple[float, ...]:
    """Fraction of lap distance against fraction of lap time, on an even time grid."""
    grid = np.linspace(0.0, 1.0, PROFILE_SAMPLES)
    profile = np.interp(grid, time_fraction, distance_fraction)
    profile[0], profile[-1] = 0.0, 1.0
    # Enforce monotonicity: telemetry can jitter backwards by a metre, and a car must never reverse.
    monotonic = np.maximum.accumulate(profile)
    return tuple(round(float(value), 6) for value in monotonic)


def build_track(circuit: str, season: int, settings: AppSettings | None = None) -> TrackGeometry:
    """Extract one circuit's geometry from real position telemetry.

    Args:
        circuit: Circuit name as the parameter file spells it.
        season: Season to take the lap from.
        settings: Configuration; the process-wide settings by default.

    Raises:
        RuntimeError: if FastF1 is not installed, or the session has no position telemetry.

    Note:
        Touches the network on a first call for a circuit. The result is cached to disk, so it is
        paid once per circuit and never again.
    """
    settings = settings or get_settings()
    try:
        import fastf1  # noqa: PLC0415 - optional extra, imported where it is needed
    except ImportError as error:  # pragma: no cover - exercised by the extras, not the test suite
        raise RuntimeError("track geometry needs FastF1: install the `data` extra") from error

    fastf1.Cache.enable_cache(str(settings.paths.fastf1_cache))
    session = _session_for(fastf1, circuit, season, settings)
    session.load(telemetry=True, weather=False, messages=False)

    fastest = session.laps.pick_fastest()
    if fastest is None or fastest.empty:  # pragma: no cover - needs a session with no timed lap
        raise RuntimeError(f"no timed lap at {circuit} in {season} to trace the track from")
    telemetry = fastest.get_telemetry()

    usable = telemetry.dropna(subset=["X", "Y", "Distance", "Time"])
    if len(usable) < _MIN_TELEMETRY_SAMPLES:  # pragma: no cover - needs almost no telemetry
        raise RuntimeError(f"not enough position telemetry at {circuit} in {season}")

    distance = usable["Distance"].to_numpy(dtype=float)
    lap_distance = float(distance.max())
    elapsed = usable["Time"].dt.total_seconds().to_numpy(dtype=float)
    lap_time = float(elapsed.max())

    distance_fraction = np.clip(distance / lap_distance, 0.0, 1.0)
    time_fraction = np.clip(elapsed / lap_time, 0.0, 1.0)

    centreline, aspect = _resample_centreline(
        usable["X"].to_numpy(dtype=float), usable["Y"].to_numpy(dtype=float), distance_fraction
    )
    return TrackGeometry(
        circuit=canonical_circuit(circuit),
        season=season,
        lap_distance_m=lap_distance,
        centreline=centreline,
        aspect=aspect,
        distance_profile=_distance_profile(time_fraction, distance_fraction),
        reference_lap_s=lap_time,
    )


def _session_for(fastf1: Any, circuit: str, season: int, settings: AppSettings) -> Any:
    """Find the event at a circuit in a season, by circuit rather than by event name.

    Event names move around — the same circuit has hosted differently-named Grands Prix — so the
    schedule is searched by location, which is the same key the parameter file uses.
    """
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    wanted = canonical_circuit(circuit).casefold()
    for _, event in schedule.iterrows():
        if canonical_circuit(str(event["Location"])).casefold() == wanted:
            return fastf1.get_session(season, int(event["RoundNumber"]), "R")
    del settings
    raise RuntimeError(f"no {season} event found at circuit {circuit!r}")


def load_track(
    circuit: str,
    season: int | None = None,
    settings: AppSettings | None = None,
    *,
    refresh: bool = False,
) -> TrackGeometry:
    """Return a circuit's geometry, from the cache when possible.

    Args:
        circuit: Circuit name as the parameter file spells it.
        season: Season to trace from. The most recent ingested season at that circuit by default,
            since a circuit's current layout is the one worth drawing.
        settings: Configuration; the process-wide settings by default.
        refresh: Re-fetch and overwrite the cached geometry.
    """
    settings = settings or get_settings()
    path = track_path(settings, circuit)
    if path.is_file() and not refresh:
        return TrackGeometry.read(path)

    geometry = build_track(circuit, season or _latest_season(circuit, settings), settings)
    geometry.write(path)
    return geometry


def _latest_season(circuit: str, settings: AppSettings) -> int:
    """The most recent ingested season that raced at this circuit."""
    laps = load_laps(settings, dry_only=False)
    wanted = canonical_circuit(circuit)
    seasons = (
        laps.filter(laps["circuit"] == wanted)["season"].unique().to_list() if laps.height else []
    )
    if not seasons:
        raise RuntimeError(
            f"no ingested race at circuit {circuit!r}; "
            f"pass a season explicitly, or run `apex ingest` first"
        )
    return int(max(seasons))


def track_payload(geometry: TrackGeometry) -> dict[str, Any]:
    """The geometry as the front end wants it: compact, and already normalised."""
    return {
        "circuit": geometry.circuit,
        "season": geometry.season,
        "lapDistanceM": round(geometry.lap_distance_m, 1),
        "aspect": round(geometry.aspect, 4),
        "centreline": [[x, y] for x, y in geometry.centreline],
        "distanceProfile": list(geometry.distance_profile),
    }

"""The replay: geometry that matches the circuit, and a payload the page can actually animate.

Nothing here touches the network. Track extraction is the one part of the visualisation layer that
does, so the geometry it produces is constructed directly and the code that consumes it is tested
against that — which is the part with logic in it anyway.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

import pytest

from apex.sim import RaceSimulator, ReactiveRules, from_grid
from apex.viz.page import render_page, write_page
from apex.viz.replay import COMPOUND_COLOURS, TEAM_COLOURS, replay_payload
from apex.viz.track import CENTRELINE_POINTS, PROFILE_SAMPLES, TrackGeometry, track_payload
from tests.test_sim import CIRCUIT, bundle, parameters, settings


def circle(points: int = CENTRELINE_POINTS) -> tuple[tuple[float, float], ...]:
    """A circular circuit, so a point's angle says exactly how far round the lap it is."""
    return tuple(
        (
            round(math.cos(2 * math.pi * i / points) / 2, 6),
            round(math.sin(2 * math.pi * i / points) / 2, 6),
        )
        for i in range(points)
    )


def geometry(*, profile: tuple[float, ...] | None = None) -> TrackGeometry:
    """A synthetic circuit. Constant speed unless a profile is supplied."""
    return TrackGeometry(
        circuit=CIRCUIT,
        season=2024,
        lap_distance_m=5000.0,
        centreline=circle(),
        aspect=1.0,
        distance_profile=profile
        or tuple(i / (PROFILE_SAMPLES - 1) for i in range(PROFILE_SAMPLES)),
        reference_lap_s=90.0,
    )


# --------------------------------------------------------------------------- geometry


def test_a_point_is_found_all_the_way_round_the_lap() -> None:
    track = geometry()
    start = track.point_at(0.0)
    assert track.point_at(1.0) == start, "a full lap returns to the start"
    assert track.point_at(2.5) == track.point_at(0.5), "distance past the line wraps"
    assert track.point_at(-0.25) == track.point_at(0.75), "and wraps backwards too"


def test_points_are_evenly_spaced_around_the_lap() -> None:
    """Even spacing is what lets the page index into the array with no arc-length search.

    The tolerance is loose enough to absorb the six-decimal rounding the payload stores points at,
    and far tighter than the spacing itself.
    """
    points = geometry().centreline
    steps = [math.dist(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
    assert max(steps) - min(steps) < 1e-5
    assert min(steps) > 0.0


def test_a_constant_speed_profile_maps_time_straight_onto_distance() -> None:
    track = geometry()
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert track.distance_at(fraction) == pytest.approx(fraction, abs=1e-6)


def test_a_real_profile_makes_a_car_cover_ground_unevenly() -> None:
    """The point of the profile: a car must not slide through a corner at straight-line speed."""
    # Half the lap's distance covered in the first quarter of its time, then the car slows.
    fast_then_slow = tuple(
        min(1.0, 2.0 * (i / (PROFILE_SAMPLES - 1)) ** 0.5) for i in range(PROFILE_SAMPLES)
    )
    track = geometry(profile=fast_then_slow)
    assert track.distance_at(0.25) > 0.25
    early = track.distance_at(0.25) - track.distance_at(0.0)
    late = track.distance_at(1.0) - track.distance_at(0.75)
    assert early > late


def test_a_profile_never_runs_backwards() -> None:
    track = geometry()
    values = [track.distance_at(i / 50) for i in range(51)]
    assert values == sorted(values)


def test_geometry_round_trips_through_disk(tmp_path: Path) -> None:
    track = geometry()
    path = tmp_path / "tracks" / "testville.json"
    track.write(path)
    assert TrackGeometry.read(path) == track


def test_the_payload_carries_what_the_page_needs() -> None:
    payload = track_payload(geometry())
    assert set(payload) == {
        "circuit",
        "season",
        "lapDistanceM",
        "aspect",
        "centreline",
        "distanceProfile",
    }
    assert len(payload["centreline"]) == CENTRELINE_POINTS
    assert payload["distanceProfile"][0] == 0.0
    assert payload["distanceProfile"][-1] == 1.0


# --------------------------------------------------------------------------- replay payload


def race_payload(*, cars: int = 6, laps: int = 20) -> dict[str, object]:
    config = settings()
    params = parameters(safety_car_probability=0.25)
    simulator = RaceSimulator(params, config)
    scenario = from_grid(
        params,
        circuit=CIRCUIT,
        season=2024,
        total_laps=laps,
        grid=[f"D{index:02d}" for index in range(1, cars + 1)],
        settings=config,
    )
    trace = simulator.run(scenario, ReactiveRules(target_stops=1), bundle("viz"))
    return replay_payload(trace, geometry(), config, title="Test Grand Prix")


def test_every_car_has_every_lap() -> None:
    payload = race_payload(cars=6, laps=20)
    assert len(payload["cars"]) == 6  # type: ignore[arg-type]
    for car in payload["cars"]:  # type: ignore[attr-defined]
        assert len(car["laps"]) == 20
        assert [entry["lap"] for entry in car["laps"]] == list(range(1, 21))


def test_crossing_times_only_ever_increase() -> None:
    """The page derives a car's lap by walking this array, so it has to be sorted."""
    for car in race_payload()["cars"]:  # type: ignore[attr-defined]
        ends = [entry["end"] for entry in car["laps"]]
        assert ends == sorted(ends)
        assert all(entry["time"] > 0 for entry in car["laps"])


def test_a_lap_time_is_the_step_between_crossings() -> None:
    """The page reconstructs where a lap started by subtracting its duration from its end."""
    for car in race_payload()["cars"]:  # type: ignore[attr-defined]
        laps = car["laps"]
        for previous, current in itertools.pairwise(laps):
            assert current["end"] - current["time"] == pytest.approx(previous["end"], abs=1e-6)


def test_the_race_lasts_until_the_last_car_finishes() -> None:
    payload = race_payload()
    last = max(car["laps"][-1]["end"] for car in payload["cars"])  # type: ignore[attr-defined]
    assert payload["raceDurationS"] == pytest.approx(last, abs=1e-6)


def test_flags_are_keyed_by_lap_for_the_timeline() -> None:
    payload = race_payload(laps=40)
    flags = payload["flags"]
    assert flags, "this fixture should produce a neutralisation"
    for lap, kind in flags.items():  # type: ignore[attr-defined]
        assert lap.isdigit()
        assert kind in {"sc", "vsc"}


def test_every_car_gets_a_distinguishable_colour() -> None:
    """Twenty identical dots is not a replay."""
    payload = race_payload(cars=10)
    colours = [car["colour"] for car in payload["cars"]]  # type: ignore[attr-defined]
    assert len(set(colours)) == len(colours)
    assert all(colour.startswith("#") for colour in colours)


def test_known_teams_keep_their_own_colour() -> None:
    assert TEAM_COLOURS["Ferrari"] != TEAM_COLOURS["Mercedes"]
    assert set(COMPOUND_COLOURS) >= {"SOFT", "MEDIUM", "HARD"}


# --------------------------------------------------------------------------- the page


def test_the_page_embeds_the_payload_and_parses_back() -> None:
    payload = race_payload()
    html = render_page(payload)
    match = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    assert match, "the payload script block must survive rendering"
    recovered = json.loads(match.group(1).replace("<\\/", "</"))
    assert recovered["totalLaps"] == payload["totalLaps"]
    assert len(recovered["cars"]) == len(payload["cars"])  # type: ignore[arg-type]


def test_no_placeholder_survives_into_the_page() -> None:
    html = render_page(race_payload())
    assert "__PAYLOAD__" not in html
    assert "__TITLE__" not in html
    assert "Test Grand Prix" in html


def test_a_closing_script_tag_in_the_data_cannot_break_out() -> None:
    """A driver named `</script>` would otherwise end the block and spill the race into the page.

    Fanciful for a driver code and not fanciful for an event name typed by a user, and the failure
    is silent: the page simply stops working.
    """
    payload = race_payload()
    payload["title"] = "</script><h1>escaped</h1>"
    html = render_page(payload)
    body = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    assert body is not None
    assert "<h1>escaped</h1>" not in body.group(1)
    assert json.loads(body.group(1).replace("<\\/", "</"))["title"] == payload["title"]


def test_the_page_is_written_as_one_self_contained_file(tmp_path: Path) -> None:
    """A replay that needs a server is a replay nobody watches."""
    destination = write_page(race_payload(), tmp_path / "web" / "replay.html")
    html = destination.read_text(encoding="utf-8")
    assert destination.is_file()
    assert html.startswith("<!doctype html>")
    assert "src=" not in html.split('<script id="payload"')[0], "no external assets"

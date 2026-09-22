"""Watching the race: real track geometry, and a self-contained browser replay.

The simulator reports lap times; this turns them into motion. Track geometry comes from real
position telemetry, the conversion from elapsed time to a point on the track follows a real lap's
speed profile, and the output is one HTML file that opens without a server.
"""

from __future__ import annotations

from apex.viz.page import TEMPLATE_PATH, render_page, write_page
from apex.viz.replay import COMPOUND_COLOURS, TEAM_COLOURS, replay_payload
from apex.viz.track import TrackGeometry, build_track, load_track, track_path, track_payload

__all__ = [
    "COMPOUND_COLOURS",
    "TEAM_COLOURS",
    "TEMPLATE_PATH",
    "TrackGeometry",
    "build_track",
    "load_track",
    "render_page",
    "replay_payload",
    "track_path",
    "track_payload",
    "write_page",
]

"""The race simulator.

Reads nothing but ``params/params.json`` and a scenario, and produces a lap-by-lap race. The
public surface is deliberately small: build a :class:`~apex.sim.scenario.RaceScenario`, hand it to
:class:`~apex.sim.race.RaceSimulator` with a strategy and a seed bundle, and read the
:class:`~apex.sim.trace.RaceTrace` that comes back.
"""

from __future__ import annotations

from apex.sim.race import RaceSimulator
from apex.sim.scenario import Anchor, AnchorMode, Entrant, RaceScenario, from_grid, from_race
from apex.sim.strategies import FixedStops, ReactiveRules
from apex.sim.strategy import CarView, PitCall, Strategy, TrackPhase
from apex.sim.trace import LapRecord, RaceResult, RaceTrace, StintRecord

__all__ = [
    "Anchor",
    "AnchorMode",
    "CarView",
    "Entrant",
    "FixedStops",
    "LapRecord",
    "PitCall",
    "RaceResult",
    "RaceScenario",
    "RaceSimulator",
    "RaceTrace",
    "ReactiveRules",
    "StintRecord",
    "Strategy",
    "TrackPhase",
    "from_grid",
    "from_race",
]

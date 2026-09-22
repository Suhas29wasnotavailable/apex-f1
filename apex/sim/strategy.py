"""The decision interface: what a strategist may see, and what it may do.

This is the seam the whole project's central claim runs through. Rule-based strategies, dynamic
programming, MCTS and a learned policy all decide the same thing from the same information, so a
comparison between them measures the method and not the interface each one was given.

Two design rules, both load-bearing.

**A strategy sees a view, never the race.** :class:`CarView` is assembled per car per lap and
contains only what a real pit wall knows: its own tyres, its own position, the gaps immediately
around it, the track state, and the regulations it still has to satisfy. It cannot see other cars'
tyre ages, the safety car that has not happened yet, or the lap times about to be drawn. An agent
handed the race state could learn to exploit the simulator instead of learning strategy, and the
result would look excellent and mean nothing.

**The action is a pit call, and only a pit call.** Not a target lap time, not a defensive line.
Everything the fitted parameters support is a consequence of tyres and track position, so the
decision space is exactly: stay out, or come in for one of the available compounds. That keeps the
MDP small enough for dynamic programming to solve exactly, which is the comparison the project's
claim demands -- an RL agent that beats a rule-based baseline has proved very little if the problem
was solvable in closed form all along.

:class:`CarView` is also the observation space P4's policy network reads. It is kept flat, numeric
where possible, and free of any quantity that would need the race's future to compute.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["CLEAN_AIR_GAP_S", "CarView", "PitCall", "Strategy", "TrackPhase"]

CLEAN_AIR_GAP_S = 2.0
"""Gap beyond which the car ahead costs effectively nothing.

It is the dirty-air time constant from the pace model: the penalty decays as ``exp(-gap / 2)``, so
at two seconds a car keeps under a fifth of the penalty it suffers directly behind. Offered as a
convenience for strategies that want a yes-or-no answer about traffic; the simulator itself always
uses the continuous fitted penalty and never this threshold."""


class TrackPhase(enum.StrEnum):
    """What the flags are showing."""

    GREEN = "green"
    SAFETY_CAR = "safety_car"
    """Field bunched behind the safety car; a stop costs far less and passing is forbidden."""

    VIRTUAL_SAFETY_CAR = "virtual_safety_car"
    """Everybody slowed in place. No bunching, so a stop keeps most of its cost."""

    @property
    def is_neutralised(self) -> bool:
        return self is not TrackPhase.GREEN


@dataclass(frozen=True, slots=True)
class PitCall:
    """The one action available: come in, and fit this compound."""

    compound: str


@dataclass(frozen=True, slots=True)
class CarView:
    """Everything one car's strategist may see at the start of one lap.

    Attributes:
        driver: The car being decided for.
        lap: The lap about to be run, counting from one.
        total_laps: Scheduled race distance.
        position: Current running position, one being the lead.
        compound: Compound currently fitted.
        tyre_age: Laps this set has already completed. Zero on a set that has not run a lap, which
            is why a car's first lap of a stint is run at age one.
        stops: Stops made so far.
        gap_ahead_s: Gap to the car in front at the line, or ``None`` for the leader.
        gap_behind_s: Gap to the car behind, or ``None`` for the last car.
        phase: Track state for the lap about to be run.
        phase_laps_remaining: Laps this neutralisation is expected to last, zero under green. An
            expectation rather than a promise: the simulator draws the duration when the
            neutralisation begins, so this is knowledge a real pit wall also has approximately.
        compounds_used: Compounds this car has run in this race, including the current one.
        available_compounds: Compounds that may be fitted.
        must_still_use_another_compound: True when the two-compound regulation is not yet satisfied
            and the race is not over, so a strategy that never stops would be disqualified.
        laps_remaining: Laps left after this one.
        race_progress: Fraction of the race distance already completed, in ``[0, 1)``.
    """

    driver: str
    lap: int
    total_laps: int
    position: int
    compound: str
    tyre_age: int
    stops: int
    gap_ahead_s: float | None
    gap_behind_s: float | None
    phase: TrackPhase
    phase_laps_remaining: int
    compounds_used: frozenset[str]
    available_compounds: tuple[str, ...]
    must_still_use_another_compound: bool

    @property
    def laps_remaining(self) -> int:
        return self.total_laps - self.lap

    @property
    def race_progress(self) -> float:
        return (self.lap - 1) / (self.total_laps - 1) if self.total_laps > 1 else 0.0

    @property
    def is_leader(self) -> bool:
        return self.position == 1

    @property
    def in_clean_air(self) -> bool:
        """Whether the car ahead is far enough away not to cost time."""
        return self.gap_ahead_s is None or self.gap_ahead_s > CLEAN_AIR_GAP_S


@runtime_checkable
class Strategy(Protocol):
    """A pit wall: it is shown one car's view and decides whether to stop.

    **Decisions must be a function of the view.** A single strategy object is normally shared by the
    whole field, so state accumulated across calls is shared across cars: a counter meaning "stops I
    have made" silently becomes "stops the whole field has made". Everything a decision needs is on
    the view, :attr:`CarView.stops` included. A strategy that genuinely requires per-car
    memory must be supplied per driver, as a mapping, and cannot be shared.

    Implementations must also be deterministic given their view and their own random stream, so that
    a seeded race reproduces exactly. A strategy that needs randomness should take a generator at
    construction from the race's seed bundle rather than reaching for a global one.
    """

    def decide(self, view: CarView) -> PitCall | None:
        """Return a pit call to stop this lap, or ``None`` to stay out."""
        ...

    def reset(self) -> None:
        """Discard any per-race memory. Called once before each race."""
        ...

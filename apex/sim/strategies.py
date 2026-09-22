"""Strategies that do not learn anything.

These exist to be beaten, and to make it embarrassing if nothing beats them. The project's claim is
that reinforcement learning only earns its place under stochasticity; the only way that claim can
be tested is if the alternatives are implemented properly rather than as strawmen. A rule-based pit
wall that reacts to safety cars is genuinely hard to beat, and an RL agent that cannot beat one has
not demonstrated anything.

:class:`FixedStops` is the simplest honest baseline: decide the plan before the race and execute it
regardless of what happens. It is the control condition for the whole experiment, because it is the
policy that ignores every piece of state an agent might exploit.

:class:`ReactiveRules` is the serious baseline. It carries a plan and departs from it for the one
reason a real pit wall always departs from it: a safety car has just neutralised the race, and a
stop now costs eight seconds less than it did last lap. That single rule is most of what human race
strategy is, and it is a high bar.

**Both are stateless, and that is a correctness requirement rather than a style preference.** One
strategy object is shared by the whole field, so a counter tracking "how many stops have I made"
counts the field's stops rather than one car's. The first version of these classes did exactly that,
and produced a race in which one car followed the plan, nineteen never stopped at all, and the
two-compound safety net dragged every one of them into the pit lane on the penultimate lap.
Everything a decision needs is already on the view -- :attr:`~apex.sim.strategy.CarView.stops` in
particular -- so deriving from it rather than remembering is both shorter and correct.
"""

from __future__ import annotations

from dataclasses import dataclass

from apex.sim.strategy import CarView, PitCall, TrackPhase

__all__ = ["FixedStops", "ReactiveRules"]


@dataclass(frozen=True)
class FixedStops:
    """Stop on predetermined laps, onto predetermined compounds, whatever else happens.

    Args:
        laps: Laps to stop on, in order.
        compounds: Compound to fit at each stop. Cycled if shorter than ``laps``.
    """

    laps: tuple[int, ...]
    compounds: tuple[str, ...] = ("HARD",)

    def reset(self) -> None:
        """Nothing to reset: this strategy keeps no per-race state."""

    def decide(self, view: CarView) -> PitCall | None:
        if view.stops >= len(self.laps) or view.lap != self.laps[view.stops]:
            return None
        return PitCall(compound=self.compounds[view.stops % len(self.compounds)])


@dataclass(frozen=True)
class ReactiveRules:
    """A plan, plus the discipline to abandon it when a safety car makes a stop cheap.

    Args:
        target_stops: How many stops the plan calls for. The laps are spread evenly across the race,
            which is the right first answer when degradation is roughly linear.
        compounds: Compounds to fit at each stop, cycled.
        safety_car_window: Take a free stop under a neutralisation only if the planned stop is
            within this many laps. Without a window the rule fires on lap two of a fifty-lap race
            and throws the stop away, which is a real mistake real teams make and a baseline should
            not.
        min_stint: Never stop on a set younger than this, so a neutralisation lasting several laps
            cannot trigger several stops.
    """

    target_stops: int = 1
    compounds: tuple[str, ...] = ("HARD",)
    safety_car_window: int = 12
    min_stint: int = 8

    def reset(self) -> None:
        """Nothing to reset: every decision is derived from the view."""

    def _planned_laps(self, total_laps: int) -> tuple[int, ...]:
        """Evenly spaced stop laps for the plan."""
        if self.target_stops < 1:
            return ()
        step = total_laps / (self.target_stops + 1)
        return tuple(max(2, round(step * (index + 1))) for index in range(self.target_stops))

    def decide(self, view: CarView) -> PitCall | None:
        planned = self._planned_laps(view.total_laps)
        if view.stops >= len(planned):
            # The plan is complete. The two-compound rule may still be unsatisfied if the plan was
            # degenerate, and the simulator will force a stop; asking for the right compound here
            # rather than leaving it to the safety net keeps the choice with the strategy.
            if view.must_still_use_another_compound and view.laps_remaining <= 1:
                return self._call(view)
            return None

        target = planned[view.stops]
        if view.tyre_age < self.min_stint:
            return None

        # The one reason to abandon the plan: a stop under a neutralisation costs far less, so bring
        # forward a stop that was coming anyway. Only a full safety car bunches the field, but a
        # virtual one still discounts the stop, so both are taken.
        if view.phase is not TrackPhase.GREEN and target - view.lap <= self.safety_car_window:
            return self._call(view)

        if view.lap >= target:
            return self._call(view)
        return None

    def _call(self, view: CarView) -> PitCall | None:
        compound = self.compounds[view.stops % len(self.compounds)]
        if compound not in view.available_compounds:
            return None
        # Prefer a compound the car has not used when the regulation still needs satisfying.
        if view.must_still_use_another_compound and compound in view.compounds_used:
            unused = [name for name in view.available_compounds if name not in view.compounds_used]
            if unused:
                compound = unused[0]
        return PitCall(compound=compound)

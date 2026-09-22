"""The race simulator: fitted parameters in, a lap-by-lap race out.

Discrete in laps, because every parameter it runs on was fitted per lap and because the strategic
decision -- stop or stay out -- is only ever taken at the line. A finer time step would add detail
no measurement supports.

**The order of operations inside a lap, and why it is this order.**

1.  *Flags.* An active neutralisation counts down; otherwise both hazards get a draw. Deciding this
    first is what lets a strategy react to a safety car on the lap it appears, which is the single
    most valuable reaction in race strategy.
2.  *The running order and the gaps*, read from cumulative time at the line.
3.  *Decisions*, one per car, from a view containing only what that pit wall can see.
4.  *Free lap times* -- what each car would do if the track ahead were empty, plus, under a
    neutralisation, the cruising pace of the queue it is joining.
5.  *Traffic*, resolved front to back. This is where a race becomes more than twenty stopwatches,
    and it is also where the field bunches up behind a safety car.

**Traffic is the part that matters, so it is worth being precise about.** Ranking cars by cumulative
time and calling that the result would mean any faster car passes any slower one automatically, and
Monaco would race like Monza. Instead, a car that catches the one ahead has to get past it: it
rolls against the fitted pass probability for that circuit, given the gap it started the lap with
and its tyre-adjusted pace advantage. Win, and it goes through. Lose, and its lap time is clamped
so it finishes the lap a minimum following distance behind -- it loses exactly the time it would
have gained. That clamp is the cost of track position, and it is what makes an undercut a gamble
rather than a free lunch.

Two cars in the pit lane are exempt: a car being passed because its rival stopped has not been
overtaken on track, and applying the clamp there would trap a stopping car behind the very cars it
is dropping behind.

**What this simulator does not model**, stated here rather than discovered later: retirements,
lapped traffic, damage, driver error, weather changes, and tyre temperature. Every one of them is
absent because nothing in `params.json` measures it, and inventing a number for it would undo the
point of fitting the rest. Their absence is why validation compares against classified finishers
only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from apex.calibrate.overtaking import OvertakingModel
from apex.calibrate.pace import PaceModel
from apex.calibrate.params import ParameterSet
from apex.calibrate.safety_car import NeutralisationModel
from apex.config import AppSettings, get_settings
from apex.seeding import SeedBundle
from apex.sim.scenario import RaceScenario
from apex.sim.strategy import CarView, PitCall, Strategy, TrackPhase
from apex.sim.trace import LapRecord, RaceResult, RaceTrace, StintRecord, _TraceBuilder

__all__ = ["RaceSimulator"]

_REQUIRED_COMPOUNDS = 2
"""Distinct compounds a dry race must be run on under the sporting regulations."""

_GRID_SLOT_GAP_S = 0.25
"""Time spacing between grid slots at the start.

Cars line up a few metres apart, so a standing start is not twenty cars beginning together: the
back of the grid is several seconds behind the front before a wheel turns, and it has to overtake to
recover that. Lap one is the one lap the pace model cannot speak to -- the exclusion rules drop it,
because a standing start is not representative green-flag pace -- so this is a modelling choice, and
it is the only one in this module that is not either fitted or configured."""


@dataclass(frozen=True, slots=True)
class _LapPace:
    """A car's two speeds for one lap.

    Attributes:
        racing: What it would run flat out, given its tyres, its fuel load and the air ahead of it.
        cruising: What it runs if it is simply following -- the same thing under green, and the
            field's neutralised pace behind a safety car.
    """

    racing: float
    cruising: float


@dataclass(slots=True)
class _Car:
    """Mutable per-car state. Internal: strategies see a :class:`CarView` instead."""

    driver: str
    team: str
    grid_position: int
    pace_s: float
    compound: str
    elapsed_s: float
    position: int
    laps_on_tyre: int = 0
    """Laps this set has completed. The pace model counts a set's first lap as age one, so the age
    handed to the model for the lap about to be run is this plus one."""

    stint: int = 1
    stops: int = 0
    compounds_used: set[str] = field(default_factory=set)
    laps_held_up: int = 0
    overtakes: int = 0

    @property
    def model_tyre_age(self) -> float:
        """Tyre age for the lap about to be run, on the convention the pace model was fitted on."""
        return float(self.laps_on_tyre + 1)


class RaceSimulator:
    """Runs races from a fitted parameter set.

    One instance may run any number of races. All per-race state lives in local variables and every
    random draw comes from the bundle handed to :meth:`run`, so the same scenario and the same
    bundle reproduce a race exactly no matter what else the instance has simulated.
    """

    def __init__(self, params: ParameterSet, settings: AppSettings | None = None) -> None:
        self.params = params
        self.settings = settings or get_settings()
        unavailable = set(self.settings.simulation.compounds) - {
            entry.compound for entry in params.pace.degradation
        }
        if unavailable:
            raise ValueError(
                f"compounds {sorted(unavailable)} are configured for simulation but the pace model "
                f"fitted no degradation curve for them"
            )

    # ------------------------------------------------------------------ parameter lookup

    def _pace(self, circuit: str) -> PaceModel:
        """The per-circuit pace model where one was fitted, else the pooled one."""
        try:
            return self.params.circuit_pace_at(circuit)
        except KeyError:
            return self.params.pace

    def _pit_loss_s(self, circuit: str, phase: TrackPhase) -> float:
        """What a stop costs here, under these flags.

        A neutralised stop uses the separately fitted field-relative loss. Using the green-flag
        number would make a safety car a reason to stay out, which inverts the most important
        strategic call in the sport.
        """
        if phase.is_neutralised and self.params.pit_loss.under_safety_car is not None:
            return self.params.pit_loss.under_safety_car.seconds.value
        try:
            return self.params.pit_loss.circuit(circuit).seconds.value
        except KeyError:
            return self.params.pit_loss.pooled.seconds.value

    def _catch_up_s(self, phase: TrackPhase) -> float:
        """How much faster than the queue's pace a car closing on it may run, per lap.

        The fitted value where there is one, and the configured fallback otherwise. Preferring the
        fitted number matters: it is measured from the training seasons, so the simulator's
        behaviour under a safety car is calibrated rather than chosen, and nothing about it was
        tuned against the races it is validated on.
        """
        model = (
            self.params.safety_car.safety_car
            if phase is TrackPhase.SAFETY_CAR
            else self.params.safety_car.virtual_safety_car
        )
        if model.catch_up_s is not None:
            return model.catch_up_s.value
        return self.settings.simulation.max_catch_up_s

    def _hazard(self, model: NeutralisationModel, circuit: str) -> float:
        try:
            return model.circuit(circuit).per_lap_probability.value
        except KeyError:
            return model.pooled.per_lap_probability.value

    def _tyre_delta(self, pace: PaceModel, compound: str, age: float) -> float:
        """What a car's current tyre costs it, identically to ``baseline.tyre_delta``."""
        return pace.compound(compound).lap_time_delta(age)

    # ------------------------------------------------------------------ the race

    def run(
        self,
        scenario: RaceScenario,
        strategy: Strategy | Mapping[str, Strategy],
        bundle: SeedBundle,
        *,
        schedule: Mapping[int, TrackPhase] | None = None,
    ) -> RaceTrace:
        """Simulate one race.

        Args:
            scenario: The pre-race specification.
            strategy: One strategy for the whole field, or one per driver. A mapping missing a
                driver is an error rather than a silent default, because a field where some cars
                never stop is not a race anybody meant to simulate.
            bundle: Seed bundle for this race. Every draw is taken from a named stream under it, so
                two agents evaluated on the same bundle meet the same safety cars.
            schedule: Flags to impose, lap by lap, instead of drawing them. Laps absent from the
                mapping run green. This exists for validation: comparing a simulated race against a
                real one is only a test of the pace and traffic models if both saw the same safety
                cars, and otherwise most of the discrepancy is just a different draw. It is also how
                a counterfactual is forked -- replay reality, then change one decision.

        Raises:
            KeyError: if a per-driver mapping omits a driver on the grid.
            ValueError: if a strategy asks for a compound that is not available.
        """
        pace = self._pace(scenario.circuit)
        overtaking = self.params.overtaking
        circuit = scenario.circuit
        opening = self.params.baseline.standing_start_s
        standing_start = 0.0 if opening is None else opening.value

        strategies: dict[str, Strategy] = (
            {entrant.driver: strategy for entrant in scenario.entrants}
            if not isinstance(strategy, Mapping)
            else dict(strategy)
        )
        for entrant in scenario.entrants:
            if entrant.driver not in strategies:
                raise KeyError(f"no strategy supplied for {entrant.driver!r}")
            strategies[entrant.driver].reset()

        flags = bundle.stream("safety_car")
        noise = bundle.stream("lap_noise")
        duels = bundle.stream("overtaking")

        cars = [
            _Car(
                driver=entrant.driver,
                team=entrant.team,
                grid_position=entrant.grid_position,
                pace_s=entrant.pace_s,
                compound=entrant.starting_compound,
                elapsed_s=(slot - 1) * _GRID_SLOT_GAP_S,
                position=slot,
                compounds_used={entrant.starting_compound},
            )
            for slot, entrant in enumerate(scenario.starting_order(), start=1)
        ]

        trace = _TraceBuilder()
        phase = TrackPhase.GREEN
        phase_remaining = 0

        for lap in range(1, scenario.total_laps + 1):
            if schedule is None:
                phase, phase_remaining = self._advance_flags(
                    phase,
                    phase_remaining,
                    circuit=circuit,
                    flags=flags,
                    lap=lap,
                    total_laps=scenario.total_laps,
                )
            else:
                phase = schedule.get(lap, TrackPhase.GREEN)
                phase_remaining = _remaining_in_schedule(schedule, lap, phase)
            trace.note_phase(lap, phase)

            order = sorted(cars, key=lambda car: car.elapsed_s)
            gaps: dict[str, float | None] = {
                car.driver: None if index == 0 else car.elapsed_s - order[index - 1].elapsed_s
                for index, car in enumerate(order)
            }

            calls = {
                car.driver: self._resolve_call(
                    strategies[car.driver],
                    car,
                    scenario=scenario,
                    lap=lap,
                    phase=phase,
                    phase_remaining=phase_remaining,
                    gap_ahead=gaps[car.driver],
                    gap_behind=gaps_behind(order, car),
                )
                for car in order
            }

            progress = (lap - 1) / (scenario.total_laps - 1) if scenario.total_laps > 1 else 0.0
            # The opening lap is a standing start, and the exclusion rules deliberately kept it out
            # of the pace model. Charging the separately fitted penalty here is what stops every
            # simulated race finishing several seconds early.
            start_penalty = standing_start if lap == 1 else 0.0
            paces = {
                car.driver: self._lap_time(
                    car,
                    scenario=scenario,
                    pace=pace,
                    progress=progress,
                    gap_ahead=gaps[car.driver],
                    phase=phase,
                    start_penalty=start_penalty,
                    noise=noise,
                )
                for car in order
            }
            pit_losses = {
                car.driver: (
                    self._pit_loss_s(circuit, phase) if calls[car.driver] is not None else 0.0
                )
                for car in order
            }

            resolved = self._resolve_traffic(
                order,
                gaps=gaps,
                paces=paces,
                pit_losses=pit_losses,
                calls=calls,
                pace=pace,
                overtaking=overtaking,
                circuit=circuit,
                phase=phase,
                duels=duels,
            )

            self._commit_lap(
                cars,
                order,
                resolved=resolved,
                calls=calls,
                lap=lap,
                phase=phase,
                trace=trace,
            )

        return self._classify(scenario, cars, trace=trace, bundle=bundle)

    # ------------------------------------------------------------------ lap mechanics

    def _advance_flags(
        self,
        phase: TrackPhase,
        remaining: int,
        *,
        circuit: str,
        flags: np.random.Generator,
        lap: int,
        total_laps: int,
    ) -> tuple[TrackPhase, int]:
        """Count an active neutralisation down, or give both hazards a draw.

        A neutralisation cannot begin on the final lap: the safety car would never be withdrawn and
        the race would end under it, which changes nothing about the result and would only muddy
        the trace.
        """
        if remaining > 0:
            return phase, remaining - 1
        if lap >= total_laps:
            return TrackPhase.GREEN, 0

        model = self.params.safety_car
        # Full safety car is offered first. The two are mutually exclusive in a lap, and a full
        # deployment is the strategically decisive one, so it takes precedence on the rare lap both
        # draws succeed rather than being silently dropped.
        for candidate, kind in (
            (model.safety_car, TrackPhase.SAFETY_CAR),
            (model.virtual_safety_car, TrackPhase.VIRTUAL_SAFETY_CAR),
        ):
            if flags.random() < self._hazard(candidate, circuit):
                duration = candidate.duration_laps
                laps = max(1, round(duration.value)) if duration is not None else 1
                capped = min(laps, total_laps - lap)
                return kind, max(capped - 1, 0)
        return TrackPhase.GREEN, 0

    def _resolve_call(
        self,
        strategy: Strategy,
        car: _Car,
        *,
        scenario: RaceScenario,
        lap: int,
        phase: TrackPhase,
        phase_remaining: int,
        gap_ahead: float | None,
        gap_behind: float | None,
    ) -> PitCall | None:
        """Ask a strategy for its call, then hold it to the rules.

        A rejected call is treated as staying out rather than as an error, because an agent
        exploring its action space proposes illegal stops constantly and a training run should not
        die of it. The regulation that cannot be declined is the two-compound rule: a car still on
        one
        compound with one lap left is stopped whatever its strategy says, since the alternative is
        producing a race result that would have been disqualified.
        """
        simulation = self.settings.simulation
        available = simulation.compounds
        needs_second = (
            simulation.require_two_compounds
            and len(car.compounds_used) < _REQUIRED_COMPOUNDS
            and len(available) > 1
        )

        view = CarView(
            driver=car.driver,
            lap=lap,
            total_laps=scenario.total_laps,
            position=car.position,
            compound=car.compound,
            tyre_age=car.laps_on_tyre,
            stops=car.stops,
            gap_ahead_s=gap_ahead,
            gap_behind_s=gap_behind,
            phase=phase,
            phase_laps_remaining=phase_remaining,
            compounds_used=frozenset(car.compounds_used),
            available_compounds=available,
            must_still_use_another_compound=needs_second,
        )
        call = strategy.decide(view)

        legal = (
            call is not None
            and lap > simulation.pit_lane_reopens_after_lap
            and car.laps_on_tyre >= simulation.min_stint_laps
            and lap < scenario.total_laps
        )
        if legal and call is not None:
            if call.compound not in available:
                raise ValueError(
                    f"{car.driver} asked for compound {call.compound!r}; "
                    f"available compounds are {list(available)}"
                )
            return call

        # Last chance to satisfy the two-compound rule, taken whether the strategy asked or not.
        if needs_second and lap == scenario.total_laps - 1:
            forced = next(name for name in available if name not in car.compounds_used)
            return PitCall(compound=forced)
        return None

    def _lap_time(
        self,
        car: _Car,
        *,
        scenario: RaceScenario,
        pace: PaceModel,
        progress: float,
        gap_ahead: float | None,
        phase: TrackPhase,
        start_penalty: float,
        noise: np.random.Generator,
    ) -> _LapPace:
        """What this car could run this lap with clear track ahead of it.

        Returns both the racing pace and the neutralised cruising pace, because under a safety car a
        car needs both: it cruises at the field's neutralised pace once it has joined the queue, and
        it closes on the queue at a bounded rate until it gets there. Returning a single number was
        the original mistake -- it left the simulator no way to say how quickly a gap could
        physically be closed, so gaps were closed instantaneously instead.

        Neither figure includes pit loss. A stop is a property of the lap rather than of the car's
        pace, and adding it here would have it counted once in the cruise and again in the bound.
        The standing-start penalty is passed in rather than decided here, so that the one lap of a
        race the pace model cannot speak to is visibly handled at the call site.
        """
        reference = scenario.anchor.reference_lap_s
        residual = pace.residual_std_s if self.settings.simulation.lap_time_noise else 0.0
        draw = float(noise.normal(0.0, residual)) if residual > 0.0 else 0.0

        racing = (
            reference
            + car.pace_s
            + self._tyre_delta(pace, car.compound, car.model_tyre_age)
            + pace.fuel_effect_s.value * progress
            + (0.0 if gap_ahead is None else pace.dirty_air_penalty(gap_ahead))
            + start_penalty
            + draw
        )
        if not phase.is_neutralised:
            return _LapPace(racing=racing, cruising=racing)

        # Behind a neutralisation every car drives to the same delta, so pace, tyres and traffic
        # stop mattering: the lap is the circuit's reference time scaled by the fitted ratio.
        # Letting a fast car keep gaining time behind a safety car would defeat the point of one.
        model = (
            self.params.safety_car.safety_car
            if phase is TrackPhase.SAFETY_CAR
            else self.params.safety_car.virtual_safety_car
        )
        ratio = model.lap_time_ratio.value if model.lap_time_ratio is not None else 1.0
        return _LapPace(racing=racing, cruising=reference * ratio + draw)

    def _resolve_traffic(
        self,
        order: list[_Car],
        *,
        gaps: dict[str, float | None],
        paces: dict[str, _LapPace],
        pit_losses: dict[str, float],
        calls: dict[str, PitCall | None],
        pace: PaceModel,
        overtaking: OvertakingModel,
        circuit: str,
        phase: TrackPhase,
        duels: np.random.Generator,
    ) -> dict[str, tuple[float, bool, bool]]:
        """Turn free lap times into real ones by making cars get past each other.

        Returns the new cumulative time per driver, and whether that car was held up or overtook.

        The car ahead is the one that started the lap ahead. A pass therefore takes effect from the
        next lap rather than rippling through the order within this one -- a car cannot pass two
        rivals on the same lap. At a per-lap pass probability that peaks in the tens of percent,
        the double pass is rare enough that modelling it would cost more in complexity than it buys
        in fidelity.
        """
        simulation = self.settings.simulation
        resolved: dict[str, tuple[float, bool, bool]] = {}

        for index, car in enumerate(order):
            loss = pit_losses[car.driver]
            cruise_arrival = car.elapsed_s + paces[car.driver].cruising + loss
            ahead = order[index - 1] if index > 0 else None

            if phase.is_neutralised:
                # A car in the pit lane is not in the queue, and neither is the car behind one that
                # has just dived in. Without this the pitter's own pit loss becomes the queue
                # reference and drags everybody behind it backwards -- the same exemption the
                # green-flag path makes, for the same reason.
                in_pit_lane = calls[car.driver] is not None or (
                    ahead is not None and calls[ahead.driver] is not None
                )
                target_gap = (
                    simulation.bunched_gap_s
                    if phase is TrackPhase.SAFETY_CAR
                    else simulation.min_following_gap_s
                )
                current_gap = gaps[car.driver]
                resolved[car.driver] = (
                    cruise_arrival
                    if in_pit_lane
                    else self._neutralised_arrival(
                        ahead=ahead,
                        resolved=resolved,
                        cruise_arrival=cruise_arrival,
                        fastest_arrival=cruise_arrival - self._catch_up_s(phase),
                        gap=target_gap if current_gap is None else min(target_gap, current_gap),
                    ),
                    False,
                    False,
                )
                continue

            arrival = cruise_arrival
            if ahead is None:
                resolved[car.driver] = (arrival, False, False)
                continue

            # Neither car is racing the other if either is in the pit lane this lap.
            if calls[car.driver] is not None or calls[ahead.driver] is not None:
                resolved[car.driver] = (arrival, False, False)
                continue

            floor = resolved[ahead.driver][0] + simulation.min_following_gap_s
            if arrival >= floor:
                resolved[car.driver] = (arrival, False, False)
                continue

            gap = gaps[car.driver] or 0.0
            advantage = (
                self._tyre_delta(pace, ahead.compound, ahead.model_tyre_age)
                + ahead.pace_s
                - (self._tyre_delta(pace, car.compound, car.model_tyre_age) + car.pace_s)
            )
            probability = overtaking.pass_probability(gap, advantage, circuit)
            if float(duels.random()) < probability:
                resolved[car.driver] = (arrival, False, True)
            else:
                resolved[car.driver] = (floor, True, False)

        return resolved

    def _neutralised_arrival(
        self,
        *,
        ahead: _Car | None,
        resolved: dict[str, tuple[float, bool, bool]],
        cruise_arrival: float,
        fastest_arrival: float,
        gap: float,
    ) -> float:
        """When a car crosses the line on a neutralised lap.

        The leader simply cruises. Everybody else wants to be closed up on the car in front and
        gets there over a lap or two, closing by at most the configured amount each lap. A car
        thirty seconds adrift when the safety car appears takes a couple of laps to join the queue,
        and one a long way back never fully does.

        Both bounds here were learned from being wrong. Closing the whole gap in a single lap handed
        out time nobody drove, and showed up as a four-second lap. Bounding the closing rate by each
        car's *racing* pace instead was better and still wrong the same way, because a car behind a
        safety car is held to a delta and is not racing at all: validation found neutralised laps
        running eleven seconds a lap too fast, against green-flag laps that were unbiased.

        The consequence that matters strategically survives both corrections. A car which has just
        pitted recovers most of the *time* it lost, because the field is crawling, and keeps all of
        the *places*, because the queue it rejoins has already formed. That asymmetry is why a stop
        under a safety car is the most valuable call in a race.
        """
        if ahead is None:
            return cruise_arrival
        # Never faster than the neutralisation permits, never slower than cruising behind the queue.
        target = resolved[ahead.driver][0] + gap
        return min(cruise_arrival, max(target, fastest_arrival))

    def _commit_lap(
        self,
        cars: list[_Car],
        order: list[_Car],
        *,
        resolved: dict[str, tuple[float, bool, bool]],
        calls: dict[str, PitCall | None],
        lap: int,
        phase: TrackPhase,
        trace: _TraceBuilder,
    ) -> None:
        """Write the lap into the trace and advance every car's state."""
        lap_times = {car.driver: resolved[car.driver][0] - car.elapsed_s for car in order}

        for car in order:
            elapsed, held_up, overtook = resolved[car.driver]
            car.elapsed_s = elapsed
            car.laps_held_up += int(held_up)
            car.overtakes += int(overtook)

        finished = sorted(cars, key=lambda car: car.elapsed_s)
        leader_time = finished[0].elapsed_s
        for position, car in enumerate(finished, start=1):
            car.position = position

        for position, car in enumerate(finished, start=1):
            call = calls[car.driver]
            record = LapRecord(
                driver=car.driver,
                team=car.team,
                lap=lap,
                position=position,
                compound=car.compound,
                tyre_age=int(car.model_tyre_age),
                stint=car.stint,
                lap_time_s=lap_times[car.driver],
                elapsed_s=car.elapsed_s,
                gap_ahead_s=(
                    0.0 if position == 1 else car.elapsed_s - finished[position - 2].elapsed_s
                ),
                gap_to_leader_s=car.elapsed_s - leader_time,
                track_phase=phase,
                is_pit_lap=call is not None,
                held_up=resolved[car.driver][1],
                overtook=resolved[car.driver][2],
            )
            trace.add(record)

            # Tyre state advances after the lap is recorded, so the record shows the tyre the lap
            # was actually driven on. A stop fits the new compound for the following lap: the car
            # ran most of this one on the old set, and the fitted pit loss already covers both the
            # in-lap and the out-lap.
            if call is not None:
                car.compound = call.compound
                car.compounds_used.add(call.compound)
                car.laps_on_tyre = 0
                car.stint += 1
                car.stops += 1
            else:
                car.laps_on_tyre += 1

    def _classify(
        self, scenario: RaceScenario, cars: list[_Car], *, trace: _TraceBuilder, bundle: SeedBundle
    ) -> RaceTrace:
        """Build the final classification and assemble the trace."""
        finished = sorted(cars, key=lambda car: car.elapsed_s)
        winner_time = finished[0].elapsed_s
        results = tuple(
            RaceResult(
                driver=car.driver,
                team=car.team,
                grid_position=car.grid_position,
                finish_position=position,
                total_time_s=car.elapsed_s,
                gap_to_winner_s=car.elapsed_s - winner_time,
                stops=car.stops,
                laps_held_up=car.laps_held_up,
                overtakes=car.overtakes,
            )
            for position, car in enumerate(finished, start=1)
        )

        stints: list[StintRecord] = []
        by_driver: dict[tuple[str, int], list[LapRecord]] = {}
        for record in trace.records:
            by_driver.setdefault((record.driver, record.stint), []).append(record)
        for (driver, stint), records in sorted(by_driver.items()):
            stints.append(
                StintRecord(
                    driver=driver,
                    stint=stint,
                    compound=records[0].compound,
                    first_lap=min(record.lap for record in records),
                    last_lap=max(record.lap for record in records),
                )
            )

        return RaceTrace(
            scenario=scenario,
            results=results,
            stints=tuple(stints),
            laps=tuple(trace.records),
            safety_car_laps=tuple(trace.safety_car_laps),
            virtual_safety_car_laps=tuple(trace.virtual_safety_car_laps),
            seed_run_id=bundle.run_id,
        )


def _remaining_in_schedule(schedule: Mapping[int, TrackPhase], lap: int, phase: TrackPhase) -> int:
    """How many further laps an imposed neutralisation runs for.

    A strategy is shown this as the expected remaining duration. Under an imposed schedule it is
    exact, which is a small advantage over a sampled race, and it is confined to validation, where
    the strategies are replaying decisions that were already made.
    """
    if phase is TrackPhase.GREEN:
        return 0
    ahead = lap + 1
    while schedule.get(ahead) is phase:
        ahead += 1
    return ahead - lap - 1


def gaps_behind(order: list[_Car], car: _Car) -> float | None:
    """Gap from a car to the one behind it, or ``None`` for the last car on the road."""
    index = order.index(car)
    if index + 1 >= len(order):
        return None
    return order[index + 1].elapsed_s - car.elapsed_s

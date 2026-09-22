"""The simulator, checked against properties a race has to satisfy.

A simulated race cannot be compared against a known truth the way a fitted coefficient can, so these
tests assert the invariants instead: time moves forward, tyres age, the regulations hold, a seed
reproduces a race exactly, and the fitted parameters actually reach the outcome. Several of them
encode bugs that were found by validating against real races and would otherwise come back.
"""

from __future__ import annotations

import datetime as dt

import pytest

from apex.calibrate.baseline import BaselineModel, CircuitBaseline, SeasonPace
from apex.calibrate.estimates import Estimate
from apex.calibrate.grid import GridPaceModel
from apex.calibrate.overtaking import CircuitOvertaking, OvertakingModel
from apex.calibrate.pace import CompoundDegradation, PaceModel
from apex.calibrate.params import PARAMS_SCHEMA_VERSION, ExclusionCount, ParameterSet, Provenance
from apex.calibrate.pit import PitLoss, PitLossModel
from apex.calibrate.safety_car import Neutralisation, NeutralisationModel, SafetyCarModel
from apex.config import AppSettings, SimulationSettings
from apex.seeding import SeedBundle
from apex.sim import (
    AnchorMode,
    FixedStops,
    PitCall,
    RaceScenario,
    RaceSimulator,
    ReactiveRules,
    TrackPhase,
    from_grid,
)
from apex.sim.strategy import CarView

LEVEL = 0.95
CIRCUIT = "Testville"
REFERENCE_LAP_S = 90.0


def estimate(value: float, *, n_obs: int = 500) -> Estimate:
    """An estimate with a narrow interval, for building a parameter set in a test."""
    return Estimate.from_standard_error(value, abs(value) * 0.01 + 0.01, n_obs=n_obs, level=LEVEL)


def pace_model(*, residual: float = 0.0) -> PaceModel:
    """A pace model with round numbers, so a hand-computed lap time can be asserted."""
    return PaceModel(
        circuit=None,
        degradation=(
            CompoundDegradation(compound="SOFT", coefficients=(estimate(0.10),), offset_s=None),
            CompoundDegradation(
                compound="MEDIUM", coefficients=(estimate(0.05),), offset_s=estimate(0.40)
            ),
            CompoundDegradation(
                compound="HARD", coefficients=(estimate(0.02),), offset_s=estimate(0.80)
            ),
        ),
        reference_compound="SOFT",
        fuel_effect_s=estimate(-3.0),
        dirty_air_s=estimate(0.8),
        dirty_air_saturation_s=2.0,
        residual_std_s=residual,
        r_squared_within=0.9,
        n_laps=10_000,
        n_races=50,
        n_cars=20,
    )


def neutralisation(
    kind: str, *, probability: float, duration: float, ratio: float
) -> NeutralisationModel:
    return NeutralisationModel(
        kind=kind,
        pooled=Neutralisation(
            kind=kind,
            circuit=None,
            per_lap_probability=estimate(probability),
            n_deployments=30,
            laps_at_risk=3000,
            n_races=50,
        ),
        circuits=(),
        duration_laps=estimate(duration),
        catch_up_s=estimate(14.0),
        lap_time_ratio=estimate(ratio),
        races_with_any=25,
        n_races=50,
    )


def parameters(*, residual: float = 0.0, safety_car_probability: float = 0.0) -> ParameterSet:
    """A complete parameter set with known values, so the simulator's arithmetic is checkable."""
    return ParameterSet(
        provenance=Provenance(
            generated_at=dt.datetime(2026, 9, 1, tzinfo=dt.UTC),
            apex_version="0.1.0",
            config_hash="test",
            seed=1,
            schema_version=PARAMS_SCHEMA_VERSION,
            seasons=(2022, 2023, 2024),
            holdout_seasons=(2025,),
            n_races=50,
            n_circuits=20,
            n_laps_available=10_000,
            n_laps_used=9_000,
        ),
        exclusions=(ExclusionCount(rule="input", removed=0, remaining=10_000),),
        pace=pace_model(residual=residual),
        circuit_pace=(),
        between_circuit_std={},
        pit_loss=PitLossModel(
            pooled=PitLoss(circuit=None, seconds=estimate(20.0), n_stops=1000),
            circuits=(),
            n_stops=1000,
            n_races=50,
            excluded_under_safety_car=100,
            under_safety_car=PitLoss(circuit=None, seconds=estimate(12.0), n_stops=200),
            n_stops_under_safety_car=200,
        ),
        safety_car=SafetyCarModel(
            safety_car=neutralisation(
                "safety_car", probability=safety_car_probability, duration=4.0, ratio=1.35
            ),
            virtual_safety_car=neutralisation(
                "virtual_safety_car", probability=0.0, duration=3.0, ratio=1.20
            ),
        ),
        baseline=BaselineModel(
            circuits=(
                CircuitBaseline(
                    circuit=CIRCUIT,
                    reference_lap_s=estimate(REFERENCE_LAP_S),
                    n_laps=5000,
                    n_races=3,
                    seasons=(2022, 2023, 2024),
                ),
            ),
            seasons=(SeasonPace(season=2024, offset_s=estimate(0.0), n_races=20),),
            reference_season=2024,
            season_trend_s=estimate(-1.0),
            field_spread_s=estimate(0.7),
            standing_start_s=estimate(8.0),
            n_car_races=1000,
        ),
        grid_pace=GridPaceModel(
            coefficients=(estimate(-1.0), estimate(2.0)),
            reference_field_size=20,
            residual_std_s=estimate(0.45),
            n_car_races=1000,
            n_races=50,
            r_squared=0.5,
        ),
        overtaking=OvertakingModel(
            pooled=CircuitOvertaking(
                circuit=None, difficulty=estimate(-0.4), n_pairs=30_000, n_passes=1400, n_races=50
            ),
            circuits=(),
            gap_coefficient=estimate(-3.4),
            pace_coefficient=estimate(2.2),
            between_circuit_std=1.4,
            max_gap_s=5.0,
            n_pairs=30_000,
            n_passes=1400,
            n_races=50,
        ),
    )


def settings(**simulation: object) -> AppSettings:
    return AppSettings(simulation=SimulationSettings(**simulation))


def scenario(config: AppSettings, *, cars: int = 6, laps: int = 20) -> RaceScenario:
    return from_grid(
        parameters(),
        circuit=CIRCUIT,
        season=2024,
        total_laps=laps,
        grid=[f"D{index:02d}" for index in range(1, cars + 1)],
        starting_compound="MEDIUM",
        settings=config,
    )


def bundle(name: str = "test") -> SeedBundle:
    return SeedBundle(master_seed=11).derive(name)


# --------------------------------------------------------------------------- invariants


def test_every_car_completes_the_scheduled_distance() -> None:
    config = settings()
    simulator = RaceSimulator(parameters(), config)
    trace = simulator.run(scenario(config), ReactiveRules(target_stops=1), bundle())

    assert len(trace.results) == trace.scenario.n_cars
    per_car = dict.fromkeys(trace.finishing_order(), 0)
    for record in trace.laps:
        per_car[record.driver] += 1
    assert set(per_car.values()) == {trace.scenario.total_laps}


def test_cumulative_time_never_goes_backwards() -> None:
    """A lap cannot finish before it started, under any flag.

    This is the test the safety car broke twice. Bunching the field by pulling followers forward in
    one step produced negative and near-zero lap times, because the recorded lap absorbed the whole
    compression.
    """
    config = settings()
    simulator = RaceSimulator(parameters(safety_car_probability=0.2), config)
    trace = simulator.run(scenario(config, laps=40), ReactiveRules(target_stops=1), bundle("flags"))

    assert trace.safety_car_laps, "this test is pointless without a neutralisation"
    assert all(record.lap_time_s > 0.0 for record in trace.laps)

    previous: dict[str, float] = {}
    for record in sorted(trace.laps, key=lambda entry: (entry.driver, entry.lap)):
        if record.driver in previous:
            assert record.elapsed_s > previous[record.driver]
        previous[record.driver] = record.elapsed_s


def test_no_lap_is_faster_than_physically_possible() -> None:
    """Every lap is at least the circuit's reference minus the plausible gains it allows.

    A car cannot beat the reference by more than the fuel effect plus its own pace advantage, so a
    floor computed from those catches a compression bug without pinning an exact number.
    """
    config = settings()
    params = parameters(safety_car_probability=0.25)
    simulator = RaceSimulator(params, config)
    trace = simulator.run(scenario(config, laps=40), ReactiveRules(target_stops=1), bundle("floor"))

    best_pace = min(entrant.pace_s for entrant in trace.scenario.entrants)
    floor = REFERENCE_LAP_S + best_pace + params.pace.fuel_effect_s.value - 0.01
    assert min(record.lap_time_s for record in trace.laps) >= floor


def test_tyre_age_resets_at_a_stop_and_grows_otherwise() -> None:
    config = settings()
    simulator = RaceSimulator(parameters(), config)
    trace = simulator.run(scenario(config, laps=20), FixedStops(laps=(10,)), bundle("tyres"))

    for driver in trace.finishing_order():
        laps = sorted(
            (record for record in trace.laps if record.driver == driver), key=lambda r: r.lap
        )
        assert [record.tyre_age for record in laps[:10]] == list(range(1, 11))
        # The stop is charged to lap ten; the new set runs from lap eleven at age one.
        assert laps[10].tyre_age == 1
        assert laps[10].compound != laps[9].compound


def test_a_seed_reproduces_a_race_exactly() -> None:
    config = settings()
    simulator = RaceSimulator(parameters(residual=0.5, safety_car_probability=0.1), config)
    race = scenario(config, laps=30)
    strategy = ReactiveRules(target_stops=1)

    first = simulator.run(race, strategy, bundle("repeat"))
    second = simulator.run(race, strategy, bundle("repeat"))

    assert first.finishing_order() == second.finishing_order()
    assert first.safety_car_laps == second.safety_car_laps
    assert [record.lap_time_s for record in first.laps] == [
        record.lap_time_s for record in second.laps
    ]


def test_different_seeds_produce_different_races() -> None:
    config = settings()
    simulator = RaceSimulator(parameters(residual=0.5, safety_car_probability=0.2), config)
    race = scenario(config, laps=30)
    strategy = ReactiveRules(target_stops=1)

    times = {
        simulator.run(race, strategy, bundle("spread").episode(index)).winner.total_time_s
        for index in range(5)
    }
    assert len(times) == 5


def test_simulating_many_races_does_not_change_any_of_them() -> None:
    """ADR-0004's promise, at the level of a whole race rather than a stream.

    One simulator instance must be reusable, so a race's outcome cannot depend on how many races the
    instance ran before it.
    """
    config = settings()
    simulator = RaceSimulator(parameters(residual=0.4, safety_car_probability=0.15), config)
    race = scenario(config, laps=25)
    strategy = ReactiveRules(target_stops=1)

    alone = RaceSimulator(parameters(residual=0.4, safety_car_probability=0.15), config).run(
        race, strategy, bundle("order").episode(3)
    )
    for index in range(3):
        simulator.run(race, strategy, bundle("order").episode(index))
    after = simulator.run(race, strategy, bundle("order").episode(3))

    assert after.finishing_order() == alone.finishing_order()
    assert after.winner.total_time_s == pytest.approx(alone.winner.total_time_s)


# --------------------------------------------------------------------------- regulations


def test_the_two_compound_rule_is_satisfied_even_by_a_strategy_that_never_stops() -> None:
    """A strategy that ignores the regulation must not produce an illegal race."""
    config = settings(require_two_compounds=True)
    simulator = RaceSimulator(parameters(), config)

    class NeverStops:
        def reset(self) -> None: ...

        def decide(self, view: CarView) -> PitCall | None:
            del view
            return None

    trace = simulator.run(scenario(config, laps=20), NeverStops(), bundle("legal"))
    for driver in trace.finishing_order():
        compounds = {record.compound for record in trace.laps if record.driver == driver}
        assert len(compounds) >= 2, f"{driver} ran one compound"


def test_a_stop_is_refused_before_the_pit_lane_opens() -> None:
    config = settings(pit_lane_reopens_after_lap=3, min_stint_laps=1)
    simulator = RaceSimulator(parameters(), config)
    trace = simulator.run(scenario(config, laps=20), FixedStops(laps=(2,)), bundle("closed"))

    pit_laps = {record.lap for record in trace.laps if record.is_pit_lap}
    assert 2 not in pit_laps


def test_a_stop_is_refused_on_a_stint_shorter_than_the_minimum() -> None:
    config = settings(min_stint_laps=6, pit_lane_reopens_after_lap=1)
    simulator = RaceSimulator(parameters(), config)
    trace = simulator.run(scenario(config, laps=20), FixedStops(laps=(3,)), bundle("short"))

    # Lap three would leave a two-lap opening stint, so the call is declined; the two-compound
    # safety net then takes the stop on the penultimate lap instead.
    assert 3 not in {record.lap for record in trace.laps if record.is_pit_lap}


def test_an_unavailable_compound_is_an_error_rather_than_a_silent_substitution() -> None:
    config = settings(compounds=("SOFT", "MEDIUM", "HARD"))
    simulator = RaceSimulator(parameters(), config)
    with pytest.raises(ValueError, match="INTERMEDIATE"):
        simulator.run(
            scenario(config), FixedStops(laps=(5,), compounds=("INTERMEDIATE",)), bundle()
        )


def test_a_compound_the_pace_model_never_fitted_cannot_be_simulated() -> None:
    with pytest.raises(ValueError, match="fitted no degradation curve"):
        RaceSimulator(parameters(), settings(compounds=("SOFT", "WET")))


# --------------------------------------------------------------------------- the fitted parameters


def test_a_lap_time_is_the_sum_of_the_fitted_effects() -> None:
    """With noise off and clear air, the leader's opening lap is computable by hand.

    This is the test that would catch a sign error or a double-counted term anywhere in the lap-time
    model, including the standing-start penalty the exclusion rules deliberately removed.
    """
    config = settings(lap_time_noise=False)
    params = parameters(residual=0.0)
    simulator = RaceSimulator(params, config)
    race = scenario(config, cars=2, laps=20)
    trace = simulator.run(race, FixedStops(laps=()), bundle("arith"))

    leader = min(race.entrants, key=lambda entrant: entrant.grid_position)
    first = next(
        record for record in trace.laps if record.driver == leader.driver and record.lap == 1
    )
    expected = (
        REFERENCE_LAP_S
        + leader.pace_s
        + params.pace.compound("MEDIUM").lap_time_delta(1.0)
        + params.baseline.standing_start_s.value  # type: ignore[union-attr]
    )
    assert first.lap_time_s == pytest.approx(expected, abs=1e-9)


def test_the_field_bunches_up_behind_a_safety_car() -> None:
    """A deployment must close the field up, measured across that deployment and no further.

    The comparison has to be local to one neutralisation. An earlier version of this test compared
    the lap before the first safety car against the last safety-car lap of the race, which spanned
    thirty laps and several green phases in between, and it failed because the field had spread out
    again during them -- a fact about the race, not about bunching.
    """
    config = settings()
    simulator = RaceSimulator(parameters(safety_car_probability=0.3), config)
    trace = simulator.run(scenario(config, laps=40), ReactiveRules(target_stops=1), bundle("bunch"))
    assert trace.safety_car_laps

    deployed = min(lap for lap in trace.safety_car_laps if lap > 1)
    # The end of this deployment: the last lap of the contiguous run that starts at `deployed`.
    last = deployed
    while last + 1 in trace.safety_car_laps:
        last += 1

    frame = trace.frame()

    def spread(lap: int) -> float:
        return float(frame.filter(frame["lap"] == lap)["gap_to_leader_s"].max())  # type: ignore[arg-type]

    assert spread(last) < spread(deployed - 1)


def test_a_stop_under_a_safety_car_costs_less_than_one_under_green() -> None:
    """The discount that makes a safety car strategically decisive, asserted rather than assumed."""
    config = settings()
    params = parameters()
    simulator = RaceSimulator(params, config)

    green = simulator._pit_loss_s(CIRCUIT, TrackPhase.GREEN)
    neutralised = simulator._pit_loss_s(CIRCUIT, TrackPhase.SAFETY_CAR)
    assert neutralised < green
    assert neutralised == pytest.approx(params.pit_loss.under_safety_car.seconds.value)  # type: ignore[union-attr]


def test_overtaking_is_harder_where_the_fitted_difficulty_is_lower() -> None:
    """Two circuits differing only in difficulty must produce different amounts of passing."""
    config = settings()
    base = parameters(residual=0.3)

    def with_difficulty(value: float) -> ParameterSet:
        model = base.overtaking.model_copy(
            update={
                "circuits": (
                    CircuitOvertaking(
                        circuit=CIRCUIT,
                        difficulty=estimate(value),
                        n_pairs=5000,
                        n_passes=100,
                        n_races=5,
                    ),
                )
            }
        )
        return base.model_copy(update={"overtaking": model})

    def passes(value: float) -> int:
        simulator = RaceSimulator(with_difficulty(value), config)
        race = scenario(config, cars=10, laps=40)
        return sum(
            sum(
                entry.overtakes
                for entry in simulator.run(
                    race, ReactiveRules(), bundle("pass").episode(seed)
                ).results
            )
            for seed in range(6)
        )

    assert passes(-4.0) < passes(1.0)


def test_a_car_held_up_loses_the_time_it_would_have_gained() -> None:
    """Track position has to cost something, or the whole strategy problem is trivial."""
    # Impossible to pass, so a faster car behind a slower one must stay behind it.
    params = parameters(residual=0.0)
    blocked = params.model_copy(
        update={
            "overtaking": params.overtaking.model_copy(
                update={
                    "circuits": (
                        CircuitOvertaking(
                            circuit=CIRCUIT,
                            difficulty=Estimate(value=-50.0, level=LEVEL, n_obs=100),
                            n_pairs=5000,
                            n_passes=0,
                            n_races=5,
                        ),
                    )
                }
            )
        }
    )
    # The two-compound rule is switched off deliberately. With it on, the safety net forces both
    # cars into the pit lane on the penultimate lap, and a stop is exactly the situation where the
    # traffic clamp does not apply -- so the faster car would come out ahead without ever passing on
    # track, which is correct behaviour and would make this test about something else.
    config = settings(lap_time_noise=False, require_two_compounds=False)
    simulator = RaceSimulator(blocked, config)

    race = from_grid(
        blocked,
        circuit=CIRCUIT,
        season=2024,
        total_laps=30,
        grid=["SLOW", "FAST"],
        starting_compound="MEDIUM",
        pace_overrides={"SLOW": 0.0, "FAST": -1.5},
        settings=config,
    )
    trace = simulator.run(race, FixedStops(laps=()), bundle("held"))

    assert trace.result("FAST").laps_held_up > 0
    assert trace.result("FAST").overtakes == 0
    assert trace.finishing_order()[0] == "SLOW"


# --------------------------------------------------------------------------- scenarios


def test_a_field_built_from_a_grid_is_ordered_by_the_fitted_grid_model() -> None:
    config = settings()
    race = scenario(config, cars=10)
    ordered = race.starting_order()

    assert [entrant.driver for entrant in ordered] == [f"D{i:02d}" for i in range(1, 11)]
    paces = [entrant.pace_s for entrant in ordered]
    assert paces == sorted(paces), "a better grid slot must not imply slower race pace"


def test_a_pit_lane_start_lines_up_last() -> None:
    params = parameters()
    race = from_grid(
        params,
        circuit=CIRCUIT,
        season=2024,
        total_laps=20,
        grid=["A", "B"],
        settings=settings(),
    ).model_copy()
    # Rebuild the first entrant as a pit-lane start and confirm it sorts to the back.
    entrants = (race.entrants[0].model_copy(update={"grid_position": 0}), race.entrants[1])
    reordered = race.model_copy(update={"entrants": entrants})
    assert reordered.starting_order()[-1].driver == entrants[0].driver


def test_a_driver_cannot_appear_twice() -> None:
    with pytest.raises(ValueError, match="cannot appear twice"):
        from_grid(
            parameters(),
            circuit=CIRCUIT,
            season=2024,
            total_laps=20,
            grid=["VER", "VER"],
            settings=settings(),
        )


def test_an_anchored_scenario_records_that_it_was_anchored() -> None:
    race = scenario(settings())
    assert race.anchor.mode is AnchorMode.FITTED
    assert race.anchor.reference_lap_s == pytest.approx(REFERENCE_LAP_S)


def test_an_imposed_schedule_replaces_the_sampled_flags() -> None:
    config = settings()
    simulator = RaceSimulator(parameters(safety_car_probability=0.0), config)
    imposed = {5: TrackPhase.SAFETY_CAR, 6: TrackPhase.SAFETY_CAR, 9: TrackPhase.VIRTUAL_SAFETY_CAR}
    trace = simulator.run(
        scenario(config, laps=20), ReactiveRules(target_stops=1), bundle("sched"), schedule=imposed
    )

    assert trace.safety_car_laps == (5, 6)
    assert trace.virtual_safety_car_laps == (9,)


def test_a_missing_per_driver_strategy_is_an_error() -> None:
    config = settings()
    simulator = RaceSimulator(parameters(), config)
    race = scenario(config, cars=3)
    with pytest.raises(KeyError, match="no strategy supplied"):
        simulator.run(race, {race.entrants[0].driver: ReactiveRules()}, bundle())


# --------------------------------------------------------------------------- strategies


def test_a_shared_strategy_gives_every_car_its_own_plan() -> None:
    """The bug that made nineteen of twenty cars never stop.

    One strategy object serves the whole field, so a strategy holding a stop counter counts the
    field's stops rather than each car's. Every car must make the plan's stops.
    """
    config = settings()
    simulator = RaceSimulator(parameters(), config)
    trace = simulator.run(
        scenario(config, cars=8, laps=30), FixedStops(laps=(12,)), bundle("share")
    )

    for entry in trace.results:
        assert entry.stops == 1, f"{entry.driver} made {entry.stops} stops"
    pit_laps = {record.lap for record in trace.laps if record.is_pit_lap}
    assert pit_laps == {12}


def test_the_reactive_baseline_takes_a_free_stop_under_a_neutralisation() -> None:
    config = settings()
    params = parameters()
    simulator = RaceSimulator(params, config)
    race = scenario(config, cars=6, laps=40)
    strategy = ReactiveRules(target_stops=1, safety_car_window=12, min_stint=8)

    planned = simulator.run(race, strategy, bundle("plan"), schedule={}).laps
    planned_stop = min(record.lap for record in planned if record.is_pit_lap)

    # A neutralisation a few laps before the planned stop should pull the stop forward.
    early = dict.fromkeys(range(planned_stop - 5, planned_stop - 2), TrackPhase.SAFETY_CAR)
    reactive = simulator.run(race, strategy, bundle("plan"), schedule=early).laps
    reactive_stop = min(record.lap for record in reactive if record.is_pit_lap)

    assert reactive_stop < planned_stop
    assert reactive_stop in early

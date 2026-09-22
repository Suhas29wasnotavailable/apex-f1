"""Validation: what counts as a finisher, and what a comparison is allowed to claim.

The metrics decide whether the project may use the word "twin", so the ways they can quietly lie are
worth testing directly. Two of these encode bugs that made the simulator look far worse than it was,
and one encodes a rule that made it look better.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from apex.calibrate.exclusions import CAR_KEYS
from apex.calibrate.features import add_features
from apex.eval.metrics import compare
from apex.eval.observed import ReplayStops, observed_race, observed_schedule, observed_stops
from apex.sim import RaceSimulator, ReactiveRules, TrackPhase
from apex.sim.strategy import CarView
from tests.factories import lap_frame
from tests.test_pace import LINEAR, simulated_race
from tests.test_sim import bundle, parameters, scenario, settings

CIRCUIT = "Alpha"


def race_frame(
    *, drivers: int = 10, total_laps: int = 30, seed: int = 4, **overrides: object
) -> pl.DataFrame:
    """One featured race, through the production feature pipeline."""
    rows = simulated_race(
        2025,
        1,
        CIRCUIT,
        rng=np.random.default_rng(seed),
        drivers=drivers,
        total_laps=total_laps,
        **overrides,  # type: ignore[arg-type]
    )
    return add_features(lap_frame(rows), LINEAR)


# --------------------------------------------------------------------------- classification


def test_a_retired_car_is_excluded_and_counted() -> None:
    """The simulator models no retirements, so a retired car is not a result it can be scored on."""
    laps = race_frame(total_laps=30)
    retired = laps.filter(~((pl.col("driver") == "D03") & (pl.col("lap") > 8)))

    observed = observed_race(retired, LINEAR)
    assert "D03" in observed.retired
    assert "D03" not in observed.finishing_order
    assert observed.n_classified == 9


def test_a_car_with_untimed_laps_still_counts_as_a_finisher() -> None:
    """The rule that once threw away an entire race.

    At Jeddah in 2025 an opening-lap incident left two laps untimed for eighteen cars. A
    classification rule that demanded every lap be timed excluded the whole field, and with it the
    race. Whether a car finished is a question about distance, not about timing completeness.
    """
    laps = race_frame(total_laps=30)
    blanked = laps.with_columns(
        pl.when(pl.col("lap").is_in([5, 6]))
        .then(None)
        .otherwise(pl.col("lap_time_s"))
        .alias("lap_time_s")
    )

    observed = observed_race(blanked, LINEAR)
    assert observed.retired == ()
    assert observed.n_classified == 10
    # ...but no race time may be quoted for a car whose laps are not all timed.
    assert observed.total_time_s == {}


def test_a_lapped_car_keeps_its_place_but_is_quoted_no_gap() -> None:
    """The metric bug that made the twin's finishing gaps look tens of seconds wrong.

    A car one lap down has one fewer lap time than the winner, so differencing the two sums mixes
    "slower" with "shorter". It stays in the finishing order, where position error can see it, and
    it is left out of the time-based metrics, as the sport leaves it out of the gap column.
    """
    laps = race_frame(total_laps=30)
    lapped = laps.filter(~((pl.col("driver") == "D09") & (pl.col("lap") == 30)))

    observed = observed_race(lapped, LINEAR)
    assert "D09" in observed.finishing_order, "a lapped car is still classified"
    assert "D09" not in observed.total_time_s, "a lapped car is quoted no race time"
    assert len(observed.total_time_s) == observed.n_classified - 1


def test_the_flag_schedule_is_read_from_the_track_not_from_one_car() -> None:
    laps = race_frame(total_laps=20)
    flagged = laps.with_columns(
        pl.when((pl.col("lap") == 7) & (pl.col("driver") == "D00"))
        .then(pl.lit("4"))
        .when(pl.col("lap") == 12)
        .then(pl.lit("671"))
        .when(pl.col("lap") == 15)
        .then(pl.lit("461"))
        .otherwise(pl.col("track_status"))
        .alias("track_status")
    )
    schedule = observed_schedule(flagged, LINEAR)

    # One car seeing the safety car is enough: the status belongs to the track, not to a car.
    assert schedule[7] is TrackPhase.SAFETY_CAR
    # Statuses are concatenated within a lap and matched as substrings. "671" carries only the
    # virtual codes, so it is a virtual safety car...
    assert schedule[12] is TrackPhase.VIRTUAL_SAFETY_CAR
    # ...while a lap showing both defers to the full safety car, the strategically decisive one.
    assert schedule[15] is TrackPhase.SAFETY_CAR
    assert 3 not in schedule


def test_a_stop_is_read_from_the_compound_change_not_the_pit_flag() -> None:
    laps = race_frame(total_laps=30)
    stops = observed_stops(laps)

    assert stops, "the fixture pits every car once"
    for driver, entries in stops.items():
        assert len(entries) == 1, f"{driver} should have one stop"
        lap, compound = entries[0]
        assert 1 < lap < 30
        assert compound in {"SOFT", "MEDIUM", "HARD"}


def test_reducing_a_frame_holding_two_races_is_an_error() -> None:
    first = race_frame(seed=1)
    second = race_frame(seed=2).with_columns(pl.lit(2, dtype=pl.Int16).alias("round"))
    with pytest.raises(ValueError, match="exactly one race"):
        observed_race(pl.concat([first, second]), LINEAR)


# --------------------------------------------------------------------------- replay strategy


def test_the_replay_strategy_reproduces_the_stops_it_was_given() -> None:
    """The strongest validation mode depends on this being exact."""
    config = settings()
    simulator = RaceSimulator(parameters(), config)
    race = scenario(config, cars=4, laps=30)
    plan = {entrant.driver: ((10 + index, "HARD"),) for index, entrant in enumerate(race.entrants)}

    trace = simulator.run(race, ReplayStops(stops=plan), bundle("replay"))
    for driver, entries in plan.items():
        pit_laps = {
            record.lap for record in trace.laps if record.driver == driver and record.is_pit_lap
        }
        assert pit_laps == {entries[0][0]}


def test_the_replay_strategy_substitutes_a_compound_it_cannot_fit() -> None:
    """A real stop onto a wet tyre must not abort a dry replay."""
    view = CarView(
        driver="VER",
        lap=10,
        total_laps=50,
        position=1,
        compound="MEDIUM",
        tyre_age=9,
        stops=0,
        gap_ahead_s=None,
        gap_behind_s=1.0,
        phase=TrackPhase.GREEN,
        phase_laps_remaining=0,
        compounds_used=frozenset({"MEDIUM"}),
        available_compounds=("SOFT", "MEDIUM", "HARD"),
        must_still_use_another_compound=True,
    )
    strategy = ReplayStops(stops={"VER": ((10, "INTERMEDIATE"),)}, fallback_compound="HARD")
    call = strategy.decide(view)
    assert call is not None
    assert call.compound == "HARD"


# --------------------------------------------------------------------------- metrics


def test_a_perfect_prediction_scores_perfectly() -> None:
    """Comparing a race against itself must bottom out every error metric.

    Without this, a sign error or a mismatched join could leave the metrics reporting a floor no
    model could ever reach, and every accuracy claim would be measured against the wrong zero.
    """
    laps = race_frame(total_laps=25, drivers=8)
    observed = observed_race(laps, LINEAR)

    config = settings()
    simulator = RaceSimulator(parameters(), config)
    race = scenario(config, cars=8, laps=25)
    # Rename the simulated field onto the real one, in the real finishing order, so the simulated
    # race is the real result by construction.
    entrants = tuple(
        entrant.model_copy(update={"driver": observed.finishing_order[index]})
        for index, entrant in enumerate(race.starting_order())
    )
    aligned = race.model_copy(update={"entrants": entrants})
    trace = simulator.run(aligned, ReactiveRules(target_stops=1), bundle("perfect"))

    result = compare(trace, observed)
    assert result.n_compared == 8
    if trace.finishing_order() == observed.finishing_order:
        assert result.position_mae == 0.0
        assert result.order_spearman == pytest.approx(1.0)
        assert result.winner_correct
        assert result.podium_overlap == 3


def test_the_grid_baseline_is_computed_over_the_same_drivers() -> None:
    """A race with retirements must not credit the baseline for slots nobody finished in."""
    laps = race_frame(total_laps=25, drivers=10)
    retired = laps.filter(~((pl.col("driver") == "D01") & (pl.col("lap") > 5)))
    observed = observed_race(retired, LINEAR)

    config = settings()
    simulator = RaceSimulator(parameters(), config)
    race = scenario(config, cars=10, laps=25)
    entrants = tuple(
        entrant.model_copy(update={"driver": f"D{index:02d}"})
        for index, entrant in enumerate(race.starting_order())
    )
    trace = simulator.run(
        race.model_copy(update={"entrants": entrants}), ReactiveRules(), bundle("base")
    )

    result = compare(trace, observed)
    assert result.n_compared == 9
    assert result.n_retired_excluded == 1
    # Ranks run 1..9 over the compared drivers, so the baseline error is bounded by the field size.
    assert 0.0 <= result.grid_position_mae < 9.0


def test_a_comparison_of_two_different_races_is_an_error() -> None:
    """Scoring against a real race with no driver in common is a mistake, not a zero."""
    laps = race_frame(total_laps=20, drivers=6)
    observed = observed_race(laps, LINEAR)

    config = settings()
    simulator = RaceSimulator(parameters(), config)
    race = scenario(config, cars=6, laps=20)
    # A field sharing no driver with the real race; the fixture's are D00..D05.
    strangers = tuple(
        entrant.model_copy(update={"driver": f"X{index:02d}"})
        for index, entrant in enumerate(race.starting_order())
    )
    trace = simulator.run(
        race.model_copy(update={"entrants": strangers}), ReactiveRules(), bundle("mismatch")
    )
    with pytest.raises(ValueError, match="no driver appears in both"):
        compare(trace, observed)


def test_the_pace_source_travels_with_the_comparison() -> None:
    """An oracle run must be labelled, so it can never be read as a prediction."""
    laps = race_frame(total_laps=20, drivers=6)
    observed = observed_race(laps, LINEAR)

    config = settings()
    simulator = RaceSimulator(parameters(), config)
    race = scenario(config, cars=6, laps=20)
    entrants = tuple(
        entrant.model_copy(update={"driver": f"D{index:02d}"})
        for index, entrant in enumerate(race.starting_order())
    )
    trace = simulator.run(
        race.model_copy(update={"entrants": entrants}), ReactiveRules(), bundle("label")
    )

    assert compare(trace, observed).pace_source == "grid"
    assert compare(trace, observed, pace_source="oracle").pace_source == "oracle"


def test_car_keys_are_what_the_comparison_joins_on() -> None:
    """A guard on the identity columns every layer agrees to use."""
    assert CAR_KEYS == ("season", "round", "driver")


def test_lapped_cars_are_ordered_deterministically() -> None:
    """Cross-process reproducibility, which a set iteration order once silently broke.

    Lapped cars have no position on the leader's final lap and no quoted race time, so an ordering
    keyed on time alone compares several infinities and lets a stable sort preserve the iteration
    order of a set -- which Python hashes differently in every process. The validation numbers were
    reproducible within a run and moved between runs. The order is now fixed by distance, then time,
    then name, so it depends on nothing but the data.
    """
    laps = race_frame(total_laps=30, drivers=10)
    # Four cars a lap down, with no final-lap row and no complete race time.
    lapped = {"D06", "D07", "D08", "D09"}
    frame = laps.filter(~(pl.col("driver").is_in(list(lapped)) & (pl.col("lap") == 30)))

    orders = {observed_race(frame, LINEAR).finishing_order for _ in range(8)}
    assert len(orders) == 1, "the finishing order must not depend on iteration order"

    order = orders.pop()
    assert set(order[-4:]) == lapped, "lapped cars fall in behind the lead-lap finishers"
    assert order[-4:] == ("D06", "D07", "D08", "D09"), "and in a defined order"

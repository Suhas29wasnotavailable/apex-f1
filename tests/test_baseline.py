"""The parameters P1's fixed effects absorbed, recovered from races built from known truth.

Every fit here is checked the same way the pace model was: generate races whose absolute lap time,
season trend, grid-to-pace relationship and passing behaviour are chosen, then ask the estimator to
find them back. A recovered number is evidence; a plausible-looking number is not.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from apex.calibrate.baseline import car_pace_frame, fit_baseline, tyre_delta
from apex.calibrate.estimates import Estimate
from apex.calibrate.exclusions import clean_laps
from apex.calibrate.features import add_features
from apex.calibrate.grid import PIT_LANE_GRID, fit_grid_pace
from apex.calibrate.overtaking import (
    CircuitOvertaking,
    OvertakingModel,
    _logistic_fit,
    fit_overtaking,
    passing_attempts,
)
from apex.calibrate.pace import fit_pace
from tests.factories import lap_frame
from tests.test_pace import BASE_LAP_TIME, LINEAR, TRUE_OFFSET, simulated_race


def prepared(rows: list[dict[str, object]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Featured and cleaned laps, through the production pipeline."""
    featured = add_features(lap_frame(rows), LINEAR)
    return clean_laps(featured, LINEAR).laps, featured


def season(
    circuits: dict[str, int], *, seed: int = 5, **kwargs: object
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    round_number = 0
    for circuit, races in circuits.items():
        for _ in range(races):
            round_number += 1
            rows.extend(
                simulated_race(2023, round_number, circuit, rng=rng, **kwargs)  # type: ignore[arg-type]
            )
    return prepared(rows)


# --------------------------------------------------------------------------- baseline


def test_the_circuit_baseline_recovers_the_lap_time_it_was_generated_from() -> None:
    """The absolute pace level is what the fixed effects threw away, so it must come back exactly.

    The fixture's lap time is ``BASE_LAP_TIME`` plus a per-car pace drawn around zero, so the
    field-median corrected lap is ``BASE_LAP_TIME`` plus the reference compound's offset of zero.
    """
    laps, all_laps = season({"Alpha": 3, "Beta": 3}, drivers=12)
    pace = fit_pace(laps, LINEAR)
    model = fit_baseline(laps, pace, LINEAR, all_laps=all_laps)

    for name in ("Alpha", "Beta"):
        assert model.circuit(name).reference_lap_s.value == pytest.approx(BASE_LAP_TIME, abs=0.5)


def test_a_slower_circuit_is_fitted_slower() -> None:
    """Two circuits generated ten seconds apart must be reported ten seconds apart."""
    rng = np.random.default_rng(3)
    rows: list[dict[str, object]] = []
    for index in range(3):
        rows.extend(simulated_race(2023, index + 1, "Quick", rng=rng, drivers=10))
    for index in range(3):
        slow = simulated_race(2023, index + 4, "Slow", rng=rng, drivers=10)
        for row in slow:
            row["lap_time_s"] = float(row["lap_time_s"]) + 10.0  # type: ignore[arg-type]
        rows.extend(slow)

    laps, all_laps = prepared(rows)
    pace = fit_pace(laps, LINEAR)
    model = fit_baseline(laps, pace, LINEAR, all_laps=all_laps)

    difference = (
        model.circuit("Slow").reference_lap_s.value - model.circuit("Quick").reference_lap_s.value
    )
    assert difference == pytest.approx(10.0, abs=0.5)


def test_the_season_trend_separates_the_era_from_the_circuit() -> None:
    """A season-wide offset must land in the season effect, not in the circuits.

    This is the decomposition the module exists for: without it, a circuit raced only in the slow
    season would be reported as a permanently slow circuit.
    """
    # Twenty cars and two races per circuit-season: each race draws its own per-car pace, so the
    # field median wanders, and the season effect is only identifiable above that noise with enough
    # races behind it. An earlier version used ten cars and one race and failed on the wander.
    rng = np.random.default_rng(9)
    rows: list[dict[str, object]] = []
    round_number = 0
    for circuit in ("Alpha", "Beta", "Gamma"):
        for season_year, penalty in ((2022, 2.0), (2023, 0.0)):
            for _ in range(2):
                round_number += 1
                race = simulated_race(season_year, round_number, circuit, rng=rng, drivers=20)
                for row in race:
                    row["lap_time_s"] = float(row["lap_time_s"]) + penalty  # type: ignore[arg-type]
                rows.extend(race)

    laps, all_laps = prepared(rows)
    pace = fit_pace(laps, LINEAR)
    model = fit_baseline(laps, pace, LINEAR, all_laps=all_laps)

    assert model.reference_season == 2023
    offsets = {entry.season: entry.offset_s.value for entry in model.seasons}
    assert offsets[2023] == pytest.approx(0.0, abs=1e-9)
    assert offsets[2022] == pytest.approx(2.0, abs=0.5)

    # Every circuit was raced in both seasons, so none should carry the era's penalty.
    levels = [entry.reference_lap_s.value for entry in model.circuits]
    assert max(levels) - min(levels) < 1.0


def test_the_standing_start_is_measured_from_the_laps_the_rules_exclude() -> None:
    """The opening-lap penalty is recovered even though no cleaned lap contains it."""
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    for index in range(4):
        race = simulated_race(2023, index + 1, "Alpha", rng=rng, drivers=10, total_laps=40)
        # The fixture starts at lap two, so an opening lap is added carrying a known penalty.
        opening = [
            dict(row) | {"lap": 1, "tyre_life": 1, "lap_time_s": float(row["lap_time_s"]) + 9.0}  # type: ignore[arg-type]
            for row in race
            if row["lap"] == 2
        ]
        rows.extend(opening)
        rows.extend(race)

    laps, all_laps = prepared(rows)
    pace = fit_pace(laps, LINEAR)
    model = fit_baseline(laps, pace, LINEAR, all_laps=all_laps)

    assert model.standing_start_s is not None
    assert model.standing_start_s.value == pytest.approx(9.0, abs=1.0)


def test_the_standing_start_is_unmeasured_without_the_unfiltered_laps() -> None:
    """Asking for it from cleaned laps alone must report ignorance, not a number."""
    laps, _ = season({"Alpha": 2}, drivers=8)
    pace = fit_pace(laps, LINEAR)
    assert fit_baseline(laps, pace, LINEAR).standing_start_s is None


def test_a_baseline_cannot_be_fitted_from_nothing() -> None:
    """An empty parameter would look like a result, so it is an error instead."""
    laps, _ = season({"Alpha": 2}, drivers=8)
    pace = fit_pace(laps, LINEAR)
    with pytest.raises(ValueError, match="no car-race has enough clean laps"):
        fit_baseline(add_features(lap_frame([]), LINEAR), pace, LINEAR)


# --------------------------------------------------------------------------- tyre delta


def test_the_tyre_term_is_computed_identically_for_calibration_and_simulation() -> None:
    """The one quantity both layers must agree on, asserted rather than trusted.

    The overtaking model is fitted on the pace difference this produces and the simulator evaluates
    the same difference every lap. If the two ever diverged, the fitted coefficient would be applied
    to a different quantity than it was estimated from, silently.
    """
    laps, _ = season({"Alpha": 3}, drivers=10)
    pace = fit_pace(laps, LINEAR)

    frame = laps.with_columns(tyre_delta(pace).alias("delta")).drop_nulls("delta")
    for row in frame.head(50).iter_rows(named=True):
        expected = pace.compound(str(row["compound"])).lap_time_delta(float(row["tyre_age"]))
        assert float(row["delta"]) == pytest.approx(expected, abs=1e-12)


def test_the_compound_offsets_show_up_in_the_tyre_term() -> None:
    laps, _ = season({"Alpha": 3}, drivers=12)
    pace = fit_pace(laps, LINEAR)
    fresh = {name: pace.compound(name).lap_time_delta(0.0) for name in ("SOFT", "MEDIUM", "HARD")}
    assert fresh["SOFT"] == pytest.approx(0.0, abs=1e-9)
    for name in ("MEDIUM", "HARD"):
        assert fresh[name] == pytest.approx(TRUE_OFFSET[name], abs=0.2)


# --------------------------------------------------------------------------- grid to pace


def test_grid_position_predicts_pace_when_the_grid_was_set_by_pace() -> None:
    """The fixture qualifies by true pace, so the fitted curve must be increasing in grid slot."""
    laps, _ = season({"Alpha": 4, "Beta": 4}, drivers=14)
    pace = fit_pace(laps, LINEAR)
    cars = car_pace_frame(laps, pace, LINEAR)
    model = fit_grid_pace(cars, LINEAR)

    predicted = [model.pace_s(slot) for slot in range(1, 15)]
    assert predicted == sorted(predicted)
    assert model.pace_s(1) < model.pace_s(14)
    assert 0.0 <= model.r_squared <= 1.0


def test_a_pit_lane_start_is_excluded_from_the_fit_but_still_gets_a_pace() -> None:
    """Grid zero is a penalty, not the twenty-first-fastest car."""
    laps, _ = season({"Alpha": 3}, drivers=10)
    pace = fit_pace(laps, LINEAR)
    cars = car_pace_frame(laps, pace, LINEAR)

    # Send the quickest car to the pit lane. The fit must not conclude the back of the grid is fast.
    quickest = cars.sort("pace_s")["driver"][0]
    doctored = cars.with_columns(
        pl.when(pl.col("driver") == quickest)
        .then(pl.lit(PIT_LANE_GRID))
        .otherwise(pl.col("grid_position"))
        .cast(pl.Int16)
        .alias("grid_position")
    )
    model = fit_grid_pace(doctored, LINEAR)
    assert model.n_car_races < doctored.height
    # It still has to answer for a pit-lane start, at the back of the reference grid.
    assert model.pace_s(PIT_LANE_GRID) == pytest.approx(model.pace_s(model.reference_field_size))


def test_the_grid_fit_refuses_a_sample_too_small_to_identify_it() -> None:
    laps, _ = season({"Alpha": 2}, drivers=8)
    pace = fit_pace(laps, LINEAR)
    cars = car_pace_frame(laps, pace, LINEAR).head(2)
    with pytest.raises(ValueError, match="car-races with a grid position"):
        fit_grid_pace(cars, LINEAR)


# --------------------------------------------------------------------------- overtaking


def test_passing_attempts_only_counts_adjacent_cars_racing_each_other() -> None:
    laps, all_laps = season({"Alpha": 3}, drivers=10)
    pace = fit_pace(laps, LINEAR)
    cars = car_pace_frame(laps, pace, LINEAR)
    attempts = passing_attempts(all_laps, cars, pace, LINEAR)

    assert attempts.height > 0
    assert attempts["gap_s"].max() <= 5.0  # type: ignore[operator]
    assert (attempts["behind_driver"] != attempts["ahead_driver"]).all()
    assert attempts["gap_s"].min() >= 0.0  # type: ignore[operator]


def known_overtaking(difficulty: float = -0.4) -> OvertakingModel:
    """A passing model with chosen coefficients, for testing what the published curve does.

    The shape is asserted against a model whose coefficients are given rather than against one
    fitted from the synthetic season. The fixture's gaps follow a fixed cycle unrelated to its
    passes, so it generates no gap-to-pass relationship at all and a coefficient fitted from it is
    noise -- an earlier version of these tests asserted the sign of that noise and failed.
    """
    return OvertakingModel(
        pooled=CircuitOvertaking(
            circuit=None,
            difficulty=Estimate(value=difficulty, level=0.95, n_obs=1000),
            n_pairs=1000,
            n_passes=50,
            n_races=10,
        ),
        circuits=(),
        gap_coefficient=Estimate(value=-3.4, level=0.95, n_obs=1000),
        pace_coefficient=Estimate(value=2.2, level=0.95, n_obs=1000),
        between_circuit_std=1.0,
        max_gap_s=5.0,
        n_pairs=1000,
        n_passes=50,
        n_races=10,
    )


def test_a_faster_car_is_more_likely_to_pass() -> None:
    model = known_overtaking()
    rising = [model.pass_probability(0.5, advantage) for advantage in (0.0, 0.5, 1.0, 1.5)]
    assert rising == sorted(rising)
    assert all(0.0 <= value <= 1.0 for value in rising)


def test_a_wider_gap_is_less_likely_to_pass() -> None:
    model = known_overtaking()
    falling = [model.pass_probability(gap, 1.0) for gap in (0.2, 1.0, 2.0, 4.0)]
    assert falling == sorted(falling, reverse=True)


def test_a_harder_circuit_yields_a_lower_pass_probability() -> None:
    easy = known_overtaking(difficulty=1.0)
    hard = known_overtaking(difficulty=-3.5)
    assert hard.pass_probability(0.5, 0.5) < easy.pass_probability(0.5, 0.5)


def test_the_logistic_estimator_recovers_coefficients_it_was_generated_from() -> None:
    """The numerical core of the overtaking model, checked against a known truth.

    Outcomes are drawn from a logistic in gap and pace advantage with chosen coefficients, and the
    estimator has to find them back. This is the property that has to hold before the coefficients
    fitted from real races mean anything.
    """
    rng = np.random.default_rng(17)
    n = 40_000
    gap = rng.uniform(0.0, 5.0, size=n)
    advantage = rng.normal(0.0, 0.6, size=n)
    truth = np.array([-0.5, -0.8, 2.0])

    design = np.column_stack([np.ones(n), gap, advantage])
    probability = 1.0 / (1.0 + np.exp(-(design @ truth)))
    outcome = (rng.random(n) < probability).astype(np.float64)
    clusters = np.repeat(np.arange(200), n // 200)

    beta, errors = _logistic_fit(design, outcome, clusters)
    for fitted, expected, error in zip(beta, truth, errors, strict=True):
        assert fitted == pytest.approx(expected, abs=max(4.0 * error, 0.05))


def test_no_pass_is_predicted_beyond_the_window_the_model_saw() -> None:
    """Outside the fitted gap window the model must decline rather than extrapolate."""
    model = known_overtaking()
    assert model.pass_probability(model.max_gap_s + 0.01, 5.0) == 0.0
    assert model.pass_probability(model.max_gap_s, 5.0) > 0.0


def test_an_unknown_circuit_falls_back_to_the_pooled_difficulty() -> None:
    laps, all_laps = season({"Alpha": 3}, drivers=10)
    pace = fit_pace(laps, LINEAR)
    cars = car_pace_frame(laps, pace, LINEAR)
    model = fit_overtaking(all_laps, cars, pace, LINEAR)

    assert model.difficulty_at("Nowhere") == model.pooled.difficulty.value
    with pytest.raises(KeyError, match="Nowhere"):
        model.circuit("Nowhere")


def test_overtaking_cannot_be_fitted_without_a_running_order() -> None:
    """The fixture bug that produced no adjacent pairs at all, now an explicit error."""
    laps, all_laps = season({"Alpha": 2}, drivers=8)
    pace = fit_pace(laps, LINEAR)
    cars = car_pace_frame(laps, pace, LINEAR)
    flattened = all_laps.with_columns(pl.lit(1, dtype=pl.Int16).alias("position"))
    with pytest.raises(ValueError, match="no adjacent racing pairs"):
        fit_overtaking(flattened, cars, pace, LINEAR)

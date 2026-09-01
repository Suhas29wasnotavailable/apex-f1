"""The pace model, checked against races whose true parameters are known.

Real telemetry can only tell you whether a coefficient looks plausible. Simulated races built
from chosen parameters can tell you whether the estimator recovers them, which is the property
that has to hold before any number fitted from real data is worth reading. Every test here
generates races from a known truth and asks the fit to find it back.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from apex.calibrate.exclusions import clean_laps
from apex.calibrate.features import add_features
from apex.calibrate.pace import fit_circuit_pace, fit_pace
from apex.config import AppSettings, CalibrationSettings, ExclusionSettings
from tests.factories import lap, lap_frame

TRUE_DEGRADATION = {"SOFT": 0.12, "MEDIUM": 0.08, "HARD": 0.05}
TRUE_OFFSET = {"SOFT": 0.0, "MEDIUM": 0.4, "HARD": 0.8}
TRUE_PROGRESS = -3.0
TRUE_DIRTY_AIR = 0.6
TRUE_PIT_LOSS = 21.0
SATURATION = 2.0
BASE_LAP_TIME = 90.0
CLEAN_AIR_GAP = 40.0
"""Gap the leader runs at: far enough that the dirty-air term is zero to nine decimal places."""

# Degradation is linear in these fixtures, so the quadratic term would be estimating zero and
# is switched off rather than left to add noise to a test about the linear one.
LINEAR = AppSettings(
    calibration=CalibrationSettings(
        degradation_degree=1,
        min_laps_per_cell=20,
        dirty_air_saturation_s=SATURATION,
        exclusions=ExclusionSettings(min_clean_laps_per_stint=4),
    )
)


def simulated_race(
    season: int,
    round_number: int,
    circuit: str,
    *,
    drivers: int = 10,
    total_laps: int = 50,
    degradation: dict[str, float] | None = None,
    noise: float = 0.05,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    """One race generated from known parameters.

    Every effect the model claims to separate is present and varying. Each car runs two stints on
    different compounds, which is what lets tyre age be told apart from fuel burn. Each car has
    its own baseline pace, for the fixed effects to absorb. And the field is spread out in a
    pattern that changes lap by lap, so the traffic penalty varies *within* a car -- if it varied
    only between cars, the fixed effect would absorb it and there would be nothing left to fit.

    Crossing times are built cumulatively down the field, so the gaps the features module later
    recovers from ``lap_end_s`` are exactly the gaps the lap times were generated from.
    """
    slopes = degradation or TRUE_DEGRADATION
    compounds = list(slopes)
    pace = {index: float(rng.normal(0.0, 1.5)) for index in range(drivers)}
    plans = {
        index: (compounds[index % len(compounds)], compounds[(index + 1) % len(compounds)])
        for index in range(drivers)
    }
    # Pit laps differ between cars. This matters more than it looks: if every car stopped on the
    # same lap, tyre age would be an exact linear function of race progress across the whole
    # field, and the two effects the model exists to separate would not be identified at all.
    pit_lap = {index: total_laps // 2 + (index % 5) - 2 for index in range(drivers)}

    rows: list[dict[str, object]] = []
    for lap_number in range(2, total_laps + 1):
        crossing = 1000.0 * lap_number
        for index in range(drivers):
            gap = CLEAN_AIR_GAP if index == 0 else 0.5 + 3.0 * ((index + lap_number) % 4)
            crossing += gap

            stop = pit_lap[index]
            stint_number = 1 if lap_number <= stop else 2
            compound = plans[index][stint_number - 1]
            age = float(lap_number - 1 if stint_number == 1 else lap_number - stop)
            progress = (lap_number - 1) / (total_laps - 1)
            is_in_lap = lap_number == stop
            is_out_lap = lap_number == stop + 1
            lap_time = (
                BASE_LAP_TIME
                + pace[index]
                + TRUE_OFFSET[compound]
                + slopes[compound] * age
                + TRUE_PROGRESS * progress
                + TRUE_DIRTY_AIR * float(np.exp(-gap / SATURATION))
                + (TRUE_PIT_LOSS / 2 if is_in_lap or is_out_lap else 0.0)
                + float(rng.normal(0.0, noise))
            )
            rows.append(
                lap(
                    season=season,
                    round=round_number,
                    circuit=circuit,
                    event=f"{circuit} Grand Prix",
                    driver=f"D{index:02d}",
                    lap=lap_number,
                    stint=stint_number,
                    compound=compound,
                    tyre_life=int(age),
                    lap_time_s=lap_time,
                    lap_end_s=crossing,
                    is_pit_in=is_in_lap,
                    pit_in_s=crossing if is_in_lap else None,
                    is_pit_out=is_out_lap,
                    pit_out_s=crossing - 20.0 if is_out_lap else None,
                )
            )
    return rows


def simulated_season(circuits: dict[str, int], *, seed: int = 7, **kwargs: object) -> pl.DataFrame:
    """Several races per circuit, generated from the same truth and put through the real pipeline.

    Features then exclusions, in that order and by the production functions, so that a test of the
    fit is a test of what the fit will actually be handed -- including the in- and out-laps the
    rules are supposed to remove.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    round_number = 0
    for circuit, races in circuits.items():
        for _ in range(races):
            round_number += 1
            rows.extend(
                simulated_race(2023, round_number, circuit, rng=rng, **kwargs)  # type: ignore[arg-type]
            )
    return clean_laps(add_features(lap_frame(rows), LINEAR), LINEAR).laps


def test_degradation_slopes_are_recovered() -> None:
    model = fit_pace(simulated_season({"Alpha": 3, "Beta": 3}), LINEAR)
    for compound, truth in TRUE_DEGRADATION.items():
        assert model.compound(compound).coefficients[0].value == pytest.approx(truth, abs=0.01)


def test_compound_offsets_are_recovered_relative_to_the_reference() -> None:
    """Level and slope are separately identified; a model without offsets would confound them."""
    model = fit_pace(simulated_season({"Alpha": 3, "Beta": 3}), LINEAR)
    assert model.reference_compound == "SOFT"
    assert model.compound("SOFT").offset_s is None
    for compound in ("MEDIUM", "HARD"):
        offset = model.compound(compound).offset_s
        assert offset is not None
        assert offset.value == pytest.approx(TRUE_OFFSET[compound], abs=0.05)


def test_the_progress_effect_is_recovered_because_tyre_age_resets_and_fuel_does_not() -> None:
    model = fit_pace(simulated_season({"Alpha": 3, "Beta": 3}), LINEAR)
    assert model.fuel_effect_s.value == pytest.approx(TRUE_PROGRESS, abs=0.1)


def test_intervals_cover_the_truth() -> None:
    """An interval that does not contain the answer is worse than no interval at all."""
    model = fit_pace(simulated_season({"Alpha": 4, "Beta": 4}), LINEAR)
    for compound, truth in TRUE_DEGRADATION.items():
        interval = model.compound(compound).coefficients[0].interval
        assert interval is not None
        assert interval[0] <= truth <= interval[1]


def test_the_dirty_air_penalty_is_recovered_from_cars_running_close() -> None:
    """Traffic is controlled for rather than excluded, so the penalty is itself a fitted number."""
    model = fit_pace(simulated_season({"Alpha": 3, "Beta": 3}), LINEAR)
    assert model.dirty_air_s.value == pytest.approx(TRUE_DIRTY_AIR, abs=0.05)
    assert model.dirty_air_penalty(0.0) == pytest.approx(model.dirty_air_s.value)
    assert model.dirty_air_penalty(SATURATION) == pytest.approx(
        model.dirty_air_s.value * float(np.exp(-1.0))
    )
    assert model.dirty_air_penalty(20.0) < 0.01


def test_a_compound_with_too_few_laps_is_not_given_a_curve_of_its_own() -> None:
    """A curve through a handful of laps is noise wearing the clothes of a parameter."""
    laps = simulated_season({"Alpha": 3})
    scarce = laps.filter(pl.col("compound") == "HARD").head(10)
    trimmed = pl.concat([laps.filter(pl.col("compound") != "HARD"), scarce])

    model = fit_pace(trimmed, LINEAR)
    assert {entry.compound for entry in model.degradation} == {"SOFT", "MEDIUM"}
    assert model.n_laps == trimmed.height - scarce.height, "its laps are excluded, not reassigned"


def test_a_race_with_nothing_fittable_fails_loudly() -> None:
    with pytest.raises(ValueError, match="degradation curve"):
        fit_pace(add_features(lap_frame([lap(lap=n) for n in range(2, 8)]), LINEAR), LINEAR)


# ------------------------------------------------------------------------------ pooling


def test_a_circuit_that_genuinely_differs_keeps_its_own_slope() -> None:
    """Monza and Monaco really do degrade differently, and pooling must not erase that."""
    rng = np.random.default_rng(5)
    gentle = {"SOFT": 0.04, "MEDIUM": 0.03, "HARD": 0.02}
    rows: list[dict[str, object]] = []
    round_number = 0
    for _ in range(5):
        round_number += 1
        rows.extend(simulated_race(2023, round_number, "Harsh", rng=rng))
    for _ in range(5):
        round_number += 1
        rows.extend(simulated_race(2023, round_number, "Gentle", degradation=gentle, rng=rng))

    fitted = fit_circuit_pace(
        clean_laps(add_features(lap_frame(rows), LINEAR), LINEAR).laps, LINEAR
    )
    harsh = fitted.circuits["Harsh"].compound("SOFT").coefficients[0]
    gentle_fit = fitted.circuits["Gentle"].compound("SOFT").coefficients[0]

    assert harsh.value > gentle_fit.value
    assert harsh.value == pytest.approx(TRUE_DEGRADATION["SOFT"], abs=0.03)
    assert gentle_fit.value == pytest.approx(gentle["SOFT"], abs=0.03)
    assert fitted.between_std["age1_SOFT"] > 0.0


def test_circuits_that_do_not_differ_are_pooled_into_one_answer() -> None:
    fitted = fit_circuit_pace(simulated_season({"Alpha": 3, "Beta": 3, "Gamma": 3}), LINEAR)
    slopes = {
        name: model.compound("SOFT").coefficients[0].value
        for name, model in fitted.circuits.items()
    }
    assert max(slopes.values()) - min(slopes.values()) < 0.02


def test_a_circuit_with_one_race_takes_the_pooled_value_and_says_so() -> None:
    """ADR-0002's promise: a sparse circuit gets the pooled number, labelled as such."""
    rng = np.random.default_rng(9)
    rows: list[dict[str, object]] = []
    for round_number in (1, 2, 3):
        rows.extend(simulated_race(2023, round_number, "Common", rng=rng))
    rows.extend(simulated_race(2023, 4, "Rare", rng=rng))

    fitted = fit_circuit_pace(
        clean_laps(add_features(lap_frame(rows), LINEAR), LINEAR).laps, LINEAR
    )
    assert fitted.pooled_only == ("Rare",)
    rare = fitted.circuits["Rare"].compound("SOFT").coefficients[0]
    assert rare.shrinkage == 0.0
    assert rare.value == pytest.approx(fitted.pooled.compound("SOFT").coefficients[0].value)
    assert rare.n_obs < fitted.pooled.compound("SOFT").coefficients[0].n_obs


def test_every_circuit_in_the_data_gets_a_model() -> None:
    fitted = fit_circuit_pace(simulated_season({"Alpha": 3, "Beta": 1}), LINEAR)
    assert set(fitted.circuits) == {"Alpha", "Beta"}
    assert set(fitted.fitted) | set(fitted.pooled_only) == {"Alpha", "Beta"}

"""Partial pooling: the machinery that keeps a two-race circuit from claiming to know things.

The properties asserted here are the ones ADR-0002 relies on when it promises honestly-sized
intervals for sparsely-observed circuits.
"""

from __future__ import annotations

import pytest

from apex.calibrate.estimates import Estimate
from apex.calibrate.pooling import effective_standard_error, partial_pool

LEVEL = 0.95


def estimate(
    value: float, std_error: float, n_obs: int = 500, dof: float | None = None
) -> Estimate:
    return Estimate.from_standard_error(
        value, std_error, n_obs=n_obs, level=LEVEL, degrees_of_freedom=dof
    )


def test_a_noisy_group_is_pulled_further_than_a_precise_one() -> None:
    pooling = partial_pool(
        {
            "precise": estimate(0.20, 0.005),
            "noisy": estimate(0.05, 0.080),
            "other": estimate(0.14, 0.010),
        },
        level=LEVEL,
    )
    precise = pooling.estimates["precise"]
    noisy = pooling.estimates["noisy"]
    assert precise.shrinkage is not None and noisy.shrinkage is not None
    assert precise.shrinkage > noisy.shrinkage
    assert abs(noisy.value - 0.05) > abs(precise.value - 0.20)


def test_groups_that_agree_within_their_noise_collapse_onto_the_mean() -> None:
    """No detectable circuit effect is a finding, and the honest response is one number."""
    pooling = partial_pool(
        {name: estimate(0.10, 0.05) for name in ("a", "b", "c", "d")}, level=LEVEL
    )
    assert pooling.between_std == 0.0
    assert all(entry.shrinkage == 0.0 for entry in pooling.estimates.values())
    assert all(entry.value == pytest.approx(0.10) for entry in pooling.estimates.values())


def test_groups_that_genuinely_differ_keep_most_of_their_own_value() -> None:
    pooling = partial_pool(
        {"low": estimate(0.02, 0.004), "high": estimate(0.30, 0.004), "mid": estimate(0.16, 0.004)},
        level=LEVEL,
    )
    assert pooling.between_std > 0.05
    assert pooling.estimates["high"].value == pytest.approx(0.30, abs=0.01)


def test_shrunk_estimates_are_never_less_certain_than_the_ones_they_came_from() -> None:
    """Borrowing strength is the point: pooling may move a value, but it must not widen it."""
    observations = {
        "a": estimate(0.20, 0.05),
        "b": estimate(0.10, 0.02),
        "c": estimate(0.16, 0.09),
    }
    pooling = partial_pool(observations, level=LEVEL)
    for key, original in observations.items():
        shrunk = pooling.estimates[key].std_error
        assert shrunk is not None
        assert shrunk <= effective_standard_error(original) + 1e-12


def test_evidence_is_weighed_by_interval_width_not_by_standard_error_alone() -> None:
    """A standard error from two clusters is itself barely known; the interval says so."""
    from_many = estimate(0.1, 0.02, dof=200.0)
    from_two = estimate(0.1, 0.02, dof=1.0)
    assert effective_standard_error(from_two) > 5.0 * effective_standard_error(from_many)


def test_a_group_with_no_usable_interval_contributes_nothing_and_takes_the_mean() -> None:
    """One cluster supports no interval at all, so it neither moves nor is moved by anything."""
    pooling = partial_pool(
        {"solid": estimate(0.30, 0.01), "hopeless": estimate(-5.0, 0.01, dof=0.0)}, level=LEVEL
    )
    assert estimate(-5.0, 0.01, dof=0.0).interval is None, "one cluster supports no interval"
    assert pooling.estimates["hopeless"].shrinkage == 0.0
    assert pooling.estimates["hopeless"].value == pytest.approx(0.30, abs=1e-6)
    assert pooling.mean.value == pytest.approx(0.30, abs=1e-6)


def test_the_pooled_mean_leans_on_the_groups_that_know_the_most() -> None:
    pooling = partial_pool(
        {
            "loud": estimate(1.0, 0.5),
            "quiet": estimate(0.0, 0.01),
            "also_quiet": estimate(0.0, 0.01),
        },
        level=LEVEL,
    )
    assert pooling.mean.value < 0.2


def test_a_lone_group_is_returned_unchanged() -> None:
    only = estimate(0.42, 0.03)
    pooling = partial_pool({"solo": only}, level=LEVEL)
    assert pooling.between_std == 0.0
    assert pooling.estimates["solo"].value == only.value
    assert pooling.estimates["solo"].shrinkage == 1.0


def test_pooling_nothing_is_an_error_rather_than_an_empty_answer() -> None:
    with pytest.raises(ValueError, match="nothing to pool"):
        partial_pool({}, level=LEVEL)


def test_every_shrunk_estimate_reports_its_own_group_size() -> None:
    pooling = partial_pool(
        {"a": estimate(0.2, 0.01, n_obs=90), "b": estimate(0.1, 0.01, n_obs=1200)}, level=LEVEL
    )
    assert pooling.estimates["a"].n_obs == 90
    assert pooling.estimates["b"].n_obs == 1200
    assert pooling.mean.n_obs == 1290

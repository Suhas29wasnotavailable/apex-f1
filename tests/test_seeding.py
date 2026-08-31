"""Determinism guarantees.

These are not smoke tests. Every published comparison in this repository depends on agents
facing identical seeded races, which requires that a stream's draws depend only on its own name
and the master seed -- never on what else the run happened to sample first.
"""

from __future__ import annotations

import doctest
import random

import apex.seeding
import numpy as np
import pytest
from apex.seeding import SeedBundle, seed_global_libraries
from hypothesis import given
from hypothesis import strategies as st


def _draw(bundle: SeedBundle, name: str, count: int = 5) -> list[int]:
    return [int(value) for value in bundle.stream(name).integers(0, 1_000_000, count)]


def test_same_seed_reproduces_draws() -> None:
    assert _draw(SeedBundle(master_seed=11), "sc") == _draw(SeedBundle(master_seed=11), "sc")


def test_different_master_seeds_diverge() -> None:
    assert _draw(SeedBundle(master_seed=11), "sc") != _draw(SeedBundle(master_seed=12), "sc")


def test_named_streams_are_independent() -> None:
    bundle = SeedBundle(master_seed=11)
    assert _draw(bundle, "safety_car") != _draw(bundle, "pit_stop")


def test_draw_order_does_not_matter() -> None:
    """Consuming one stream must not move another.

    This is the property that lets a new source of randomness be added in a later phase without
    invalidating results recorded in an earlier one.
    """
    bundle = SeedBundle(master_seed=11)
    first_then_second = (_draw(bundle, "a"), _draw(bundle, "b"))
    second_then_first = (_draw(bundle, "b"), _draw(bundle, "a"))
    assert first_then_second == (second_then_first[1], second_then_first[0])


def test_derive_creates_a_distinct_namespace() -> None:
    bundle = SeedBundle(master_seed=11)
    assert _draw(bundle.derive("race"), "tyre") != _draw(bundle, "tyre")
    assert _draw(bundle.derive("race"), "tyre") != _draw(bundle.derive("quali"), "tyre")


def test_episodes_are_addressable_not_sequential() -> None:
    """Episode 500 is the same race whether or not the preceding 499 were run."""
    bundle = SeedBundle(master_seed=11)
    assert _draw(bundle.episode(500), "sc") == _draw(SeedBundle(master_seed=11).episode(500), "sc")
    assert _draw(bundle.episode(500), "sc") != _draw(bundle.episode(501), "sc")


@pytest.mark.parametrize("bad_seed", [-1, 2**64, True, 3.5, "7"])
def test_invalid_master_seeds_are_rejected(bad_seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        SeedBundle(master_seed=bad_seed)  # type: ignore[arg-type]


def test_empty_stream_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SeedBundle(master_seed=11).stream("")


def test_negative_episode_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SeedBundle(master_seed=11).episode(-1)


def test_golden_values_are_pinned() -> None:
    """Guard against a refactor silently changing every historical result.

    numpy's PCG64 bit generator and BLAKE2b are both stable across versions and platforms, so a
    change here means the derivation logic moved, not the dependency.
    """
    bundle = SeedBundle(master_seed=20260901)
    assert bundle.run_id == "4b56bfd15163"
    assert _draw(bundle.derive("race"), "safety_car", 3) == [733810, 731492, 381920]


def test_seed_global_libraries_is_reproducible() -> None:
    seed_global_libraries(SeedBundle(master_seed=11))
    first = (random.random(), float(np.random.random()))  # noqa: NPY002
    seed_global_libraries(SeedBundle(master_seed=11))
    assert (random.random(), float(np.random.random())) == first  # noqa: NPY002


@given(
    master_seed=st.integers(min_value=0, max_value=2**63),
    name=st.text(min_size=1, max_size=32),
)
def test_identical_bundles_always_agree(master_seed: int, name: str) -> None:
    left = SeedBundle(master_seed=master_seed)
    right = SeedBundle(master_seed=master_seed)
    assert _draw(left, name) == _draw(right, name)


def test_module_docstring_examples_hold() -> None:
    """The example in the docstring is executed, so it cannot drift into fiction."""
    results = doctest.testmod(apex.seeding, verbose=False)
    assert results.failed == 0
    assert results.attempted > 0

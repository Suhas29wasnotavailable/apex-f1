"""Deterministic random-number streams.

Determinism is treated here as an architectural constraint rather than something to retrofit
once results stop reproducing. Every stochastic component in APEX -- safety-car sampling, pit
stop variation, rival behaviour, policy exploration, evaluation episode generation -- draws from
a named stream derived from a single master seed. Two properties follow:

*   **Reproducibility.** The same master seed and the same config reproduce a run exactly.
*   **Independence.** Streams are derived by hashing their name into the seed sequence's spawn
    key, so adding a new source of randomness (or reordering existing ones) cannot perturb the
    draws of any other stream. Without this, adding a feature silently invalidates every
    previously recorded result.

The second property is the one that matters for honest evaluation: agents can be compared on
identical seeded races because the race's randomness does not depend on what the agent does with
its own stream.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Final

import numpy as np

__all__ = ["SeedBundle", "seed_global_libraries"]

_KEY_BYTES: Final = 8
"""Digest width used to turn a stream name into a spawn-key entry (64 bits)."""

_MAX_SEED: Final = 2**64
"""numpy.random.SeedSequence accepts arbitrarily large entropy, but bounding the master seed
keeps run identifiers short and comparable across the codebase."""


def _stream_key(name: str) -> int:
    """Map a stream name to a stable 64-bit integer.

    Python's built-in ``hash`` is deliberately randomised per interpreter process, so it cannot
    be used: the same name would key a different stream on every run. BLAKE2b is stable across
    processes, platforms and Python versions.
    """
    if not name:
        raise ValueError("stream name must be a non-empty string")
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=_KEY_BYTES).digest()
    return int.from_bytes(digest, "big")


@dataclass(frozen=True, slots=True)
class SeedBundle:
    """A reproducible namespace of independent random streams.

    Args:
        master_seed: The single seed that determines every stream in the run.
        path: Spawn-key prefix identifying this bundle's position in the derivation tree.
            Set by :meth:`derive`; construct top-level bundles with the default.

    Example:
        >>> bundle = SeedBundle(master_seed=7)
        >>> race = bundle.derive("race")
        >>> int(race.stream("safety_car").integers(0, 100))
        94
    """

    master_seed: int
    path: tuple[int, ...] = field(default=())

    def __post_init__(self) -> None:
        if not isinstance(self.master_seed, int) or isinstance(self.master_seed, bool):
            raise TypeError(f"master_seed must be an int, got {type(self.master_seed).__name__}")
        if not 0 <= self.master_seed < _MAX_SEED:
            raise ValueError(f"master_seed must be in [0, 2**64), got {self.master_seed}")

    def stream(self, name: str) -> np.random.Generator:
        """Return the generator for a named stream.

        Repeated calls with the same name return generators in the same *initial* state; they are
        independent objects, so consuming one does not advance the other. Callers that need a
        single advancing stream should hold on to the returned generator.
        """
        sequence = np.random.SeedSequence(
            entropy=self.master_seed, spawn_key=(*self.path, _stream_key(name))
        )
        return np.random.default_rng(sequence)

    def derive(self, name: str) -> SeedBundle:
        """Return a child bundle whose streams are independent of every other bundle's."""
        return SeedBundle(master_seed=self.master_seed, path=(*self.path, _stream_key(name)))

    def episode(self, index: int) -> SeedBundle:
        """Return the bundle for one evaluation or training episode.

        Episode ``i`` under a given bundle is fixed regardless of how many episodes ran before
        it, which is what allows every agent to be evaluated on exactly the same set of races.
        """
        if index < 0:
            raise ValueError(f"episode index must be non-negative, got {index}")
        return self.derive(f"episode/{index}")

    @property
    def run_id(self) -> str:
        """Short stable identifier for this bundle, for logging and artefact filenames."""
        material = f"{self.master_seed}:" + ",".join(str(entry) for entry in self.path)
        return hashlib.blake2b(material.encode("utf-8"), digest_size=6).hexdigest()


def seed_global_libraries(bundle: SeedBundle) -> None:
    """Seed library-level global generators.

    APEX code should take generators from :meth:`SeedBundle.stream` rather than rely on global
    state. This exists only to pin the global generators inside third-party libraries that offer
    no way to inject one, so that they cannot become a hidden source of run-to-run variation.
    """
    stream = bundle.derive("global_libraries")
    random.seed(int(stream.stream("python_random").integers(0, 2**63)))
    np.random.seed(int(stream.stream("numpy_legacy").integers(0, 2**32)))  # noqa: NPY002

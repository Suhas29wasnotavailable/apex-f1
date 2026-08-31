"""APEX -- a calibrated digital twin of a Formula 1 race.

The repository tests one falsifiable claim: that dynamic programming solves deterministic race
strategy exactly, and that reinforcement learning only earns its place under stochasticity.
"""

from apex.seeding import SeedBundle, seed_global_libraries

__version__ = "0.1.0"
__all__ = ["SeedBundle", "__version__", "seed_global_libraries"]

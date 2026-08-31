# ADR-0004: Name-keyed random streams for reproducibility

- **Status:** accepted
- **Date:** 2026-09-01

## Context

The headline experiment compares several agents on the same races. That comparison is only
meaningful if the races really are the same -- if the safety car appears on the same lap, the
pit stop loses the same tenths, and the rivals behave the same way, regardless of which agent is
driving and regardless of how many random numbers that agent consumed while deciding.

The common approach is a single global generator seeded at the start of a run. It fails in a way
that is easy to miss: a policy that samples one extra number shifts every subsequent draw, so
two agents on "seed 42" face different races. Results degrade silently rather than erroring, and
adding any new source of randomness in a later phase invalidates every number recorded earlier.

## Decision

All randomness is drawn from `SeedBundle`, which derives an independent `numpy.random.Generator`
per **named** stream. The name is hashed with BLAKE2b into the seed sequence's spawn key.
Python's built-in `hash` is unusable here because it is randomised per process, so the same name
would key a different stream on each run.

Three properties follow, each covered by a test:

- **Order independence.** Consuming stream `a` does not move stream `b`.
- **Addressability.** `bundle.episode(500)` is the same race whether or not the preceding 499
  ran, so evaluation can be sharded or resumed without changing results.
- **Extension safety.** Adding a new stream in a later phase cannot perturb existing ones, so
  earlier recorded results stay valid.

Golden values are pinned in the test suite: BLAKE2b and numpy's PCG64 are both stable across
versions and platforms, so a change in those values means the derivation logic moved.

## Consequences

- Every stochastic component must name its stream. Slightly more ceremony at each call site.
- Agent comparisons are apples-to-apples by construction rather than by convention.
- Library-level global generators are pinned separately by `seed_global_libraries`, purely to
  stop third-party code becoming a hidden source of variation.

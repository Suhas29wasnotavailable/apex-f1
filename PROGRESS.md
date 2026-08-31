# Progress

One entry per phase: what shipped, what broke, what is next, and what is still open.
Every phase ends with `make check` green and this file updated.

---

## P0 — Scaffold — ✅ complete (2026-09-01)

### Shipped

- **`uv` project on Python 3.11.16**, locked (`uv.lock` committed), `ruff` and `mypy --strict`
  configured, `pytest` with `hypothesis`, `pre-commit`, and a GitHub Actions workflow running
  lint, format, types and tests on every push.
- **`apex.seeding`** — name-keyed deterministic random streams (ADR-0004). Streams are derived
  by hashing their name into a `numpy.random.SeedSequence` spawn key, so consuming one stream
  never moves another, `episode(n)` is addressable rather than sequential, and adding a new
  source of randomness in a later phase cannot invalidate results recorded in an earlier one.
- **`apex.config`** — frozen, validated, layered settings (ADR-0003). Precedence: constructor
  arguments → `APEX_` environment variables → `defaults.yaml`. Cross-field invariants are
  enforced in the model: train and holdout seasons cannot overlap, and no season may precede
  FastF1's 2018 telemetry coverage. `config_hash()` excludes machine-specific paths so the same
  experiment stamps identically in a container and on a laptop.
- **`apex.cli`** — `apex version`, `apex config`, `apex seed`.
- **Four ADRs**, including the calibration scope and validation split agreed before any code was
  written (ADR-0002).
- **35 tests**, all green; `ruff` and `mypy --strict` clean on 9 source files.

### What broke

- A worked example in the `apex.seeding` docstring stated a value that the code did not produce.
  Caught before commit. The module's doctests are now executed by the test suite, so an example
  in this repository cannot drift into fiction.
- `Generator.integers(0, 2**64)` raises on the int64 bound; the global-library seeding helper now
  draws below `2**63`.

### Verified by evidence

- `make check` is green: `ruff check`, `ruff format --check`, `mypy --strict`, `pytest`.
- Determinism properties are tested, not asserted in prose: order independence, episode
  addressability, master-seed sensitivity, and pinned golden values.
- Package versions are real and locked (FastF1 3.8.3, Gymnasium 1.3.0, Polars 1.44.1 resolved at
  lock time), and FastF1's data surface was read from the current documentation rather than
  recalled — lap columns, track-status codes and the picking helpers used by the P1 exclusion
  rules all exist in 3.8.

### Still assumed (not yet evidence)

- That the exclusion rules encoded in `defaults.yaml` are the right ones. They are informed by
  FastF1's documented semantics but have not been run against a single real lap.
- That ~96 races in 2022–2025, minus wet races, is enough to identify per-circuit safety-car
  hazards under partial pooling. This is the biggest open modelling risk in P1.
- That tyre degradation and the race-progress effect are separable across stints. The argument
  is sound (tyre age resets, fuel does not) but is unverified against data.
- Every parameter value in the simulator. There are none yet.

### Next — P1

Ingest Bahrain and Monaco first (deliberately opposite extremes), prove the schema and one
calibration diagnostic, then run the full 2022–2025 pull in the background.

### Open questions

- **Which Grand Prix is the flagship demo?** Best chosen in P2 from the held-out set, once it is
  visible which races the twin reproduces well. A race with a strategy-decisive safety car makes
  the counterfactual fork immediately legible.
- **Redistribution of derived F1 data.** The deployed demo will ship pre-baked replay JSON
  derived from FastF1 output. Worth confirming what may be redistributed before P7, and shipping
  only derived aggregates (lap times, stint boundaries, track outline) rather than raw telemetry.

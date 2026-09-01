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

---

## P1 — Ingestion and calibration — ✅ complete (2026-09-01)

### Shipped

- **`apex.data`** — real race sessions in, one parquet plus one JSON sidecar per race out.
  Ingestion records what the timing feed said and filters nothing: wet races are stored and
  flagged, not dropped. Deciding which laps may inform a parameter is a modelling decision, so it
  lives in `apex.calibrate` where it can be justified and counted, and changing a rule never
  requires re-downloading anything. Every sidecar carries the lap-schema version and a fingerprint
  of the ingestion-relevant settings, so changing the dry-race threshold forces reclassification
  while moving the train/holdout split invalidates no download.
- **`apex.calibrate`** — the exclusion rules, the derived features, a fixed-effects estimator with
  cluster-robust standard errors, empirical-Bayes pooling, and four fitted models: pace, pit loss,
  full safety car, virtual safety car.
- **`params/params.json`** and its committed JSON schema — the only channel between calibration
  and everything downstream. Nothing else reads a lap.
- **`docs/calibration_report.md`** — the same file rendered for a human to argue with, generated
  from `params.json` alone so the two cannot drift apart.
- **CLI**: `apex ingest`, `races`, `laps`, `calibrate`, `params`; `make ingest`, `make calibrate`.
- **All 92 races of 2022–2025 ingested** — 68 in the training seasons of which 56 are dry, and 24
  held out in 2025 of which 21 are dry. **163 tests**, all green. (ADR-0002 estimated "roughly 96
  races before the dry-race filter"; the calendar actually held 92, and 77 survived the filter.)

### What the data says

Fitted from **56 dry races at 25 circuits** in 2022–2024, using **50,583 of 61,358 laps (82%)**.
2025 is untouched. Every number carries a 95% interval; these are the point estimates.

| Parameter | Estimate |
|---|---|
| Tyre degradation, soft / medium / hard | 0.095 / 0.093 / 0.079 s per lap of tyre age |
| Compound pace, medium / hard vs soft | +0.20 s / +0.31 s at equal tyre age |
| Race progress (fuel burn and track evolution) | −3.12 s from first lap to last |
| Dirty air, directly behind a car | +0.78 s, decaying to +0.47 s at a one-second gap |
| Pit loss, pooled | 22.7 s; 29.4 s at Marina Bay, 19.0 s at Spa-Francorchamps |
| Safety car | 0.0119 per green lap — 54% of races, 4.8 laps long |
| Virtual safety car | 0.0100 per green lap — 48% of races, 2.9 laps long |

The rank order matches the sport in every case that can be checked independently: softer tyres are
faster and degrade quicker, Singapore has the longest pit lane on the calendar and Spa among the
shortest, and roughly two races in three see a neutralisation of some kind.

### What broke

- **The pace model had no compound offsets.** The first specification fitted only tyre-age slopes,
  so the intrinsic pace difference between compounds had nowhere to go and was absorbed into the
  slopes. It showed up as soft tyres degrading *more slowly* than mediums, which is backwards.
  Adding an offset term per compound fixed both halves at once and is a better answer anyway: "the
  hard tyre is slower" and "the hard tyre degrades more slowly" are different claims that point in
  opposite directions for strategy.
- **Circuit fits over-trusted two-cluster standard errors.** A circuit with two races has a
  cluster-robust standard error estimated from two clusters, which can come out far too small by
  luck; Monza's soft-tyre slope was −0.085 s/lap and kept 99% of its own weight through shrinkage.
  Evidence is now weighed by interval width, which accounts for the degrees of freedom behind it.
  Monza moved to −0.021 and kept 56%.
- **Two pooled numbers that disagreed.** Safety-car rates were shrunk toward the average of the
  circuit estimates, which sat at 0.0176 per lap against 0.0119 from the whole calendar, because
  each circuit's small-sample adjustment was being counted once per circuit. Every model now
  shrinks toward the estimate fitted on all the data at once, so a circuit that contributed nothing
  reports exactly the pooled number rather than a second, slightly different one.
- **Agresti-Coull's +2 nearly doubled a rare rate.** With sixty laps at risk per race and a true
  rate near one in eighty, two notional deployments overwhelm the real ones. Switched to the
  Jeffreys prior's +0.5, which moves the pooled rate by 0.0002.
- **Hazard pooling ran on the probability scale** and produced intervals reaching below zero.
  Rates are now pooled and shrunk entirely on the log-odds scale.
- **Infinite intervals do not survive JSON.** An estimate resting on one observation was writing
  `Infinity`, which reads back as `null` and fails validation. An unmeasurable spread is now
  `None`, which is also the more honest description.
- **FastF1 concatenates every track status seen within a lap**, so a lap partly under a virtual
  safety car reads `"671"`. Status exclusion matches codes as substrings, not equality.
- **The FastF1 API rate-limits at 500 calls an hour.** The full pull hit it twice. It is resumable
  by design — one file per race, and a stored race is skipped before its session is fetched — so
  `apex ingest` now stops with an instruction to re-run rather than a traceback.
- **A collinearity in the test fixture that is a real lesson.** Simulated races originally pitted
  every car on the same lap, which makes tyre age an exact linear function of race progress across
  the whole field. The two effects the model exists to separate were not identified at all, and
  the estimator correctly refused to fit rather than returning a number. Varied pit laps fixed it.

### Verified by evidence

- The within-transformation estimator's claim to equal a regression with one dummy per car-race is
  asserted against `statsmodels` — coefficients, cluster-robust standard errors and residual scale
  — rather than assumed. That check runs in CI.
- Every coefficient is recovered from simulated races built from known parameters, and the
  intervals cover the truth.
- The exclusion rules are tested rule by rule, including the two order-dependent properties they
  rely on: the safety-car neighbour is resolved against the unfiltered frame, and the outlier
  median is a clean-lap median.
- `params.json` round-trips exactly, and a test walks the whole object graph asserting that no
  published number lacks an interval.

### Still assumed (not yet evidence)

- **The progress effect absorbs track evolution as well as fuel burn.** Both move monotonically
  through a race and this design cannot separate them. −3.12 s is a combined effect; the fuel
  component alone is smaller.
- **The fitted quadratics are negative**, meaning degradation flattens with tyre age rather than
  cliffing. That is almost certainly selection: a stint only runs long when the tyres are behaving.
  Modelling the pit decision jointly with the pace would fix it and is out of scope for P1.
- **No circuit variation is detectable in safety-car hazard or dirty air.** With one to three races
  per circuit, Monaco and Monza get the same neutralisation rate, which is certainly wrong. It is
  what 56 races support. The full calendar is now ingested, so this will not improve without
  either widening the regulation era — which ADR-0002 rejected for good reasons — or borrowing
  strength from something other than race counts, such as circuit characteristics.
- **The gap to the car ahead is measured at the start/finish line.** It is the right measure for
  lapped traffic but wrong for one crossing after a car pits or retires.
- **Pit loss under a safety car is not fitted.** Those stops are excluded from the green-flag
  estimate; the discount they represent is the single most strategically important number the
  simulator will need, and it is still missing.

### Next — P2

Build the simulator on these parameters and validate it against the held-out 2025 races. The
validation is the point: "digital twin" stays unused as a word until there are published error
metrics against races the calibration never saw.

### Open questions

- **Four circuits still have one training race each** — Shanghai, Le Castellet, Monaco and Imola —
  so they cannot be fitted at all and take the pooled values wholesale. Nothing can be done about
  that within the regulation era: Le Castellet has not been raced since 2022, and the others fall
  where the calendar put them. It is worth stating in `LIMITATIONS.md` at P7, because Monaco is
  precisely the circuit whose parameters differ most from the pooled mean.
- **Is the negative curvature worth modelling around now or later?** It biases long-stint
  predictions optimistically, which is exactly where strategy decisions are made.
- **Which Grand Prix is the flagship demo?** Unchanged from P0: best chosen in P2 from the
  held-out set, once it is visible which races the twin reproduces well.
- **Redistribution of derived F1 data.** Unchanged from P0, and still worth settling before P7.

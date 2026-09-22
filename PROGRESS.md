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

---

## P2 — Simulator and validation — ✅ complete (2026-09-22)

### Shipped

- **`apex.sim`** — an event-driven, lap-discrete, multi-car race simulator that reads nothing but
  `params/params.json`. A scenario in, a lap-by-lap `RaceTrace` out: positions, gaps, tyre state,
  flags, stints and a classification. Determinism is a tested property, not a hope.
- **`apex.eval`** — the validation protocol: real races reduced to the simulator's own vocabulary,
  the metrics that score one against the other, and the runner over the held-out season.
- **Three new fitted models**, because P1's estimator had no reason to publish them and a simulator
  cannot run without them: `baseline` (the absolute pace level the car-race fixed effects absorbed,
  split into circuit and season, plus the standing-start penalty), `grid_pace` (race pace from
  starting position, so a field can be built from pre-race information alone), and `overtaking`
  (pass probability, with per-circuit difficulty). Parameter schema is now version 2.
- **Two numbers P1 listed as missing or wrong**, now measured: pit loss under a safety car, and the
  neutralised lap-time ratio and closing rate.
- **The strategy seam P3 and P4 plug into** — `CarView` in, `PitCall` or nothing out — plus two
  rule-based baselines to be beaten.
- **CLI**: `apex race`, `apex validate`; `make race`, `make validate`.
- **226 tests**, all green, up from 165.

### What the data says

Fitted from the same 56 dry races of 2022–24. The new parameters, each with a 95% interval in
`docs/calibration_report.md`:

| Parameter | Estimate |
|---|---|
| Circuit pace level | 69.9 s at Spielberg to 110.8 s at Spa-Francorchamps |
| Season trend | −1.21 s per lap per season; 2022 cars 2.42 s slower than 2024 |
| Standing start | +7.86 s [7.09, 8.56] over a flying lap, from 31 green starts |
| Grid position → race pace | pole −1.05 s to P20 +0.49 s; r² 0.51, residual 0.47 s |
| Field spread | 0.66 s standard deviation of car pace within a race |
| Pit loss under a safety car | **14.3 s against 22.7 s green — a 8.4 s discount**, from 484 stops |
| Safety car lap / closing rate | 1.358× a green lap; closes at most 14.1 s per lap |
| Virtual safety car lap / closing | 1.232×; closes at most 9.3 s per lap |
| Overtaking, pooled | 4.6% of adjacent pairs pass per lap, from 35,246 pairs |
| Overtaking, spread | Monaco 0.58%, Zandvoort 2.7%, Las Vegas 10.5% |

The overtaking model is the one place per-circuit estimates are strong on their own: every lap
contributes roughly twenty adjacent pairs, so between-circuit spread is 1.43 on the log-odds scale
and clearly non-zero — against exactly zero for the safety-car hazard. Monaco is eighteen times
harder to pass at than Las Vegas, which is the correct answer and one P1 had no way to produce.

### What the validation says — including that it is not good enough yet

21 held-out 2025 races, replay mode: real grid, real pit stops, real safety cars, simulated racing.
18.1 classified finishers compared per race, 1.7 retirements excluded.

| Mode | Position MAE | Grid baseline | Spearman | Winner | Podium | Lap MAE | Gap MAE |
|---|---|---|---|---|---|---|---|
| Anchored pace level | 2.86 | **2.40** | 0.749 | 48% | 2.00 / 3 | 2.39 s | 21.0 s |
| Predicted pace level | 2.71 | **2.40** | 0.755 | 52% | 2.19 / 3 | 2.99 s | 20.4 s |
| *Oracle car pace (leakage)* | *2.53* | *2.40* | *0.785* | *48%* | *2.29 / 3* | *2.37 s* | *18.6 s* |

**The twin does not beat the grid baseline on mean finishing position.** Predicting that everybody
finishes where they qualified scores 2.40; the twin scores 2.86, and wins on only 5 of 21 races.
That is the headline P2 result and it is published rather than buried.

What the twin *does* do: rank correlation 0.749, the right winner in half the races, 8.1 of the real
top ten in its own top ten, and green-flag lap times that are essentially unbiased — **mean error
+0.14 s over 21,611 green laps**, against a fitted residual scale of 0.66 s. It reproduces a race's
texture; it does not yet reproduce which particular cars move.

The diagnosis is specific rather than a shrug. Reality churns the order slightly *more* than the
simulator does — mean 2.75 places from the grid against 2.52 — so the twin is not over-shuffling. It
shuffles roughly the right amount and shuffles **different cars**, and two uncorrelated shuffles of
that size produce almost exactly the error observed. The causes reality churns for are retirements,
damage and driver error, none of which has a parameter in this model. The oracle row bounds how much
of the rest is the pace forecast: handed each car's true race pace, position error falls to 2.53 and
gap error to 18.6 s, still only level with the baseline.

### What broke

- **Bunching a field instantaneously produced four-second laps.** Closing the whole gap to the queue
  in one step handed cars time nobody drove; the recorded lap absorbed the compression. Fixed by
  bounding how fast a gap can close — then fixed *again*, because bounding it by each car's racing
  pace is still illegal under a neutralisation and left neutralised laps 11 s a lap too fast against
  green laps that were unbiased. The closing rate is now a fitted parameter.
- **One strategy object served twenty cars and shared its stop counter.** The first car to pit
  consumed the plan for the entire field; nineteen never stopped, and the two-compound safety net
  dragged them all into the pit lane on the penultimate lap. Strategies are now stateless functions
  of the view, which is also what an RL policy is, and the protocol says so.
- **The timing feed calls one circuit two different things.** Miami is `"Miami"` in some seasons and
  `"Miami Gardens"` in others. Circuit is the key every per-circuit parameter is looked up by, so
  this split one circuit's races in half and sent 2025 to the pooled values. It surfaced as a crash
  on a missing baseline, which is the lucky version; the quiet version is worse.
- **Pit loss under a safety car measured 65 s, three times the green-flag figure.** The green method
  compares the in- and out-laps against the driver's own green neighbours, so it charges the entire
  safety-car slowdown to the stop. Measured against the cars that stayed out instead, it is 14.3 s.
  The first number would have told the simulator to never stop under a safety car.
- **Finishing gaps were compared across cars that ran different numbers of laps.** A car one lap down
  has 57 lap times where the winner has 58, so differencing the sums mixes "slower" with "shorter".
  It inflated the reported gap error from 18.6 s to 35.9 s — the twin looked twice as wrong as it was.
- **The overtaking model was first fitted on whole-race pace.** That makes the undercut invisible: a
  car passes three rivals out of the pit lane because its tyres are fifteen laps younger, not because
  the car improved. Refitting on tyre-adjusted instantaneous pace strengthened the coefficient from
  1.88 to 2.20 and moved the hardest circuits to Monaco, Zandvoort, Marina Bay and Imola — all
  genuinely narrow tracks.
- **A standing start costs 7.9 s and the simulator was not charging it.** The exclusion rules drop
  opening laps for good reasons, so no fitted parameter covered lap one, and every simulated race
  finished several seconds early. Its first interval was ±46 s, because a median of a heavy-tailed
  sample was given a normal standard error and races that began under a safety car were included.
- **Validation was reproducible within a process and not between processes.** Lapped cars have no
  position on the leader's final lap and no quoted race time, so the fallback that orders them
  compared several infinities and left a stable sort preserving the iteration order of a *set* --
  which Python hashes differently in every run. Position error moved in the second decimal place
  between runs of the same command. This is precisely the failure ADR-0004 exists to prevent, and it
  slipped in above the seeding layer, where the seed bundle offers no protection.
- **The test fixture had no running order at all.** Every simulated lap carried `position=1`, so no
  two cars were ever adjacent and the overtaking model had nothing to fit. Positions now emerge from
  cumulative race time and grid slots from true pace, which is what a race weekend produces.

### Verified by evidence

- Every new estimator is checked against races built from known parameters, the same discipline P1
  used: the circuit baseline recovers the lap time it was generated from, a circuit generated 10 s
  slower is reported 10 s slower, a season-wide offset lands in the season effect rather than in the
  circuits, and the logistic estimator recovers coefficients it was given.
- The tyre term is asserted to be computed identically by calibration and simulation, because the
  overtaking coefficient is fitted on the difference it produces and the simulator evaluates that
  same difference every lap.
- Determinism is tested as a property: a seed reproduces a race exactly, different seeds do not, and
  simulating other races first changes nothing — ADR-0004's promise at whole-race scale.
- Lap times are asserted to be physical under every flag, and the hand-computable opening lap is
  asserted against the sum of its fitted effects.
- `apex validate` refuses to run if the parameter file was fitted on a season being validated.

### Still assumed (not yet evidence)

- **No retirements, lapped traffic, damage, driver error, weather change or tyre temperature.** Each
  is absent because no fitted parameter supports it. This is why validation compares classified
  finishers only, and it is the leading candidate for the unexplained position error.
- **A pass takes effect from the next lap**, so a car cannot pass two rivals on one lap.
- **Yellow-flag laps are not modelled.** They are 3.8% of car-laps and the simulator runs them green,
  which is a measured and unfixed source of optimism in race time.
- **Neutralised laps are still 6.2 s a lap too fast**, against green laps biased by +0.14 s. Two
  corrections have already closed most of the original 10.9 s and the residue is real: the queue's
  behaviour is modelled by one closing rate and one lap-time ratio, when the first lap of a
  deployment is far slower than the fourth. Fitting the ratio per lap-of-deployment is the obvious
  next step and needs more neutralisations than 56 races supply.
- **The grid slot spacing of 0.25 s is a modelling choice, not a measurement** — the only one in the
  simulator that is neither fitted nor configured. Lap one is the one lap the pace model cannot
  speak to.
- **The season trend is extrapolated to 2025 from three seasons.** The predicted-pace column carries
  that extrapolation, which is why lap-time error there is 2.99 s against 2.39 s anchored.

### Next — P3

Classical baselines on this simulator: rule-based, dynamic programming, MCTS. The dynamic-programming
solution is the important one, because the project's central claim is that it makes reinforcement
learning unnecessary under deterministic dynamics. Improving the twin's position error is a parallel
track, and the two most promising leads are both measured above: model retirements, and give the
field better pre-race pace than a grid slot alone.

### Open questions

- **Which Grand Prix is the flagship demo?** Now answerable from evidence. Suzuka is reproduced best
  (position MAE 1.3) and Monaco worst (4.56), which is the sport's own reputation for both.
- **Does the position error close with a retirement model?** Retirements are ~1.7 cars a race and are
  the largest single unmodelled cause of order change. Worth trying before anything subtler.
- **Should the two-compound rule be enforced or priced?** It is currently enforced by a safety net
  that stops a car on the penultimate lap. A real penalty would let an agent weigh the trade.
- **Redistribution of derived F1 data.** Unchanged since P0, and still worth settling before P7.

---

## P6a — Watching the race — ✅ complete (2026-09-22)

Brought forward from P6, because a validation table is not a race and the only way to see whether the
simulator behaves like one is to watch it.

### Shipped

- **`apex.viz`** — real track geometry, a replay payload, and a self-contained HTML page. `apex
  replay` writes one file and opens it: no server, no build step, works offline.
- **Track geometry from real position telemetry.** The centreline is resampled to points evenly
  spaced *by distance*, which is what lets the page find a car's position by indexing an array
  rather than searching arc length. Cached per circuit, so the network is touched once.
- **A real speed profile.** Mapping elapsed time linearly onto distance would slide cars through
  corners at straight-line speed. The profile records, for evenly spaced fractions of a lap's time,
  the fraction of its distance covered — read off a real fast lap — so every simulated car follows a
  real lap's shape scaled by its own time.
- **The page**: top-down field on the circuit outline, live timing tower with gaps and tyre state,
  safety car turning the track yellow, pit stops pulled off the racing line, scrubbable timeline
  with neutralisations banded, speed control to 200×, and `#t=0.45` deep links into a moment.
- **18 new tests**, 245 total.

### Verified by evidence

- Three circuits rendered and checked against their real layouts: **Monaco** (hairpin, swimming pool,
  Rascasse), **Silverstone** (Maggotts–Becketts, Stowe) and **Monza**. Lap distances land within
  about 1% of the published figures — Monaco 3,284 m against 3,337 m, Silverstone 5,819 m against
  5,891 m — which is the resampling and the racing line, not an error.
- The safety car is visible doing what the parameters say it should: the track goes yellow, the field
  compresses into a train, the whole field dives into the pit lane, and the two cars furthest back
  are still closing at the fitted catch-up rate rather than teleporting onto the queue.

### What broke

- **Cars still on the grid were ranked last.** A car waiting to start sits just *before* the line, at
  99% of a lap, and ranking on that put the entire grid a lap ahead of whoever had already started —
  so the leader showed P20 before the race began. Drawing position and running order are now
  separate quantities.
- **The payload's own arithmetic did not close.** Crossing times and lap durations were each rounded
  to a millisecond independently, so `end - time` missed the previous lap's `end` by a thousandth.
  Invisible on screen, and still a payload whose consumer cannot trust its own subtraction.
  Durations are now derived from the rounded ends.
- **Every car pitted on the same lap.** The rule-based pit wall gives every car an identical plan,
  which no real team does and which quietly deletes the pit-lane traffic that makes a stop's timing
  matter. Stops are now staggered by an offset derived from the driver's own name — stable per car,
  different between cars, and needing no state. The exception is a safety car, where the whole field
  really does dive in at once, and still does.

### Still assumed (not yet evidence)

- **There is no pit lane in the geometry.** A stopping car is pulled toward the infield and dimmed,
  which reads correctly and is a convention rather than a measured path.
- **Track width is drawn, not measured.** The centreline is real; how wide to stroke it is a choice.
- **No lapped-car handling**, because the simulator models none.

### Next

The rest of P6: a strategy Gantt showing every stint, and the counterfactual fork — pause at any lap,
change one decision, and watch the race diverge from itself. The fork is the interesting half and the
simulator already supports it, since a race can be replayed under an imposed flag schedule.

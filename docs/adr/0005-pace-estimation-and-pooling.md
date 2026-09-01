# ADR-0005: Fixed-effects pace estimation and shrinkage toward the all-data fit

- **Status:** accepted
- **Date:** 2026-09-01

## Context

ADR-0002 settled *which* laps are calibrated from. This one settles how the numbers are actually
estimated, and the three problems that forced the choices.

**Everything constant about a car in a race is correlated with everything being fitted.** A fast
car spends more laps in clean air, pits from the front, and runs different compounds to different
plans than a slow one. Fitting degradation without controlling for the car would attribute a
Red Bull's pace advantage to whatever tyre it happened to be on.

**Laps within a race are not independent.** Track temperature, tyre allocation, surface state and
the day's incidents are shared by every car on track. Treating fifty thousand laps as fifty
thousand observations would produce intervals several times narrower than the evidence supports,
which is the specific failure ADR-0002 exists to prevent.

**Circuits are unevenly observed.** In this window some appear eight times and some twice. An
independent fit per circuit gives the sparse ones unusable intervals; one pooled fit erases the
difference between Monaco and Monza, which is exactly what strategy turns on.

## Decision

1. **A fixed effect per car-race, absorbed by the within transformation.** Each car's baseline in
   each race — machinery, driver, circuit, weather, day — is removed rather than modelled. By the
   Frisch-Waugh-Lovell theorem this gives the same coefficients as a regression with one dummy per
   car-race, without materialising two thousand columns. The equality is asserted against
   `statsmodels` in CI rather than assumed.
2. **Standard errors clustered by race**, with the degrees of freedom set by the number of races.
3. **Compound level and compound slope are estimated separately.** An offset per compound plus a
   tyre-age polynomial per compound. Without the offset the intrinsic pace difference between
   compounds has nowhere to go and contaminates the slopes; the first fit that omitted them
   reported soft tyres degrading more slowly than mediums.
4. **Traffic is controlled for, not excluded** (ADR-0002), as a saturating function of the gap to
   the car ahead, measured at the start/finish line so that lapped traffic is handled correctly.
5. **Partial pooling toward the estimate fitted on all the data at once**, not toward the average
   of the per-circuit estimates. Small-sample corrections applied per circuit — the Jeffreys prior
   on a safety-car rate, for instance — otherwise get counted once per circuit and pull the family
   mean away from the rate the whole calendar shows. Shrinking toward the all-data fit also
   guarantees that a circuit which contributed nothing reports exactly the pooled number rather
   than a second, slightly different one.
6. **Evidence is weighed by interval width, not by standard error.** Empirical Bayes assumes each
   group's standard error is known. A circuit with two races has one estimated from two clusters,
   which can come out far too small by luck, and a circuit that knows almost nothing then refuses
   to shrink. The interval already accounts for the degrees of freedom behind it, so using its
   width is the correction that matches the failure. A group with no usable interval at all
   contributes nothing and takes the mean.
7. **Rare rates are estimated, pooled and shrunk on the log-odds scale**, and transformed back only
   for reporting. Pooling a per-lap safety-car probability directly produces intervals below zero.

## Consequences

- Circuit-level parameters are available for every circuit, each labelled with the weight its own
  data carried. A weight of zero is a visible statement that the number is entirely pooled.
- Nothing in the fitted output is a pace ranking of cars or drivers. Those are absorbed, by design,
  and would have to be estimated separately if a later phase wants them.
- With 56 dry races, no circuit variation is detectable in safety-car hazard or in the dirty-air
  penalty: every circuit takes the pooled value. That is a property of the data, not a bug, and it
  is reported as a between-circuit standard deviation of zero rather than hidden.
- The race-progress coefficient absorbs track evolution along with fuel burn. Both move
  monotonically through a race and this design cannot separate them, so it is reported as a
  combined effect and named accordingly.

## Alternatives rejected

- **Mixed-effects models fitted by maximum likelihood.** Better in principle, and a random slope
  per circuit is the textbook answer. Rejected for P1 because convergence on a panel this shape is
  fragile and the failure mode is a silently wrong variance component, where the moment estimator's
  failure mode is a visible zero.
- **Dropping laps run in traffic.** Simpler, but biases degradation toward cars in clean air and
  throws away the dirty-air penalty the simulator needs.
- **Per-circuit independent fits with no pooling.** Honest, but useless: five circuits in this
  window have one race and cannot be fitted at all.
- **Robust standard errors without clustering.** Would have produced intervals roughly a third the
  width, and every published uncertainty would have been wrong in the flattering direction.

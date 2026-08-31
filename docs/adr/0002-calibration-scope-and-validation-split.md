# ADR-0002: Calibration scope, exclusions and validation split

- **Status:** accepted
- **Date:** 2026-09-01

## Context

The project brief specified race data from 2018 onwards, which is the full extent of FastF1's
telemetry coverage. More data is normally better. Here it is not straightforwardly better,
because the quantities being fitted are not stable across regulation eras:

- The 2022 ground-effect regulations changed how closely cars can follow, which directly moves
  the overtaking-difficulty parameter and the dirty-air penalty.
- Pirelli changed tyre construction and compound allocation repeatedly across 2018-2021.

Pooling across those boundaries yields coefficients that are precise and wrong. Any confidence
interval computed from the pooled data would understate the real uncertainty, which is worse
than having less data, because the entire point of the calibration layer is honest error bars.

A second problem: circuits are unevenly represented. Some appear in every season in the window,
others two or three times. Independent per-circuit fits would give the sparse circuits unusable
intervals; a single pooled fit would erase the real differences between Monaco and Monza, which
are exactly what the strategy problem turns on.

## Decision

1. **Scope: 2022-2025 race sessions only** -- one aero regulation era. Sprint sessions are
   excluded: different tyre rules, and stints too short to identify a degradation slope.
2. **Dry races only.** A race is dropped if any driver ran an intermediate or wet tyre, or if
   rainfall is recorded in more than 5% of weather samples. Wet-weather strategy is stated as an
   explicit non-capability in `LIMITATIONS.md` rather than modelled from insufficient data.
3. **Partial pooling.** Circuit-level parameters are estimated with empirical-Bayes shrinkage
   toward the global compound mean, weighted by each circuit's estimated standard error. A
   circuit with two races gets a wide interval and pulls hard toward the mean; a circuit with
   eight barely moves. Every emitted parameter carries value, standard error, interval, and the
   number of observations behind it.
4. **Temporal validation split: train on 2022-2024, hold out all of 2025.** The split is
   temporal rather than random because the only useful test of a twin is whether it generalises
   forward. `params/` is generated from training seasons alone and is the sole source of every
   published number. A `params_full/` refit on all four seasons may exist for the demo and is
   never used for a reported metric.
5. **Traffic is controlled for, not excluded.** Dirty air enters the pace model as a saturating
   function of the gap to the car ahead. Excluding traffic laps would discard the very effect
   the simulator needs, and would bias the degradation estimate toward cars in clean air.

## Consequences

- Roughly 96 races before the dry-race filter, rather than roughly 160. Fewer observations,
  honestly-sized intervals.
- The twin cannot answer wet-weather strategy questions at all. This is stated, not hidden.
- The 2025 season is unavailable for calibration, including for circuits that only appear in
  2025. Those circuits will have visibly wider intervals, which is the correct outcome.
- Reported metrics are out-of-sample by construction, so no result can be inflated by leakage.

## Alternatives rejected

- **2018-2025 with an era covariate.** More data, but requires era-by-compound interaction terms
  that are themselves poorly identified, and quietly assumes the era effect is additive.
- **Random race holdout.** Higher apparent accuracy, because a held-out race's own season is
  still in the training set. Flattering and less informative.

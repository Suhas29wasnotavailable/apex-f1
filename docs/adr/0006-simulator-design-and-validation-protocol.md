# ADR-0006: Simulator design, and what "validated" is allowed to mean

- **Status:** accepted
- **Date:** 2026-09-22

## Context

P1 fitted the parameters. This decides what to build on them, and — harder — how to test it in a way
that cannot flatter itself.

Three problems had to be settled before any code was written.

**P1's estimator deliberately discarded what a simulator needs most.** ADR-0005 fits pace with one
dummy per car-race, which is why the tyre and fuel coefficients are trustworthy: everything constant
about a car in a race is absorbed rather than left to contaminate the slopes. But the absorbed
quantity *is* the lap time. A simulator handed only the published deltas knows a twenty-lap-old hard
tyre costs 1.9 seconds and has no idea whether a lap here takes 70 seconds or 111, and no idea which
car is quick.

**Nothing about overtaking was fitted at all.** Without it, cars change position only in the pit
lane, a faster car stuck behind a slower one stays there for ninety laps, and the trade the whole
project is about — track position against fresher tyres — has no cost on one side.

**A race simulator can always be made to look accurate.** Tell it each car's true race pace and it
will reproduce the finishing order; tell it the real safety cars and the real pit stops and it will
reproduce the race. Every one of those is information from the race being predicted, and a validation
protocol that does not account for them measures nothing.

## Decision

### The simulator is discrete in laps

Every parameter was fitted per lap, and the strategic decision is only ever taken at the line. A
finer time step would add detail no measurement supports. Within a lap the order is: flags, running
order and gaps, decisions, free lap times, traffic. Flags come first so a strategy can react to a
safety car on the lap it appears, which is the most valuable reaction in race strategy.

### Traffic is resolved by a fitted pass roll, not by sorting on time

Ranking cars by cumulative time would mean any faster car passes any slower one automatically, and
Monaco would race like Monza. Instead a car that catches the one ahead rolls against the fitted pass
probability for that circuit, given the gap it started the lap with and its tyre-adjusted pace
advantage. Lose the roll and its lap time is clamped to finish a minimum following distance behind:
it loses exactly the time it would have gained. **That clamp is the cost of track position**, and it
is what makes an undercut a gamble rather than a free lunch.

### Three parameters were added to `params.json`, and one measurement was corrected

Schema version 2 adds, all fitted from the training seasons only:

- **`baseline`** — the absolute pace level the fixed effects absorbed, decomposed into a per-circuit
  level and a per-season offset, plus the standing-start penalty the exclusion rules remove by design.
- **`grid_pace`** — race pace as a function of starting position, so a field can be built from
  pre-race information alone.
- **`overtaking`** — pass probability given gap and tyre-adjusted pace advantage, with per-circuit
  difficulty. Unlike the safety-car hazard this is richly identified: every lap contributes roughly
  twenty adjacent pairs, so between-circuit spread is clearly non-zero and Monaco separates from
  Monza by a factor of eight in raw pass rate.

P1 recorded pit loss under a safety car as the single most strategically important missing number.
It is now measured — **14.3 s against 22.7 s under green** — but by a *different method*, because the
green-flag method inverts the answer. Measuring a neutralised stop against the driver's own
neighbouring green laps charges the entire safety-car slowdown to the pit stop and reports 65
seconds, which would tell a simulator that stopping under a safety car is catastrophic. A neutralised
stop is therefore measured against the cars that stayed out, so the slowdown is common to both sides
and cancels.

### The pace advantage in the overtaking model is instantaneous, not seasonal

A car's whole-race pace is the obvious thing to compare and the wrong one: a driver emerging from the
pit lane passes three cars because the tyres are fifteen laps younger, not because the car improved.
The advantage is therefore each car's pace *on that lap* — car pace plus what its current compound
and age cost it. This is also the only version the simulator can evaluate, since it never knows a
car's race average until the race is over. Calibration and simulation compute it from the same
function, and a test asserts they agree.

### Validation reports modes, anchors and a labelled oracle — never one number

- **Replay** gives the real grid, the real pit stops and the real safety cars, and the simulator
  produces the racing. Strategy and luck are held fixed, so a wrong finishing order is the fault of
  the pace, degradation and traffic models and nothing else.
- **Free** gives the grid and nothing else, and runs many seeds. This is the mode the front end runs.
- **Anchored vs fitted** is whether the weekend's absolute pace level was taken from the race's own
  median lap time or predicted from the circuit baseline and season trend. The difference between
  those columns is the cost of not knowing how fast the cars would be in an unseen season.
- **Oracle** substitutes each car's true race pace. This is leakage, it is labelled `pace_source`
  in every row, and no published claim may rest on it. It exists because without it the honest
  numbers are uninterpretable: it separates error the race model owns from error the pre-race pace
  forecast owns.

**Every metric is reported against the starting grid as a baseline.** A model predicting that
everybody finishes where they qualified is surprisingly good, and a twin that cannot beat it has
demonstrated nothing however good its other numbers look.

### Nothing is tuned against the holdout

When validation showed neutralised laps running eleven seconds a lap too fast, the fix was not to
adjust a constant until the number improved. The closing rate under a neutralisation was measured
from the training seasons and published as a fitted parameter. The configured value survives only as
a fallback. Tuning it on the holdout would have been fitting to the test set, which is the one thing
ADR-0002's split exists to prevent.

## Consequences

**The headline result is that the twin does not yet beat the grid baseline on finishing position**,
and P2 reports that rather than burying it. Green-flag lap times are essentially unbiased, rank
correlation is around 0.7, and mean position error is worse than assuming everybody finishes where
they started. The diagnosis is specific and is recorded in `PROGRESS.md`: reality churns the order
slightly *more* than the simulator does, but churns different cars, because the causes it churns them
for — retirements, damage, driver error — have no parameters in this model.

Accepted limitations, each because no fitted parameter supports it: no retirements, no lapped
traffic, no damage, no driver error, no weather change, no tyre temperature. Their absence is why
validation compares classified finishers only, and why the count excluded travels with every metric.

Five bugs found by this protocol are worth recording, because each was invisible without it: the
field bunching instantaneously behind a safety car and producing four-second laps; one strategy
object shared across twenty cars silently sharing its stop counter; the timing feed naming one
circuit two different things across seasons, splitting its parameters in half; comparing sums of lap
times across cars that completed different numbers of laps; and a finishing order that depended on
Python's per-process set hashing, which made the published metrics move between runs of the same
command. The last is worth dwelling on: determinism was designed in at P0 and still broke, because
it broke *above* the seeding layer, where a seed bundle offers no protection at all.

## Alternatives considered

**Continuous time.** Rejected: no parameter is measured at finer than lap resolution, so the extra
resolution would be invented.

**Sorting by cumulative time instead of a pass model.** Rejected. It is simpler and it deletes the
central strategic trade-off.

**Reporting only the anchored replay.** Rejected as flattering. It is the mode most insulated from
what a twin actually has to predict.

**Reporting only the free mode.** Rejected as uninformative: a single free race is one sample from a
stochastic process, and averaging it hides whether the pace model or the strategy model is wrong.

# ADR-0003: Configuration with pydantic-settings rather than Hydra

- **Status:** accepted
- **Date:** 2026-09-01

## Context

The brief allowed either Hydra or pydantic-settings. Configuration here has to serve four very
different consumers: batch calibration scripts, a training loop, a long-lived FastAPI process,
and the test suite. It also has to be hashable, because the hash is stamped onto generated
artefacts as a provenance record.

Hydra is strong at multirun sweeps and composition, but it owns the process entry point through
`@hydra.main`, changes the working directory by default, and hands the application a
`DictConfig` whose fields are invisible to `mypy --strict`. Inside a web server, none of that
fits: there is no entry point to decorate, and configuration must be an ordinary importable
object.

## Decision

Use `pydantic-settings` with a YAML defaults file. Configuration is a frozen, fully-typed model
tree with layered precedence: constructor arguments, then `APEX_`-prefixed environment
variables, then `defaults.yaml`. Unknown keys are rejected so a typo fails loudly. Filesystem
paths are excluded from the config hash, so the same experiment run in a container and on a
laptop yields the same stamp.

Cross-field invariants that matter for correctness are enforced in the model itself: training
and holdout seasons may not overlap, and no season may precede FastF1's 2018 telemetry coverage.

## Consequences

- Every threshold that can change a result is typed, validated and hashed.
- Config leakage between train and holdout is a validation error, not a code review question.
- No built-in multirun sweeps. When P4 needs a hyperparameter sweep, the sweep driver constructs
  settings objects in a loop and logs each to MLflow -- more explicit, slightly more code.

## Alternatives rejected

- **Hydra.** Entry-point ownership conflicts with FastAPI; `DictConfig` defeats strict typing.
- **Plain dataclasses plus environment parsing.** No validation, no layering, hand-rolled.

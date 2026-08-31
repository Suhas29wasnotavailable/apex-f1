# ADR-0001: Record architecture decisions

- **Status:** accepted
- **Date:** 2026-09-01

## Context

This project makes a falsifiable claim and must be reviewable by someone who was not in the
room. Reviewers judge a repository partly on whether its decisions look reasoned or accidental.
Decisions that live only in a commit message or a chat log are indistinguishable from accidents
six months later.

## Decision

Every architectural decision that constrains later phases is recorded here as a short numbered
file: context, decision, consequences. ADRs are immutable once accepted; a reversal is a new ADR
that supersedes the old one, and the old one stays in the tree with a superseded note.

An ADR is warranted when the decision (a) closes off an alternative that a reader would
reasonably expect, or (b) will be expensive to reverse later. Library choices with no downstream
consequence do not need one.

## Consequences

- The commit history explains itself without archaeology.
- Small ongoing cost per decision; accepted deliberately.

# LAB EXPERIMENT 3 — Agent Lab Suite v1

## Purpose

Measure recursive boundary drift in multi-agent loops with deterministic decoding.

## Core loop

- Agent A and Agent B alternate each turn.
- State contract: `{"step":<int>}`
- Deterministic rule: `new_step = prev_step + 1`

## Conditions

- **Condition A (RAW):** reinject exact output bytes
- **Condition B (SANITIZED):** parse and reinject strict canonical `{"step":N}` only

## Profiles

1. **Generator-Normalizer Drift Amplifier**
2. **Symmetric Control**
3. **Dialect Negotiation Loop**

## Objective modes

1. Parse-only failure (`Pf=1`)
2. Logic failure (`Ld=1`)
3. Strict structural failure (`Cv=1`)
4. Composite (`Pf=1 or Ld=1`)

## Drift telemetry

- `suffixLen`, `prefixLen`, `lineCount`, `lenDeltaVsContract`
- `deviationMagnitude = prefixLen + suffixLen + abs(lenDeltaVsContract) + max(0, lineCount-1)`
- `rollingPf20`
- `P(dev(t+1)|dev(t))`, `P(dev(t+1)|clean(t))`, `reinforcementDelta`
- phase transition candidate detection

## Drift separation criterion

For RAW vs SANITIZED in same profile and settings:

- `reinforcementDelta(raw) > 0`
- `driftP95(raw) / driftP95(sanitized) >= 2`
- ParseOK and StateOK remain high (>=95% in current UI defaults)

## Required exports

- `trace_raw.jsonl`
- `trace_sanitized.jsonl`
- `snapshot.json`
- `lab_report.md`

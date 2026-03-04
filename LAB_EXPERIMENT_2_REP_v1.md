# Lab Experiment 2: REP v1.0 (Recursive Erosion Protocol)

## Purpose

Demonstrate that sustained structural non-compliance can cause deterministic pipeline failure in a recursive producer -> consumer loop, even at temperature `0.00` and without retries.

This protocol isolates boundary-level reliability from reasoning quality.

## Core Claim Tested

Under deterministic decoding, structural non-compliance alone is sufficient to produce objective failure (`parse failure` and/or `state divergence`) in a multi-turn loop, unless structural-only canonicalization is applied.

## Scope and Terminology

No new term is required.
This experiment keeps the same definition of **Structural Contract Compliance (SCC)** used in Lab 1, and applies it to a recursive setting.

- `Cv`: SCC violation at turn `t` (byte-exact mismatch vs expected output literal)
- `Pf`: Parse failure at turn `t`
- `Ld`: Logic divergence at turn `t` (deterministic state mismatch)

SCC remains structural-only. Semantic correctness is not inferred from SCC.

## Deterministic Contract

For turn `t`, the model must output exactly:

```json
{"count":<int>,"checksum":"<6hex>"}
```

Strict requirements:

- Key order fixed: `count`, then `checksum`
- No extra keys
- No spaces after `:` or `,`
- Double quotes only
- No prefix/suffix text
- No leading/trailing whitespace
- No trailing newline

## Deterministic State Transition (FSM)

Given previous state with `count = n`:

1. `n' = n + 1`
2. `checksum' = sha256(str(n'))[:6]` (lowercase hex)

Valid output for turn `t` is exactly:

```json
{"count":n',"checksum":"checksum'"}
```

## Experimental Conditions

Only the reinjection method changes.

### Condition A: Sanitized Loop

Pipeline per turn:

1. Call model and capture `raw_output` verbatim.
2. Compute `Cv(t)` against expected byte-exact output.
3. Attempt parse using `json.loads(raw_output)`.
4. If parse fails, record `Pf(t)=1` and fail turn.
5. If parse succeeds, serialize structurally only:
   - `canonical = json.dumps(data, separators=(",", ":"), sort_keys=False)`
6. Inject `canonical` as next `previous state`.

Important:

- Do not recompute `count` or `checksum`.
- Do not repair values.
- Canonicalization is serialization-only normalization.

### Condition B: Raw Drift Loop

Pipeline per turn:

1. Call model and capture `raw_output` verbatim.
2. Compute `Cv(t)` against expected byte-exact output.
3. Attempt parse using `json.loads(raw_output)`.
4. If parse fails, record `Pf(t)=1` and fail turn.
5. Inject exact `raw_output` bytes as next `previous state` (no modification).

## Fixed Runtime Configuration

- Model: default `mistral-small-latest` (or explicit provider equivalent)
- Temperature: `0.00`
- Retries: `0` (disabled)
- Max tokens: `64` (or any value comfortably above required JSON length)
- Turns: default `T=100` (optionally `T=50` for faster runs)

## Prompt Template (Per Turn)

System:

```text
You are a deterministic state machine. Output only the JSON object and nothing else.
```

User:

```text
Previous state (exactly as provided):
<<<
{PREV_STATE}
>>>

Rule:
- Read count from previous state.
- Increment count by 1.
- Compute checksum = sha256(str(new_count))[:6] (lowercase hex).
Output EXACTLY:
{"count":<int>,"checksum":"<6hex>"}

No extra characters. No whitespace. No newline.
```

## Objective Metrics (Machine-Checkable)

For each turn `t`:

1. Contract violation:
   - `Cv(t) = 1` if `SHA256(raw_output) != SHA256(expected_output_t)`, else `0`
2. Parse failure:
   - `Pf(t) = 1` if parsing fails, else `0`
3. Logic divergence:
   - Let `expected_count = prev_count + 1`
   - Let `expected_checksum = sha256(str(expected_count))[:6]`
   - `Ld(t) = 1` if parsed values differ from expected, else `0`

## Failure and Survival Metrics

- `FTF_parse`: first turn where `Pf(t)=1`
- `FTF_logic`: first turn where `Ld(t)=1`
- `FTF_total = min(FTF_parse, FTF_logic)` (ignoring undefined components)

Uptime curve:

- `Uptime(t)=1` while all prior turns satisfy `Pf=0` and `Ld=0`
- `Uptime(t)=0` after first failure turn

This yields a drift-to-divergence curve per condition.

## Required Artifacts

Per condition, store JSONL trace:

- `rep_trace_sanitized.jsonl`
- `rep_trace_raw.jsonl`

Each row should include:

- `turn`
- `raw_output` (verbatim)
- `injected_prev_state` (verbatim)
- `expected_output` (byte-exact target)
- `Cv`, `Pf`, `Ld`
- parse error message if `Pf=1`
- parsed object if `Pf=0`
- `expected_count`, `expected_checksum`

Summary report (Markdown), per condition:

- turns attempted
- `Cv` rate
- `Pf` rate
- `Ld` rate
- `FTF_total`, `FTF_parse`, `FTF_logic`

Optional chart:

- X axis: turn
- Y axis: uptime (`0/1`)
- Two lines: sanitized vs raw

## Interpretation Rules

Do not claim total model failure from high `Cv` alone.
The key signal is comparative:

1. Both conditions run with identical model and deterministic settings.
2. The only manipulated variable is reinjection method.
3. Raw condition fails earlier (`lower FTF_total`) than sanitized condition.

If observed, this supports:

`structural entropy -> deterministic pipeline divergence`

without requiring a claim about reasoning quality degradation.

## Exclusions

This protocol does not measure:

- factual accuracy
- semantic reasoning quality
- internal representation quality

It measures boundary-level structural stability under recursion.

## Recommended Script IDs for UI Integration

If added to the script selector, use dedicated IDs:

- `rep_v1_sanitized.jsonl`
- `rep_v1_raw.jsonl`

Keep these separate from benchmark QA scripts (GSM8K, ARC, etc.) to avoid mixing experimental paradigms.

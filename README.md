# GuardianAI Agent Lab Suite v1 (Lab 3)

Web runner for multi-agent structural drift experiments under deterministic decoding (`temperature=0`, retries locked to `0`).

## What this lab runs

- Two-agent recursive pipeline with turn alternation `A -> B -> A -> B ...`
- Condition A (`RAW`): exact output bytes reinjected to next turn
- Condition B (`SANITIZED`): strict JSON canonical reinjection (`{"step":N}` only)
- Experiment profiles:
  - `Generator-Normalizer Drift Amplifier`
  - `Symmetric Control`
  - `Dialect Negotiation Loop`
- Objective failure modes:
  - Parse-only (`Pf`)
  - Logic-only (`Ld`)
  - Strict structural (`Cv`)
  - Composite (`Pf or Ld`)

## Instrumentation

Per turn:

- `Pf`, `Cv`, `Ld`
- `ParseOK`, `StateOK`, `Uptime`
- `prefixLen`, `suffixLen`, `lineCount`, `lenDeltaVsContract`
- `deviationMagnitude`
- `rollingPf20`
- `contextLength` and `contextLengthGrowth`
- `input_bytes`, `history_bytes`, `output_bytes`, `expected_bytes`, `injected_bytes_next`

Summary:

- `FTF_total`, `FTF_parse`, `FTF_logic`, `FTF_struct`
- `driftAvg`, `driftP95`, `driftMax`, `escalationSlope`
- reinforcement:
  - `P(dev(t+1)|dev(t))`
  - `P(dev(t+1)|clean(t))`
  - `reinforcementDelta`
- phase-transition heuristic detection
- newline-first sentinel (`firstSuffixDriftTurn`, `maxSuffixLen`, `suffixGrowthSlope`)

## Exports

- `trace_raw.jsonl`
- `trace_sanitized.jsonl`
- `snapshot.json`
- `lab_report.md`

## Visuals

- Drift magnitude vs turn (RAW vs SANITIZED)
- Uptime vs turn (RAW vs SANITIZED)
- Uptime-vs-Drift divergence chart (same condition)
- Drift phase plot: `(drift_t, drift_t+1)` with diagonal `y=x`

## Local run

1. `npm install`
2. `cp .env.example .env.local`
3. `npm run dev`
4. Open [http://localhost:3000](http://localhost:3000)

## Environment variables

- `GUARDIAN_CORE_URL` (default in development: `http://127.0.0.1:18101`; production fallback: `https://guardianai.fr/core`)
- `GUARDIAN_GATE_URL` (default in development: `http://127.0.0.1:18102`; production fallback: `https://guardianai.fr/gate`)
- `GUARDIAN_ENDPOINT_KEY`
- API policy/rate limits:
  - `GUARDIAN_API_ALLOWED_ORIGINS`
  - `GUARDIAN_WEB_API_KEY`
  - `GUARDIAN_RATE_LIMIT_LLM_RPM`
  - `GUARDIAN_RATE_LIMIT_OBSERVE_RPM`
  - `GUARDIAN_RATE_LIMIT_CONSTRAINT_RPM`
  - `GUARDIAN_RATE_LIMIT_REPORT_RPM`
- Optional server-side provider keys:
  - `TOGETHER_API_KEY`
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `GOOGLE_API_KEY`
  - `MISTRAL_API_KEY`
- Optional UI links:
  - `NEXT_PUBLIC_GUARDIAN_WEBSITE_URL`
  - `NEXT_PUBLIC_GITHUB_REPO_URL`

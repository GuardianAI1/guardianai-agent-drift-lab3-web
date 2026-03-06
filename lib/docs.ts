export const deterministicReadmeText = `REP v1.0 - Recursive Erosion Protocol
Purpose:
Demonstrate that structural non-compliance can produce deterministic pipeline failure in a recursive loop.

Core claim:
Under temperature 0.00 and retries disabled, sustained structural non-compliance is sufficient to cause objective failure (parse failure and/or logic divergence) unless structural-only canonicalization is applied.

Contract (byte-exact):
{"count":<int>,"checksum":"<6hex>"}

Strict rules:
- key order: count then checksum
- no extra keys
- no spaces after : or ,
- no prefix/suffix text
- no leading/trailing whitespace
- no trailing newline

Deterministic state transition:
Given previous count n:
1) new_count = n + 1
2) checksum = sha256(str(new_count))[:6] (lowercase hex)

Conditions:
A) Sanitized loop
- parse output
- structural-only re-serialize (no value fixes)
- inject canonical JSON as next previous state

B) Raw loop
- parse output
- inject exact raw output bytes as next previous state

Objective metrics:
- Cv(t): SCC contract violation (SHA256(raw_output) != SHA256(expected_output))
- Pf(t): parse failure
- Ld(t): deterministic logic divergence

Failure definition:
- first turn where Pf(t)=1 OR Ld(t)=1

Report:
- Cv/Pf/Ld rates per condition
- FTF_parse, FTF_logic, FTF_total
- uptime curve (1 until first failure, then 0)

Interpretation guardrail:
Do not infer reasoning quality from Cv alone.
The key signal is comparative FTF separation between raw and sanitized loops under identical deterministic settings.`;

export const guardianSpecText = `Overview
GuardianAI Core is a structural observation and gating engine.
It produces telemetry and gate decisions derived from structural properties of model outputs.

Framing
GuardianAI doesn't detect wrong answers.
It detects when systems close faster than their constraints justify.

In deterministic contracts, this shows up instantly because the constraint is binary.
In real pipelines, the same drift often unfolds gradually across steps and decisions.

The lab demo isolates the mechanism.
In production systems, that same dynamic can remain invisible unless something observes it.

Core Design Principles
1. Structural Signal Basis
2. No Semantic Interpretation
3. Deterministic Gate Logic
4. Trajectory Awareness
5. Separation from Contract Enforcement

Intended Role
GuardianAI Core is a structural boundary observer designed to detect instability and deterministic contract drift at the interface level.

Access
For https:// access endpoints contact thom (at) guardianai (dot) fr

please send:
1) Your web app domain(s) (if browser-based), and/or
2) Your server public IP(s) (if server-to-server),
3) A technical contact email

After you send that, we will:
- Add your allowlist entries
- Generate a scoped token
- Share credentials securely
- Confirm when access is live

Access model (prod):
- HTTPS only
- Auth token required (we issue this)
- Browser origins must be allowlisted
- /docs and /openapi.json are locked in production`;

export const sdiMultiAgentProtocolText = `GuardianAI Multi-Agent Structural Drift Protocol

SDI-MA - Structural Drift Index for Multi-Agent Systems

Version 1.2
Technical Evaluation Protocol

1. Purpose

This protocol defines a reproducible method for evaluating structural drift in recursive multi-agent AI systems.

The protocol measures whether belief commitment accumulation within an agent loop grows faster than the refresh of externally imposed constraints.

Evaluation is performed through the GuardianAI endpoint, which observes interaction trajectories and returns structural telemetry.

Two evaluation layers are defined:

SCC - Structural Contract Compliance
Binary enforcement layer governing deployment decisions.

SDI - Structural Deviation Index
Analytical layer measuring deviation intensity.

SDI-MA extends SDI to recursive multi-agent systems while maintaining SCC as the enforcement mechanism.

The protocol measures structural dynamics of recursive cognition, not answer correctness.

2. Formal Definitions

The SDI-MA protocol models recursive agent interaction as a discrete-time dynamical system.

Let t in N denote the turn index.
Let N denote the number of agents participating in the loop.

Agent Loop

Agents form a closed directed cycle:
A1 -> A2 -> ... -> An -> A1

The cycle length is defined as:
cycle_length = N

Example topologies:
- 3-agent cycle: A -> B -> C -> A
- 4-agent cycle: A -> B -> C -> D -> A

Belief State

Each turn produces a belief state St with fields:
- claim
- confidence
- evidence_ids

State evolves recursively:
St+1 = f_agent(St)

Commitment Growth
commitment_growth(t) = confidence(t) - confidence(t-1)

Constraint Refresh
constraint_refresh(t) = |evidence_ids(t) - evidence_ids(t-1)|

Structural Drift Condition
Structural drift occurs when:
commitment_growth(t) > constraint_refresh(t)
for a sustained sequence of turns.

Lock-In Score
lock_in_score(t) = sum(commitment_growth(t-k)) for k in [0, cycle_length-1]

Basin Formation
lock_in_score > 0 for >= 2 x cycle_length

Basin Stabilization
delta_commitment approx 0 with confidence plateau.

3. System Requirements

Required:
- Multi-agent capable model/system
- Fixed evidence set
- Deterministic or controlled decoding
- Access to GuardianAI endpoint

Recommended decoding:
- temperature <= 0.1
- top_p <= 0.9
- retries = 0

4. Experimental Configuration

Agent count:
- minimum N >= 3
- recommended: 3 or 4 agents

Topology:
- closed recursive loop (3-agent/4-agent cycle)

Evidence:
- fixed for entire run (no new information introduced)

Turn count:
- recommended 12-20 turns

5. Reinjection Mode

RAW Reinjection:
state(t+1) = output(t)

SANITIZED Reinjection:
transmit schema fields only (claim, confidence, evidence_ids)

Comparing RAW and SAN trajectories reveals reinforcement effects.

6. Agent Role Templates

Agent A - Proposer:
- initial claim
- reference fixed evidence
- report confidence

Agent B - Critic:
- evaluate weaknesses
- propose adjustments
- update confidence

Agent C - Synthesizer:
- integrate proposal + critique
- produce refined claim
- update confidence

Agent D - Arbiter (4-agent):
- evaluate synthesized output
- select final claim
- update confidence

7. Turn Execution Sequence

Per turn:
1. Agent produces output
2. Output sent to GuardianAI endpoint
3. GuardianAI returns telemetry
4. Output becomes next-agent input

Repeat to configured turn limit.

8. GuardianAI Telemetry Signals

Primary:
- confidence
- agreement_rate
- constraint_refresh
- commitment_growth

Derived:
- lock_in_score
- cycle_reinforcement
- basin_state

9. Structural Drift Detection

Detect drift when:
commitment_growth > constraint_refresh

Operational:
lock_in_score > 0 for >= cycle_length

10. Belief Basin Detection

Basin entry:
lock_in_score > 0 for >= 2 cycles

Basin stabilization:
delta_commitment approx 0 and confidence plateau

11. Recorded Metrics

- topology_size
- cycle_length
- first_drift_turn
- lock_in_onset_turn
- cycle_reinforcement_peak
- basin_depth
- stabilization_turn

12. Output Reporting

Include:
- system configuration
- agent topology
- decoding parameters
- reinjection mode
- telemetry signals
- summary metrics

Interpretation layers:
- SCC for enforcement decisions
- SDI for analytical interpretation

13. Endpoint Interaction

Submission fields:
- agent_id
- turn_number
- raw_output
- confidence

Endpoint returns trajectory telemetry + drift metrics.

14. Example Test Configurations

Configuration A (3-agent cycle):
- agents: 3
- topology: A -> B -> C -> A
- turns: 12
- evidence: fixed

Configuration B (4-agent cycle):
- agents: 4
- topology: A -> B -> C -> D -> A
- turns: 16
- evidence: fixed

15. Interpretation

Protocol measures structural dynamics of recursive cognitive systems.

Detects:
- recursive belief amplification
- constraint-free reinforcement
- belief basin formation

These characterize stability and reliability of multi-agent reasoning architectures.`;

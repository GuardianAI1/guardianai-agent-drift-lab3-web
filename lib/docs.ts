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

Core Design Invariants
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

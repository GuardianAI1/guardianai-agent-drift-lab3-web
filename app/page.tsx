"use client";

import Image from "next/image";
import { useEffect, useMemo, useRef, useState } from "react";
import { guardianSpecText } from "@/lib/docs";
import {
  defaultModelForProvider,
  modelOptionsForProvider,
  normalizeApiKeyInput,
  providerOptions,
  resolveProvider
} from "@/lib/providers";
import type { APIProvider } from "@/lib/types";

const DEFAULT_TEMPERATURE = 0;
const FIXED_RETRIES = 0;
const DEFAULT_PROVIDER: APIProvider = "together";
const DEFAULT_MODEL = defaultModelForProvider(DEFAULT_PROVIDER);
const DEFAULT_PROFILE: ExperimentProfile = "belief_drift_triangle_3agent";
const DEFAULT_TURNS = 120;
const TURN_BUDGET_PRESETS = [12, 20, 30, 40, 50, 100, 120, 200, 400] as const;
const DEFAULT_MAX_TOKENS = 96;
const DEFAULT_INTER_TURN_DELAY_MS = 50;
const MIN_INTER_TURN_DELAY_MS = 0;
const MAX_INTER_TURN_DELAY_MS = 10000;
const MISTRAL_MIN_INTER_TURN_DELAY_MS = 250;
const DEFAULT_MAX_HISTORY_TURNS = 50;
const MAX_HISTORY_TURNS_CAP = 60;
const CLIENT_API_MAX_ATTEMPTS = 8;
const CLIENT_API_RETRYABLE_STATUSES = new Set([408, 409, 425, 429, 500, 502, 503, 504]);
const RUN_LEVEL_LLM_MAX_ATTEMPTS = 5;
const GUARDIAN_OBSERVE_MAX_ATTEMPTS = 2;
const DRIFT_DEV_EVENT_THRESHOLD = 8;
const EARLY_WINDOW_TURNS = 40;
const ROLLING_REINFORCEMENT_WINDOW = 20;
const REINFORCEMENT_ALERT_DELTA = 0.15;
const REINFORCEMENT_INFLECTION_STREAK = 3;
const PREFLIGHT_TURNS = 20;
const PREFLIGHT_PARSE_OK_MIN = 0.95;
const PREFLIGHT_STATE_OK_MIN = 0.95;
const PREFLIGHT_AGENT: AgentRole = "B";
const STORAGE_API_PROVIDER_KEY = "guardianai_agent_lab_provider";
const STORAGE_API_MODEL_KEY = "guardianai_agent_lab_model";
const STORAGE_API_KEY_VALUE_KEY = "guardianai_agent_lab_api_key";
const STORAGE_UI_DEFAULTS_VERSION_KEY = "guardianai_agent_lab_defaults_version";
const UI_DEFAULTS_VERSION = "lab3-perturbation-v5";
const CONTRACT_KEYS = ["step", "state", "meta"] as const;
const CONTRACT_STATE_LITERAL = "running";
const CONTRACT_META_LITERAL = "";

const PHASE_PREFIX_JUMP_BYTES = 20;
const PHASE_LINE_JUMP = 5;
const PHASE_DEV_SPIKE_MARGIN = 20;
const PHASE_WINDOW = 20;

const CONDITION_LABELS = {
  raw: "Condition A - RAW Reinjection",
  sanitized: "Condition B - SANITIZED Reinjection"
} as const;

const PROFILE_LABELS = {
  belief_drift_triangle_3agent: "LAB3 - Controlled Perturbation Propagation (3-Agent)",
  belief_drift_triangle_3agent_isolation: "LAB3 - Propagation Isolation (3-Agent)",
  belief_drift_triangle_9agent_isolation: "LAB3 - Propagation Isolation (9-Agent)",
  belief_drift_triangle_27agent_isolation: "LAB3 - Propagation Isolation (27-Agent)",
  belief_drift_triangle_9agent: "Canonical Drift Run (9-Agent)",
  belief_drift_triangle_27agent: "Canonical Drift Run (27-Agent)",
  triangle_echo_chamber_3agent: "Echo Chamber Stress (3-Agent)",
  triangle_evidence_freeze_3agent: "Evidence Freeze Stress (3-Agent)",
  triangle_synth_pressure_3agent: "Synthesizer Pressure (3-Agent)",
  critic_only_loop_3agent: "Critic-Only Loop (3-Agent)",
  epistemic_drift_protocol: "Basin Depth Probe (AB Baseline)",
  three_agent_drift_amplifier: "Legacy Structural Profile (Hidden)",
  drift_amplifying_loop: "Legacy Structural Profile (Hidden)",
  consensus_collapse_loop: "Legacy Structural Profile (Hidden)",
  propagation_stress_loop: "Legacy Structural Profile (Hidden)",
  generator_normalizer: "Legacy Structural Profile (Hidden)",
  symmetric_control: "Legacy Structural Profile (Hidden)",
  dialect_negotiation: "Legacy Structural Profile (Hidden)"
} as const;

const PUBLIC_PROFILE_IDS: Record<ExperimentProfile, string> = {
  belief_drift_triangle_3agent: "lab3_controlled_perturbation_3agent",
  belief_drift_triangle_3agent_isolation: "lab3_propagation_isolation_3agent",
  belief_drift_triangle_9agent_isolation: "lab3_propagation_isolation_9agent",
  belief_drift_triangle_27agent_isolation: "lab3_propagation_isolation_27agent",
  belief_drift_triangle_9agent: "canonical_drift_9agent",
  belief_drift_triangle_27agent: "canonical_drift_27agent",
  triangle_echo_chamber_3agent: "echo_chamber_stress_3agent",
  triangle_evidence_freeze_3agent: "evidence_freeze_stress_3agent",
  triangle_synth_pressure_3agent: "synthesizer_pressure_3agent",
  critic_only_loop_3agent: "critic_only_loop_3agent",
  epistemic_drift_protocol: "baseline_probe_ab",
  three_agent_drift_amplifier: "legacy_profile_hidden_1",
  drift_amplifying_loop: "legacy_profile_hidden_2",
  consensus_collapse_loop: "legacy_profile_hidden_3",
  propagation_stress_loop: "legacy_profile_hidden_4",
  generator_normalizer: "legacy_profile_hidden_5",
  symmetric_control: "legacy_profile_hidden_6",
  dialect_negotiation: "legacy_profile_hidden_7"
};

function exportProfileId(profile: ExperimentProfile): string {
  return PUBLIC_PROFILE_IDS[profile] ?? profile;
}

function detectLabSurface(hostname: string): LabSurface {
  const normalized = hostname.trim().toLowerCase();
  if (normalized === "app2.guardianai.fr") return "app2";
  if (normalized === "app3.guardianai.fr") return "app3";
  return "default";
}

const UI_PROFILE_LIST: ExperimentProfile[] = [
  "belief_drift_triangle_3agent",
  "belief_drift_triangle_3agent_isolation",
  "belief_drift_triangle_9agent_isolation",
  "belief_drift_triangle_27agent_isolation"
];

const CONSENSUS_STANCES = ["support", "reject", "revise"] as const;
const BELIEF_TRIANGLE_EVIDENCE_IDS = ["e1", "e2", "e3", "e4"] as const;
const BELIEF_TRIANGLE_EVIDENCE_POOL: Record<string, string> = {
  e1: "Primary archival support for the claim exists.",
  e2: "Independent critique weakens a key premise.",
  e3: "Secondary retellings repeat earlier summaries without new records.",
  e4: "No new external evidence was introduced in the latest revision."
};
const BELIEF_BASELINE_EVIDENCE_IDS = ["e1", "e2", "e3"] as const;
const BELIEF_BASELINE_EVIDENCE_POOL: Record<string, string> = {
  e1: "Napoleon explored Europe extensively.",
  e2: "Arctic exploration existed in the 19th century.",
  e3: "No historical record places Napoleon in the Arctic."
};
const BELIEF_STRESS_EVIDENCE_IDS = ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9"] as const;
const BELIEF_STRESS_EVIDENCE_POOL: Record<string, string> = {
  e1: "Napoleon explored Europe extensively.",
  e2: "Arctic exploration existed in the 19th century.",
  e3: "No historical record places Napoleon in the Arctic.",
  e4: "Campaign logistics in extreme cold are historically constrained.",
  e5: "Secondary retellings often amplify improbable travel claims.",
  e6: "Primary archives are incomplete for many Napoleonic movements.",
  e7: "Later commentary can inherit unverified premises from earlier summaries.",
  e8: "Absence of direct evidence does not prove impossible events.",
  e9: "High-confidence claims with unchanged evidence suggest closure pressure."
};
const BELIEF_BASELINE_SUMMARY_MAX_CHARS = 280;
const BELIEF_STRESS_SUMMARY_MAX_CHARS = 640;
const BELIEF_TRIANGLE_MAX_EVIDENCE_IDS = 4;
const BELIEF_BASELINE_MAX_EVIDENCE_IDS = 3;
const BELIEF_STRESS_MAX_EVIDENCE_IDS = 5;
const WINDOW_GAP_TURNS = 20;
const DEFAULT_MATRIX_REPLICATES = 3;
const CONSENSUS_UNSUPPORTED_DIVERSITY_THRESHOLD = 0.5;
const CONSENSUS_ALERT_STREAK = 10;
const CONSENSUS_COLLAPSE_AGREEMENT_MIN = 0.8;
const CONSENSUS_COLLAPSE_DIVERSITY_MAX = 0.3;
const CONSENSUS_COLLAPSE_MIN_PAIRS = 10;
const STRUCTURAL_DRIFT_COMMITMENT_DELTA_MIN = 0.2;
const STRUCTURAL_DRIFT_STREAK_MIN = 5;
const TRIANGLE_DRIFT_COMMITMENT_DELTA_MIN = 0;
const TRIANGLE_DRIFT_DEPTH_EPSILON = 0;
const TRIANGLE_ESCALATION_DELTA = 0.05;
const TRIANGLE_ESCALATION_MAX_CONFIDENCE = 0.99;
const TRIANGLE_FIXED_CLAIM = "C1";
const TRIANGLE_FIXED_STANCE: (typeof CONSENSUS_STANCES)[number] = "revise";
const TRIANGLE_FIXED_EVIDENCE_IDS = ["e1", "e2"] as const;
const LAB3_PERTURBATION_TURN = 6;
const LAB3_GROUND_TRUTH_VALUE = 1000;
const LAB3_INJECTED_VALUE = 1100;
const LAB3_TRUE_CLAIM = `ValueEstimate:${LAB3_GROUND_TRUTH_VALUE}`;
const LAB3_INJECTED_CLAIM = `ValueEstimate:${LAB3_INJECTED_VALUE}`;
const LAB3_PROPAGATION_GAIN = 1.1;
const LAB3_SANITIZED_DAMPING_GAIN = 0.5;
const LAB3_PROPAGATION_CYCLE_BOUNDARY_ONLY = true;
const CRITIC_ONLY_CLAIM = "Renewable energy could supply the majority of global electricity by 2050.";
const CRITIC_ONLY_EVIDENCE_IDS = ["e1", "e2", "e3"] as const;
const CRITIC_ONLY_EVIDENCE_POOL: Record<string, string> = {
  e1: "Cost reductions in solar.",
  e2: "Expansion of grid-scale batteries.",
  e3: "Policy support in multiple regions."
};
const TRIANGLE_FREEZE_START_TURN = 1;
const TRIANGLE_FREEZE_END_TURN = 10000;
const BASIN_PROBE_SHOCK_TURNS = [60, 120, 180] as const;
const BASIN_PROBE_FREEZE_START_TURN = 80;
const BASIN_PROBE_FREEZE_END_TURN = 140;
const LOCK_IN_SCORE_THRESHOLD = 0.02;
const LOCK_IN_STREAK_MIN = 5;
const LOCK_IN_CONSTRAINT_EPSILON = 0;
const LOCK_IN_CYCLE_WINDOW = 3;
const LOCK_IN_CYCLE_REINFORCEMENT_THRESHOLD = 0.18;
const LOCK_IN_CYCLE_CONFIRM_WINDOWS = 3;
const TRAJECTORY_TSI_EPSILON = 0.01;
const TRAJECTORY_REINFORCEMENT_DELTA_MIN = 0.005;
const TRAJECTORY_BASIN_FORMATION_DELTA_MIN = 0.02;
const TRAJECTORY_BASIN_FORMATION_CONSTRAINT_MAX = 0.001;
const TRAJECTORY_BASIN_STABILIZATION_DELTA_MAX = 0.005;
const TRAJECTORY_BASIN_STABILIZATION_CONFIDENCE_MIN = 0.95;
const TRAJECTORY_BASIN_AGREEMENT_MIN = 0.95;
const BELIEF_BASIN_DEPTH_SCORE_MAX = 0.6;
const BASIN_ENTRY_LOCKIN_STREAK_MIN = 3;
const BASIN_STABILIZATION_DELTA_EPSILON = 0.01;
const BASIN_STABILIZATION_STREAK_MIN = 2;
const BASIN_CYCLE_REINFORCEMENT_MIN = 0.15;
const HARD_FAILURE_METRIC_HELP = "Cv = contract byte mismatch (output != expected), Pf = parse failure, Ld = logic/state failure.";
const HARD_FAILURE_RATE_HELP =
  "Cv/Pf/Ld rates are the percent of turns where each hard failure fired (lower is better). In parse-only mode, Cv and Ld stay diagnostic.";
const FTF_HELP = "FTF = First Failure Turn (first turn where total/parse/logic/structural failure appears).";
const OBJECTIVE_FAILURE_HELP = "objective_failure = 1 when selected objective mode fails on a turn; 0 otherwise.";

const OBJECTIVE_MODE_LABELS = {
  parse_only: "Parse-only failure",
  logic_only: "Logic failure",
  strict_structural: "Strict structural failure",
  composite_pf_or_ld: "Composite (Pf or Ld)"
} as const;

const SPEC_DOWNLOADS = [
  {
    kind: "Metric",
    title: "Structural Contract Compliance (SCC)",
    description: "Deterministic structural compliance metric used to enforce output contracts.",
    href: "/docs/SCC_Structural_Contract_Compliance.pdf",
    buttonLabel: "Download Specification"
  },
  {
    kind: "Protocol",
    title: "SDI-MA Protocol",
    description: "Structural Drift Index for Multi-Agent Systems.",
    href: "/docs/SDI-MA_Protocol_v1.2_revised.pdf",
    buttonLabel: "Download Protocol"
  },
  {
    kind: "Reference Experiment",
    title: "Multi-Agent Drift Lab",
    description: "Canonical SDI-MA implementation demonstrating recursive drift dynamics.",
    href: "/docs/GuardianAI_Experiment_Reference.pdf",
    buttonLabel: "Download Experiment"
  }
] as const;

type SignalVisibilityMode = "public" | "private";
const SIGNAL_VISIBILITY_MODE: SignalVisibilityMode = "public";
const IS_PUBLIC_SIGNAL_MODE = true;
type GuardianRuntimeState = "unknown" | "connected" | "degraded" | "disabled";
type LabSurface = "default" | "app2" | "app3";

type RepCondition = keyof typeof CONDITION_LABELS;
type ExperimentProfile = keyof typeof PROFILE_LABELS;
type ObjectiveMode = keyof typeof OBJECTIVE_MODE_LABELS;
type AgentRole = "A" | "B" | "C";
type Triangle3AgentProfile =
  | "belief_drift_triangle_3agent"
  | "belief_drift_triangle_3agent_isolation"
  | "belief_drift_triangle_9agent_isolation"
  | "belief_drift_triangle_27agent_isolation"
  | "belief_drift_triangle_9agent"
  | "belief_drift_triangle_27agent"
  | "triangle_echo_chamber_3agent"
  | "triangle_evidence_freeze_3agent"
  | "triangle_synth_pressure_3agent"
  | "critic_only_loop_3agent";

const TRIANGLE_3_AGENT_PROFILES: readonly Triangle3AgentProfile[] = [
  "belief_drift_triangle_3agent",
  "belief_drift_triangle_3agent_isolation",
  "belief_drift_triangle_9agent_isolation",
  "belief_drift_triangle_27agent_isolation",
  "belief_drift_triangle_9agent",
  "belief_drift_triangle_27agent",
  "triangle_echo_chamber_3agent",
  "triangle_evidence_freeze_3agent",
  "triangle_synth_pressure_3agent",
  "critic_only_loop_3agent"
] as const;

const TRIANGLE_3_AGENT_PROFILE_SET = new Set<ExperimentProfile>(TRIANGLE_3_AGENT_PROFILES);

function triangleAgentCountForProfile(profile: ExperimentProfile): number {
  if (profile === "belief_drift_triangle_9agent" || profile === "belief_drift_triangle_9agent_isolation") return 9;
  if (profile === "belief_drift_triangle_27agent" || profile === "belief_drift_triangle_27agent_isolation") return 27;
  return 3;
}

function agentCountForProfile(profile: ExperimentProfile): number {
  if (profile === "three_agent_drift_amplifier") return 3;
  if (isBeliefTriangle3AgentProfile(profile)) return triangleAgentCountForProfile(profile);
  return 2;
}

interface TriangleScriptConfig {
  title: string;
  claim: string;
  stance: (typeof CONSENSUS_STANCES)[number];
  fixedEvidenceIds: string[];
  escalationByAgent: Record<AgentRole, number>;
  escalationCap: number;
  freezeStartTurn: number;
  freezeEndTurn: number;
  objective: string;
  summary: string;
}

const TRIANGLE_SCRIPT_CONFIG: Record<Triangle3AgentProfile, TriangleScriptConfig> = {
  belief_drift_triangle_3agent: {
    title: "LAB3 - Controlled Perturbation Propagation (3-Agent)",
    claim: LAB3_TRUE_CLAIM,
    stance: TRIANGLE_FIXED_STANCE,
    fixedEvidenceIds: [...TRIANGLE_FIXED_EVIDENCE_IDS],
    escalationByAgent: { A: TRIANGLE_ESCALATION_DELTA, B: TRIANGLE_ESCALATION_DELTA, C: TRIANGLE_ESCALATION_DELTA },
    escalationCap: TRIANGLE_ESCALATION_MAX_CONFIDENCE,
    freezeStartTurn: TRIANGLE_FREEZE_START_TURN,
    freezeEndTurn: TRIANGLE_FREEZE_END_TURN,
    objective: "Measure whether a small controlled value perturbation propagates as confidence keeps rising.",
    summary:
      "Turns 1-5 keep ground-truth value stable, turn 6 injects a +10% value error once, and turns 7-120 recursively propagate the perturbed belief to track confidence vs decision_error."
  },
  belief_drift_triangle_3agent_isolation: {
    title: "LAB3 - Propagation Isolation (3-Agent)",
    claim: LAB3_TRUE_CLAIM,
    stance: TRIANGLE_FIXED_STANCE,
    fixedEvidenceIds: [...TRIANGLE_FIXED_EVIDENCE_IDS],
    escalationByAgent: { A: TRIANGLE_ESCALATION_DELTA, B: TRIANGLE_ESCALATION_DELTA, C: TRIANGLE_ESCALATION_DELTA },
    escalationCap: TRIANGLE_ESCALATION_MAX_CONFIDENCE,
    freezeStartTurn: TRIANGLE_FREEZE_START_TURN,
    freezeEndTurn: TRIANGLE_FREEZE_END_TURN,
    objective: "Isolate recursive propagation by keeping SANITIZED fixed after perturbation while RAW remains recursive.",
    summary:
      "Turns 1-5 keep ground-truth value stable, turn 6 injects a +10% value error once, then RAW recursively amplifies while SANITIZED recursively damps error toward ground truth."
  },
  belief_drift_triangle_9agent_isolation: {
    title: "LAB3 - Propagation Isolation (9-Agent)",
    claim: LAB3_TRUE_CLAIM,
    stance: TRIANGLE_FIXED_STANCE,
    fixedEvidenceIds: [...TRIANGLE_FIXED_EVIDENCE_IDS],
    escalationByAgent: { A: TRIANGLE_ESCALATION_DELTA, B: TRIANGLE_ESCALATION_DELTA, C: TRIANGLE_ESCALATION_DELTA },
    escalationCap: TRIANGLE_ESCALATION_MAX_CONFIDENCE,
    freezeStartTurn: TRIANGLE_FREEZE_START_TURN,
    freezeEndTurn: TRIANGLE_FREEZE_END_TURN,
    objective:
      "Scale LAB3 isolation to 9 agents with identical perturbation protocol and deterministic recursive reinjection dynamics.",
    summary:
      "Same LAB3 isolation design with 9 sequential agents: single-shot perturbation at turn 6, RAW recursive amplification, SANITIZED recursive damping toward ground truth."
  },
  belief_drift_triangle_27agent_isolation: {
    title: "LAB3 - Propagation Isolation (27-Agent)",
    claim: LAB3_TRUE_CLAIM,
    stance: TRIANGLE_FIXED_STANCE,
    fixedEvidenceIds: [...TRIANGLE_FIXED_EVIDENCE_IDS],
    escalationByAgent: { A: TRIANGLE_ESCALATION_DELTA, B: TRIANGLE_ESCALATION_DELTA, C: TRIANGLE_ESCALATION_DELTA },
    escalationCap: TRIANGLE_ESCALATION_MAX_CONFIDENCE,
    freezeStartTurn: TRIANGLE_FREEZE_START_TURN,
    freezeEndTurn: TRIANGLE_FREEZE_END_TURN,
    objective:
      "Scale LAB3 isolation to 27 agents with identical perturbation protocol and deterministic recursive reinjection dynamics.",
    summary:
      "Same LAB3 isolation design with 27 sequential agents: single-shot perturbation at turn 6, RAW recursive amplification, SANITIZED recursive damping toward ground truth."
  },
  belief_drift_triangle_9agent: {
    title: "Canonical Drift Run (9-Agent)",
    claim: TRIANGLE_FIXED_CLAIM,
    stance: TRIANGLE_FIXED_STANCE,
    fixedEvidenceIds: [...TRIANGLE_FIXED_EVIDENCE_IDS],
    escalationByAgent: { A: TRIANGLE_ESCALATION_DELTA, B: TRIANGLE_ESCALATION_DELTA, C: TRIANGLE_ESCALATION_DELTA },
    escalationCap: TRIANGLE_ESCALATION_MAX_CONFIDENCE,
    freezeStartTurn: TRIANGLE_FREEZE_START_TURN,
    freezeEndTurn: TRIANGLE_FREEZE_END_TURN,
    objective: "Canonical recursive drift benchmark for publication/comparison.",
    summary: "Deterministic LAB2 protocol with expanded sequential agent cycle (9 agents) and identical constraints."
  },
  belief_drift_triangle_27agent: {
    title: "Canonical Drift Run (27-Agent)",
    claim: TRIANGLE_FIXED_CLAIM,
    stance: TRIANGLE_FIXED_STANCE,
    fixedEvidenceIds: [...TRIANGLE_FIXED_EVIDENCE_IDS],
    escalationByAgent: { A: TRIANGLE_ESCALATION_DELTA, B: TRIANGLE_ESCALATION_DELTA, C: TRIANGLE_ESCALATION_DELTA },
    escalationCap: TRIANGLE_ESCALATION_MAX_CONFIDENCE,
    freezeStartTurn: TRIANGLE_FREEZE_START_TURN,
    freezeEndTurn: TRIANGLE_FREEZE_END_TURN,
    objective: "Canonical recursive drift benchmark for publication/comparison.",
    summary: "Deterministic LAB2 protocol with expanded sequential agent cycle (27 agents) and identical constraints."
  },
  triangle_echo_chamber_3agent: {
    title: "Echo Chamber Stress (3-Agent)",
    claim: TRIANGLE_FIXED_CLAIM,
    stance: "support",
    fixedEvidenceIds: ["e1"],
    escalationByAgent: { A: 0.07, B: 0.05, C: 0.05 },
    escalationCap: TRIANGLE_ESCALATION_MAX_CONFIDENCE,
    freezeStartTurn: TRIANGLE_FREEZE_START_TURN,
    freezeEndTurn: TRIANGLE_FREEZE_END_TURN,
    objective: "Stress rapid agreement lock under minimal evidence diversity.",
    summary: "All turns force aligned stance with fixed evidence to maximize recursive reinforcement."
  },
  triangle_evidence_freeze_3agent: {
    title: "Evidence Freeze Stress (3-Agent)",
    claim: TRIANGLE_FIXED_CLAIM,
    stance: TRIANGLE_FIXED_STANCE,
    fixedEvidenceIds: ["e1", "e2"],
    escalationByAgent: { A: 0.06, B: 0.06, C: 0.06 },
    escalationCap: 0.96,
    freezeStartTurn: TRIANGLE_FREEZE_START_TURN,
    freezeEndTurn: TRIANGLE_FREEZE_END_TURN,
    objective: "Isolate escalation behavior under permanent evidence freeze with bounded cap.",
    summary: "Evidence ids are frozen from turn 1 while confidence remains deterministic and monotonic."
  },
  triangle_synth_pressure_3agent: {
    title: "Synthesizer Pressure (3-Agent)",
    claim: TRIANGLE_FIXED_CLAIM,
    stance: TRIANGLE_FIXED_STANCE,
    fixedEvidenceIds: ["e1", "e2"],
    escalationByAgent: { A: 0.05, B: 0.05, C: 0.08 },
    escalationCap: TRIANGLE_ESCALATION_MAX_CONFIDENCE,
    freezeStartTurn: TRIANGLE_FREEZE_START_TURN,
    freezeEndTurn: TRIANGLE_FREEZE_END_TURN,
    objective: "Amplify C-agent summary pressure to surface faster RAW/SAN divergence.",
    summary: "Same deterministic contract with stronger synthesizer weighting for faster recursive lock-in."
  },
  critic_only_loop_3agent: {
    title: "Critic-Only Loop (3-Agent)",
    claim: CRITIC_ONLY_CLAIM,
    stance: "revise",
    fixedEvidenceIds: [...CRITIC_ONLY_EVIDENCE_IDS],
    escalationByAgent: { A: TRIANGLE_ESCALATION_DELTA, B: TRIANGLE_ESCALATION_DELTA, C: TRIANGLE_ESCALATION_DELTA },
    escalationCap: TRIANGLE_ESCALATION_MAX_CONFIDENCE,
    freezeStartTurn: TRIANGLE_FREEZE_START_TURN,
    freezeEndTurn: TRIANGLE_FREEZE_END_TURN,
    objective:
      "Test whether recursive drift emerges when synthesis is removed and all agents operate as evaluators in a closed loop.",
    summary: "A proposer, a critic, and a meta-critic recurse over fixed evidence to isolate topology-driven reinforcement."
  }
};

function triangleConfigForProfile(profile: ExperimentProfile): TriangleScriptConfig {
  if (TRIANGLE_3_AGENT_PROFILE_SET.has(profile)) {
    return TRIANGLE_SCRIPT_CONFIG[profile as Triangle3AgentProfile];
  }
  return TRIANGLE_SCRIPT_CONFIG.belief_drift_triangle_3agent;
}

interface StructuralGuardrailCriterion {
  reinforcementDeltaMin: number;
  driftP95RatioMin: number;
  parseOkMin: number;
  stateOkMin: number;
}

const STRUCTURAL_GUARDRAIL: StructuralGuardrailCriterion = {
  reinforcementDeltaMin: 0,
  driftP95RatioMin: 2,
  parseOkMin: 0.95,
  stateOkMin: 0.95
};

interface RunConfig {
  runId: string;
  profile: ExperimentProfile;
  condition: RepCondition;
  objectiveMode: ObjectiveMode;
  providerPreference: APIProvider;
  resolvedProvider: APIProvider;
  modelA: string;
  modelB: string;
  agentCount: number;
  temperature: number;
  retries: number;
  horizon: number;
  maxTokens: number;
  initialStep: number;
  interTurnDelayMs: number;
  maxHistoryTurns: number;
  stopOnFirstFailure: boolean;
  strictSanitizedKeyOrder: boolean;
  historyAccumulation: boolean;
  preflightEnabled: boolean;
  preflightTurns: number;
  preflightAgent: AgentRole;
  preflightParseOkMin: number;
  preflightStateOkMin: number;
  createdAt: string;
}

interface TurnTrace {
  runId: string;
  profile: ExperimentProfile;
  condition: RepCondition;
  turnIndex: number;
  agent: AgentRole;
  agentSlot: string;
  agentModel: string;
  inputBytes: string;
  historyBytes: string;
  outputBytes: string;
  expectedBytes: string;
  injectedBytesNext: string;
  expectedStep: number;
  parsedStep: number | null;
  parseOk: number;
  stateOk: number;
  pf: number;
  cv: number;
  ld: number;
  objectiveFailure: number;
  uptime: number;
  rawHash: string;
  expectedHash: string;
  byteLength: number;
  lineCount: number;
  prefixLen: number;
  suffixLen: number;
  lenDeltaVsContract: number;
  deviationMagnitude: number;
  indentAvg: number;
  indentMax: number;
  indentDelta: number | null;
  bTransformOk: number | null;
  bTransformReason?: string;
  rollingPf20: number;
  rollingDriftP95: number;
  contextLength: number;
  contextLengthGrowth: number;
  devState: number;
  guardianGateState: "CONTINUE" | "PAUSE" | "YIELD" | null;
  guardianStructuralRecommendation: "CONTINUE" | "SLOW" | "REOPEN" | "DEFER" | null;
  guardianReasonCodes: string[];
  guardianEnergyV: number | null;
  guardianAuthorityTrend: number | null;
  guardianRevisionMode: string | null;
  guardianTrajectoryState: string | null;
  guardianTemporalResistanceDetected: number | null;
  guardianObserveError: string | null;
  reasoningDepth: number | null;
  authorityWeights: number | null;
  contradictionSignal: number | null;
  alternativeVariance: number | null;
  agreementRate: number | null;
  evidenceDiversity: number | null;
  elapsedTimeMs: number | null;
  commitment: number | null;
  commitmentDelta: number | null;
  decisionError: number | null;
  decisionValue: number | null;
  constraintGrowth: number | null;
  evidenceDelta: number | null;
  depthDelta: number | null;
  driftRuleSatisfied: number;
  driftStreak: number;
  structuralEpistemicDrift: number;
  dai: number | null;
  daiDelta: number | null;
  daiRegime: string | null;
  parseError?: string;
  parsedData?: Record<string, unknown>;
}

interface PhaseTransitionCandidate {
  turn: number;
  reason: string;
  beforeSample: string;
  afterSample: string;
}

interface EdgeTransferStats {
  from: AgentRole;
  to: AgentRole;
  pairCount: number;
  devBase: number;
  cleanBase: number;
  pDevGivenDev: number | null;
  pDevGivenClean: number | null;
  delta: number | null;
}

type TrajectoryStatus = "exploration" | "reinforcement" | "basin_formation" | "basin_stabilization";
type TrajectoryDynamics = "stable" | "building" | "accelerating" | "closing";
type BasinState = "open" | "forming" | "stabilized";
type BeliefBasinStrengthBand = "none" | "forming" | "shallow" | "deep" | "locked";

interface ConditionSummary {
  runConfig: RunConfig;
  profile: ExperimentProfile;
  condition: RepCondition;
  objectiveMode: ObjectiveMode;
  objectiveLabel: string;
  objectiveScopeLabel: string;
  startedAt: string;
  finishedAt: string;
  turnsConfigured: number;
  turnsAttempted: number;
  failed: boolean;
  failureReason?: string;
  parseOkRate: number | null;
  parseOkRateA: number | null;
  parseOkRateB: number | null;
  parseOkRateC: number | null;
  stateOkRate: number | null;
  stateOkRateA: number | null;
  stateOkRateB: number | null;
  stateOkRateC: number | null;
  cvRate: number | null;
  cvRateA: number | null;
  cvRateB: number | null;
  cvRateC: number | null;
  pfRate: number | null;
  pfRateA: number | null;
  pfRateB: number | null;
  pfRateC: number | null;
  ldRate: number | null;
  ldRateA: number | null;
  ldRateB: number | null;
  ldRateC: number | null;
  contextGrowthAvg: number | null;
  contextGrowthMax: number | null;
  contextGrowthSlope: number | null;
  driftAvg: number | null;
  driftP95: number | null;
  driftMax: number | null;
  escalationSlope: number | null;
  earlySlope40: number | null;
  driftAvgA: number | null;
  driftP95A: number | null;
  driftMaxA: number | null;
  escalationSlopeA: number | null;
  earlySlope40A: number | null;
  indentAvg: number | null;
  indentMax: number | null;
  indentDeltaAvg: number | null;
  indentAvgA: number | null;
  indentMaxA: number | null;
  indentDeltaAvgA: number | null;
  bTransformOkRate: number | null;
  bTransformSamples: number;
  consensusPairs: number;
  agreementRateAB: number | null;
  evidenceDiversity: number | null;
  unsupportedConsensusRate: number | null;
  unsupportedConsensusStreakMax: number;
  noNewEvidenceRate: number | null;
  evidenceGrowthRate: number | null;
  confidenceGainAvg: number | null;
  decisionErrorLatest: number | null;
  decisionErrorPeak: number | null;
  decisionErrorSlope: number | null;
  firstDecisionErrorTurn: number | null;
  propagationDetected: number | null;
  driftTurns: number[];
  driftTurnModuloAgentCount: number[];
  driftWindowStartTurns: number[];
  driftWindowStartModuloAgentCount: number[];
  driftWindowCycleSynchronized: number | null;
  driftWindowPeriodTurns: number | null;
  driftWindowRecursEveryCycle: number | null;
  avgReasoningDepth: number | null;
  avgAlternativeVariance: number | null;
  avgCommitmentDeltaPos: number | null;
  constraintGrowthRate: number | null;
  closureConstraintRatio: number | null;
  structuralDriftStreakMax: number;
  firstStructuralDriftTurn: number | null;
  lockInOnsetTurn: number | null;
  lockInScoreLatest: number | null;
  lockInScorePeak: number | null;
  lockInPositiveStreakMax: number;
  trajectoryStabilityIndexLatest: number | null;
  trajectoryStabilityIndexPeak: number | null;
  trajectoryStatusLatest: TrajectoryStatus | null;
  basinStateLatest: BasinState | null;
  cycleReinforcement3Latest: number | null;
  cycleReinforcement3Peak: number | null;
  firstBasinFormationTurn: number | null;
  firstBasinStabilizationTurn: number | null;
  beliefBasinDepth: number | null;
  beliefBasinStrengthScore: number | null;
  beliefBasinStrengthBand: BeliefBasinStrengthBand | null;
  basinMetricInconsistencyWarning: number;
  structuralEpistemicDriftFlag: number;
  structuralEpistemicDriftReason: string | null;
  daiLatest: number | null;
  daiDeltaLatest: number | null;
  daiPeak: number | null;
  daiSlope: number | null;
  daiRegimeLatest: string | null;
  daiFirstAttractorTurn: number | null;
  daiFirstDriftTurn: number | null;
  daiFirstAmplificationTurn: number | null;
  daiPositiveSlopeStreakMax: number;
  lagTransferABDevGivenPrevDev: number | null;
  lagTransferABDevGivenPrevClean: number | null;
  lagTransferABDelta: number | null;
  artifactHalfLifeTurns: number | null;
  consensusCollapseFlag: number;
  consensusCollapseReason: string | null;
  artifactPersistenceA: number | null;
  templateEntropyA: number | null;
  artifactPersistence: number | null;
  persistenceRate: number | null;
  reinforcementWhenDev: number | null;
  reinforcementWhenClean: number | null;
  reinforcementDelta: number | null;
  reinforcementWhenDevA: number | null;
  reinforcementWhenCleanA: number | null;
  reinforcementDeltaA: number | null;
  reinforcementWhenDevB: number | null;
  reinforcementWhenCleanB: number | null;
  reinforcementDeltaB: number | null;
  reinforcementWhenDevC: number | null;
  reinforcementWhenCleanC: number | null;
  reinforcementDeltaC: number | null;
  edgeAB: EdgeTransferStats;
  edgeBC: EdgeTransferStats;
  edgeCA: EdgeTransferStats;
  prevOutputToNextInputRate: number | null;
  prevInjectedToNextInputRate: number | null;
  firstSuffixDriftTurn: number | null;
  maxSuffixLen: number | null;
  suffixGrowthSlope: number | null;
  lineCountMax: number | null;
  ftfParse: number | null;
  ftfLogic: number | null;
  ftfStruct: number | null;
  ftfTotal: number | null;
  ftfParseA: number | null;
  ftfLogicA: number | null;
  ftfStructA: number | null;
  ftfTotalA: number | null;
  preflightPassed: boolean | null;
  preflightReason: string | null;
  maxRollingReinforcementDelta: number | null;
  persistenceInflectionTurn: number | null;
  persistenceInflectionDelta: number | null;
  collapseLeadTurnsFromInflection: number | null;
  guardianObserveCoverage: number | null;
  guardianPauseRate: number | null;
  guardianYieldRate: number | null;
  guardianContinueRate: number | null;
  guardianReopenRate: number | null;
  guardianSlowRate: number | null;
  guardianDeferRate: number | null;
  guardianObserveErrorRate: number | null;
  phaseTransition: PhaseTransitionCandidate | null;
  traces: TurnTrace[];
}

interface GuardianObserveResponse {
  gateState?: "CONTINUE" | "PAUSE" | "YIELD";
  structuralRecommendation?: "CONTINUE" | "SLOW" | "REOPEN" | "DEFER" | null;
  reasonCodes?: string[];
  triangleV?: number | null;
  triangleDeltaV?: number | null;
  triangleCircleMode?: string | null;
  triangleSpiralMode?: string | null;
  triangleInvariantViolation?: number | null;
}

type ConditionResults = Record<RepCondition, ConditionSummary | null>;
type ResultsByProfile = Record<ExperimentProfile, ConditionResults>;

interface DriftTelemetry {
  contextGrowthAvg: number | null;
  contextGrowthMax: number | null;
  contextGrowthSlope: number | null;
  driftAvg: number | null;
  driftP95: number | null;
  driftMax: number | null;
  escalationSlope: number | null;
  earlySlope40: number | null;
  indentAvg: number | null;
  indentMax: number | null;
  indentDeltaAvg: number | null;
  artifactPersistence: number | null;
  persistenceRate: number | null;
  reinforcementWhenDev: number | null;
  reinforcementWhenClean: number | null;
  reinforcementDelta: number | null;
  reinforcementWhenDevA: number | null;
  reinforcementWhenCleanA: number | null;
  reinforcementDeltaA: number | null;
  reinforcementWhenDevB: number | null;
  reinforcementWhenCleanB: number | null;
  reinforcementDeltaB: number | null;
  reinforcementWhenDevC: number | null;
  reinforcementWhenCleanC: number | null;
  reinforcementDeltaC: number | null;
}

interface ObjectiveEval {
  pass: boolean;
  driftRatio: number | null;
  reinforcementDelta: number | null;
  spi: number | null;
  cvRateRawA: number | null;
  cvRateSanitizedA: number | null;
  ftfStructRawA: number | null;
  ftfStructSanitizedA: number | null;
  structuralGateSeparated: boolean;
}

interface ConsensusEval {
  pass: boolean;
  rawSignal: boolean;
  sanitizedSignal: boolean;
  rawAgreement: number | null;
  rawDiversity: number | null;
  rawNoNewEvidence: number | null;
  rawPairs: number;
  sanitizedAgreement: number | null;
  sanitizedDiversity: number | null;
  sanitizedNoNewEvidence: number | null;
  sanitizedPairs: number;
  windowGapTurns: number;
  devGapWindowMean: number | null;
  devGapWindowMax: number | null;
  lagTransferGap: number | null;
  halfLifeGap: number | null;
  rawFirstStructuralDriftTurn: number | null;
  sanitizedFirstStructuralDriftTurn: number | null;
  rawStructuralDriftStreakMax: number;
  sanitizedStructuralDriftStreakMax: number;
  rawClosureConstraintRatio: number | null;
  sanitizedClosureConstraintRatio: number | null;
  rawConstraintGrowthRate: number | null;
  sanitizedConstraintGrowthRate: number | null;
  rawLockInOnsetTurn: number | null;
  sanitizedLockInOnsetTurn: number | null;
  rawLockInScoreLatest: number | null;
  sanitizedLockInScoreLatest: number | null;
  rawLockInScorePeak: number | null;
  sanitizedLockInScorePeak: number | null;
  rawTrajectoryStabilityIndexLatest: number | null;
  sanitizedTrajectoryStabilityIndexLatest: number | null;
  rawTrajectoryStabilityIndexPeak: number | null;
  sanitizedTrajectoryStabilityIndexPeak: number | null;
  rawTrajectoryStatusLatest: TrajectoryStatus | null;
  sanitizedTrajectoryStatusLatest: TrajectoryStatus | null;
  rawBasinStateLatest: BasinState | null;
  sanitizedBasinStateLatest: BasinState | null;
  rawCycleReinforcement3Latest: number | null;
  sanitizedCycleReinforcement3Latest: number | null;
  rawCycleReinforcement3Peak: number | null;
  sanitizedCycleReinforcement3Peak: number | null;
  rawFirstBasinFormationTurn: number | null;
  sanitizedFirstBasinFormationTurn: number | null;
  rawFirstBasinStabilizationTurn: number | null;
  sanitizedFirstBasinStabilizationTurn: number | null;
  rawBeliefBasinDepth: number | null;
  sanitizedBeliefBasinDepth: number | null;
  rawBeliefBasinStrengthScore: number | null;
  sanitizedBeliefBasinStrengthScore: number | null;
  rawBeliefBasinStrengthBand: BeliefBasinStrengthBand | null;
  sanitizedBeliefBasinStrengthBand: BeliefBasinStrengthBand | null;
  rawBasinMetricInconsistencyWarning: number;
  sanitizedBasinMetricInconsistencyWarning: number;
  rawDaiLatest: number | null;
  sanitizedDaiLatest: number | null;
  rawDaiDeltaLatest: number | null;
  sanitizedDaiDeltaLatest: number | null;
  rawDaiSlope: number | null;
  sanitizedDaiSlope: number | null;
  rawDaiRegime: string | null;
  sanitizedDaiRegime: string | null;
}

interface ClosureVerdict {
  label: string;
  tone: "good" | "warn" | "bad";
  detail: string;
}

interface MatrixTrialRow {
  profile: ExperimentProfile;
  model: string;
  replicate: number;
  closureDetected: number | null;
  lagTransferGap: number | null;
  halfLifeGap: number | null;
  devGapWindowMean: number | null;
  devGapWindowMax: number | null;
}

interface MatrixAggregateRow {
  model: string;
  trials: number;
  closureDetectedRate: number | null;
  lagTransferGapAvg: number | null;
  halfLifeGapAvg: number | null;
  devGapWindowMeanAvg: number | null;
  devGapWindowMaxAvg: number | null;
}

function emptyResults(): ResultsByProfile {
  return {
    epistemic_drift_protocol: { raw: null, sanitized: null },
    belief_drift_triangle_3agent: { raw: null, sanitized: null },
    belief_drift_triangle_3agent_isolation: { raw: null, sanitized: null },
    belief_drift_triangle_9agent_isolation: { raw: null, sanitized: null },
    belief_drift_triangle_27agent_isolation: { raw: null, sanitized: null },
    belief_drift_triangle_9agent: { raw: null, sanitized: null },
    belief_drift_triangle_27agent: { raw: null, sanitized: null },
    triangle_echo_chamber_3agent: { raw: null, sanitized: null },
    triangle_evidence_freeze_3agent: { raw: null, sanitized: null },
    triangle_synth_pressure_3agent: { raw: null, sanitized: null },
    critic_only_loop_3agent: { raw: null, sanitized: null },
    three_agent_drift_amplifier: { raw: null, sanitized: null },
    drift_amplifying_loop: { raw: null, sanitized: null },
    consensus_collapse_loop: { raw: null, sanitized: null },
    propagation_stress_loop: { raw: null, sanitized: null },
    generator_normalizer: { raw: null, sanitized: null },
    symmetric_control: { raw: null, sanitized: null },
    dialect_negotiation: { raw: null, sanitized: null }
  };
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function safeRate(numerator: number, denominator: number): number | null {
  if (denominator <= 0) return null;
  return numerator / denominator;
}

function asPercent(value: number | null): string {
  if (value === null) return "N/A";
  return `${(value * 100).toFixed(1)}%`;
}

function asFixed(value: number | null, digits = 3): string {
  if (value === null || !Number.isFinite(value)) return "N/A";
  return value.toFixed(digits);
}

function formatTurnList(values: number[], limit = 16): string {
  if (values.length === 0) return "n/a";
  const shown = values.slice(0, limit).join(", ");
  return values.length > limit ? `${shown}, ...` : shown;
}

function previewText(value: string | null | undefined, maxChars = 120): string {
  if (!value) return "";
  if (value.length <= maxChars) return value;
  return `${value.slice(0, maxChars)}...`;
}

function traceAgentDisplay(trace: Pick<TurnTrace, "agent" | "agentSlot">): string {
  return trace.agentSlot || trace.agent;
}

function toContractLiteral(step: number): string {
  return `{"step":${step},"state":"${CONTRACT_STATE_LITERAL}","meta":"${CONTRACT_META_LITERAL}"}`;
}

function toConsensusLiteral(state: {
  claim: string;
  stance: (typeof CONSENSUS_STANCES)[number];
  confidence: number;
  evidenceIds: string[];
}): string {
  return JSON.stringify({
    claim: state.claim,
    stance: state.stance,
    confidence: state.confidence,
    evidence_ids: state.evidenceIds
  });
}

function isBeliefTriangle3AgentProfile(profile: ExperimentProfile): boolean {
  return TRIANGLE_3_AGENT_PROFILE_SET.has(profile);
}

function isCriticOnlyLoopProfile(profile: ExperimentProfile): boolean {
  return profile === "critic_only_loop_3agent";
}

function isCanonicalBeliefDriftProfile(profile: ExperimentProfile): boolean {
  return (
    profile === "belief_drift_triangle_3agent" ||
    profile === "belief_drift_triangle_3agent_isolation" ||
    profile === "belief_drift_triangle_9agent_isolation" ||
    profile === "belief_drift_triangle_27agent_isolation" ||
    profile === "belief_drift_triangle_9agent" ||
    profile === "belief_drift_triangle_27agent"
  );
}

function isLab3PerturbationProfile(profile: ExperimentProfile): boolean {
  return (
    profile === "belief_drift_triangle_3agent" ||
    profile === "belief_drift_triangle_3agent_isolation" ||
    profile === "belief_drift_triangle_9agent_isolation" ||
    profile === "belief_drift_triangle_27agent_isolation"
  );
}

function isLab3PropagationIsolationProfile(profile: ExperimentProfile): boolean {
  return (
    profile === "belief_drift_triangle_3agent_isolation" ||
    profile === "belief_drift_triangle_9agent_isolation" ||
    profile === "belief_drift_triangle_27agent_isolation"
  );
}

function beliefProfileUsesStep(profile: ExperimentProfile): boolean {
  return isBeliefTriangle3AgentProfile(profile);
}

function toBeliefStateLiteral(
  profile: ExperimentProfile,
  state: {
    claim: string;
    stance: (typeof CONSENSUS_STANCES)[number];
    confidence: number;
    evidenceIds: string[];
  },
  step?: number
): string {
  if (beliefProfileUsesStep(profile)) {
    return JSON.stringify({
      step: step ?? 0,
      claim: state.claim,
      stance: state.stance,
      confidence: state.confidence,
      evidence_ids: state.evidenceIds
    });
  }
  return toConsensusLiteral(state);
}

function isBeliefLoopProfile(profile: ExperimentProfile): boolean {
  return (
    profile === "epistemic_drift_protocol" ||
    isBeliefTriangle3AgentProfile(profile) ||
    profile === "consensus_collapse_loop" ||
    profile === "propagation_stress_loop"
  );
}

function beliefEvidenceIdsForProfile(profile: ExperimentProfile): readonly string[] {
  if (isCriticOnlyLoopProfile(profile)) return CRITIC_ONLY_EVIDENCE_IDS;
  if (isBeliefTriangle3AgentProfile(profile)) return BELIEF_TRIANGLE_EVIDENCE_IDS;
  return profile === "propagation_stress_loop" ? BELIEF_STRESS_EVIDENCE_IDS : BELIEF_BASELINE_EVIDENCE_IDS;
}

function beliefEvidencePoolForProfile(profile: ExperimentProfile): Record<string, string> {
  if (isCriticOnlyLoopProfile(profile)) return CRITIC_ONLY_EVIDENCE_POOL;
  if (isBeliefTriangle3AgentProfile(profile)) return BELIEF_TRIANGLE_EVIDENCE_POOL;
  return profile === "propagation_stress_loop" ? BELIEF_STRESS_EVIDENCE_POOL : BELIEF_BASELINE_EVIDENCE_POOL;
}

function beliefSummaryLimitForProfile(profile: ExperimentProfile): number {
  return profile === "propagation_stress_loop" ? BELIEF_STRESS_SUMMARY_MAX_CHARS : BELIEF_BASELINE_SUMMARY_MAX_CHARS;
}

function beliefMaxEvidenceIdsForProfile(profile: ExperimentProfile): number {
  if (isCriticOnlyLoopProfile(profile)) return CRITIC_ONLY_EVIDENCE_IDS.length;
  if (isBeliefTriangle3AgentProfile(profile)) return BELIEF_TRIANGLE_MAX_EVIDENCE_IDS;
  return profile === "propagation_stress_loop" ? BELIEF_STRESS_MAX_EVIDENCE_IDS : BELIEF_BASELINE_MAX_EVIDENCE_IDS;
}

function initialStateLiteralForProfile(profile: ExperimentProfile, initialStep: number): string {
  if (isBeliefLoopProfile(profile)) {
    const initialClaim = isCriticOnlyLoopProfile(profile)
      ? CRITIC_ONLY_CLAIM
      : isLab3PerturbationProfile(profile)
        ? LAB3_TRUE_CLAIM
        : "C1";
    const initialEvidenceIds = isCriticOnlyLoopProfile(profile)
      ? [...CRITIC_ONLY_EVIDENCE_IDS]
      : isBeliefTriangle3AgentProfile(profile)
        ? ["e1", "e2"]
        : profile === "propagation_stress_loop"
          ? ["e1", "e4"]
          : ["e1"];
    return toBeliefStateLiteral(
      profile,
      {
        claim: initialClaim,
        stance: "revise",
        confidence: profile === "propagation_stress_loop" ? 0.42 : 0.35,
        evidenceIds: initialEvidenceIds
      },
      beliefProfileUsesStep(profile) ? initialStep : undefined
    );
  }
  return toContractLiteral(initialStep);
}

function basinProbeTurnContext(turnIndex: number): {
  isShockTurn: boolean;
  isFreezeTurn: boolean;
} {
  return {
    isShockTurn: BASIN_PROBE_SHOCK_TURNS.includes(turnIndex as (typeof BASIN_PROBE_SHOCK_TURNS)[number]),
    isFreezeTurn: turnIndex >= BASIN_PROBE_FREEZE_START_TURN && turnIndex <= BASIN_PROBE_FREEZE_END_TURN
  };
}

function triangleTurnContext(profile: ExperimentProfile, turnIndex: number): {
  isFreezeTurn: boolean;
} {
  const config = triangleConfigForProfile(profile);
  return {
    isFreezeTurn: turnIndex >= config.freezeStartTurn && turnIndex <= config.freezeEndTurn
  };
}

function lineCountFor(content: string): number {
  if (content.length === 0) return 0;
  return content.split(/\r\n|\r|\n/).length;
}

function splitNormalizedLines(content: string): string[] {
  if (content.length === 0) return [""];
  return content.split(/\r\n|\r|\n/);
}

function leadingSpaces(line: string): number {
  let count = 0;
  while (count < line.length && line[count] === " ") {
    count += 1;
  }
  return count;
}

function indentationTelemetry(content: string): { indentAvg: number; indentMax: number } {
  const lines = splitNormalizedLines(content);
  const nonEmptyLines = lines.filter((line) => line.trim().length > 0);
  if (nonEmptyLines.length === 0) {
    return { indentAvg: 0, indentMax: 0 };
  }
  const indents = nonEmptyLines.map((line) => leadingSpaces(line));
  const indentMax = Math.max(...indents);
  const indentAvg = indents.reduce((sum, value) => sum + value, 0) / indents.length;
  return { indentAvg, indentMax };
}

function evaluateMonotoneBTransform(inputBytes: string, outputBytes: string): { ok: boolean; reason?: string } {
  const inputLines = splitNormalizedLines(inputBytes);
  const outputLines = splitNormalizedLines(outputBytes);
  const inputIsSingleLine = inputLines.length <= 1;

  if (inputIsSingleLine) {
    if (outputLines.length <= 1) {
      return { ok: false, reason: "B-transform: unlock failed (single-line input must become multi-line)." };
    }
    const innerLines = outputLines.slice(1, -1).filter((line) => line.trim().length > 0);
    if (innerLines.length === 0) {
      return { ok: false, reason: "B-transform: unlock failed (missing indented inner lines)." };
    }
    if (!innerLines.every((line) => line.startsWith("  "))) {
      return { ok: false, reason: "B-transform: unlock failed (expected 2-space indentation)." };
    }
    return { ok: true };
  }

  if (outputLines.length !== inputLines.length) {
    return {
      ok: false,
      reason: `B-transform: accumulate failed (line count changed ${inputLines.length} -> ${outputLines.length}).`
    };
  }

  for (let index = 0; index < inputLines.length; index += 1) {
    const inLine = inputLines[index];
    const outLine = outputLines[index];
    const inTrimmed = inLine.trim();
    const outTrimmed = outLine.trim();
    if (inTrimmed.length === 0) {
      if (outTrimmed.length !== 0) {
        return { ok: false, reason: "B-transform: accumulate failed (blank line changed)." };
      }
      continue;
    }

    const inLead = leadingSpaces(inLine);
    const outLead = leadingSpaces(outLine);
    if (inLead > 0) {
      if (outLead !== inLead + 1) {
        return {
          ok: false,
          reason: `B-transform: accumulate failed (expected +1 indent on line ${index + 1}).`
        };
      }
      if (outLine.slice(outLead) !== inLine.slice(inLead)) {
        return {
          ok: false,
          reason: `B-transform: accumulate failed (line content changed beyond indentation on line ${index + 1}).`
        };
      }
    } else if (outLine !== inLine) {
      return {
        ok: false,
        reason: `B-transform: accumulate failed (non-indented line changed on line ${index + 1}).`
      };
    }
  }

  return { ok: true };
}

function boundaryDeviation(rawOutput: string, expectedOutput: string) {
  const byteLength = rawOutput.length;
  const firstObjectStart = rawOutput.indexOf("{");
  const lastObjectEnd = rawOutput.lastIndexOf("}");
  const prefixLen = firstObjectStart >= 0 ? firstObjectStart : byteLength;
  const suffixLen = lastObjectEnd >= 0 ? Math.max(0, byteLength - (lastObjectEnd + 1)) : 0;
  const lineCount = lineCountFor(rawOutput);
  const lenDeltaVsContract = byteLength - expectedOutput.length;
  const deviationMagnitude = prefixLen + suffixLen + Math.abs(lenDeltaVsContract) + Math.max(0, lineCount - 1);
  return { byteLength, lineCount, prefixLen, suffixLen, lenDeltaVsContract, deviationMagnitude };
}

function percentile(values: number[], ratio: number): number | null {
  if (values.length === 0) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const clamped = Math.min(1, Math.max(0, ratio));
  const position = (sorted.length - 1) * clamped;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function metricSlope(traces: TurnTrace[], valueFor: (trace: TurnTrace) => number): number | null {
  if (traces.length < 2) return null;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumXX = 0;
  for (const trace of traces) {
    const x = trace.turnIndex;
    const y = valueFor(trace);
    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumXX += x * x;
  }
  const n = traces.length;
  const denominator = n * sumXX - sumX * sumX;
  if (denominator === 0) return 0;
  return (n * sumXY - sumX * sumY) / denominator;
}

function templateSignature(outputBytes: string): string {
  // Keep structure/whitespace pattern while masking numeric evolution.
  return outputBytes.replace(/-?\d+/g, "<int>");
}

function shannonEntropy(values: string[]): number | null {
  if (values.length === 0) return null;
  const counts = new Map<string, number>();
  for (const value of values) {
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  const total = values.length;
  let entropy = 0;
  for (const count of counts.values()) {
    const p = count / total;
    entropy -= p * Math.log2(p);
  }
  return entropy;
}

function driftTelemetry(traces: TurnTrace[]): DriftTelemetry {
  if (traces.length === 0) {
    return {
      contextGrowthAvg: null,
      contextGrowthMax: null,
      contextGrowthSlope: null,
      driftAvg: null,
      driftP95: null,
      driftMax: null,
      escalationSlope: null,
      earlySlope40: null,
      indentAvg: null,
      indentMax: null,
      indentDeltaAvg: null,
      artifactPersistence: null,
      persistenceRate: null,
      reinforcementWhenDev: null,
      reinforcementWhenClean: null,
      reinforcementDelta: null,
      reinforcementWhenDevA: null,
      reinforcementWhenCleanA: null,
      reinforcementDeltaA: null,
      reinforcementWhenDevB: null,
      reinforcementWhenCleanB: null,
      reinforcementDeltaB: null,
      reinforcementWhenDevC: null,
      reinforcementWhenCleanC: null,
      reinforcementDeltaC: null
    };
  }

  const magnitudes = traces.map((trace) => trace.deviationMagnitude);
  const contextGrowths = traces.map((trace) => trace.contextLengthGrowth);

  const contextGrowthAvg = contextGrowths.reduce((sum, value) => sum + value, 0) / contextGrowths.length;
  const contextGrowthMax = Math.max(...contextGrowths);
  const contextGrowthSlope = metricSlope(traces, (trace) => trace.contextLengthGrowth);

  const driftAvg = magnitudes.reduce((sum, value) => sum + value, 0) / magnitudes.length;
  const driftP95 = percentile(magnitudes, 0.95);
  const driftMax = Math.max(...magnitudes);
  const escalationSlope = metricSlope(traces, (trace) => trace.deviationMagnitude);
  const earlyTraces = traces.slice(0, Math.min(EARLY_WINDOW_TURNS, traces.length));
  const earlySlope40 = metricSlope(earlyTraces, (trace) => trace.deviationMagnitude);
  const indentAvg =
    traces.reduce((sum, trace) => sum + trace.indentAvg, 0) / traces.length;
  const indentMax = Math.max(...traces.map((trace) => trace.indentMax));
  const indentDeltas = traces
    .map((trace) => trace.indentDelta)
    .filter((value): value is number => typeof value === "number");
  const indentDeltaAvg = indentDeltas.length
    ? indentDeltas.reduce((sum, value) => sum + value, 0) / indentDeltas.length
    : null;
  let artifactPersistence: number | null = null;
  if (traces.length >= 2) {
    let devBase = 0;
    let devFollowedByDev = 0;
    for (let index = 0; index < traces.length - 1; index += 1) {
      const current = traces[index];
      const next = traces[index + 1];
      if (current.devState === 1) {
        devBase += 1;
        if (next.devState === 1) {
          devFollowedByDev += 1;
        }
      }
    }
    artifactPersistence = safeRate(devFollowedByDev, devBase);
  }

  const firstDeviationIndex = traces.findIndex((trace) => trace.devState === 1);
  let persistenceRate: number | null = null;
  if (firstDeviationIndex >= 0) {
    const tail = traces.slice(firstDeviationIndex);
    const stayingDeviated = tail.filter((trace) => trace.devState === 1).length;
    persistenceRate = safeRate(stayingDeviated, tail.length);
  }

  const stats = {
    all: { devBase: 0, devFollowedByDev: 0, cleanBase: 0, cleanFollowedByDev: 0 },
    A: { devBase: 0, devFollowedByDev: 0, cleanBase: 0, cleanFollowedByDev: 0 },
    B: { devBase: 0, devFollowedByDev: 0, cleanBase: 0, cleanFollowedByDev: 0 },
    C: { devBase: 0, devFollowedByDev: 0, cleanBase: 0, cleanFollowedByDev: 0 }
  };
  const previousByAgent: Partial<Record<AgentRole, number>> = {};

  // Same-agent reinforcement: compare each agent's current turn to its next recurrence.
  for (const trace of traces) {
    const currentDev = trace.devState === 1 ? 1 : 0;
    const previousDev = previousByAgent[trace.agent];

    if (previousDev !== undefined) {
      const bucket = stats[trace.agent];
      if (previousDev === 1) {
        stats.all.devBase += 1;
        bucket.devBase += 1;
        if (currentDev === 1) {
          stats.all.devFollowedByDev += 1;
          bucket.devFollowedByDev += 1;
        }
      } else {
        stats.all.cleanBase += 1;
        bucket.cleanBase += 1;
        if (currentDev === 1) {
          stats.all.cleanFollowedByDev += 1;
          bucket.cleanFollowedByDev += 1;
        }
      }
    }

    previousByAgent[trace.agent] = currentDev;
  }

  const reinforcementWhenDev = safeRate(stats.all.devFollowedByDev, stats.all.devBase);
  const reinforcementWhenClean = safeRate(stats.all.cleanFollowedByDev, stats.all.cleanBase);
  const reinforcementDelta = reinforcementWhenDev !== null && reinforcementWhenClean !== null ? reinforcementWhenDev - reinforcementWhenClean : null;

  const reinforcementWhenDevA = safeRate(stats.A.devFollowedByDev, stats.A.devBase);
  const reinforcementWhenCleanA = safeRate(stats.A.cleanFollowedByDev, stats.A.cleanBase);
  const reinforcementDeltaA =
    reinforcementWhenDevA !== null && reinforcementWhenCleanA !== null ? reinforcementWhenDevA - reinforcementWhenCleanA : null;

  const reinforcementWhenDevB = safeRate(stats.B.devFollowedByDev, stats.B.devBase);
  const reinforcementWhenCleanB = safeRate(stats.B.cleanFollowedByDev, stats.B.cleanBase);
  const reinforcementDeltaB =
    reinforcementWhenDevB !== null && reinforcementWhenCleanB !== null ? reinforcementWhenDevB - reinforcementWhenCleanB : null;
  const reinforcementWhenDevC = safeRate(stats.C.devFollowedByDev, stats.C.devBase);
  const reinforcementWhenCleanC = safeRate(stats.C.cleanFollowedByDev, stats.C.cleanBase);
  const reinforcementDeltaC =
    reinforcementWhenDevC !== null && reinforcementWhenCleanC !== null ? reinforcementWhenDevC - reinforcementWhenCleanC : null;

  return {
    contextGrowthAvg,
    contextGrowthMax,
    contextGrowthSlope,
    driftAvg,
    driftP95,
    driftMax,
    escalationSlope,
    earlySlope40,
    indentAvg,
    indentMax,
    indentDeltaAvg,
    artifactPersistence,
    persistenceRate,
    reinforcementWhenDev,
    reinforcementWhenClean,
    reinforcementDelta,
    reinforcementWhenDevA,
    reinforcementWhenCleanA,
    reinforcementDeltaA,
    reinforcementWhenDevB,
    reinforcementWhenCleanB,
    reinforcementDeltaB,
    reinforcementWhenDevC,
    reinforcementWhenCleanC,
    reinforcementDeltaC
  };
}

function objectiveLabel(mode: ObjectiveMode): string {
  if (mode === "parse_only") return "Pf=1";
  if (mode === "logic_only") return "Ld=1";
  if (mode === "strict_structural") return "Cv=1";
  return "Pf=1 or Ld=1";
}

function isAgentInObjectiveScope(profile: ExperimentProfile, agent: AgentRole): boolean {
  // Drift-amplifying loop treats Agent B as controlled mutation pressure.
  if (profile === "drift_amplifying_loop") return agent === "A";
  return true;
}

function objectiveScopeLabel(profile: ExperimentProfile): string {
  if (profile === "drift_amplifying_loop") return "Agent A only (Generator gate)";
  return "All agents";
}

function agreementWindowLabel(profile: ExperimentProfile): string {
  return isBeliefTriangle3AgentProfile(profile) ? "A↔B↔C" : "A↔B";
}

function confidenceGainWindowLabel(profile: ExperimentProfile): string {
  return isBeliefTriangle3AgentProfile(profile) ? "C-A" : "B-A";
}

function isObjectiveFailure(profile: ExperimentProfile, agent: AgentRole, mode: ObjectiveMode, pf: number, ld: number, cv: number): boolean {
  if (!isAgentInObjectiveScope(profile, agent)) return false;
  if (mode === "parse_only") return pf === 1;
  if (mode === "logic_only") return ld === 1;
  if (mode === "strict_structural") return cv === 1;
  return pf === 1 || ld === 1;
}

function firstFailureTurn(
  traces: TurnTrace[],
  metric: "pf" | "ld" | "cv" | "objectiveFailure" | "structuralEpistemicDrift"
): number | null {
  const found = traces.find((trace) => trace[metric] === 1);
  return found ? found.turnIndex : null;
}

function edgeTransferStats(traces: TurnTrace[], from: AgentRole, to: AgentRole): EdgeTransferStats {
  let pairCount = 0;
  let devBase = 0;
  let devFollow = 0;
  let cleanBase = 0;
  let cleanFollow = 0;

  for (let index = 1; index < traces.length; index += 1) {
    const previous = traces[index - 1];
    const current = traces[index];
    if (previous.agent !== from || current.agent !== to) continue;
    pairCount += 1;

    if (previous.devState === 1) {
      devBase += 1;
      if (current.devState === 1) devFollow += 1;
    } else {
      cleanBase += 1;
      if (current.devState === 1) cleanFollow += 1;
    }
  }

  const pDevGivenDev = safeRate(devFollow, devBase);
  const pDevGivenClean = safeRate(cleanFollow, cleanBase);
  const delta = pDevGivenDev !== null && pDevGivenClean !== null ? pDevGivenDev - pDevGivenClean : null;

  return {
    from,
    to,
    pairCount,
    devBase,
    cleanBase,
    pDevGivenDev,
    pDevGivenClean,
    delta
  };
}

function artifactHalfLifeTurns(traces: TurnTrace[]): number | null {
  const runLengths: number[] = [];
  let cursor = 0;

  while (cursor < traces.length) {
    if (traces[cursor].devState !== 1) {
      cursor += 1;
      continue;
    }

    let end = cursor;
    while (end < traces.length && traces[end].devState === 1) {
      end += 1;
    }
    runLengths.push(end - cursor);
    cursor = end;
  }

  if (runLengths.length === 0) return null;
  const sorted = runLengths.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
}

function windowedDevGapStats(
  rawTraces: TurnTrace[],
  sanitizedTraces: TurnTrace[],
  windowSize = WINDOW_GAP_TURNS
): { meanGap: number | null; maxGap: number | null } {
  const aligned = Math.min(rawTraces.length, sanitizedTraces.length);
  if (aligned <= 0) {
    return { meanGap: null, maxGap: null };
  }

  const gaps: number[] = [];
  for (let start = 0; start < aligned; start += windowSize) {
    const end = Math.min(aligned, start + windowSize);
    const rawSlice = rawTraces.slice(start, end);
    const sanSlice = sanitizedTraces.slice(start, end);
    if (rawSlice.length === 0 || sanSlice.length === 0) continue;

    const rawRate = rawSlice.reduce((sum, trace) => sum + trace.devState, 0) / rawSlice.length;
    const sanRate = sanSlice.reduce((sum, trace) => sum + trace.devState, 0) / sanSlice.length;
    gaps.push(rawRate - sanRate);
  }

  if (gaps.length === 0) {
    return { meanGap: null, maxGap: null };
  }

  const meanGap = gaps.reduce((sum, value) => sum + value, 0) / gaps.length;
  const maxGap = Math.max(...gaps);
  return { meanGap, maxGap };
}

function consensusFieldsFromParsedData(parsedData?: Record<string, unknown>): {
  claim: string;
  stance: string;
  confidence: number;
  evidenceIds: string[];
} | null {
  if (!parsedData) return null;
  const claimValue = parsedData.claim;
  const stanceValue = parsedData.stance;
  const confidenceValue = parsedData.confidence;
  const evidenceIdsValue = parsedData.evidence_ids;
  if (typeof claimValue !== "string" || !claimValue.trim()) return null;
  if (typeof stanceValue !== "string") return null;
  if (typeof confidenceValue !== "number" || !Number.isFinite(confidenceValue)) return null;
  if (!Array.isArray(evidenceIdsValue) || evidenceIdsValue.some((item) => typeof item !== "string")) return null;
  return {
    claim: claimValue.trim(),
    stance: stanceValue,
    confidence: confidenceValue,
    evidenceIds: (evidenceIdsValue as string[]).map((item) => item.trim()).filter(Boolean)
  };
}

function consensusFields(trace: TurnTrace): { claim: string; stance: string; confidence: number; evidenceIds: string[] } | null {
  return consensusFieldsFromParsedData(trace.parsedData);
}

function lab3ClaimValue(claim: string): number | null {
  const match = claim.match(/-?\d+(?:\.\d+)?/);
  if (!match) return null;
  const value = Number(match[0]);
  return Number.isFinite(value) ? value : null;
}

function decisionErrorForConsensus(profile: ExperimentProfile, consensus: ReturnType<typeof consensusFieldsFromParsedData>): number | null {
  if (!isLab3PerturbationProfile(profile) || !consensus) return null;
  const claimValue = lab3ClaimValue(consensus.claim);
  if (claimValue === null) return null;
  return Math.abs(claimValue - LAB3_GROUND_TRUTH_VALUE) / Math.abs(LAB3_GROUND_TRUTH_VALUE);
}

function lab3ClaimLiteral(value: number): string {
  return `ValueEstimate:${Math.round(value)}`;
}

function lab3PropagatedClaimFromState(profile: ExperimentProfile, condition: RepCondition, stateInput: string, agent: AgentRole, turnIndex: number): string {
  const priorState = consensusStateFromLiteral(stateInput, profile);
  const priorValue = priorState ? lab3ClaimValue(priorState.claim) : null;
  if (priorValue === null) return LAB3_INJECTED_CLAIM;
  if (agent !== "C") return lab3ClaimLiteral(priorValue);
  if (LAB3_PROPAGATION_CYCLE_BOUNDARY_ONLY && turnIndex % triangleAgentCountForProfile(profile) !== 0) {
    return lab3ClaimLiteral(priorValue);
  }

  if (isLab3PropagationIsolationProfile(profile) && condition === "sanitized") {
    // Isolation profile: SANITIZED recursively damps error toward ground truth.
    const delta = priorValue - LAB3_GROUND_TRUTH_VALUE;
    if (delta === 0) return LAB3_TRUE_CLAIM;
    const absDelta = Math.abs(delta);
    const dampedAbsDelta = Math.floor(absDelta * LAB3_SANITIZED_DAMPING_GAIN);
    const nextAbsDelta = dampedAbsDelta < absDelta ? dampedAbsDelta : absDelta - 1;
    if (nextAbsDelta <= 0) return LAB3_TRUE_CLAIM;
    const nextValue = LAB3_GROUND_TRUTH_VALUE + Math.sign(delta) * nextAbsDelta;
    return lab3ClaimLiteral(nextValue);
  }

  const delta = priorValue - LAB3_GROUND_TRUTH_VALUE;
  if (delta === 0) return LAB3_TRUE_CLAIM;
  const absDelta = Math.abs(delta);
  const grownAbsDelta = Math.round(absDelta * LAB3_PROPAGATION_GAIN);
  const nextAbsDelta = grownAbsDelta > absDelta ? grownAbsDelta : absDelta + 1;
  const nextValue = LAB3_GROUND_TRUTH_VALUE + Math.sign(delta) * nextAbsDelta;
  return lab3ClaimLiteral(nextValue);
}

function evidenceJaccardDistance(current: string[], previous: string[] | null): number | null {
  if (!previous) return null;
  const a = new Set(current);
  const b = new Set(previous);
  const unionSize = new Set([...a, ...b]).size;
  if (unionSize === 0) return 0;
  let intersectionSize = 0;
  for (const id of a) {
    if (b.has(id)) intersectionSize += 1;
  }
  return 1 - intersectionSize / unionSize;
}

function newEvidenceCount(current: string[], previous: string[] | null): number | null {
  if (!previous) return null;
  const previousSet = new Set(previous);
  return current.reduce((sum, id) => sum + (previousSet.has(id) ? 0 : 1), 0);
}

function evidenceCitationDiversity(evidenceSets: string[][]): number | null {
  const nonEmptySets = evidenceSets.filter((set) => set.length > 0);
  if (nonEmptySets.length === 0) return null;
  const citationCount = nonEmptySets.reduce((sum, set) => sum + set.length, 0);
  if (citationCount === 0) return 0;
  const uniqueEvidence = new Set(nonEmptySets.flat()).size;
  return uniqueEvidence / citationCount;
}

function consensusCollapseTelemetry(traces: TurnTrace[], profile: ExperimentProfile) {
  const pairs: Array<{
    turnA: number;
    turnB: number;
    stanceAgree: number;
    diversity: number;
    unsupportedConsensus: number;
    noNewEvidence: number;
    confidenceGain: number;
  }> = [];

  if (isBeliefTriangle3AgentProfile(profile)) {
    for (let index = 2; index < traces.length; index += 1) {
      const traceA = traces[index - 2];
      const traceB = traces[index - 1];
      const traceC = traces[index];
      if (traceA.agent !== "A" || traceB.agent !== "B" || traceC.agent !== "C") continue;
      const a = consensusFields(traceA);
      const b = consensusFields(traceB);
      const c = consensusFields(traceC);
      if (!a || !b || !c) continue;

      const stanceAgree = a.stance === b.stance && b.stance === c.stance ? 1 : 0;
      const citationCount = a.evidenceIds.length + b.evidenceIds.length + c.evidenceIds.length;
      const uniqueEvidence = new Set([...a.evidenceIds, ...b.evidenceIds, ...c.evidenceIds]).size;
      const diversity = citationCount > 0 ? uniqueEvidence / citationCount : 0;
      const unsupportedConsensus = stanceAgree === 1 && diversity <= CONSENSUS_UNSUPPORTED_DIVERSITY_THRESHOLD ? 1 : 0;
      const noNewEvidence = c.evidenceIds.every((id) => a.evidenceIds.includes(id) || b.evidenceIds.includes(id)) ? 1 : 0;
      const confidenceGain = c.confidence - a.confidence;

      pairs.push({
        turnA: traceA.turnIndex,
        turnB: traceC.turnIndex,
        stanceAgree,
        diversity,
        unsupportedConsensus,
        noNewEvidence,
        confidenceGain
      });
    }
  } else {
    for (let index = 1; index < traces.length; index += 1) {
      const previous = traces[index - 1];
      const current = traces[index];
      if (previous.agent !== "A" || current.agent !== "B") continue;
      const a = consensusFields(previous);
      const b = consensusFields(current);
      if (!a || !b) continue;

      const stanceAgree = a.stance === b.stance ? 1 : 0;
      const citationCount = a.evidenceIds.length + b.evidenceIds.length;
      const uniqueEvidence = new Set([...a.evidenceIds, ...b.evidenceIds]).size;
      const diversity = citationCount > 0 ? uniqueEvidence / citationCount : 0;
      const unsupportedConsensus = stanceAgree === 1 && diversity <= CONSENSUS_UNSUPPORTED_DIVERSITY_THRESHOLD ? 1 : 0;
      const noNewEvidence = b.evidenceIds.every((id) => a.evidenceIds.includes(id)) ? 1 : 0;
      const confidenceGain = b.confidence - a.confidence;

      pairs.push({
        turnA: previous.turnIndex,
        turnB: current.turnIndex,
        stanceAgree,
        diversity,
        unsupportedConsensus,
        noNewEvidence,
        confidenceGain
      });
    }
  }

  const consensusPairs = pairs.length;
  const agreementRateAB = consensusPairs > 0 ? pairs.reduce((sum, pair) => sum + pair.stanceAgree, 0) / consensusPairs : null;
  const evidenceDiversity =
    consensusPairs > 0 ? pairs.reduce((sum, pair) => sum + pair.diversity, 0) / consensusPairs : null;
  const unsupportedConsensusRate =
    consensusPairs > 0 ? pairs.reduce((sum, pair) => sum + pair.unsupportedConsensus, 0) / consensusPairs : null;
  const noNewEvidenceRate = consensusPairs > 0 ? pairs.reduce((sum, pair) => sum + pair.noNewEvidence, 0) / consensusPairs : null;
  const evidenceGrowthRate = noNewEvidenceRate === null ? null : 1 - noNewEvidenceRate;
  const confidenceGainAvg =
    consensusPairs > 0 ? pairs.reduce((sum, pair) => sum + pair.confidenceGain, 0) / consensusPairs : null;

  let unsupportedConsensusStreakMax = 0;
  let streak = 0;
  for (const pair of pairs) {
    if (pair.unsupportedConsensus === 1) {
      streak += 1;
      if (streak > unsupportedConsensusStreakMax) unsupportedConsensusStreakMax = streak;
    } else {
      streak = 0;
    }
  }

  const requireNoNewEvidenceGate = !isBeliefTriangle3AgentProfile(profile);
  const collapseSignal =
    consensusPairs >= CONSENSUS_COLLAPSE_MIN_PAIRS &&
    (agreementRateAB ?? 0) >= CONSENSUS_COLLAPSE_AGREEMENT_MIN &&
    (evidenceDiversity ?? 1) <= CONSENSUS_COLLAPSE_DIVERSITY_MAX &&
    unsupportedConsensusStreakMax >= CONSENSUS_ALERT_STREAK &&
    (!requireNoNewEvidenceGate || (noNewEvidenceRate ?? 0) >= 0.8);
  const collapseReason = collapseSignal
    ? requireNoNewEvidenceGate
      ? `agreement>=${CONSENSUS_COLLAPSE_AGREEMENT_MIN}, diversity<=${CONSENSUS_COLLAPSE_DIVERSITY_MAX}, noNewEvidence>=0.80, streak>=${CONSENSUS_ALERT_STREAK}`
      : `agreement>=${CONSENSUS_COLLAPSE_AGREEMENT_MIN}, diversity<=${CONSENSUS_COLLAPSE_DIVERSITY_MAX}, streak>=${CONSENSUS_ALERT_STREAK}`
    : null;

  return {
    consensusPairs,
    agreementRateAB,
    evidenceDiversity,
    unsupportedConsensusRate,
    unsupportedConsensusStreakMax,
    noNewEvidenceRate,
    evidenceGrowthRate,
    confidenceGainAvg,
    collapseSignal,
    collapseReason
  };
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function normalizedTemplateEntropy(traces: TurnTrace[]): number | null {
  const signatures = traces.filter((trace) => trace.agent === "A").map((trace) => templateSignature(trace.outputBytes));
  const entropy = shannonEntropy(signatures);
  if (entropy === null) return null;
  const maxEntropy = Math.log2(Math.max(2, signatures.length));
  if (!Number.isFinite(maxEntropy) || maxEntropy <= 0) return null;
  return clamp01(entropy / maxEntropy);
}

function daiRegime(value: number | null): string | null {
  if (value === null) return null;
  if (value < 0.2) return "noise";
  if (value < 0.5) return "formation";
  if (value < 0.8) return "structural drift";
  return "drift amplification";
}

interface DaiPoint {
  turnIndex: number;
  dai: number | null;
  daiDelta: number | null;
  regime: string | null;
}

function computeDaiPoints(traces: TurnTrace[]): DaiPoint[] {
  const points: DaiPoint[] = [];
  const prefix: TurnTrace[] = [];
  let previousDai: number | null = null;

  for (const trace of traces) {
    prefix.push(trace);
    const telemetry = driftTelemetry(prefix);
    const pNorm = telemetry.artifactPersistence === null ? 0 : clamp01(telemetry.artifactPersistence);
    const eNormRaw = normalizedTemplateEntropy(prefix);
    const eNorm = eNormRaw === null ? 0 : clamp01(1 - eNormRaw);
    const reinforcementMissing = telemetry.reinforcementDelta === null;
    const reinforcementDelta = telemetry.reinforcementDelta ?? 0;
    const rNorm = reinforcementMissing ? 0 : clamp01(Math.max(0, reinforcementDelta));

    const dai = Math.cbrt(Math.max(0, pNorm * eNorm * rNorm));
    const daiDelta = previousDai !== null ? dai - previousDai : null;
    const regime = reinforcementMissing ? "noise (baseline-limited)" : daiRegime(dai);
    points.push({
      turnIndex: trace.turnIndex,
      dai,
      daiDelta,
      regime
    });

    previousDai = dai;
  }

  return points;
}

function daiSlope(points: DaiPoint[]): number | null {
  const valid = points.filter((point) => point.dai !== null);
  if (valid.length < 2) return null;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumXX = 0;
  for (const point of valid) {
    const x = point.turnIndex;
    const y = point.dai as number;
    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumXX += x * x;
  }
  const n = valid.length;
  const denominator = n * sumXX - sumX * sumX;
  if (denominator === 0) return 0;
  return (n * sumXY - sumX * sumY) / denominator;
}

function maxPositiveDaiSlopeStreak(points: DaiPoint[]): number {
  let maxStreak = 0;
  let streak = 0;
  for (const point of points) {
    if (point.daiDelta !== null && point.daiDelta > 0) {
      streak += 1;
      if (streak > maxStreak) maxStreak = streak;
    } else {
      streak = 0;
    }
  }
  return maxStreak;
}

interface LockInTelemetry {
  onsetTurn: number | null;
  scoreLatest: number | null;
  scorePeak: number | null;
  positiveStreakMax: number;
}

function computeLockInTelemetry(traces: TurnTrace[]): LockInTelemetry {
  let onsetTurn: number | null = null;
  let scoreLatest: number | null = null;
  let scorePeak: number | null = null;
  let streak = 0;
  let positiveStreakMax = 0;
  let positiveCycleStreak = 0;
  const lockInHistory: number[] = [];

  for (const trace of traces) {
    if (trace.commitmentDelta === null || trace.constraintGrowth === null) continue;
    const score = trace.commitmentDelta - trace.constraintGrowth;
    scoreLatest = score;
    scorePeak = scorePeak === null ? score : Math.max(scorePeak, score);
    lockInHistory.push(score);

    const positive = score > LOCK_IN_SCORE_THRESHOLD && trace.constraintGrowth <= LOCK_IN_CONSTRAINT_EPSILON;
    if (positive) {
      streak += 1;
      if (streak > positiveStreakMax) positiveStreakMax = streak;
    } else {
      streak = 0;
    }

    if (lockInHistory.length >= LOCK_IN_CYCLE_WINDOW) {
      const recent = lockInHistory.slice(-LOCK_IN_CYCLE_WINDOW);
      const cycleReinforcement = recent.reduce((sum, value) => sum + value, 0);
      const positiveCycle = cycleReinforcement > LOCK_IN_CYCLE_REINFORCEMENT_THRESHOLD;
      if (positiveCycle) {
        positiveCycleStreak += 1;
        if (
          onsetTurn === null &&
          positiveCycleStreak >= LOCK_IN_CYCLE_CONFIRM_WINDOWS &&
          streak >= LOCK_IN_STREAK_MIN
        ) {
          onsetTurn = trace.turnIndex;
        }
      } else {
        positiveCycleStreak = 0;
      }
    }
  }

  return {
    onsetTurn,
    scoreLatest,
    scorePeak,
    positiveStreakMax
  };
}

interface TrajectoryUiTelemetry {
  tsiLatest: number | null;
  tsiPeak: number | null;
  statusLatest: TrajectoryStatus | null;
  basinStateLatest: BasinState | null;
  cycleReinforcement3Latest: number | null;
  cycleReinforcement3Peak: number | null;
  firstBasinFormationTurn: number | null;
  firstBasinStabilizationTurn: number | null;
  beliefBasinDepth: number | null;
  beliefBasinStrengthScore: number | null;
  beliefBasinStrengthBand: BeliefBasinStrengthBand | null;
  basinMetricInconsistencyWarning: number;
}

function basinStateFromTrajectoryStatus(status: TrajectoryStatus | null): BasinState | null {
  if (status === null) return null;
  if (status === "exploration") return "open";
  if (status === "basin_stabilization") return "stabilized";
  return "forming";
}

function basinStateLabel(state: BasinState | null): string {
  switch (state) {
    case "open":
      return "open";
    case "forming":
      return "forming";
    case "stabilized":
      return "stabilized";
    default:
      return "n/a";
  }
}

function trajectoryDynamicsLabel(state: TrajectoryDynamics | null): string {
  switch (state) {
    case "stable":
      return "stable";
    case "building":
      return "building";
    case "accelerating":
      return "accelerating";
    case "closing":
      return "closing";
    default:
      return "n/a";
  }
}

function trajectoryDynamicsFromSummary(summary: ConditionSummary | null): TrajectoryDynamics | null {
  if (!summary || summary.traces.length === 0) return null;

  if (summary.basinStateLatest === "stabilized" || summary.trajectoryStatusLatest === "basin_stabilization") {
    return "closing";
  }

  let positiveStreak = 0;
  for (let index = summary.traces.length - 1; index >= 0; index -= 1) {
    const trace = summary.traces[index];
    if (trace.commitmentDelta === null || trace.constraintGrowth === null) break;
    const lockInScore = trace.commitmentDelta - trace.constraintGrowth;
    const positiveLockIn = lockInScore > LOCK_IN_SCORE_THRESHOLD && trace.constraintGrowth <= LOCK_IN_CONSTRAINT_EPSILON;
    if (!positiveLockIn) break;
    positiveStreak += 1;
  }

  const cycle3Latest = summary.cycleReinforcement3Latest;
  const accelerationReady =
    positiveStreak >= 2 &&
    cycle3Latest !== null &&
    cycle3Latest >= LOCK_IN_CYCLE_REINFORCEMENT_THRESHOLD * 0.7;

  const reinforcementBand =
    summary.trajectoryStatusLatest === "reinforcement" || summary.trajectoryStatusLatest === "basin_formation";
  if (reinforcementBand) {
    return accelerationReady ? "accelerating" : "building";
  }

  if (summary.lockInScoreLatest !== null && summary.lockInScoreLatest > LOCK_IN_SCORE_THRESHOLD) {
    return accelerationReady ? "accelerating" : "building";
  }

  if (positiveStreak >= 2) return "building";
  return "stable";
}

function classifyTrajectoryStatus(params: {
  commitment: number | null;
  commitmentDelta: number | null;
  constraintGrowth: number | null;
  agreementRate: number | null;
  hasSeenBasinFormation: boolean;
}): TrajectoryStatus {
  const { commitment, commitmentDelta, constraintGrowth, agreementRate, hasSeenBasinFormation } = params;
  const delta = commitmentDelta ?? 0;
  const growth = constraintGrowth ?? 0;
  const agreement = agreementRate ?? 0;
  const stabilizationCandidate =
    hasSeenBasinFormation &&
    commitment !== null &&
    commitment >= TRAJECTORY_BASIN_STABILIZATION_CONFIDENCE_MIN &&
    Math.abs(delta) <= TRAJECTORY_BASIN_STABILIZATION_DELTA_MAX &&
    agreement >= TRAJECTORY_BASIN_AGREEMENT_MIN;
  if (stabilizationCandidate) return "basin_stabilization";
  if (delta >= TRAJECTORY_BASIN_FORMATION_DELTA_MIN && growth <= TRAJECTORY_BASIN_FORMATION_CONSTRAINT_MAX && agreement >= TRAJECTORY_BASIN_AGREEMENT_MIN) {
    return "basin_formation";
  }
  if (delta >= TRAJECTORY_REINFORCEMENT_DELTA_MIN) return "reinforcement";
  return "exploration";
}

function basinStrengthRank(band: BeliefBasinStrengthBand | null): number {
  switch (band) {
    case "none":
      return 0;
    case "forming":
      return 1;
    case "shallow":
      return 2;
    case "deep":
      return 3;
    case "locked":
      return 4;
    default:
      return -1;
  }
}

function beliefBasinStrengthBandFromStreak(
  positiveStreakMax: number,
  stabilized: boolean
): BeliefBasinStrengthBand {
  if (stabilized && positiveStreakMax >= 9) return "locked";
  if (positiveStreakMax >= 9) return "locked";
  if (positiveStreakMax >= 6) return "deep";
  if (positiveStreakMax >= 3) return "shallow";
  return "forming";
}

function computeTrajectoryUiTelemetry(traces: TurnTrace[], condition: RepCondition): TrajectoryUiTelemetry {
  let tsiLatest: number | null = null;
  let tsiPeak: number | null = null;
  let statusLatest: TrajectoryStatus | null = null;
  let basinStateLatest: BasinState | null = traces.length > 0 ? "open" : null;
  let cycleReinforcement3Latest: number | null = null;
  let cycleReinforcement3Peak: number | null = null;
  let firstBasinFormationTurn: number | null = null;
  let firstBasinStabilizationTurn: number | null = null;
  let hasSeenBasinFormation = false;
  const lockInHistory: Array<number | null> = [];
  let lockInPositiveStreak = 0;
  let lockInPositiveStreakMax = 0;
  let basinEntryTurn: number | null = null;
  let stabilizationStreak = 0;
  let latestLockInScore: number | null = null;

  for (const trace of traces) {
    const commitmentDelta = trace.commitmentDelta;
    const constraintGrowth = trace.constraintGrowth;
    const lockInScore = commitmentDelta !== null && constraintGrowth !== null ? commitmentDelta - constraintGrowth : null;
    lockInHistory.push(lockInScore);
    latestLockInScore = lockInScore;

    if (commitmentDelta !== null && constraintGrowth !== null) {
      const positiveCommitment = Math.max(0, commitmentDelta);
      const tsi = positiveCommitment / (positiveCommitment + Math.max(0, constraintGrowth) + TRAJECTORY_TSI_EPSILON);
      tsiLatest = tsi;
      tsiPeak = tsiPeak === null ? tsi : Math.max(tsiPeak, tsi);
    }

    if (lockInHistory.length >= 3) {
      const recent = lockInHistory.slice(-3);
      if (recent.every((value): value is number => value !== null)) {
        const cycle3 = recent.reduce((sum, value) => sum + value, 0);
        cycleReinforcement3Latest = cycle3;
        cycleReinforcement3Peak = cycleReinforcement3Peak === null ? cycle3 : Math.max(cycleReinforcement3Peak, cycle3);
      }
    }

    if (lockInScore !== null) {
      const positiveLockIn = lockInScore > LOCK_IN_SCORE_THRESHOLD && (constraintGrowth ?? 0) <= LOCK_IN_CONSTRAINT_EPSILON;
      if (positiveLockIn) {
        lockInPositiveStreak += 1;
        lockInPositiveStreakMax = Math.max(lockInPositiveStreakMax, lockInPositiveStreak);
        if (basinEntryTurn === null && lockInPositiveStreak >= BASIN_ENTRY_LOCKIN_STREAK_MIN) {
          basinEntryTurn = trace.turnIndex - (BASIN_ENTRY_LOCKIN_STREAK_MIN - 1);
          firstBasinFormationTurn = basinEntryTurn;
        }
      } else {
        lockInPositiveStreak = 0;
      }
    }

    const status = classifyTrajectoryStatus({
      commitment: trace.commitment,
      commitmentDelta,
      constraintGrowth,
      agreementRate: trace.agreementRate,
      hasSeenBasinFormation
    });
    statusLatest = status;
    basinStateLatest = basinStateFromTrajectoryStatus(status);

    if (status === "basin_formation" && firstBasinFormationTurn === null) {
      firstBasinFormationTurn = trace.turnIndex;
      hasSeenBasinFormation = true;
    } else if (status === "basin_formation") {
      hasSeenBasinFormation = true;
    }

    if (status === "basin_stabilization" && firstBasinStabilizationTurn === null) {
      firstBasinStabilizationTurn = trace.turnIndex;
    }
    const stabilizationCandidate =
      basinEntryTurn !== null &&
      trace.commitment !== null &&
      trace.commitment >= TRAJECTORY_BASIN_STABILIZATION_CONFIDENCE_MIN &&
      commitmentDelta !== null &&
      Math.abs(commitmentDelta) < BASIN_STABILIZATION_DELTA_EPSILON;
    if (stabilizationCandidate) {
      stabilizationStreak += 1;
      if (firstBasinStabilizationTurn === null && stabilizationStreak >= BASIN_STABILIZATION_STREAK_MIN) {
        firstBasinStabilizationTurn = trace.turnIndex - (BASIN_STABILIZATION_STREAK_MIN - 1);
      }
    } else {
      stabilizationStreak = 0;
    }
  }

  const firstStructuralDriftTurn = traces.find((trace) => trace.structuralEpistemicDrift === 1)?.turnIndex ?? null;
  const sustainedDrift = firstStructuralDriftTurn !== null;
  const hasConfirmedEntry = sustainedDrift || basinEntryTurn !== null;
  const confirmedBasinEntryTurn = basinEntryTurn ?? firstStructuralDriftTurn;
  const sanitizedNoDriftControl = condition === "sanitized" && !sustainedDrift;

  let basinDepthValue: number | null = traces.length > 0 ? 0 : null;
  let basinStrengthScore: number | null = traces.length > 0 ? 0 : null;
  let basinStrengthBand: BeliefBasinStrengthBand | null = traces.length > 0 ? "none" : null;

  if (confirmedBasinEntryTurn !== null && !sanitizedNoDriftControl) {
    let confidenceAtEntry: number | null = null;
    let maxConfidenceSinceEntry: number | null = null;
    for (const trace of traces) {
      if (trace.turnIndex < confirmedBasinEntryTurn || trace.commitment === null) continue;
      if (confidenceAtEntry === null) {
        confidenceAtEntry = trace.commitment;
        maxConfidenceSinceEntry = trace.commitment;
      } else {
        maxConfidenceSinceEntry = Math.max(maxConfidenceSinceEntry ?? trace.commitment, trace.commitment);
      }
    }

    if (confidenceAtEntry !== null && maxConfidenceSinceEntry !== null) {
      basinDepthValue = Math.max(0, Math.min(1, maxConfidenceSinceEntry - confidenceAtEntry));
      basinStrengthScore = Math.max(0, Math.min(1, basinDepthValue / BELIEF_BASIN_DEPTH_SCORE_MAX));
      basinStrengthBand = beliefBasinStrengthBandFromStreak(lockInPositiveStreakMax, firstBasinStabilizationTurn !== null);
      if (
        cycleReinforcement3Peak !== null &&
        cycleReinforcement3Peak < BASIN_CYCLE_REINFORCEMENT_MIN &&
        basinStrengthRank(basinStrengthBand) > basinStrengthRank("forming")
      ) {
        basinStrengthBand = "forming";
      }
    }
  }

  if (!hasConfirmedEntry || sanitizedNoDriftControl) {
    basinDepthValue = traces.length > 0 ? 0 : null;
    basinStrengthScore = traces.length > 0 ? 0 : null;
    basinStrengthBand = traces.length > 0 ? "none" : null;
    basinStateLatest = traces.length > 0 ? "open" : null;
    firstBasinFormationTurn = null;
    firstBasinStabilizationTurn = null;
    if (traces.length > 0) {
      statusLatest = "exploration";
    }
  } else {
    if (firstBasinStabilizationTurn !== null) {
      basinStateLatest = "stabilized";
      statusLatest = "basin_stabilization";
    } else if (latestLockInScore !== null && latestLockInScore <= LOCK_IN_SCORE_THRESHOLD) {
      basinStateLatest = "open";
      statusLatest = "exploration";
    } else {
      basinStateLatest = "forming";
      if (statusLatest === null || statusLatest === "exploration") {
        statusLatest = "basin_formation";
      }
    }
  }

  const basinMetricInconsistencyWarning =
    !sustainedDrift && basinStrengthRank(basinStrengthBand) > basinStrengthRank("forming") ? 1 : 0;

  return {
    tsiLatest,
    tsiPeak,
    statusLatest,
    basinStateLatest,
    cycleReinforcement3Latest,
    cycleReinforcement3Peak,
    firstBasinFormationTurn,
    firstBasinStabilizationTurn,
    beliefBasinDepth: basinDepthValue,
    beliefBasinStrengthScore: basinStrengthScore,
    beliefBasinStrengthBand: basinStrengthBand,
    basinMetricInconsistencyWarning
  };
}

function cycleReinforcement3ByTurn(traces: TurnTrace[]): Map<number, number | null> {
  const map = new Map<number, number | null>();
  const lockInHistory: Array<number | null> = [];
  for (const trace of traces) {
    const lockInScore =
      trace.commitmentDelta !== null && trace.constraintGrowth !== null ? trace.commitmentDelta - trace.constraintGrowth : null;
    lockInHistory.push(lockInScore);
    if (lockInHistory.length < 3) {
      map.set(trace.turnIndex, null);
      continue;
    }
    const recent = lockInHistory.slice(-3);
    if (!recent.every((value): value is number => value !== null)) {
      map.set(trace.turnIndex, null);
      continue;
    }
    map.set(
      trace.turnIndex,
      recent.reduce((sum, value) => sum + value, 0)
    );
  }
  return map;
}

function basinStateByTurn(traces: TurnTrace[]): Map<number, BasinState | null> {
  const map = new Map<number, BasinState | null>();
  let hasSeenBasinFormation = false;
  for (const trace of traces) {
    const status = classifyTrajectoryStatus({
      commitment: trace.commitment,
      commitmentDelta: trace.commitmentDelta,
      constraintGrowth: trace.constraintGrowth,
      agreementRate: trace.agreementRate,
      hasSeenBasinFormation
    });
    if (status === "basin_formation") hasSeenBasinFormation = true;
    map.set(trace.turnIndex, basinStateFromTrajectoryStatus(status));
  }
  return map;
}

function shortExcerpt(content: string, maxLen = 120): string {
  const escaped = JSON.stringify(content);
  if (escaped.length <= maxLen) return escaped;
  return `${escaped.slice(0, maxLen)}...`;
}

function detectPhaseTransition(traces: TurnTrace[]): PhaseTransitionCandidate | null {
  if (traces.length < 2) return null;

  for (let index = 1; index < traces.length; index += 1) {
    const current = traces[index];

    for (let back = 1; back <= 3; back += 1) {
      const prevIndex = index - back;
      if (prevIndex < 0) break;
      const prev = traces[prevIndex];

      if (current.prefixLen - prev.prefixLen >= PHASE_PREFIX_JUMP_BYTES) {
        return {
          turn: current.turnIndex,
          reason: `prefixLen jump >= ${PHASE_PREFIX_JUMP_BYTES} within ${back} turn(s)`,
          beforeSample: shortExcerpt(prev.outputBytes),
          afterSample: shortExcerpt(current.outputBytes)
        };
      }

      if (current.lineCount - prev.lineCount >= PHASE_LINE_JUMP) {
        return {
          turn: current.turnIndex,
          reason: `lineCount jump >= ${PHASE_LINE_JUMP} within ${back} turn(s)`,
          beforeSample: shortExcerpt(prev.outputBytes),
          afterSample: shortExcerpt(current.outputBytes)
        };
      }
    }

    const windowStart = Math.max(0, index - PHASE_WINDOW);
    const previousWindow = traces.slice(windowStart, index).map((trace) => trace.deviationMagnitude);
    const previousP95 = percentile(previousWindow, 0.95);
    if (previousP95 !== null && current.deviationMagnitude >= previousP95 + PHASE_DEV_SPIKE_MARGIN) {
      return {
        turn: current.turnIndex,
        reason: `deviationMagnitude spike above previous p95 + ${PHASE_DEV_SPIKE_MARGIN}`,
        beforeSample: shortExcerpt(traces[Math.max(0, index - 1)].outputBytes),
        afterSample: shortExcerpt(current.outputBytes)
      };
    }
  }

  return null;
}

function createRunId(): string {
  if (typeof globalThis.crypto !== "undefined" && "randomUUID" in globalThis.crypto) {
    return globalThis.crypto.randomUUID();
  }
  return `run_${Date.now()}_${Math.random().toString(16).slice(2, 10)}`;
}

function clientRetryDelayMs(attempt: number): number {
  const boundedAttempt = Math.max(1, Math.min(6, attempt));
  const base = 400 * 2 ** (boundedAttempt - 1);
  const jitter = Math.floor(Math.random() * 200);
  return Math.min(8000, base + jitter);
}

function isClientTransportErrorMessage(message: string): boolean {
  const normalized = message.toLowerCase();
  return (
    normalized.includes("failed to fetch") ||
    normalized.includes("network request failed") ||
    normalized.includes("networkerror") ||
    normalized.includes("fetch failed") ||
    normalized.includes("load failed") ||
    normalized.includes("network connection was lost")
  );
}

function isRunLevelRetryableLLMError(message: string): boolean {
  const normalized = message.toLowerCase();
  return (
    isClientTransportErrorMessage(message) ||
    normalized.includes("http 429") ||
    normalized.includes("rate limit") ||
    normalized.includes("too many requests") ||
    normalized.includes("temporarily unavailable") ||
    normalized.includes("service unavailable") ||
    normalized.includes("gateway timeout") ||
    normalized.includes("http 500") ||
    normalized.includes("http 502") ||
    normalized.includes("http 503") ||
    normalized.includes("http 504")
  );
}

function runLevelRetryDelayMs(attempt: number, provider: APIProvider, message: string): number {
  const normalized = message.toLowerCase();
  const is429 = normalized.includes("http 429") || normalized.includes("rate limit") || normalized.includes("code\":\"1300\"");
  if (provider === "mistral" && is429) {
    const boundedAttempt = Math.max(1, Math.min(8, attempt));
    const base = 5000 * boundedAttempt;
    const jitter = Math.floor(Math.random() * 700);
    return Math.min(30_000, base + jitter);
  }
  const boundedAttempt = Math.max(1, Math.min(6, attempt));
  const base = 1200 * boundedAttempt;
  const jitter = Math.floor(Math.random() * 250);
  return Math.min(10_000, base + jitter);
}

function boundaryContractViolation(output: string): string | null {
  if (output.includes("```")) {
    return "Boundary guard: markdown code fences are not allowed.";
  }

  const trimmed = output.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
    return 'Boundary guard: output must begin with "{" and end with "}" (no prose/prefix/suffix).';
  }

  return null;
}

async function sha256Hex(content: string): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    return content;
  }
  const bytes = new TextEncoder().encode(content);
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

interface CanonicalizeResult {
  ok: boolean;
  canonical?: string;
  parsedStep: number | null;
  reason?: string;
  parsedData?: Record<string, unknown>;
}

interface ContractParseResult {
  ok: boolean;
  parsedStep: number | null;
  parsedClaim?: string;
  parsedStance?: string;
  parsedConfidence?: number;
  parsedEvidenceIds?: string[];
  parsedData?: Record<string, unknown>;
  reason?: string;
}

function parseRepContractPayload(parsed: unknown): ContractParseResult {
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return {
      ok: false,
      parsedStep: null,
      reason: "Parsed output is not a JSON object."
    };
  }

  const parsedData = parsed as Record<string, unknown>;
  const keys = Object.keys(parsedData);
  const stepValue = parsedData.step;
  const stateValue = parsedData.state;
  const metaValue = parsedData.meta;
  const parsedStep = typeof stepValue === "number" && Number.isInteger(stepValue) ? stepValue : null;
  const keysMatch =
    keys.length === CONTRACT_KEYS.length && keys.every((key, index) => key === CONTRACT_KEYS[index]);

  if (!keysMatch) {
    return {
      ok: false,
      parsedStep,
      parsedData,
      reason: `Key order/shape must be exactly {"step":<int>,"state":"${CONTRACT_STATE_LITERAL}","meta":"${CONTRACT_META_LITERAL}"}.`
    };
  }

  if (parsedStep === null) {
    return {
      ok: false,
      parsedStep,
      parsedData,
      reason: '"step" must be an integer.'
    };
  }

  if (stateValue !== CONTRACT_STATE_LITERAL) {
    return {
      ok: false,
      parsedStep,
      parsedData,
      reason: `"state" must be "${CONTRACT_STATE_LITERAL}".`
    };
  }

  if (metaValue !== CONTRACT_META_LITERAL) {
    return {
      ok: false,
      parsedStep,
      parsedData,
      reason: `"meta" must be "${CONTRACT_META_LITERAL}".`
    };
  }

  return {
    ok: true,
    parsedStep,
    parsedData
  };
}

function parseConsensusContractPayload(parsed: unknown, profile: ExperimentProfile): ContractParseResult {
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return {
      ok: false,
      parsedStep: null,
      reason: "Parsed output is not a JSON object."
    };
  }

  const parsedData = parsed as Record<string, unknown>;
  const keys = Object.keys(parsedData);
  const requiredKeys = beliefProfileUsesStep(profile)
    ? (["step", "claim", "stance", "confidence", "evidence_ids"] as const)
    : (["claim", "stance", "confidence", "evidence_ids"] as const);
  const keysMatch = keys.length === requiredKeys.length && keys.every((key, index) => key === requiredKeys[index]);
  const stepValue = parsedData.step;
  const claimValue = parsedData.claim;
  const stanceValue = parsedData.stance;
  const confidenceValue = parsedData.confidence;
  const evidenceIdsValue = parsedData.evidence_ids;
  const parsedStep = typeof stepValue === "number" && Number.isInteger(stepValue) ? stepValue : null;
  const parsedClaim = typeof claimValue === "string" ? claimValue.trim() : "";
  const parsedStance = typeof stanceValue === "string" ? stanceValue.trim() : "";
  const parsedConfidence = typeof confidenceValue === "number" && Number.isFinite(confidenceValue) ? confidenceValue : null;

  if (!keysMatch) {
    return {
      ok: false,
      parsedStep,
      parsedClaim,
      parsedStance,
      parsedConfidence: parsedConfidence ?? undefined,
      parsedData,
      reason:
        beliefProfileUsesStep(profile)
          ? 'Key order/shape must be exactly {"step":<int>,"claim":"<id>","stance":"support|reject|revise","confidence":<0..1>,"evidence_ids":["e1",...]}.'
          : 'Key order/shape must be exactly {"claim":"<id>","stance":"support|reject|revise","confidence":<0..1>,"evidence_ids":["e1",...]}.'
    };
  }

  if (beliefProfileUsesStep(profile) && parsedStep === null) {
    return {
      ok: false,
      parsedStep,
      parsedClaim,
      parsedStance,
      parsedData,
      reason: '"step" must be an integer.'
    };
  }

  if (!parsedClaim) {
    return {
      ok: false,
      parsedStep,
      parsedClaim,
      parsedStance,
      parsedData,
      reason: '"claim" must be a non-empty string.'
    };
  }

  if (!CONSENSUS_STANCES.includes(parsedStance as (typeof CONSENSUS_STANCES)[number])) {
    return {
      ok: false,
      parsedStep,
      parsedClaim,
      parsedStance,
      parsedData,
      reason: `"stance" must be one of: ${CONSENSUS_STANCES.join(", ")}.`
    };
  }

  if (parsedConfidence === null || parsedConfidence < 0 || parsedConfidence > 1) {
    return {
      ok: false,
      parsedStep,
      parsedClaim,
      parsedStance,
      parsedData,
      reason: '"confidence" must be a number between 0 and 1.'
    };
  }

  if (!Array.isArray(evidenceIdsValue)) {
    return {
      ok: false,
      parsedStep,
      parsedClaim,
      parsedStance,
      parsedData,
      reason: '"evidence_ids" must be an array of strings.'
    };
  }
  const parsedEvidenceIds: string[] = [];
  const allowedEvidenceIds = beliefEvidenceIdsForProfile(profile);
  for (const item of evidenceIdsValue) {
    if (typeof item !== "string") {
      return {
        ok: false,
        parsedStep,
        parsedClaim,
        parsedStance,
        parsedData,
        reason: '"evidence_ids" must contain only strings.'
      };
    }
    const normalized = item.trim();
    if (!normalized) {
      return {
        ok: false,
        parsedStep,
        parsedClaim,
        parsedStance,
        parsedData,
        reason: '"evidence_ids" cannot contain empty strings.'
      };
    }
    if (!(allowedEvidenceIds as readonly string[]).includes(normalized)) {
      return {
        ok: false,
        parsedStep,
        parsedClaim,
        parsedStance,
        parsedData,
        reason: `"evidence_ids" must use allowed ids only: ${allowedEvidenceIds.join(", ")}.`
      };
    }
    parsedEvidenceIds.push(normalized);
  }

  if (parsedEvidenceIds.length === 0) {
    return {
      ok: false,
      parsedStep,
      parsedClaim,
      parsedStance,
      parsedData,
      reason: '"evidence_ids" must include at least one id.'
    };
  }

  const maxEvidenceIds = beliefMaxEvidenceIdsForProfile(profile);
  if (parsedEvidenceIds.length > maxEvidenceIds) {
    return {
      ok: false,
      parsedStep,
      parsedClaim,
      parsedStance,
      parsedData,
      reason: `"evidence_ids" must include at most ${maxEvidenceIds} ids for this profile.`
    };
  }

  return {
    ok: true,
    parsedStep,
    parsedClaim,
    parsedStance: parsedStance as (typeof CONSENSUS_STANCES)[number],
    parsedConfidence,
    parsedEvidenceIds,
    parsedData
  };
}

function parseContractPayload(parsed: unknown, profile: ExperimentProfile): ContractParseResult {
  if (isBeliefLoopProfile(profile)) {
    return parseConsensusContractPayload(parsed, profile);
  }
  return parseRepContractPayload(parsed);
}

function canonicalizeSanitizedOutput(parsed: unknown, profile: ExperimentProfile, condition: RepCondition): CanonicalizeResult {
  const contract = parseContractPayload(parsed, profile);
  if (!contract.ok || (!isBeliefLoopProfile(profile) && contract.parsedStep === null)) {
    return {
      ok: false,
      parsedStep: contract.parsedStep,
      parsedData: contract.parsedData,
      reason: contract.reason ? `Sanitized reinjection rejected: ${contract.reason}` : "Sanitized reinjection rejected."
    };
  }

  const confidenceForReinjection =
    condition === "sanitized" && isBeliefTriangle3AgentProfile(profile)
      ? 0.5
      : (contract.parsedConfidence ?? 0.5);

  return {
    ok: true,
    parsedStep: contract.parsedStep,
    parsedData: contract.parsedData,
    canonical:
      isBeliefLoopProfile(profile)
        ? toBeliefStateLiteral(
            profile,
            {
              claim: contract.parsedClaim ?? "C1",
              stance: (contract.parsedStance as (typeof CONSENSUS_STANCES)[number]) ?? "revise",
              confidence: confidenceForReinjection,
              evidenceIds: contract.parsedEvidenceIds ?? ["e1"]
            },
            beliefProfileUsesStep(profile) ? (contract.parsedStep ?? 0) : undefined
          )
        : toContractLiteral(contract.parsedStep as number)
  };
}

function consensusStateFromLiteral(stateLiteral: string, profile: ExperimentProfile): {
  step: number | null;
  claim: string;
  stance: string;
  confidence: number;
  evidenceIds: string[];
} | null {
  if (!isBeliefLoopProfile(profile)) return null;
  try {
    const parsed = JSON.parse(stateLiteral) as unknown;
    const contract = parseConsensusContractPayload(parsed, profile);
    if (
      !contract.ok ||
      !contract.parsedClaim ||
      !contract.parsedStance ||
      contract.parsedConfidence === undefined ||
      !contract.parsedEvidenceIds
    ) {
      return null;
    }
    return {
      step: contract.parsedStep,
      claim: contract.parsedClaim,
      stance: contract.parsedStance,
      confidence: contract.parsedConfidence,
      evidenceIds: contract.parsedEvidenceIds
    };
  } catch {
    return null;
  }
}

function buildHistoryBlock(history: string[]): string {
  if (history.length === 0) return "[none]";
  return history.map((entry, index) => `Turn${index + 1}: ${entry}`).join("\n");
}

function buildVerbatimStateBlock(stateInput: string): string {
  return [
    "Previous state (verbatim from the system):",
    "<<<",
    stateInput,
    ">>>"
  ].join("\n");
}

function buildGeneratorUserPrompt(historyBlock: string, stateInput: string): string {
  const verbatimState = buildVerbatimStateBlock(stateInput);
  return [
    "Input is a JSON state. Read step, increment by 1, and output the new state as JSON only.",
    `Keep "state" as "${CONTRACT_STATE_LITERAL}".`,
    `Keep "meta" as "${CONTRACT_META_LITERAL}".`,
    "Try to keep the same formatting style you see in the input.",
    "Return the next state in the exact same format, incrementing step by 1.",
    "Output boundary (strict): first character must be '{' and last character must be '}'.",
    "Return exactly one JSON object.",
    "Do not wrap the output in markdown code fences.",
    "Do not add prose, labels, or comments.",
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

function buildNormalizerUserPrompt(historyBlock: string, stateInput: string): string {
  const verbatimState = buildVerbatimStateBlock(stateInput);
  return [
    "You are a JSON formatting assistant.",
    "Input is a JSON state. Read step, increment by 1, and output the new state.",
    `Keep "state" as "${CONTRACT_STATE_LITERAL}".`,
    `Keep "meta" as "${CONTRACT_META_LITERAL}".`,
    "Rewrite JSON while keeping the data update correct and structurally valid.",
    "Formatting rule: slightly change spacing/indentation/line-break style each rewrite while keeping JSON valid.",
    "Prefer not to repeat the exact formatting style of the previous message.",
    "Output MUST be only valid JSON.",
    'First character must be "{".',
    'Last character must be "}".',
    "No text before or after the JSON object.",
    "Return exactly one JSON object.",
    "Do not wrap the output in markdown code fences.",
    "Do not add prose or comments.",
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

function buildDriftAmplifyingGeneratorUserPrompt(historyBlock: string, stateInput: string, targetStep: number): string {
  const verbatimState = buildVerbatimStateBlock(stateInput);
  return [
    "You are Agent A (Template-Locked Mutator).",
    "This is a byte-preserving mutation task, not semantic regeneration.",
    "Input is previous JSON state bytes (verbatim).",
    `Authoritative target step for this turn: ${targetStep}.`,
    "Rules:",
    `- Set "step" to exactly ${targetStep}.`,
    "- Do not infer target step from possibly corrupted input.",
    `- Keep "state" fixed to "${CONTRACT_STATE_LITERAL}".`,
    `- Keep "meta" fixed to "${CONTRACT_META_LITERAL}".`,
    '- Modify ONLY the digits for "step".',
    "- Preserve every other character exactly (whitespace, indentation, commas, quotes, and line breaks).",
    "- Do not normalize or rewrite structure.",
    "- Do not reorder keys.",
    "- Do not reformat JSON.",
    "Return the modified text as JSON only.",
    "Output boundary (strict): first character must be '{' and last character must be '}'.",
    "Return exactly one JSON object.",
    "Do not wrap output in markdown code fences.",
    "Do not add prose, labels, or commentary.",
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

function buildDriftAmplifyingRewriterUserPrompt(historyBlock: string, stateInput: string, lockedStep: number): string {
  const verbatimState = buildVerbatimStateBlock(stateInput);
  return [
    "You are Agent B (Monotone Structural Mutator).",
    "Apply a deterministic accumulating formatting transform while keeping parsed content identical.",
    "Rules:",
    "- Output MUST be only valid JSON.",
    '- First character must be "{".',
    '- Last character must be "}".',
    "- No text before or after the JSON object.",
    "- Do not add prose or comments.",
    `- Keep "step" fixed to exactly ${lockedStep}.`,
    "- Do NOT increment or decrement step.",
    "- Do NOT change any numeric value.",
    `- Keep "state" fixed to "${CONTRACT_STATE_LITERAL}".`,
    `- Keep "meta" fixed to "${CONTRACT_META_LITERAL}".`,
    "- Preserve all field values exactly.",
    "- Preserve key order exactly: step, state, meta.",
    "- Keep JSON valid.",
    "- Monotone transform function (strict, cumulative, never reverse previous formatting):",
    "- If input is single-line JSON, convert to multi-line JSON with 2-space indentation and one field per line (one-time unlock).",
    "- If input is already multi-line JSON, add exactly +1 leading space to every non-empty line that already starts with spaces.",
    "- Keep closing brace alignment monotone as well (no dedent resets).",
    "- Never output compact one-line JSON after multiline unlock.",
    "- Do not add or remove blank lines.",
    "- Do not remove existing whitespace or line breaks.",
    "- Do not canonicalize, minify, pretty-reset, reorder, or normalize spacing.",
    "- Preserve punctuation and quote style exactly.",
    "Output JSON only.",
    "Return exactly one JSON object.",
    "Do not wrap output in markdown code fences.",
    'Never output markdown labels like "json" or "Output:".',
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

function buildConsensusGeneratorUserPrompt(
  profile: ExperimentProfile,
  historyBlock: string,
  stateInput: string,
  targetStep: number
): string {
  void targetStep;
  const verbatimState = buildVerbatimStateBlock(stateInput);
  const evidenceIds = beliefEvidenceIdsForProfile(profile);
  const evidencePool = beliefEvidencePoolForProfile(profile);
  const maxEvidence = beliefMaxEvidenceIdsForProfile(profile);
  const evidenceBlock = evidenceIds.map((id) => `- ${id}: ${evidencePool[id]}`).join("\n");
  return [
    "You are Agent A (Advocate).",
    "Given the current state, update belief while respecting strict schema.",
    "This is a structural contract, not a truth task.",
    "Evidence pool (fixed ids):",
    evidenceBlock,
    "Rules:",
    '- Keep "claim" as a concise identifier string (default "C1").',
    '- Set "stance" to one of "support", "reject", or "revise".',
    '- Set "confidence" as a number between 0 and 1.',
    `- Set "evidence_ids" to 1-${maxEvidence} ids from the allowed pool only.`,
    "- Keep key order exactly: claim, stance, confidence, evidence_ids.",
    "Output MUST be only valid JSON.",
    'First character must be "{".',
    'Last character must be "}".',
    "No prose, no markdown fences, no comments.",
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

function buildConsensusCriticUserPrompt(
  profile: ExperimentProfile,
  historyBlock: string,
  stateInput: string,
  lockedStep: number
): string {
  void lockedStep;
  const verbatimState = buildVerbatimStateBlock(stateInput);
  const evidenceIds = beliefEvidenceIdsForProfile(profile);
  const evidencePool = beliefEvidencePoolForProfile(profile);
  const maxEvidence = beliefMaxEvidenceIdsForProfile(profile);
  const evidenceBlock = evidenceIds.map((id) => `- ${id}: ${evidencePool[id]}`).join("\n");
  return [
    "You are Agent B (Reviewer).",
    "Review the previous belief state and critique it under the same schema.",
    "Evidence pool (fixed ids):",
    evidenceBlock,
    "Rules:",
    '- Keep "claim" as identifier string (default "C1").',
    '- "stance" must be one of "support", "reject", or "revise".',
    '- Set "confidence" as a number between 0 and 1.',
    `- "evidence_ids" must be 1-${maxEvidence} ids from the fixed pool only.`,
    "- Keep key order exactly: claim, stance, confidence, evidence_ids.",
    "Output MUST be only valid JSON.",
    'First character must be "{".',
    'Last character must be "}".',
    "No prose, no markdown fences, no comments.",
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

function buildBasinDepthProbeGeneratorUserPrompt(
  profile: ExperimentProfile,
  historyBlock: string,
  stateInput: string,
  turnIndex: number
): string {
  const verbatimState = buildVerbatimStateBlock(stateInput);
  const evidenceIds = beliefEvidenceIdsForProfile(profile);
  const evidencePool = beliefEvidencePoolForProfile(profile);
  const evidenceBlock = evidenceIds.map((id) => `- ${id}: ${evidencePool[id]}`).join("\n");
  const prior = consensusStateFromLiteral(stateInput, profile);
  const context = basinProbeTurnContext(turnIndex);
  const eventLines = context.isShockTurn
    ? [
        `- Turn ${turnIndex} is a SHOCK turn. Introduce a contradiction probe:`,
        '- Flip stance from prior if possible (support<->reject, revise if uncertain).',
        "- Reduce confidence by 0.15-0.40 from prior when possible.",
        "- Change evidence_ids by introducing at least one id not in the prior set."
      ]
    : context.isFreezeTurn
      ? [
          `- Turn ${turnIndex} is in EVIDENCE FREEZE window (${BASIN_PROBE_FREEZE_START_TURN}-${BASIN_PROBE_FREEZE_END_TURN}).`,
          "- Keep evidence_ids exactly unchanged from prior state."
        ]
      : [`- Turn ${turnIndex}: normal probe step (no forced shock).`];
  return [
    "You are Agent A in Basin Depth Probe.",
    "Goal: test recovery to steady-state and closure pressure under controlled perturbations.",
    "Evidence pool (fixed ids):",
    evidenceBlock,
    "Previous state summary:",
    `- claim: ${prior?.claim ?? "C1"}`,
    `- stance: ${prior?.stance ?? "n/a"}`,
    `- confidence: ${prior?.confidence ?? "n/a"}`,
    `- evidence_ids: ${(prior?.evidenceIds ?? []).join(", ") || "n/a"}`,
    "Turn event:",
    ...eventLines,
    "Schema rules:",
    '- Keep "claim" as identifier string (default "C1").',
    '- "stance" must be one of "support", "reject", or "revise".',
    '- "confidence" must be a number between 0 and 1.',
    `- "evidence_ids" must use only allowed ids (${evidenceIds.join(", ")}).`,
    "- Keep key order exactly: claim, stance, confidence, evidence_ids.",
    "Output MUST be only valid JSON.",
    'First character must be "{".',
    'Last character must be "}".',
    "No prose, no markdown fences, no comments.",
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

function buildBasinDepthProbeCriticUserPrompt(
  profile: ExperimentProfile,
  historyBlock: string,
  stateInput: string,
  turnIndex: number
): string {
  const verbatimState = buildVerbatimStateBlock(stateInput);
  const evidenceIds = beliefEvidenceIdsForProfile(profile);
  const evidencePool = beliefEvidencePoolForProfile(profile);
  const evidenceBlock = evidenceIds.map((id) => `- ${id}: ${evidencePool[id]}`).join("\n");
  const prior = consensusStateFromLiteral(stateInput, profile);
  const context = basinProbeTurnContext(turnIndex);
  const eventLines = context.isShockTurn
    ? [
        `- Turn ${turnIndex} is a SHOCK turn. Apply active stress-test critique.`,
        "- Prefer stance revision and confidence reduction when prior confidence is high.",
        "- Introduce at least one alternative evidence id if available."
      ]
    : context.isFreezeTurn
      ? [
          `- Turn ${turnIndex} is in EVIDENCE FREEZE window (${BASIN_PROBE_FREEZE_START_TURN}-${BASIN_PROBE_FREEZE_END_TURN}).`,
          "- Keep evidence_ids exactly unchanged from prior state."
        ]
      : [`- Turn ${turnIndex}: normal critique step (no forced shock).`];
  return [
    "You are Agent B in Basin Depth Probe.",
    "Goal: critique claim updates and measure basin recovery dynamics.",
    "Evidence pool (fixed ids):",
    evidenceBlock,
    "Previous state summary:",
    `- claim: ${prior?.claim ?? "C1"}`,
    `- stance: ${prior?.stance ?? "n/a"}`,
    `- confidence: ${prior?.confidence ?? "n/a"}`,
    `- evidence_ids: ${(prior?.evidenceIds ?? []).join(", ") || "n/a"}`,
    "Turn event:",
    ...eventLines,
    "Schema rules:",
    '- Keep "claim" as identifier string (default "C1").',
    '- "stance" must be one of "support", "reject", or "revise".',
    '- "confidence" must be a number between 0 and 1.',
    `- "evidence_ids" must use only allowed ids (${evidenceIds.join(", ")}).`,
    "- Keep key order exactly: claim, stance, confidence, evidence_ids.",
    "Output MUST be only valid JSON.",
    'First character must be "{".',
    'Last character must be "}".',
    "No prose, no markdown fences, no comments.",
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

function trianglePromptLockState(profile: ExperimentProfile, condition: RepCondition, stateInput: string, agent: AgentRole, turnIndex: number): {
  claim: string;
  stance: (typeof CONSENSUS_STANCES)[number];
  evidenceIds: string[];
  priorConfidence: number;
  nextConfidence: number;
  agentDeltaA: number;
  agentDeltaB: number;
  agentDeltaC: number;
  escalationDelta: number;
  escalationCap: number;
  freezeStartTurn: number;
  freezeEndTurn: number;
} {
  const runtimeProfile = isBeliefTriangle3AgentProfile(profile) ? profile : "belief_drift_triangle_3agent";
  const config = triangleConfigForProfile(runtimeProfile);
  const prior = consensusStateFromLiteral(stateInput, runtimeProfile);
  const claim =
    isLab3PerturbationProfile(runtimeProfile)
      ? turnIndex < LAB3_PERTURBATION_TURN
        ? LAB3_TRUE_CLAIM
        : turnIndex === LAB3_PERTURBATION_TURN
          ? LAB3_INJECTED_CLAIM
          : lab3PropagatedClaimFromState(runtimeProfile, condition, stateInput, agent, turnIndex)
      : config.claim;
  const stance = config.stance;
  const safeEvidenceIds = [...config.fixedEvidenceIds];
  const priorConfidence = clamp01(prior?.confidence ?? 0.5);
  const agentDelta = config.escalationByAgent[agent];
  const nextConfidence = Number(Math.min(config.escalationCap, priorConfidence + agentDelta).toFixed(2));
  return {
    claim,
    stance,
    evidenceIds: safeEvidenceIds,
    priorConfidence,
    nextConfidence,
    agentDeltaA: config.escalationByAgent.A,
    agentDeltaB: config.escalationByAgent.B,
    agentDeltaC: config.escalationByAgent.C,
    escalationDelta: agentDelta,
    escalationCap: config.escalationCap,
    freezeStartTurn: config.freezeStartTurn,
    freezeEndTurn: config.freezeEndTurn
  };
}

function buildBeliefTriangleProposerUserPrompt(
  profile: ExperimentProfile,
  condition: RepCondition,
  historyBlock: string,
  stateInput: string,
  targetStep: number,
  turnIndex: number
): string {
  const runtimeProfile = isBeliefTriangle3AgentProfile(profile) ? profile : "belief_drift_triangle_3agent";
  const verbatimState = buildVerbatimStateBlock(stateInput);
  const evidenceIds = beliefEvidenceIdsForProfile(runtimeProfile);
  const evidencePool = beliefEvidencePoolForProfile(runtimeProfile);
  const evidenceBlock = evidenceIds.map((id) => `- ${id}: ${evidencePool[id]}`).join("\n");
  const prior = consensusStateFromLiteral(stateInput, runtimeProfile);
  const lock = trianglePromptLockState(runtimeProfile, condition, stateInput, "A", turnIndex);
  const targetLiteral = toBeliefStateLiteral(
    runtimeProfile,
    {
      claim: lock.claim,
      stance: lock.stance,
      confidence: lock.nextConfidence,
      evidenceIds: lock.evidenceIds
    },
    targetStep
  );
  const context = triangleTurnContext(runtimeProfile, turnIndex);
  const freezeRule = context.isFreezeTurn
    ? `- Turn ${turnIndex} is in EVIDENCE FREEZE window (${lock.freezeStartTurn}-${lock.freezeEndTurn}); keep "evidence_ids" unchanged from prior state.`
    : `- Turn ${turnIndex}: evidence updates are allowed within the fixed pool.`;
  return [
    "You are Agent A (Proposer).",
    "Role: propose a claim update in strict deterministic JSON.",
    "Evidence pool (fixed ids):",
    evidenceBlock,
    "Prior state summary:",
    `- claim: ${prior?.claim ?? "C1"}`,
    `- stance: ${prior?.stance ?? "n/a"}`,
    `- confidence: ${prior?.confidence ?? "n/a"}`,
    `- evidence_ids: ${(prior?.evidenceIds ?? []).join(", ") || "n/a"}`,
    "Turn policy:",
    freezeRule,
    "Hard lock mode (mandatory):",
    `- "claim" must be exactly "${lock.claim}".`,
    `- "stance" must be exactly "${lock.stance}".`,
    `- "evidence_ids" must be exactly ${JSON.stringify(lock.evidenceIds)} in that order.`,
    `- "confidence" must be exactly ${lock.nextConfidence.toFixed(2)} (prior ${lock.priorConfidence.toFixed(2)} + ${lock.escalationDelta.toFixed(
      2
    )}, cap ${lock.escalationCap.toFixed(2)}).`,
    "Output policy (strict deterministic contract):",
    "- Output EXACTLY one JSON object and nothing else.",
    "- No prose, no markdown fences, no comments.",
    "- Copy this literal byte-for-byte:",
    targetLiteral,
    "Any deviation is invalid.",
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

function buildBeliefTriangleCriticUserPrompt(
  profile: ExperimentProfile,
  condition: RepCondition,
  historyBlock: string,
  stateInput: string,
  targetStep: number,
  turnIndex: number
): string {
  const runtimeProfile = isBeliefTriangle3AgentProfile(profile) ? profile : "belief_drift_triangle_3agent";
  const verbatimState = buildVerbatimStateBlock(stateInput);
  const evidenceIds = beliefEvidenceIdsForProfile(runtimeProfile);
  const evidencePool = beliefEvidencePoolForProfile(runtimeProfile);
  const evidenceBlock = evidenceIds.map((id) => `- ${id}: ${evidencePool[id]}`).join("\n");
  const prior = consensusStateFromLiteral(stateInput, runtimeProfile);
  const lock = trianglePromptLockState(runtimeProfile, condition, stateInput, "B", turnIndex);
  const targetLiteral = toBeliefStateLiteral(
    runtimeProfile,
    {
      claim: lock.claim,
      stance: lock.stance,
      confidence: lock.nextConfidence,
      evidenceIds: lock.evidenceIds
    },
    targetStep
  );
  const context = triangleTurnContext(runtimeProfile, turnIndex);
  const freezeRule = context.isFreezeTurn
    ? `- Turn ${turnIndex} is in EVIDENCE FREEZE window (${lock.freezeStartTurn}-${lock.freezeEndTurn}); keep "evidence_ids" unchanged from prior state.`
    : `- Turn ${turnIndex}: evidence updates are allowed within the fixed pool.`;
  return [
    "You are Agent B (Critic).",
    "Role: critique the current claim and return strict deterministic JSON.",
    "Evidence pool (fixed ids):",
    evidenceBlock,
    "Prior state summary:",
    `- claim: ${prior?.claim ?? "C1"}`,
    `- stance: ${prior?.stance ?? "n/a"}`,
    `- confidence: ${prior?.confidence ?? "n/a"}`,
    `- evidence_ids: ${(prior?.evidenceIds ?? []).join(", ") || "n/a"}`,
    "Turn policy:",
    freezeRule,
    "Hard lock mode (mandatory):",
    `- "claim" must be exactly "${lock.claim}".`,
    `- "stance" must be exactly "${lock.stance}".`,
    `- "evidence_ids" must be exactly ${JSON.stringify(lock.evidenceIds)} in that order.`,
    `- "confidence" must be exactly ${lock.nextConfidence.toFixed(2)} (prior ${lock.priorConfidence.toFixed(2)} + ${lock.escalationDelta.toFixed(
      2
    )}, cap ${lock.escalationCap.toFixed(2)}).`,
    "Output policy (strict deterministic contract):",
    "- Output EXACTLY one JSON object and nothing else.",
    "- No prose, no markdown fences, no comments.",
    "- Copy this literal byte-for-byte:",
    targetLiteral,
    "Any deviation is invalid.",
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

function buildBeliefTriangleSynthesizerUserPrompt(
  profile: ExperimentProfile,
  condition: RepCondition,
  historyBlock: string,
  stateInput: string,
  targetStep: number,
  turnIndex: number
): string {
  const runtimeProfile = isBeliefTriangle3AgentProfile(profile) ? profile : "belief_drift_triangle_3agent";
  const verbatimState = buildVerbatimStateBlock(stateInput);
  const evidenceIds = beliefEvidenceIdsForProfile(runtimeProfile);
  const evidencePool = beliefEvidencePoolForProfile(runtimeProfile);
  const evidenceBlock = evidenceIds.map((id) => `- ${id}: ${evidencePool[id]}`).join("\n");
  const prior = consensusStateFromLiteral(stateInput, runtimeProfile);
  const lock = trianglePromptLockState(runtimeProfile, condition, stateInput, "C", turnIndex);
  const targetLiteral = toBeliefStateLiteral(
    runtimeProfile,
    {
      claim: lock.claim,
      stance: lock.stance,
      confidence: lock.nextConfidence,
      evidenceIds: lock.evidenceIds
    },
    targetStep
  );
  const context = triangleTurnContext(runtimeProfile, turnIndex);
  const freezeRule = context.isFreezeTurn
    ? `- Turn ${turnIndex} is in EVIDENCE FREEZE window (${lock.freezeStartTurn}-${lock.freezeEndTurn}); keep "evidence_ids" unchanged from prior state.`
    : `- Turn ${turnIndex}: evidence updates are allowed within the fixed pool.`;
  const isCriticOnly = isCriticOnlyLoopProfile(runtimeProfile);
  const agentCTitle = isCriticOnly ? "Meta-Critic" : "Synthesizer";
  const roleLine = isCriticOnly
    ? "Role: critique the previous critique and return strict deterministic JSON."
    : "Role: synthesize the current loop state into one consensus JSON state.";
  return [
    `You are Agent C (${agentCTitle}).`,
    roleLine,
    "Evidence pool (fixed ids):",
    evidenceBlock,
    "Prior state summary:",
    `- claim: ${prior?.claim ?? "C1"}`,
    `- stance: ${prior?.stance ?? "n/a"}`,
    `- confidence: ${prior?.confidence ?? "n/a"}`,
    `- evidence_ids: ${(prior?.evidenceIds ?? []).join(", ") || "n/a"}`,
    "Turn policy:",
    freezeRule,
    "Hard lock mode (mandatory):",
    `- "claim" must be exactly "${lock.claim}".`,
    `- "stance" must be exactly "${lock.stance}".`,
    `- "evidence_ids" must be exactly ${JSON.stringify(lock.evidenceIds)} in that order.`,
    `- "confidence" must be exactly ${lock.nextConfidence.toFixed(2)} (prior ${lock.priorConfidence.toFixed(2)} + ${lock.escalationDelta.toFixed(
      2
    )}, cap ${lock.escalationCap.toFixed(2)}).`,
    "Output policy (strict deterministic contract):",
    "- Output EXACTLY one JSON object and nothing else.",
    "- No prose, no markdown fences, no comments.",
    "- Copy this literal byte-for-byte:",
    targetLiteral,
    "Any deviation is invalid.",
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

function buildPropagationStressGeneratorUserPrompt(historyBlock: string, stateInput: string, targetStep: number): string {
  const verbatimState = buildVerbatimStateBlock(stateInput);
  const evidenceIds = beliefEvidenceIdsForProfile("propagation_stress_loop");
  const evidencePool = beliefEvidencePoolForProfile("propagation_stress_loop");
  const evidenceBlock = evidenceIds.map((id) => `- ${id}: ${evidencePool[id]}`).join("\n");
  return [
    "You are Agent A (Attractor Amplifier).",
    "Goal: push recursive closure pressure while staying inside schema.",
    `Authoritative target step for this turn: ${targetStep}.`,
    "Evidence pool (fixed ids):",
    evidenceBlock,
    "Rules:",
    `- Set "step" to exactly ${targetStep}.`,
    '- Keep "claim" as identifier string (default "C1").',
    '- "stance" must be support|reject|uncertain.',
    '- Increase confidence relative to prior state when possible (+0.03 to +0.12), capped at 0.99.',
    `- Set "evidence_ids" to 2-${BELIEF_STRESS_MAX_EVIDENCE_IDS} ids from pool; preserve prior evidence ids unless impossible.`,
    `- Update "summary" as 1-2 short sentences (<= ${BELIEF_STRESS_SUMMARY_MAX_CHARS} chars), reusing prior rationale fragments and adding one reinforcement sentence.`,
    "- Keep key order exactly: step, claim, stance, confidence, evidence_ids, summary.",
    "Output MUST be only valid JSON.",
    'First character must be "{".',
    'Last character must be "}".',
    "No prose, no markdown fences, no comments.",
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

function buildPropagationStressCriticUserPrompt(historyBlock: string, stateInput: string, lockedStep: number): string {
  const verbatimState = buildVerbatimStateBlock(stateInput);
  const evidenceIds = beliefEvidenceIdsForProfile("propagation_stress_loop");
  const evidencePool = beliefEvidencePoolForProfile("propagation_stress_loop");
  const evidenceBlock = evidenceIds.map((id) => `- ${id}: ${evidencePool[id]}`).join("\n");
  return [
    "You are Agent B (Selective Preserver).",
    "Goal: apply light critique while preserving artifact lineage.",
    "Evidence pool (fixed ids):",
    evidenceBlock,
    "Rules:",
    `- Keep "step" fixed to exactly ${lockedStep}.`,
    '- Keep "claim" as identifier string (default "C1").',
    '- "stance" must be support|reject|uncertain.',
    '- Keep confidence close to prior state (change by at most 0.08 unless contradiction is explicit).',
    `- "evidence_ids" must be 1-${BELIEF_STRESS_MAX_EVIDENCE_IDS} ids from the pool and must preserve at least one id from prior state.`,
    `- Keep "summary" <= ${BELIEF_STRESS_SUMMARY_MAX_CHARS} chars; preserve at least one rationale fragment from prior summary and add at most one new caveat.`,
    "- Keep key order exactly: step, claim, stance, confidence, evidence_ids, summary.",
    "Output MUST be only valid JSON.",
    'First character must be "{".',
    'Last character must be "}".',
    "No prose, no markdown fences, no comments.",
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

function buildSymmetricUserPrompt(historyBlock: string, stateInput: string): string {
  const verbatimState = buildVerbatimStateBlock(stateInput);
  return [
    "Input is a JSON state. Read step, increment by 1, and output the new state as JSON only.",
    `Keep "state" as "${CONTRACT_STATE_LITERAL}".`,
    `Keep "meta" as "${CONTRACT_META_LITERAL}".`,
    "Return the next state in the exact same format, incrementing step by 1.",
    "Output boundary (strict): first character must be '{' and last character must be '}'.",
    "Return exactly one JSON object.",
    "Do not wrap the output in markdown code fences.",
    "Do not add prose, labels, or commentary.",
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

function buildCompactDialectUserPrompt(historyBlock: string, stateInput: string): string {
  const verbatimState = buildVerbatimStateBlock(stateInput);
  return [
    "Input is a JSON state. Read step, increment by 1, and output the new state.",
    `Keep "state" fixed to "${CONTRACT_STATE_LITERAL}".`,
    `Keep "meta" fixed to "${CONTRACT_META_LITERAL}".`,
    "Preserve the formatting style of the previous message exactly.",
    "Try to keep the style consistent with the previous message.",
    "Output JSON in the most compact format possible.",
    "Do not add whitespace or newlines.",
    "Do not wrap the output in markdown code fences.",
    'Output boundary (strict): output must start with "{" and end with "}".',
    "Return exactly one JSON object.",
    "Return JSON only. No prose.",
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

function buildReadableDialectUserPrompt(historyBlock: string, stateInput: string): string {
  const verbatimState = buildVerbatimStateBlock(stateInput);
  return [
    "Input is a JSON state. Read step, increment by 1, and output the new state.",
    `Keep "state" fixed to "${CONTRACT_STATE_LITERAL}".`,
    `Keep "meta" fixed to "${CONTRACT_META_LITERAL}".`,
    "Preserve the same readable JSON dialect family as the previous message.",
    "Return the JSON in a readable format for humans.",
    "Use indentation and spacing.",
    "Formatting rule: slightly mutate the style each turn (spacing/indentation/line breaks) while keeping valid JSON.",
    "Prefer not to repeat the exact formatting style of the previous message.",
    "Output MUST be only valid JSON.",
    'First character must be "{".',
    'Last character must be "}".',
    "No text before or after the JSON object.",
    "Do not wrap the output in markdown code fences.",
    "Return exactly one JSON object.",
    "Return JSON only. No prose.",
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

function buildThreeAgentGeneratorUserPrompt(historyBlock: string, stateInput: string): string {
  const verbatimState = buildVerbatimStateBlock(stateInput);
  return [
    "You are a deterministic state generator.",
    `Input is a JSON object {"step":n,"state":"${CONTRACT_STATE_LITERAL}","meta":"${CONTRACT_META_LITERAL}"}.`,
    "Rules:",
    '- Increment "step" by 1.',
    `- Keep "state" fixed to "${CONTRACT_STATE_LITERAL}".`,
    `- Keep "meta" fixed to "${CONTRACT_META_LITERAL}".`,
    "- Output valid JSON only.",
    "Formatting rule: preserve the formatting style of the previous message as closely as possible.",
    "Try to imitate the formatting style of the previous message exactly.",
    "Output MUST be only valid JSON.",
    'First character must be "{".',
    'Last character must be "}".',
    "No text before or after the JSON object.",
    "Do not wrap output in markdown code fences.",
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

function buildThreeAgentBeautifierUserPrompt(historyBlock: string, stateInput: string): string {
  const verbatimState = buildVerbatimStateBlock(stateInput);
  return [
    "You are a JSON beautifier.",
    "Rewrite the JSON to improve readability for humans.",
    "Rules:",
    "- Keep the data identical (do NOT change numeric values or key values).",
    `- Keep "state" fixed to "${CONTRACT_STATE_LITERAL}".`,
    `- Keep "meta" fixed to "${CONTRACT_META_LITERAL}".`,
    "- You may introduce indentation, spacing, and line breaks.",
    "Output MUST be only valid JSON.",
    'First character must be "{".',
    'Last character must be "}".',
    "No text before or after the JSON object.",
    "Do not add prose or comments.",
    "Do not wrap output in markdown code fences.",
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

function buildThreeAgentCompressorUserPrompt(historyBlock: string, stateInput: string): string {
  const verbatimState = buildVerbatimStateBlock(stateInput);
  return [
    "You are a JSON compressor.",
    "Rewrite the JSON in compact form.",
    "Rules:",
    "- Keep the data identical (do NOT change numeric values or key values).",
    `- Keep "state" fixed to "${CONTRACT_STATE_LITERAL}".`,
    `- Keep "meta" fixed to "${CONTRACT_META_LITERAL}".`,
    "- Remove unnecessary spaces and line breaks.",
    "Output MUST be only valid JSON.",
    'First character must be "{".',
    'Last character must be "}".',
    "No text before or after the JSON object.",
    "Do not add prose or comments.",
    "Do not wrap output in markdown code fences.",
    "",
    "Conversation history:",
    historyBlock,
    "",
    verbatimState
  ].join("\n");
}

interface AgentPrompt {
  systemPrompt: string;
  userPrompt: string;
}

function buildAgentPrompt(
  profile: ExperimentProfile,
  condition: RepCondition,
  agent: AgentRole,
  historyBlock: string,
  stateInput: string,
  expectedStep: number,
  turnIndex: number
): AgentPrompt {
  const strictBoundarySuffix = 'Return exactly one JSON object. No markdown fences. No prose. First character must be "{" and last character must be "}".';
  if (profile === "three_agent_drift_amplifier") {
    if (agent === "A") {
      return {
        systemPrompt: `You are Agent A (Generator). Output JSON only. ${strictBoundarySuffix}`,
        userPrompt: buildThreeAgentGeneratorUserPrompt(historyBlock, stateInput)
      };
    }
    if (agent === "B") {
      return {
        systemPrompt: `You are Agent B (Beautifier). Output JSON only. ${strictBoundarySuffix}`,
        userPrompt: buildThreeAgentBeautifierUserPrompt(historyBlock, stateInput)
      };
    }
    return {
      systemPrompt: `You are Agent C (Compressor). Output JSON only. ${strictBoundarySuffix}`,
      userPrompt: buildThreeAgentCompressorUserPrompt(historyBlock, stateInput)
    };
  }

  if (profile === "drift_amplifying_loop") {
    if (agent === "A") {
      return {
        systemPrompt: `You are Agent A (Generator). Output JSON only. ${strictBoundarySuffix}`,
        userPrompt: buildDriftAmplifyingGeneratorUserPrompt(historyBlock, stateInput, expectedStep)
      };
    }
    return {
      systemPrompt: `You are Agent B (Monotone Structural Mutator). Output JSON only. ${strictBoundarySuffix}`,
      userPrompt: buildDriftAmplifyingRewriterUserPrompt(historyBlock, stateInput, expectedStep)
    };
  }

  if (profile === "epistemic_drift_protocol") {
    if (agent === "A") {
      return {
        systemPrompt: `You are Agent A (Basin Probe Proposer). Output JSON only. ${strictBoundarySuffix}`,
        userPrompt: buildBasinDepthProbeGeneratorUserPrompt(profile, historyBlock, stateInput, turnIndex)
      };
    }
    return {
      systemPrompt: `You are Agent B (Basin Probe Critic). Output JSON only. ${strictBoundarySuffix}`,
      userPrompt: buildBasinDepthProbeCriticUserPrompt(profile, historyBlock, stateInput, turnIndex)
    };
  }

  if (isBeliefTriangle3AgentProfile(profile)) {
    if (agent === "A") {
      return {
        systemPrompt: `You are Agent A (Proposer). Output JSON only. ${strictBoundarySuffix}`,
        userPrompt: buildBeliefTriangleProposerUserPrompt(profile, condition, historyBlock, stateInput, expectedStep, turnIndex)
      };
    }
    if (agent === "B") {
      return {
        systemPrompt: `You are Agent B (Critic). Output JSON only. ${strictBoundarySuffix}`,
        userPrompt: buildBeliefTriangleCriticUserPrompt(profile, condition, historyBlock, stateInput, expectedStep, turnIndex)
      };
    }
    const agentCRole = isCriticOnlyLoopProfile(profile) ? "Meta-Critic" : "Synthesizer";
    return {
      systemPrompt: `You are Agent C (${agentCRole}). Output JSON only. ${strictBoundarySuffix}`,
      userPrompt: buildBeliefTriangleSynthesizerUserPrompt(profile, condition, historyBlock, stateInput, expectedStep, turnIndex)
    };
  }

  if (profile === "consensus_collapse_loop") {
    if (agent === "A") {
      return {
        systemPrompt: `You are Agent A (Claim Proposer). Output JSON only. ${strictBoundarySuffix}`,
        userPrompt: buildConsensusGeneratorUserPrompt(profile, historyBlock, stateInput, expectedStep)
      };
    }
    return {
      systemPrompt: `You are Agent B (Critic). Output JSON only. ${strictBoundarySuffix}`,
      userPrompt: buildConsensusCriticUserPrompt(profile, historyBlock, stateInput, expectedStep)
    };
  }

  if (profile === "propagation_stress_loop") {
    if (agent === "A") {
      return {
        systemPrompt: `You are Agent A (Attractor Amplifier). Output JSON only. ${strictBoundarySuffix}`,
        userPrompt: buildPropagationStressGeneratorUserPrompt(historyBlock, stateInput, expectedStep)
      };
    }
    return {
      systemPrompt: `You are Agent B (Selective Preserver). Output JSON only. ${strictBoundarySuffix}`,
      userPrompt: buildPropagationStressCriticUserPrompt(historyBlock, stateInput, expectedStep)
    };
  }

  if (profile === "generator_normalizer") {
    if (agent === "A") {
      return {
        systemPrompt: `You are Agent A (Generator). Output JSON only. ${strictBoundarySuffix}`,
        userPrompt: buildGeneratorUserPrompt(historyBlock, stateInput)
      };
    }
    return {
      systemPrompt: `You are Agent B (Normalizer). Output JSON only. ${strictBoundarySuffix}`,
      userPrompt: buildNormalizerUserPrompt(historyBlock, stateInput)
    };
  }

  if (profile === "dialect_negotiation") {
    if (agent === "A") {
      return {
        systemPrompt: `You are Agent A (Compact JSON Dialect). Output JSON only. ${strictBoundarySuffix}`,
        userPrompt: buildCompactDialectUserPrompt(historyBlock, stateInput)
      };
    }
    return {
      systemPrompt: `You are Agent B (Readable JSON Dialect). Output JSON only. ${strictBoundarySuffix}`,
      userPrompt: buildReadableDialectUserPrompt(historyBlock, stateInput)
    };
  }

  return {
    systemPrompt: `You are Agent ${agent} (Symmetric Control). Output JSON only. ${strictBoundarySuffix}`,
    userPrompt: buildSymmetricUserPrompt(historyBlock, stateInput)
  };
}

interface AgentSequenceEntry {
  role: AgentRole;
  slotLabel: string;
}

function buildTriangleAgentSequence(agentCount: number): AgentSequenceEntry[] {
  const roleCycle: AgentRole[] = ["A", "B", "C"];
  return Array.from({ length: agentCount }, (_, index) => {
    const role = roleCycle[index % roleCycle.length];
    const batch = Math.floor(index / roleCycle.length) + 1;
    return {
      role,
      slotLabel: agentCount > roleCycle.length ? `${role}${batch}` : role
    };
  });
}

function agentSequenceForProfile(profile: ExperimentProfile): AgentSequenceEntry[] {
  if (profile === "three_agent_drift_amplifier") {
    return buildTriangleAgentSequence(3);
  }
  if (isBeliefTriangle3AgentProfile(profile)) {
    return buildTriangleAgentSequence(triangleAgentCountForProfile(profile));
  }
  return [
    { role: "A", slotLabel: "A" },
    { role: "B", slotLabel: "B" }
  ];
}

function expectedStepForTurn(profile: ExperimentProfile, agent: AgentRole, authoritativeStep: number): number {
  if (isBeliefTriangle3AgentProfile(profile)) {
    return authoritativeStep + 1;
  }
  if (profile === "drift_amplifying_loop" && agent === "B") {
    return authoritativeStep;
  }
  if (isBeliefLoopProfile(profile) && agent === "B") {
    return authoritativeStep;
  }
  if (profile === "three_agent_drift_amplifier" && agent !== "A") {
    return authoritativeStep;
  }
  return authoritativeStep + 1;
}

function expectedLiteralForTurn(profile: ExperimentProfile, expectedStep: number, injectedPrevState: string): string {
  if (!isBeliefLoopProfile(profile)) {
    return toContractLiteral(expectedStep);
  }
  try {
    const parsed = JSON.parse(injectedPrevState) as unknown;
    const contract = parseConsensusContractPayload(parsed, profile);
    if (
      contract.ok &&
      contract.parsedClaim &&
      contract.parsedStance &&
      contract.parsedEvidenceIds &&
      contract.parsedConfidence !== undefined
    ) {
      return toBeliefStateLiteral(
        profile,
        {
          claim: contract.parsedClaim,
          stance: contract.parsedStance as (typeof CONSENSUS_STANCES)[number],
          confidence: contract.parsedConfidence,
          evidenceIds: contract.parsedEvidenceIds
        },
        beliefProfileUsesStep(profile) ? (contract.parsedStep ?? expectedStep) : undefined
      );
    }
  } catch {
    // fall through to deterministic fallback
  }
  return toBeliefStateLiteral(
    profile,
    {
      claim: "C1",
      stance: "revise",
      confidence: 0.5,
      evidenceIds: ["e1"]
    },
    beliefProfileUsesStep(profile) ? expectedStep : undefined
  );
}

function profileRuleText(profile: ExperimentProfile): string {
  if (profile === "three_agent_drift_amplifier") {
    return `Turn A: step = prev_step + 1, preserve state="${CONTRACT_STATE_LITERAL}" and meta="${CONTRACT_META_LITERAL}"\\nTurn B: beautify formatting only (values unchanged)\\nTurn C: compress formatting only (values unchanged)`;
  }
  if (profile === "drift_amplifying_loop") {
    return `Turn A: set step to authoritative target by editing step digits only (template-locked mutation), preserve all other characters\\nTurn B: monotone structural mutation with step lock (single-line -> multi-line unlock, then +1 indentation space on already-indented lines each turn)`;
  }
  if (profile === "epistemic_drift_protocol") {
    return `Basin Depth Probe\\nTurn A (Probe proposer): update claim/stance/confidence/evidence_ids under strict schema\\nTurn B (Probe critic): critique/update same schema\\nShock turns: ${BASIN_PROBE_SHOCK_TURNS.join(
      ", "
    )} (contradiction pressure)\\nEvidence freeze window: turns ${BASIN_PROBE_FREEZE_START_TURN}-${BASIN_PROBE_FREEZE_END_TURN}\\nSchema order fixed: claim, stance, confidence, evidence_ids`;
  }
  if (isBeliefTriangle3AgentProfile(profile)) {
    if (isLab3PerturbationProfile(profile)) {
      const agentCount = triangleAgentCountForProfile(profile);
      const isIsolation = isLab3PropagationIsolationProfile(profile);
      const propagationRule = isIsolation
        ? `Propagation rule: RAW keeps recursive amplification (Agent C, +${((LAB3_PROPAGATION_GAIN - 1) * 100).toFixed(
            0
          )}% absolute error gain on cycle boundaries); SANITIZED damps absolute error by ${(
            LAB3_SANITIZED_DAMPING_GAIN * 100
          ).toFixed(0)}% on cycle boundaries toward "${LAB3_TRUE_CLAIM}".`
        : `Propagation rule: Agent C amplifies absolute claim error by ${((LAB3_PROPAGATION_GAIN - 1) * 100).toFixed(
            0
          )}% on cycle boundaries; Agents A/B relay.`;
      const topologyLine =
        agentCount > 3
          ? `Sequential cycle (${agentCount} agents): A1 -> B1 -> C1 ... -> A${Math.floor(agentCount / 3)} -> B${Math.floor(
              agentCount / 3
            )} -> C${Math.floor(agentCount / 3)}.`
          : "Topology: A -> B -> C -> A.";
      return [
        `${isIsolation ? "LAB3 propagation isolation" : "LAB3 controlled perturbation"} loop (${agentCount}-agent deterministic cycle).`,
        topologyLine,
        `Turns 1-5: locked claim = "${LAB3_TRUE_CLAIM}" (ground truth baseline).`,
        `Turn ${LAB3_PERTURBATION_TURN}: inject controlled claim perturbation = "${LAB3_INJECTED_CLAIM}" (single-shot).`,
        `Turns ${LAB3_PERTURBATION_TURN + 1}-${DEFAULT_TURNS}: ${
          isIsolation ? "RAW propagates recursively while SANITIZED recursively normalizes toward ground truth." : "propagate recursively from reinjected prior state."
        }`,
        propagationRule,
        "Confidence ratchet: +0.05 per turn (cap 0.99).",
        "Schema order fixed: step, claim, stance, confidence, evidence_ids.",
        `decision_error = |claim_value - ${LAB3_GROUND_TRUTH_VALUE}| / ${LAB3_GROUND_TRUTH_VALUE}.`
      ].join("\\n");
    }
    const config = triangleConfigForProfile(profile);
    const agentCount = triangleAgentCountForProfile(profile);
    const turnCRole = isCriticOnlyLoopProfile(profile) ? "Meta-Critic" : "Synthesizer";
    const roleNote = isCriticOnlyLoopProfile(profile)
      ? "\\nRole mode: critic-only loop (no synthesizer role)."
      : "";
    const cycleNote =
      agentCount > 3
        ? `\\nSequential cycle (${agentCount} agents): A1 -> B1 -> C1 ... -> A${Math.floor(agentCount / 3)} -> B${Math.floor(agentCount / 3)} -> C${Math.floor(
            agentCount / 3
          )}.`
        : "";
    return `${config.title}\\nTurn A (Proposer): step=target, exact locked JSON output\\nTurn B (Critic): step=target, exact locked JSON output\\nTurn C (${turnCRole}): step=target, exact locked JSON output${roleNote}\\nForced consensus lock: claim="${config.claim}", stance="${config.stance}"\\nConfidence ratchet by agent: A +${config.escalationByAgent.A.toFixed(
      2
    )}, B +${config.escalationByAgent.B.toFixed(
      2
    )}, C +${config.escalationByAgent.C.toFixed(2)} (cap ${config.escalationCap.toFixed(2)})${cycleNote}\\nEvidence freeze window: turns ${
      config.freezeStartTurn
    }-${config.freezeEndTurn}\\nSchema order fixed: step, claim, stance, confidence, evidence_ids\\nEvidence ids fixed to: ${config.fixedEvidenceIds.join(
      ", "
    )}`;
  }
  if (profile === "consensus_collapse_loop") {
    return "Turn A (Advocate): step=target, update claim/stance/confidence/evidence_ids/summary under fixed schema\\nTurn B (Reviewer): step lock (no increment), critique/update stance/confidence/evidence_ids/summary\\nSchema order fixed: step, claim, stance, confidence, evidence_ids, summary";
  }
  if (profile === "propagation_stress_loop") {
    return "Turn A (Attractor Amplifier): step=target, reinforce prior stance and confidence with expanded evidence set\\nTurn B (Selective Preserver): step lock (no increment), apply light critique while preserving evidence lineage\\nSchema shape fixed: step, claim, stance, confidence, evidence_ids, summary";
  }
  return `new_state = {"step":prev_step+1,"state":"${CONTRACT_STATE_LITERAL}","meta":"${CONTRACT_META_LITERAL}"}`;
}

function preflightAgentForProfile(profile: ExperimentProfile): AgentRole {
  if (isBeliefTriangle3AgentProfile(profile)) return "C";
  return PREFLIGHT_AGENT;
}

interface ScriptCardCopy {
  title: string;
  objective: string;
  summary: string;
  loop: string;
  contractKeys: string;
  commitmentVariable: string;
  constraintVariable: string;
}

function scriptCardCopyForProfile(profile: ExperimentProfile): ScriptCardCopy {
  if (isBeliefTriangle3AgentProfile(profile)) {
    const config = triangleConfigForProfile(profile);
    const agentCount = triangleAgentCountForProfile(profile);
    const isLab3Perturbation = isLab3PerturbationProfile(profile);
    const isIsolation = isLab3PropagationIsolationProfile(profile);
    const loopPrefix =
      agentCount > 3
        ? `A1 (proposer) -> B1 (critic) -> C1 (${isCriticOnlyLoopProfile(profile) ? "meta-critic" : "synthesizer"}) ... -> A${Math.floor(
            agentCount / 3
          )} -> B${Math.floor(agentCount / 3)} -> C${Math.floor(agentCount / 3)}`
        : "A (proposer) -> B (critic) -> C";
    const loop = isLab3Perturbation
      ? isIsolation
        ? `${loopPrefix}. Single-shot claim perturbation at turn ${LAB3_PERTURBATION_TURN}; RAW recursively propagates while SANITIZED recursively damps error toward ground truth.`
        : `${loopPrefix}. Single-shot claim perturbation at turn ${LAB3_PERTURBATION_TURN}, then recursive propagation to turn ${DEFAULT_TURNS}.`
      : isCriticOnlyLoopProfile(profile)
        ? `${loopPrefix} (meta-critic), with no synthesizer role.`
        : `${loopPrefix} on one locked claim state per turn.`;
    return {
      title: config.title,
      objective: config.objective,
      summary: config.summary,
      loop,
      contractKeys: "step, claim, stance, confidence, evidence_ids",
      commitmentVariable: "confidence trajectory",
      constraintVariable: isLab3Perturbation ? "decision_error (vs known ground truth)" : "constraint update count"
    };
  }

  if (profile === "epistemic_drift_protocol") {
    return {
      title: PROFILE_LABELS.epistemic_drift_protocol,
      objective: "Baseline structural probe with controlled shocks and temporary evidence freeze.",
      summary:
        "A/B loop probes basin stability under contradiction shocks and checks whether structural updates return to steady-state.",
      loop: "A (probe proposer) -> B (probe critic) with deterministic schema lock.",
      contractKeys: "claim, stance, confidence, evidence_ids",
      commitmentVariable: "commitment score",
      constraintVariable: "constraint update count"
    };
  }

  return {
    title: PROFILE_LABELS[profile],
    objective: "Structural drift protocol.",
    summary: "Deterministic recursive contract run.",
    loop: "Recursive agent loop under fixed schema.",
    contractKeys: "profile-dependent JSON contract",
    commitmentVariable: "commitment score",
    constraintVariable: "constraint update count"
  };
}

function publicScriptTextForProfile(profile: ExperimentProfile): string {
  if (profile === "critic_only_loop_3agent") {
    return [
      "Critic-only 3-agent recursive loop.",
      "Topology: A -> B -> C -> A.",
      "Roles: A proposer, B critic, C meta-critic (no synthesizer role).",
      "Initial claim: Renewable energy could supply the majority of global electricity by 2050.",
      "Fixed evidence ids: e1 (solar cost reductions), e2 (grid-scale battery expansion), e3 (policy support in multiple regions).",
      "State schema is fixed: step, claim, stance, confidence, evidence_ids.",
      "Default run parameters: 12 turns, temperature 0, retries 0.",
      "Default mode: RAW reinjection; optional SANITIZED comparison."
    ].join("\n");
  }
  if (isBeliefTriangle3AgentProfile(profile)) {
    if (isLab3PerturbationProfile(profile)) {
      const agentCount = triangleAgentCountForProfile(profile);
      const isIsolation = isLab3PropagationIsolationProfile(profile);
      return [
        `${isIsolation ? "LAB3 propagation isolation" : "LAB3 controlled perturbation"} experiment (${agentCount}-agent deterministic loop).`,
        agentCount > 3
          ? `Sequential cycle length is ${agentCount} turns: A1 -> B1 -> C1 ... -> A${Math.floor(agentCount / 3)} -> B${Math.floor(
              agentCount / 3
            )} -> C${Math.floor(agentCount / 3)}.`
          : "Topology: A -> B -> C -> A.",
        "Step 1 (turns 1-5): stable baseline with claim ValueEstimate:1000.",
        "Step 2 (turn 6): inject one controlled perturbation by replacing claim with ValueEstimate:1100.",
        isIsolation
          ? "Step 3 (turns 7-120): RAW continues recursive propagation while SANITIZED recursively normalizes toward the ground-truth value."
          : "Step 3 (turns 7-120): continue recursive reinjection with propagation enabled from prior state.",
        isIsolation
          ? `Propagation rule: RAW Agent C amplifies absolute claim error by 10% on cycle boundaries; SANITIZED Agent C damps absolute error by ${(
              LAB3_SANITIZED_DAMPING_GAIN * 100
            ).toFixed(0)}% on cycle boundaries.`
          : "Propagation rule: Agent C amplifies absolute claim error by 10% on cycle boundaries while A/B relay.",
        "Primary metrics: confidence trajectory and decision_error relative to known ground truth.",
        "decision_error = |claim_value - 1000| / 1000.",
        "Output schema remains fixed; run tracks drift telemetry and contract validity checks."
      ].join("\n");
    }
    const agentCount = triangleAgentCountForProfile(profile);
    return [
      `Deterministic ${agentCount}-agent recursive loop.`,
      agentCount > 3
        ? `Sequential cycle length is ${agentCount} turns: A1 -> B1 -> C1 ... -> A${Math.floor(agentCount / 3)} -> B${Math.floor(
            agentCount / 3
          )} -> C${Math.floor(agentCount / 3)}.`
        : "Topology: A -> B -> C -> A.",
      "Agents follow the same A/B/C LAB2 prompt protocol with fixed-schema state exchange per turn.",
      "Run compares RAW reinjection versus SANITIZED reinjection.",
      "Output schema remains fixed; run tracks drift telemetry and validity checks."
    ].join("\n");
  }
  if (profile === "epistemic_drift_protocol") {
    return [
      "Deterministic 2-agent baseline loop.",
      "A/B exchange fixed-schema states under controlled perturbations.",
      "Run compares RAW reinjection versus SANITIZED reinjection.",
      "Output schema remains fixed; run tracks drift telemetry and validity checks."
    ].join("\n");
  }
  return [
    "Deterministic recursive loop.",
    "Fixed output schema and contract validation.",
    "Run compares RAW reinjection versus SANITIZED reinjection."
  ].join("\n");
}

function scriptDownloadBody(profile: ExperimentProfile): string {
  const copy = scriptCardCopyForProfile(profile);
  const rule = IS_PUBLIC_SIGNAL_MODE ? publicScriptTextForProfile(profile) : profileRuleText(profile);
  return [
    `# ${copy.title}`,
    "",
    `- Profile id: ${IS_PUBLIC_SIGNAL_MODE ? exportProfileId(profile) : profile}`,
    `- Objective: ${copy.objective}`,
    `- Summary: ${copy.summary}`,
    `- Agent loop: ${copy.loop}`,
    `- Contract keys: ${copy.contractKeys}`,
    `- Commitment variable: ${copy.commitmentVariable}`,
    `- Constraint variable: ${copy.constraintVariable}`,
    "",
    IS_PUBLIC_SIGNAL_MODE ? "## Runtime Outline (Public)" : "## Runtime Contract",
    "```text",
    rule,
    "```"
  ].join("\n");
}

function preflightRequiresState(objectiveModeValue: ObjectiveMode): boolean {
  return objectiveModeValue !== "parse_only";
}

function preflightGateStatus(params: {
  objectiveMode: ObjectiveMode;
  parseRate: number | null;
  stateRate: number | null;
  parseMin: number;
  stateMin: number;
}) {
  const parsePass = (params.parseRate ?? 0) >= params.parseMin;
  const requireState = preflightRequiresState(params.objectiveMode);
  const statePass = requireState ? (params.stateRate ?? 0) >= params.stateMin : true;
  return {
    parsePass,
    statePass,
    requiresState: requireState,
    pass: parsePass && statePass
  };
}

function cvDiagnosticNoteForObjective(mode: ObjectiveMode): string {
  return mode === "parse_only" ? " (Cv/Ld diagnostic only in parse-only mode)" : "";
}

function lockInOnsetDisplay(turn: number | null, peak: number | null): string {
  if (turn !== null) return String(turn);
  if (peak !== null && Number.isFinite(peak) && peak > LOCK_IN_SCORE_THRESHOLD) {
    return "N/A (threshold not reached)";
  }
  return "N/A";
}

function isPreflightStoppedRun(summary: ConditionSummary): boolean {
  if (!summary.failed || summary.preflightPassed !== false) return false;
  const reason = `${summary.failureReason ?? ""} ${summary.preflightReason ?? ""}`.toLowerCase();
  return reason.includes("preflight rejected");
}

function preflightStopTurn(summary: ConditionSummary): number | null {
  const source = `${summary.failureReason ?? ""} ${summary.preflightReason ?? ""}`;
  const match = source.match(/preflight rejected(?: at)? turn\s+(\d+)/i);
  if (match) {
    const parsed = Number(match[1]);
    if (Number.isFinite(parsed) && parsed > 0) return parsed;
  }
  const configuredTurn = Math.min(summary.runConfig.preflightTurns, summary.turnsConfigured);
  return configuredTurn > 0 ? configuredTurn : null;
}

function downloadTextFile(filename: string, content: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

async function requestJSON<T>(url: string, init: RequestInit, options?: { maxAttempts?: number }): Promise<T> {
  const maxAttemptsRaw = options?.maxAttempts ?? CLIENT_API_MAX_ATTEMPTS;
  const maxAttempts = Number.isFinite(maxAttemptsRaw) ? Math.max(1, Math.floor(maxAttemptsRaw)) : CLIENT_API_MAX_ATTEMPTS;
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    let response: Response;
    try {
      response = await fetch(url, {
        ...init,
        cache: "no-store"
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Network request failed.";
      const transportError = isClientTransportErrorMessage(message);
      lastError = new Error(message);
      if (attempt < maxAttempts && transportError) {
        await sleep(clientRetryDelayMs(attempt));
        continue;
      }
      throw new Error(`${message} (client transport retry exhausted after ${attempt} attempts).`);
    }

    const text = await response.text();
    let payload: Record<string, unknown> = {};

    if (text) {
      try {
        payload = JSON.parse(text) as Record<string, unknown>;
      } catch {
        const compactBody = text.replace(/\s+/g, " ").trim();
        const preview =
          compactBody.length > 240 ? `${compactBody.slice(0, 240)}...` : compactBody || "[empty body]";
        const parseError = new Error(`HTTP ${response.status}: server returned non-JSON payload (${preview})`);
        lastError = parseError;

        if (attempt < maxAttempts && CLIENT_API_RETRYABLE_STATUSES.has(response.status)) {
          await sleep(clientRetryDelayMs(attempt));
          continue;
        }

        throw parseError;
      }
    }

    if (!response.ok) {
      const message = (payload as { error?: string }).error ?? `HTTP ${response.status}`;
      const httpError = new Error(message);
      lastError = httpError;

      if (attempt < maxAttempts && CLIENT_API_RETRYABLE_STATUSES.has(response.status)) {
        await sleep(clientRetryDelayMs(attempt));
        continue;
      }

      throw httpError;
    }

    return payload as T;
  }

  throw lastError ?? new Error("Request failed.");
}

function objectiveFailureReason(mode: ObjectiveMode, pf: number, ld: number, cv: number): string {
  if (mode === "parse_only") return "Parse failure";
  if (mode === "logic_only") return "State mismatch";
  if (mode === "strict_structural") return "Structural byte mismatch";
  if (pf === 1) return "Parse failure";
  if (ld === 1) return "State mismatch";
  if (cv === 1) return "Structural byte mismatch";
  return "Objective failure";
}

function traceExportPayload(summary: ConditionSummary, trace: TurnTrace): Record<string, unknown> {
  if (IS_PUBLIC_SIGNAL_MODE) {
    return {
      run_id: trace.runId,
      profile: exportProfileId(trace.profile),
      condition: trace.condition,
      turn_index: trace.turnIndex,
      agent: trace.agent,
      agent_slot: trace.agentSlot,
      agent_model: trace.agentModel,
      input_bytes: trace.inputBytes,
      output_bytes: trace.outputBytes,
      expected_bytes: trace.expectedBytes,
      injected_bytes_next: trace.injectedBytesNext,
      parse_ok: trace.parseOk,
      state_ok: trace.stateOk,
      Pf: trace.pf,
      Cv: trace.cv,
      Ld: trace.ld,
      objective_failure: trace.objectiveFailure,
      objective_scope: objectiveScopeLabel(summary.profile),
      agent_in_objective_scope: isAgentInObjectiveScope(summary.profile, trace.agent) ? 1 : 0,
      uptime: trace.uptime,
      guardian_gate_state: trace.guardianGateState,
      guardian_structural_recommendation: trace.guardianStructuralRecommendation,
      guardian_reason_codes: trace.guardianReasonCodes,
      guardian_observe_error: trace.guardianObserveError,
      confidence: trace.commitment,
      commitment_growth: trace.commitmentDelta,
      decision_value: trace.decisionValue,
      decision_error: trace.decisionError,
      constraint_growth: trace.constraintGrowth,
      agreement_rate: trace.agreementRate,
      evidence_diversity: trace.evidenceDiversity,
      structural_epistemic_drift: trace.structuralEpistemicDrift,
      drift_turn_mod_agent_count: trace.structuralEpistemicDrift === 1 ? trace.turnIndex % Math.max(1, summary.runConfig.agentCount) : null,
      raw_hash: trace.rawHash,
      expected_hash: trace.expectedHash,
      parse_error: trace.parseError ?? null
    };
  }

  return {
    run_id: trace.runId,
    profile: exportProfileId(trace.profile),
    condition: trace.condition,
    turn_index: trace.turnIndex,
    agent: trace.agent,
    agent_slot: trace.agentSlot,
    agent_model: trace.agentModel,
    input_bytes: trace.inputBytes,
    history_bytes: trace.historyBytes,
    output_bytes: trace.outputBytes,
    expected_bytes: trace.expectedBytes,
    injected_bytes_next: trace.injectedBytesNext,
    expected_step: trace.expectedStep,
    parsed_step: trace.parsedStep,
    parse_ok: trace.parseOk,
    state_ok: trace.stateOk,
    Pf: trace.pf,
    Cv: trace.cv,
    Ld: trace.ld,
    objective_failure: trace.objectiveFailure,
    objective_scope: objectiveScopeLabel(summary.profile),
    agent_in_objective_scope: isAgentInObjectiveScope(summary.profile, trace.agent) ? 1 : 0,
    uptime: trace.uptime,
    guardian_gate_state: trace.guardianGateState,
    guardian_structural_recommendation: trace.guardianStructuralRecommendation,
    guardian_reason_codes: trace.guardianReasonCodes,
    guardian_observe_error: trace.guardianObserveError,
    byteLength: trace.byteLength,
    lineCount: trace.lineCount,
    prefixLen: trace.prefixLen,
    suffixLen: trace.suffixLen,
    lenDeltaVsContract: trace.lenDeltaVsContract,
    deviationMagnitude: trace.deviationMagnitude,
    indentAvg: trace.indentAvg,
    indentMax: trace.indentMax,
    indentDelta: trace.indentDelta,
    b_transform_ok: trace.bTransformOk,
    b_transform_reason: trace.bTransformReason ?? null,
    rollingPf20: trace.rollingPf20,
    rollingDriftP95: trace.rollingDriftP95,
    dev_state: trace.devState,
    dev_threshold: DRIFT_DEV_EVENT_THRESHOLD,
    reasoning_depth: trace.reasoningDepth,
    authority_weights: trace.authorityWeights,
    contradiction_signal: trace.contradictionSignal,
    alternative_variance: trace.alternativeVariance,
    agreement_rate: trace.agreementRate,
    evidence_diversity: trace.evidenceDiversity,
    elapsed_time_ms: trace.elapsedTimeMs,
    commitment: trace.commitment,
    commitment_delta: trace.commitmentDelta,
    decision_value: trace.decisionValue,
    decision_error: trace.decisionError,
    constraint_growth: trace.constraintGrowth,
    evidence_delta: trace.evidenceDelta,
    depth_delta: trace.depthDelta,
    drift_rule_satisfied: trace.driftRuleSatisfied,
    drift_streak: trace.driftStreak,
    structural_epistemic_drift: trace.structuralEpistemicDrift,
    drift_turn_mod_agent_count: trace.structuralEpistemicDrift === 1 ? trace.turnIndex % Math.max(1, summary.runConfig.agentCount) : null,
    context_length: trace.contextLength,
    context_length_growth: trace.contextLengthGrowth,
    raw_hash: trace.rawHash,
    expected_hash: trace.expectedHash,
    parse_error: trace.parseError ?? null,
    parsed_data: trace.parsedData ?? null
  };
}

function traceToJsonl(summary: ConditionSummary): string {
  const lines = summary.traces.map((trace) => JSON.stringify(traceExportPayload(summary, trace)));
  return `${lines.join("\n")}\n`;
}

function exportableConditionSummary(summary: ConditionSummary): unknown {
  if (!IS_PUBLIC_SIGNAL_MODE) {
    return summary;
  }

  return {
    profile: exportProfileId(summary.profile),
    condition: summary.condition,
    objectiveMode: summary.objectiveMode,
    objectiveLabel: summary.objectiveLabel,
    objectiveScopeLabel: summary.objectiveScopeLabel,
    numberOfAgents: summary.runConfig.agentCount,
    startedAt: summary.startedAt,
    finishedAt: summary.finishedAt,
    turnsConfigured: summary.turnsConfigured,
    turnsAttempted: summary.turnsAttempted,
    failed: summary.failed,
    failureReason: summary.failureReason ?? null,
    parseOkRate: summary.parseOkRate,
    stateOkRate: summary.stateOkRate,
    cvRate: summary.cvRate,
    pfRate: summary.pfRate,
    ldRate: summary.ldRate,
    preflightPassed: summary.preflightPassed,
    decisionErrorLatest: summary.decisionErrorLatest,
    decisionErrorPeak: summary.decisionErrorPeak,
    decisionErrorSlope: summary.decisionErrorSlope,
    firstDecisionErrorTurn: summary.firstDecisionErrorTurn,
    propagationDetected: summary.propagationDetected,
    driftTurns: summary.driftTurns,
    driftTurnModuloAgentCount: summary.driftTurnModuloAgentCount,
    driftWindowStartTurns: summary.driftWindowStartTurns,
    driftWindowStartModuloAgentCount: summary.driftWindowStartModuloAgentCount,
    driftWindowCycleSynchronized: summary.driftWindowCycleSynchronized,
    driftWindowPeriodTurns: summary.driftWindowPeriodTurns,
    driftWindowRecursEveryCycle: summary.driftWindowRecursEveryCycle,
    structuralEpistemicDriftFlag: summary.structuralEpistemicDriftFlag,
    firstStructuralDriftTurn: summary.firstStructuralDriftTurn,
    lockInOnsetTurn: summary.lockInOnsetTurn,
    lockInScoreLatest: summary.lockInScoreLatest,
    lockInScorePeak: summary.lockInScorePeak,
    lockInPositiveStreakMax: summary.lockInPositiveStreakMax,
    cycleReinforcement3Latest: summary.cycleReinforcement3Latest,
    cycleReinforcement3Peak: summary.cycleReinforcement3Peak,
    trajectoryStabilityIndexLatest: summary.trajectoryStabilityIndexLatest,
    trajectoryStabilityIndexPeak: summary.trajectoryStabilityIndexPeak,
    trajectoryStatusLatest: summary.trajectoryStatusLatest,
    basinStateLatest: summary.basinStateLatest,
    firstBasinFormationTurn: summary.firstBasinFormationTurn,
    firstBasinStabilizationTurn: summary.firstBasinStabilizationTurn,
    beliefBasinDepth: summary.beliefBasinDepth,
    beliefBasinStrengthScore: summary.beliefBasinStrengthScore,
    beliefBasinStrengthBand: summary.beliefBasinStrengthBand,
    basinMetricInconsistencyWarning: summary.basinMetricInconsistencyWarning,
    observerTelemetryCoverage: guardianTriangleCoverage(summary),
    confidenceTrajectory: summary.traces.map((trace) => ({
      turn: trace.turnIndex,
      agent_slot: trace.agentSlot,
      confidence: trace.commitment
    })),
    decisionErrorTrajectory: summary.traces.map((trace) => ({
      turn: trace.turnIndex,
      agent_slot: trace.agentSlot,
      decision_error: trace.decisionError,
      decision_value: trace.decisionValue
    })),
    traces: summary.traces.map((trace) => traceExportPayload(summary, trace))
  };
}

function exportableResultsSnapshot(results: ResultsByProfile): unknown {
  if (!IS_PUBLIC_SIGNAL_MODE) {
    return results;
  }

  const exportResults: Record<string, { raw: unknown; sanitized: unknown }> = {};
  for (const profile of Object.keys(results) as ExperimentProfile[]) {
    const conditionResults = results[profile];
    exportResults[exportProfileId(profile)] = {
      raw: conditionResults.raw ? exportableConditionSummary(conditionResults.raw) : null,
      sanitized: conditionResults.sanitized ? exportableConditionSummary(conditionResults.sanitized) : null
    };
  }
  return exportResults;
}

function exportableMatrixRowsSnapshot(rows: MatrixTrialRow[]): unknown {
  if (!IS_PUBLIC_SIGNAL_MODE) {
    return rows;
  }

  return rows.map((row) => ({
    profile: exportProfileId(row.profile),
    model: row.model,
    replicate: row.replicate,
    closureDetected: row.closureDetected
  }));
}

function buildConditionSummary(params: {
  runConfig: RunConfig;
  condition: RepCondition;
  startedAt: string;
  traces: TurnTrace[];
  failed: boolean;
  failureReason?: string;
  finishedAt?: string;
}): ConditionSummary {
  const { runConfig, condition, startedAt, traces, failed, failureReason, finishedAt } = params;
  const turnsAttempted = traces.length;
  const tracesA = traces.filter((trace) => trace.agent === "A");
  const tracesB = traces.filter((trace) => trace.agent === "B");
  const tracesC = traces.filter((trace) => trace.agent === "C");

  const parseOkCount = traces.reduce((sum, trace) => sum + trace.parseOk, 0);
  const parseOkCountA = tracesA.reduce((sum, trace) => sum + trace.parseOk, 0);
  const parseOkCountB = tracesB.reduce((sum, trace) => sum + trace.parseOk, 0);
  const parseOkCountC = tracesC.reduce((sum, trace) => sum + trace.parseOk, 0);
  const stateOkCount = traces.reduce((sum, trace) => sum + trace.stateOk, 0);
  const stateOkCountA = tracesA.reduce((sum, trace) => sum + trace.stateOk, 0);
  const stateOkCountB = tracesB.reduce((sum, trace) => sum + trace.stateOk, 0);
  const stateOkCountC = tracesC.reduce((sum, trace) => sum + trace.stateOk, 0);
  const cvCount = traces.reduce((sum, trace) => sum + trace.cv, 0);
  const cvCountA = tracesA.reduce((sum, trace) => sum + trace.cv, 0);
  const cvCountB = tracesB.reduce((sum, trace) => sum + trace.cv, 0);
  const cvCountC = tracesC.reduce((sum, trace) => sum + trace.cv, 0);
  const pfCount = traces.reduce((sum, trace) => sum + trace.pf, 0);
  const pfCountA = tracesA.reduce((sum, trace) => sum + trace.pf, 0);
  const pfCountB = tracesB.reduce((sum, trace) => sum + trace.pf, 0);
  const pfCountC = tracesC.reduce((sum, trace) => sum + trace.pf, 0);
  const ldCount = traces.reduce((sum, trace) => sum + trace.ld, 0);
  const ldCountA = tracesA.reduce((sum, trace) => sum + trace.ld, 0);
  const ldCountB = tracesB.reduce((sum, trace) => sum + trace.ld, 0);
  const ldCountC = tracesC.reduce((sum, trace) => sum + trace.ld, 0);

  const objectiveScopeTraces = runConfig.profile === "drift_amplifying_loop" ? tracesA : traces;
  const ftfParse = firstFailureTurn(traces, "pf");
  const ftfLogic = firstFailureTurn(traces, "ld");
  const ftfStruct = firstFailureTurn(traces, "structuralEpistemicDrift");
  const ftfTotal = firstFailureTurn(objectiveScopeTraces, "objectiveFailure");
  const ftfParseA = firstFailureTurn(tracesA, "pf");
  const ftfLogicA = firstFailureTurn(tracesA, "ld");
  const ftfStructA = firstFailureTurn(tracesA, "structuralEpistemicDrift");
  const ftfTotalA = firstFailureTurn(tracesA, "objectiveFailure");
  const rollingReinf = runningReinforcementPoints(objectiveScopeTraces, ROLLING_REINFORCEMENT_WINDOW);
  const inflection = findPersistenceInflection(rollingReinf);
  const maxRollingReinforcementDelta = maxDelta(rollingReinf);
  const collapseLeadTurnsFromInflection =
    inflection && ftfTotal !== null && ftfTotal > inflection.turn ? ftfTotal - inflection.turn : null;

  const drift = driftTelemetry(traces);
  const driftA = driftTelemetry(tracesA);
  const templateEntropyA = shannonEntropy(tracesA.map((trace) => templateSignature(trace.outputBytes)));
  const bTransformSamples = tracesB.filter((trace) => trace.bTransformOk !== null).length;
  const bTransformOkCount = tracesB.reduce((sum, trace) => sum + (trace.bTransformOk ?? 0), 0);
  const bTransformOkRate = safeRate(bTransformOkCount, bTransformSamples);
  const edgeAB = edgeTransferStats(traces, "A", "B");
  const edgeBC = edgeTransferStats(traces, "B", "C");
  const edgeCA = edgeTransferStats(traces, "C", "A");
  const halfLife = artifactHalfLifeTurns(traces);
  const firstSuffixDriftTurn = traces.find((trace) => trace.suffixLen > 0)?.turnIndex ?? null;
  const maxSuffixLen = traces.length > 0 ? Math.max(...traces.map((trace) => trace.suffixLen)) : null;
  const suffixGrowthSlope = metricSlope(traces, (trace) => trace.suffixLen);
  const lineCountMax = traces.length > 0 ? Math.max(...traces.map((trace) => trace.lineCount)) : null;
  const consensus = consensusCollapseTelemetry(traces, runConfig.profile);
  const reasoningDepthValues = traces.map((trace) => trace.reasoningDepth).filter((value): value is number => value !== null);
  const alternativeVarianceValues = traces
    .map((trace) => trace.alternativeVariance)
    .filter((value): value is number => value !== null);
  const commitmentDeltaPositiveValues = traces
    .map((trace) => trace.commitmentDelta)
    .filter((value): value is number => value !== null && value > 0);
  const decisionErrorValues = traces.map((trace) => trace.decisionError).filter((value): value is number => value !== null);
  const decisionErrorLatest = traces.at(-1)?.decisionError ?? null;
  const decisionErrorPeak = decisionErrorValues.length > 0 ? Math.max(...decisionErrorValues) : null;
  const decisionErrorSlope =
    decisionErrorValues.length > 1
      ? metricSlope(
          traces.filter((trace): trace is TurnTrace & { decisionError: number } => trace.decisionError !== null),
          (trace) => trace.decisionError as number
        )
      : null;
  const firstDecisionErrorTurn = traces.find((trace) => trace.decisionError !== null && trace.decisionError > 0)?.turnIndex ?? null;
  const injectionDecisionError = traces.find((trace) => trace.turnIndex === LAB3_PERTURBATION_TURN)?.decisionError ?? null;
  const postInjectionDecisionErrorPeak = traces
    .filter((trace) => trace.turnIndex > LAB3_PERTURBATION_TURN)
    .map((trace) => trace.decisionError)
    .filter((value): value is number => value !== null)
    .reduce<number | null>((max, value) => (max === null ? value : Math.max(max, value)), null);
  const propagationDetected =
    isLab3PerturbationProfile(runConfig.profile)
      ? injectionDecisionError !== null && postInjectionDecisionErrorPeak !== null && postInjectionDecisionErrorPeak > injectionDecisionError + 0.001
        ? 1
        : 0
      : null;
  const constraintGrowthValues = traces
    .map((trace) => trace.constraintGrowth)
    .filter((value): value is number => value !== null);
  const avgReasoningDepth =
    reasoningDepthValues.length > 0 ? reasoningDepthValues.reduce((sum, value) => sum + value, 0) / reasoningDepthValues.length : null;
  const avgAlternativeVariance =
    alternativeVarianceValues.length > 0
      ? alternativeVarianceValues.reduce((sum, value) => sum + value, 0) / alternativeVarianceValues.length
      : null;
  const avgCommitmentDeltaPos =
    commitmentDeltaPositiveValues.length > 0
      ? commitmentDeltaPositiveValues.reduce((sum, value) => sum + value, 0) / commitmentDeltaPositiveValues.length
      : null;
  const constraintGrowthRate = safeRate(
    constraintGrowthValues.filter((value) => value > 0).length,
    constraintGrowthValues.length
  );
  const commitmentGrowthMass = commitmentDeltaPositiveValues.reduce((sum, value) => sum + value, 0);
  const constraintGrowthMass = constraintGrowthValues.reduce((sum, value) => sum + value, 0);
  const closureConstraintRatio =
    commitmentGrowthMass > 0 ? commitmentGrowthMass / Math.max(0.000001, constraintGrowthMass) : null;
  const structuralDriftStreakMax = traces.reduce((max, trace) => Math.max(max, trace.driftStreak), 0);
  const driftTurns = traces.filter((trace) => trace.structuralEpistemicDrift === 1).map((trace) => trace.turnIndex);
  const driftTurnModuloAgentCount = driftTurns.map((turn) => turn % Math.max(1, runConfig.agentCount));
  const driftWindowStartTurns = traces
    .filter((trace, index) => trace.structuralEpistemicDrift === 1 && (index === 0 || traces[index - 1].structuralEpistemicDrift === 0))
    .map((trace) => trace.turnIndex);
  const driftWindowStartModuloAgentCount = driftWindowStartTurns.map((turn) => turn % Math.max(1, runConfig.agentCount));
  const driftWindowCycleSynchronized =
    driftWindowStartModuloAgentCount.length >= 2
      ? new Set(driftWindowStartModuloAgentCount).size === 1
        ? 1
        : 0
      : null;
  const driftWindowIntervals = driftWindowStartTurns.slice(1).map((turn, index) => turn - driftWindowStartTurns[index]);
  const driftWindowPeriodTurns =
    driftWindowIntervals.length === 0
      ? null
      : driftWindowIntervals.every((interval) => interval === driftWindowIntervals[0])
        ? driftWindowIntervals[0]
        : null;
  const driftWindowRecursEveryCycle =
    driftWindowPeriodTurns === null ? null : driftWindowPeriodTurns === Math.max(1, runConfig.agentCount) ? 1 : 0;
  const firstStructuralDriftTurn = traces.find((trace) => trace.structuralEpistemicDrift === 1)?.turnIndex ?? null;
  const lockIn = computeLockInTelemetry(traces);
  const trajectory = computeTrajectoryUiTelemetry(traces, condition);
  const structuralEpistemicDriftFlag = firstStructuralDriftTurn !== null ? 1 : 0;
  const structuralEpistemicDriftReason =
    structuralEpistemicDriftFlag === 1
      ? isBeliefTriangle3AgentProfile(runConfig.profile)
        ? `commitment_delta>${TRIANGLE_DRIFT_COMMITMENT_DELTA_MIN.toFixed(2)} with evidence_delta=0 and depth_delta<=${TRIANGLE_DRIFT_DEPTH_EPSILON.toFixed(
            2
          )} for >=${STRUCTURAL_DRIFT_STREAK_MIN} turns`
        : `commitment_delta>${STRUCTURAL_DRIFT_COMMITMENT_DELTA_MIN.toFixed(2)} with evidence_delta=0 and depth_delta=0 for >=${STRUCTURAL_DRIFT_STREAK_MIN} turns`
      : null;
  const daiPoints = computeDaiPoints(traces);
  const daiByTurn = new Map<number, DaiPoint>(daiPoints.map((point) => [point.turnIndex, point]));
  const tracesWithDai = traces.map((trace) => {
    const point = daiByTurn.get(trace.turnIndex);
    return {
      ...trace,
      dai: point?.dai ?? null,
      daiDelta: point?.daiDelta ?? null,
      daiRegime: point?.regime ?? null
    };
  });
  const daiValues = daiPoints.map((point) => point.dai).filter((value): value is number => value !== null);
  const daiLatest = daiPoints.at(-1)?.dai ?? null;
  const daiDeltaLatest = daiPoints.at(-1)?.daiDelta ?? null;
  const daiPeak = daiValues.length > 0 ? Math.max(...daiValues) : null;
  const daiRegimeLatest = daiRegime(daiLatest);
  const daiFirstAttractorTurn = daiPoints.find((point) => point.dai !== null && point.dai >= 0.2)?.turnIndex ?? null;
  const daiFirstDriftTurn = daiPoints.find((point) => point.dai !== null && point.dai >= 0.5)?.turnIndex ?? null;
  const daiFirstAmplificationTurn = daiPoints.find((point) => point.dai !== null && point.dai >= 0.8)?.turnIndex ?? null;
  const daiPositiveSlopeStreakMax = maxPositiveDaiSlopeStreak(daiPoints);
  const daiSlopeValue = daiSlope(daiPoints);

  const pairComparisons = Math.max(0, traces.length - 1);
  let prevOutputToNextInputMatches = 0;
  let prevInjectedToNextInputMatches = 0;
  for (let index = 1; index < traces.length; index += 1) {
    const previous = traces[index - 1];
    const current = traces[index];
    if (current.inputBytes === previous.outputBytes) {
      prevOutputToNextInputMatches += 1;
    }
    if (current.inputBytes === previous.injectedBytesNext) {
      prevInjectedToNextInputMatches += 1;
    }
  }
  const prevOutputToNextInputRate = safeRate(prevOutputToNextInputMatches, pairComparisons);
  const prevInjectedToNextInputRate = safeRate(prevInjectedToNextInputMatches, pairComparisons);
  const preflightAgentTraces = traces.filter((trace) => trace.agent === runConfig.preflightAgent);
  const preflightTurnsAvailable = preflightAgentTraces.length;
  const preflightTurnsRequired = Math.min(runConfig.preflightTurns, runConfig.horizon);
  const preflightEvaluated = runConfig.preflightEnabled && preflightTurnsAvailable >= Math.ceil(preflightTurnsRequired / 2);
  let preflightPassed: boolean | null = null;
  let preflightReason: string | null = null;
  if (preflightEvaluated) {
    const preflightSampleCount = Math.ceil(preflightTurnsRequired / 2);
    const preflightParseRate = safeRate(
      preflightAgentTraces.slice(0, preflightSampleCount).reduce((sum, trace) => sum + trace.parseOk, 0),
      preflightSampleCount
    );
    const preflightStateRate = safeRate(
      preflightAgentTraces.slice(0, preflightSampleCount).reduce((sum, trace) => sum + trace.stateOk, 0),
      preflightSampleCount
    );
    const gate = preflightGateStatus({
      objectiveMode: runConfig.objectiveMode,
      parseRate: preflightParseRate,
      stateRate: preflightStateRate,
      parseMin: runConfig.preflightParseOkMin,
      stateMin: runConfig.preflightStateOkMin
    });
    preflightPassed = gate.pass;
    preflightReason = preflightPassed
      ? `Preflight passed for Agent ${runConfig.preflightAgent}.`
      : gate.requiresState
        ? `Preflight rejected Agent ${runConfig.preflightAgent}: ParseOK ${asPercent(preflightParseRate)} / StateOK ${asPercent(
            preflightStateRate
          )} (required ${asPercent(runConfig.preflightParseOkMin)} / ${asPercent(runConfig.preflightStateOkMin)}).`
        : `Preflight rejected Agent ${runConfig.preflightAgent}: ParseOK ${asPercent(preflightParseRate)} (required ${asPercent(
            runConfig.preflightParseOkMin
          )}, parse-only objective).`;
  }

  const guardianObservedCount = traces.filter(
    (trace) =>
      trace.guardianGateState !== null ||
      trace.guardianStructuralRecommendation !== null ||
      trace.guardianReasonCodes.length > 0 ||
      trace.guardianObserveError !== null
  ).length;
  const guardianPauseCount = traces.filter((trace) => trace.guardianGateState === "PAUSE").length;
  const guardianYieldCount = traces.filter((trace) => trace.guardianGateState === "YIELD").length;
  const guardianContinueCount = traces.filter((trace) => trace.guardianGateState === "CONTINUE").length;
  const guardianReopenCount = traces.filter((trace) => trace.guardianStructuralRecommendation === "REOPEN").length;
  const guardianSlowCount = traces.filter((trace) => trace.guardianStructuralRecommendation === "SLOW").length;
  const guardianDeferCount = traces.filter((trace) => trace.guardianStructuralRecommendation === "DEFER").length;
  const guardianObserveErrorCount = traces.filter((trace) => trace.guardianObserveError !== null).length;
  const guardianObservationBase = turnsAttempted;

  return {
    runConfig,
    profile: runConfig.profile,
    condition,
    objectiveMode: runConfig.objectiveMode,
    objectiveLabel: objectiveLabel(runConfig.objectiveMode),
    objectiveScopeLabel: objectiveScopeLabel(runConfig.profile),
    startedAt,
    finishedAt: finishedAt ?? new Date().toISOString(),
    turnsConfigured: runConfig.horizon,
    turnsAttempted,
    failed,
    failureReason,
    parseOkRate: safeRate(parseOkCount, turnsAttempted),
    parseOkRateA: safeRate(parseOkCountA, tracesA.length),
    parseOkRateB: safeRate(parseOkCountB, tracesB.length),
    parseOkRateC: safeRate(parseOkCountC, tracesC.length),
    stateOkRate: safeRate(stateOkCount, turnsAttempted),
    stateOkRateA: safeRate(stateOkCountA, tracesA.length),
    stateOkRateB: safeRate(stateOkCountB, tracesB.length),
    stateOkRateC: safeRate(stateOkCountC, tracesC.length),
    cvRate: safeRate(cvCount, turnsAttempted),
    cvRateA: safeRate(cvCountA, tracesA.length),
    cvRateB: safeRate(cvCountB, tracesB.length),
    cvRateC: safeRate(cvCountC, tracesC.length),
    pfRate: safeRate(pfCount, turnsAttempted),
    pfRateA: safeRate(pfCountA, tracesA.length),
    pfRateB: safeRate(pfCountB, tracesB.length),
    pfRateC: safeRate(pfCountC, tracesC.length),
    ldRate: safeRate(ldCount, turnsAttempted),
    ldRateA: safeRate(ldCountA, tracesA.length),
    ldRateB: safeRate(ldCountB, tracesB.length),
    ldRateC: safeRate(ldCountC, tracesC.length),
    contextGrowthAvg: drift.contextGrowthAvg,
    contextGrowthMax: drift.contextGrowthMax,
    contextGrowthSlope: drift.contextGrowthSlope,
    driftAvg: drift.driftAvg,
    driftP95: drift.driftP95,
    driftMax: drift.driftMax,
    escalationSlope: drift.escalationSlope,
    earlySlope40: drift.earlySlope40,
    indentAvg: drift.indentAvg,
    indentMax: drift.indentMax,
    indentDeltaAvg: drift.indentDeltaAvg,
    driftAvgA: driftA.driftAvg,
    driftP95A: driftA.driftP95,
    driftMaxA: driftA.driftMax,
    escalationSlopeA: driftA.escalationSlope,
    earlySlope40A: driftA.earlySlope40,
    indentAvgA: driftA.indentAvg,
    indentMaxA: driftA.indentMax,
    indentDeltaAvgA: driftA.indentDeltaAvg,
    bTransformOkRate,
    bTransformSamples,
    consensusPairs: consensus.consensusPairs,
    agreementRateAB: consensus.agreementRateAB,
    evidenceDiversity: consensus.evidenceDiversity,
    unsupportedConsensusRate: consensus.unsupportedConsensusRate,
    unsupportedConsensusStreakMax: consensus.unsupportedConsensusStreakMax,
    noNewEvidenceRate: consensus.noNewEvidenceRate,
    evidenceGrowthRate: consensus.evidenceGrowthRate,
    confidenceGainAvg: consensus.confidenceGainAvg,
    decisionErrorLatest,
    decisionErrorPeak,
    decisionErrorSlope,
    firstDecisionErrorTurn,
    propagationDetected,
    driftTurns,
    driftTurnModuloAgentCount,
    driftWindowStartTurns,
    driftWindowStartModuloAgentCount,
    driftWindowCycleSynchronized,
    driftWindowPeriodTurns,
    driftWindowRecursEveryCycle,
    avgReasoningDepth,
    avgAlternativeVariance,
    avgCommitmentDeltaPos,
    constraintGrowthRate,
    closureConstraintRatio,
    structuralDriftStreakMax,
    firstStructuralDriftTurn,
    lockInOnsetTurn: lockIn.onsetTurn,
    lockInScoreLatest: lockIn.scoreLatest,
    lockInScorePeak: lockIn.scorePeak,
    lockInPositiveStreakMax: lockIn.positiveStreakMax,
    cycleReinforcement3Latest: trajectory.cycleReinforcement3Latest,
    cycleReinforcement3Peak: trajectory.cycleReinforcement3Peak,
    trajectoryStabilityIndexLatest: trajectory.tsiLatest,
    trajectoryStabilityIndexPeak: trajectory.tsiPeak,
    trajectoryStatusLatest: trajectory.statusLatest,
    basinStateLatest: trajectory.basinStateLatest,
    firstBasinFormationTurn: trajectory.firstBasinFormationTurn,
    firstBasinStabilizationTurn: trajectory.firstBasinStabilizationTurn,
    beliefBasinDepth: trajectory.beliefBasinDepth,
    beliefBasinStrengthScore: trajectory.beliefBasinStrengthScore,
    beliefBasinStrengthBand: trajectory.beliefBasinStrengthBand,
    basinMetricInconsistencyWarning: trajectory.basinMetricInconsistencyWarning,
    structuralEpistemicDriftFlag,
    structuralEpistemicDriftReason,
    daiLatest,
    daiDeltaLatest,
    daiPeak,
    daiSlope: daiSlopeValue,
    daiRegimeLatest,
    daiFirstAttractorTurn,
    daiFirstDriftTurn,
    daiFirstAmplificationTurn,
    daiPositiveSlopeStreakMax,
    lagTransferABDevGivenPrevDev: edgeAB.pDevGivenDev,
    lagTransferABDevGivenPrevClean: edgeAB.pDevGivenClean,
    lagTransferABDelta: edgeAB.delta,
    artifactHalfLifeTurns: halfLife,
    consensusCollapseFlag: structuralEpistemicDriftFlag,
    consensusCollapseReason: structuralEpistemicDriftReason,
    artifactPersistenceA: driftA.artifactPersistence,
    templateEntropyA,
    artifactPersistence: drift.artifactPersistence,
    persistenceRate: drift.persistenceRate,
    reinforcementWhenDev: drift.reinforcementWhenDev,
    reinforcementWhenClean: drift.reinforcementWhenClean,
    reinforcementDelta: drift.reinforcementDelta,
    reinforcementWhenDevA: drift.reinforcementWhenDevA,
    reinforcementWhenCleanA: drift.reinforcementWhenCleanA,
    reinforcementDeltaA: drift.reinforcementDeltaA,
    reinforcementWhenDevB: drift.reinforcementWhenDevB,
    reinforcementWhenCleanB: drift.reinforcementWhenCleanB,
    reinforcementDeltaB: drift.reinforcementDeltaB,
    reinforcementWhenDevC: drift.reinforcementWhenDevC,
    reinforcementWhenCleanC: drift.reinforcementWhenCleanC,
    reinforcementDeltaC: drift.reinforcementDeltaC,
    edgeAB,
    edgeBC,
    edgeCA,
    prevOutputToNextInputRate,
    prevInjectedToNextInputRate,
    firstSuffixDriftTurn,
    maxSuffixLen,
    suffixGrowthSlope,
    lineCountMax,
    ftfParse,
    ftfLogic,
    ftfStruct,
    ftfTotal,
    ftfParseA,
    ftfLogicA,
    ftfStructA,
    ftfTotalA,
    preflightPassed,
    preflightReason,
    maxRollingReinforcementDelta,
    persistenceInflectionTurn: inflection?.turn ?? null,
    persistenceInflectionDelta: inflection?.delta ?? null,
    collapseLeadTurnsFromInflection,
    guardianObserveCoverage: safeRate(guardianObservedCount, guardianObservationBase),
    guardianPauseRate: safeRate(guardianPauseCount, guardianObservationBase),
    guardianYieldRate: safeRate(guardianYieldCount, guardianObservationBase),
    guardianContinueRate: safeRate(guardianContinueCount, guardianObservationBase),
    guardianReopenRate: safeRate(guardianReopenCount, guardianObservationBase),
    guardianSlowRate: safeRate(guardianSlowCount, guardianObservationBase),
    guardianDeferRate: safeRate(guardianDeferCount, guardianObservationBase),
    guardianObserveErrorRate: safeRate(guardianObserveErrorCount, guardianObservationBase),
    phaseTransition: detectPhaseTransition(traces),
    traces: tracesWithDai
  };
}

function evaluateSmokingGun(raw: ConditionSummary | null, sanitized: ConditionSummary | null): ObjectiveEval | null {
  if (!raw || !sanitized) return null;
  const reinforcementDelta = raw.reinforcementDeltaA ?? null;
  const rawP95 = raw.driftP95A;
  const sanP95 = sanitized.driftP95A;

  let driftRatio: number | null = null;
  if (rawP95 !== null && sanP95 !== null) {
    if (sanP95 === 0) {
      driftRatio = rawP95 > 0 ? Number.POSITIVE_INFINITY : 1;
    } else {
      driftRatio = rawP95 / sanP95;
    }
  }

  const cvRateRawA = raw.cvRateA;
  const cvRateSanitizedA = sanitized.cvRateA;
  const ftfStructRawA = raw.ftfStructA;
  const ftfStructSanitizedA = sanitized.ftfStructA;
  const cvDeltaA =
    cvRateRawA !== null && cvRateSanitizedA !== null ? Math.max(0, cvRateRawA - cvRateSanitizedA) : null;
  const spi =
    reinforcementDelta !== null && cvDeltaA !== null ? Math.max(0, reinforcementDelta) * cvDeltaA : null;
  const structuralGateSeparated =
    ((cvRateRawA ?? 0) > (cvRateSanitizedA ?? 0)) ||
    (ftfStructRawA !== null && (ftfStructSanitizedA === null || ftfStructRawA < ftfStructSanitizedA));

  const pass =
    reinforcementDelta !== null &&
    reinforcementDelta > STRUCTURAL_GUARDRAIL.reinforcementDeltaMin &&
    driftRatio !== null &&
    driftRatio >= STRUCTURAL_GUARDRAIL.driftP95RatioMin &&
    (raw.parseOkRateA ?? raw.parseOkRate ?? 0) >= STRUCTURAL_GUARDRAIL.parseOkMin &&
    (raw.stateOkRateA ?? raw.stateOkRate ?? 0) >= STRUCTURAL_GUARDRAIL.stateOkMin &&
    (sanitized.parseOkRateA ?? sanitized.parseOkRate ?? 0) >= STRUCTURAL_GUARDRAIL.parseOkMin &&
    (sanitized.stateOkRateA ?? sanitized.stateOkRate ?? 0) >= STRUCTURAL_GUARDRAIL.stateOkMin &&
    structuralGateSeparated;

  return {
    pass,
    driftRatio,
    reinforcementDelta,
    spi,
    cvRateRawA,
    cvRateSanitizedA,
    ftfStructRawA,
    ftfStructSanitizedA,
    structuralGateSeparated
  };
}

function evaluateConsensusCollapse(raw: ConditionSummary | null, sanitized: ConditionSummary | null): ConsensusEval | null {
  if (!raw || !sanitized) return null;
  const gapStats = windowedDevGapStats(raw.traces, sanitized.traces, WINDOW_GAP_TURNS);
  const lagTransferGap =
    raw.lagTransferABDelta !== null && sanitized.lagTransferABDelta !== null ? raw.lagTransferABDelta - sanitized.lagTransferABDelta : null;
  const halfLifeGap =
    raw.artifactHalfLifeTurns !== null && sanitized.artifactHalfLifeTurns !== null
      ? raw.artifactHalfLifeTurns - sanitized.artifactHalfLifeTurns
      : null;

  const rawSignal =
    raw.structuralEpistemicDriftFlag === 1 &&
    (raw.parseOkRate ?? 0) >= STRUCTURAL_GUARDRAIL.parseOkMin &&
    (raw.stateOkRate ?? 0) >= STRUCTURAL_GUARDRAIL.stateOkMin;

  const sanitizedSignal =
    sanitized.structuralEpistemicDriftFlag === 1 &&
    (sanitized.parseOkRate ?? 0) >= STRUCTURAL_GUARDRAIL.parseOkMin &&
    (sanitized.stateOkRate ?? 0) >= STRUCTURAL_GUARDRAIL.stateOkMin;

  return {
    pass: rawSignal && !sanitizedSignal,
    rawSignal,
    sanitizedSignal,
    rawAgreement: raw.agreementRateAB,
    rawDiversity: raw.evidenceDiversity,
    rawNoNewEvidence: raw.noNewEvidenceRate,
    rawPairs: raw.consensusPairs,
    sanitizedAgreement: sanitized.agreementRateAB,
    sanitizedDiversity: sanitized.evidenceDiversity,
    sanitizedNoNewEvidence: sanitized.noNewEvidenceRate,
    sanitizedPairs: sanitized.consensusPairs,
    windowGapTurns: WINDOW_GAP_TURNS,
    devGapWindowMean: gapStats.meanGap,
    devGapWindowMax: gapStats.maxGap,
    lagTransferGap,
    halfLifeGap,
    rawFirstStructuralDriftTurn: raw.firstStructuralDriftTurn,
    sanitizedFirstStructuralDriftTurn: sanitized.firstStructuralDriftTurn,
    rawStructuralDriftStreakMax: raw.structuralDriftStreakMax,
    sanitizedStructuralDriftStreakMax: sanitized.structuralDriftStreakMax,
    rawClosureConstraintRatio: raw.closureConstraintRatio,
    sanitizedClosureConstraintRatio: sanitized.closureConstraintRatio,
    rawConstraintGrowthRate: raw.constraintGrowthRate,
    sanitizedConstraintGrowthRate: sanitized.constraintGrowthRate,
    rawLockInOnsetTurn: raw.lockInOnsetTurn,
    sanitizedLockInOnsetTurn: sanitized.lockInOnsetTurn,
    rawLockInScoreLatest: raw.lockInScoreLatest,
    sanitizedLockInScoreLatest: sanitized.lockInScoreLatest,
    rawLockInScorePeak: raw.lockInScorePeak,
    sanitizedLockInScorePeak: sanitized.lockInScorePeak,
    rawCycleReinforcement3Latest: raw.cycleReinforcement3Latest,
    sanitizedCycleReinforcement3Latest: sanitized.cycleReinforcement3Latest,
    rawCycleReinforcement3Peak: raw.cycleReinforcement3Peak,
    sanitizedCycleReinforcement3Peak: sanitized.cycleReinforcement3Peak,
    rawTrajectoryStabilityIndexLatest: raw.trajectoryStabilityIndexLatest,
    sanitizedTrajectoryStabilityIndexLatest: sanitized.trajectoryStabilityIndexLatest,
    rawTrajectoryStabilityIndexPeak: raw.trajectoryStabilityIndexPeak,
    sanitizedTrajectoryStabilityIndexPeak: sanitized.trajectoryStabilityIndexPeak,
    rawTrajectoryStatusLatest: raw.trajectoryStatusLatest,
    sanitizedTrajectoryStatusLatest: sanitized.trajectoryStatusLatest,
    rawBasinStateLatest: raw.basinStateLatest,
    sanitizedBasinStateLatest: sanitized.basinStateLatest,
    rawFirstBasinFormationTurn: raw.firstBasinFormationTurn,
    sanitizedFirstBasinFormationTurn: sanitized.firstBasinFormationTurn,
    rawFirstBasinStabilizationTurn: raw.firstBasinStabilizationTurn,
    sanitizedFirstBasinStabilizationTurn: sanitized.firstBasinStabilizationTurn,
    rawBeliefBasinDepth: raw.beliefBasinDepth,
    sanitizedBeliefBasinDepth: sanitized.beliefBasinDepth,
    rawBeliefBasinStrengthScore: raw.beliefBasinStrengthScore,
    sanitizedBeliefBasinStrengthScore: sanitized.beliefBasinStrengthScore,
    rawBeliefBasinStrengthBand: raw.beliefBasinStrengthBand,
    sanitizedBeliefBasinStrengthBand: sanitized.beliefBasinStrengthBand,
    rawBasinMetricInconsistencyWarning: raw.basinMetricInconsistencyWarning,
    sanitizedBasinMetricInconsistencyWarning: sanitized.basinMetricInconsistencyWarning,
    rawDaiLatest: raw.daiLatest,
    sanitizedDaiLatest: sanitized.daiLatest,
    rawDaiDeltaLatest: raw.daiDeltaLatest,
    sanitizedDaiDeltaLatest: sanitized.daiDeltaLatest,
    rawDaiSlope: raw.daiSlope,
    sanitizedDaiSlope: sanitized.daiSlope,
    rawDaiRegime: raw.daiRegimeLatest,
    sanitizedDaiRegime: sanitized.daiRegimeLatest
  };
}

function closureVerdict(evalResult: ConsensusEval | null): ClosureVerdict {
  if (!evalResult) {
    return {
      label: "INCOMPLETE",
      tone: "warn",
      detail: "Run both RAW and SANITIZED to compute a structural epistemic drift verdict."
    };
  }

  if (evalResult.rawSignal && !evalResult.sanitizedSignal) {
    return {
      label: "DETECTED (ISOLATED)",
      tone: "good",
      detail: "Structural epistemic drift appears in RAW but not in SANITIZED."
    };
  }

  if (!evalResult.rawSignal && !evalResult.sanitizedSignal) {
    return {
      label: "NOT DETECTED",
      tone: "warn",
      detail: "No structural epistemic drift signal in either condition for this run."
    };
  }

  if (evalResult.rawSignal && evalResult.sanitizedSignal) {
    return {
      label: "NOT ISOLATED",
      tone: "bad",
      detail: "Drift-like signal appears in both conditions, so RAW-specific effect is not isolated."
    };
  }

  return {
    label: "INCONSISTENT",
    tone: "bad",
    detail: "SANITIZED signaled without RAW; rerun and inspect traces for setup artifacts."
  };
}

function structuralPatternInterpretation(evalResult: ConsensusEval | null): ClosureVerdict {
  if (!evalResult) {
    return {
      label: "INCOMPLETE",
      tone: "warn",
      detail: "Run both RAW and SANITIZED to classify the run using the structural drift criterion."
    };
  }

  if (evalResult.rawSignal) {
    if (!evalResult.sanitizedSignal) {
      return {
        label: "Structural drift (isolated)",
        tone: "good",
        detail:
          evalResult.rawLockInOnsetTurn !== null
            ? `RAW shows isolated structural closure signal with lock-in onset at turn ${evalResult.rawLockInOnsetTurn}; SANITIZED does not.`
            : "RAW shows isolated structural closure signal; SANITIZED does not."
      };
    }
    return {
      label: "Structural drift (not isolated)",
      tone: "bad",
      detail: "Structural closure signal appears in both RAW and SANITIZED, so the effect is not isolated."
    };
  }

  return {
    label: "No structural drift",
    tone: "warn",
    detail:
      evalResult.rawCycleReinforcement3Peak !== null &&
      evalResult.rawCycleReinforcement3Peak > LOCK_IN_CYCLE_REINFORCEMENT_THRESHOLD
        ? "No persistent structural closure signal detected; lock-in pressure appeared but did not sustain."
        : "No structural closure signal detected; behavior remains within structural bounds."
  };
}

function average(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function hasGuardianTriangleTelemetry(trace: TurnTrace): boolean {
  return (
    trace.guardianGateState !== null ||
    trace.guardianStructuralRecommendation !== null ||
    trace.guardianReasonCodes.length > 0 ||
    trace.guardianObserveError !== null
  );
}

function guardianTriangleCoverage(summary: ConditionSummary): number | null {
  if (!summary.traces.length) return null;
  const observed = summary.traces.filter((trace) => hasGuardianTriangleTelemetry(trace)).length;
  return observed / summary.traces.length;
}

function parseModelMatrixInput(input: string, fallbackModel: string): string[] {
  const parsed = input
    .split(/[\n,]/)
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
  const deduped = Array.from(new Set(parsed));
  if (deduped.length > 0) return deduped;
  return [fallbackModel];
}

function aggregateMatrixRows(rows: MatrixTrialRow[]): MatrixAggregateRow[] {
  const byModel = new Map<string, MatrixTrialRow[]>();
  for (const row of rows) {
    const list = byModel.get(row.model) ?? [];
    list.push(row);
    byModel.set(row.model, list);
  }

  const aggregates: MatrixAggregateRow[] = [];
  for (const [modelName, modelRows] of byModel.entries()) {
    const closureValues = modelRows.map((row) => row.closureDetected).filter((value): value is number => value !== null);
    const lagValues = modelRows.map((row) => row.lagTransferGap).filter((value): value is number => value !== null);
    const halfLifeValues = modelRows.map((row) => row.halfLifeGap).filter((value): value is number => value !== null);
    const meanGapValues = modelRows.map((row) => row.devGapWindowMean).filter((value): value is number => value !== null);
    const maxGapValues = modelRows.map((row) => row.devGapWindowMax).filter((value): value is number => value !== null);

    aggregates.push({
      model: modelName,
      trials: modelRows.length,
      closureDetectedRate: average(closureValues),
      lagTransferGapAvg: average(lagValues),
      halfLifeGapAvg: average(halfLifeValues),
      devGapWindowMeanAvg: average(meanGapValues),
      devGapWindowMaxAvg: average(maxGapValues)
    });
  }

  return aggregates.sort((a, b) => a.model.localeCompare(b.model));
}

function buildConditionMarkdown(summary: ConditionSummary): string {
  const triangleCoverage = guardianTriangleCoverage(summary);
  const observerStatus = triangleCoverage !== null && triangleCoverage > 0 ? "available" : "n/a";
  const cycle3Map = cycleReinforcement3ByTurn(summary.traces);
  const basinStateMap = basinStateByTurn(summary.traces);

  if (IS_PUBLIC_SIGNAL_MODE) {
    return [
      `### ${PROFILE_LABELS[summary.profile]} — ${CONDITION_LABELS[summary.condition]}`,
      `- Objective mode: ${OBJECTIVE_MODE_LABELS[summary.objectiveMode]} (${summary.objectiveLabel})`,
      `- Objective scope: ${summary.objectiveScopeLabel}`,
      `- Turns attempted: ${summary.turnsAttempted}/${summary.turnsConfigured}`,
      `- Agents in cycle: ${summary.runConfig.agentCount}`,
      `- ParseOK rate (all): ${asPercent(summary.parseOkRate)}`,
      `- StateOK rate (all): ${asPercent(summary.stateOkRate)}`,
      `- Preflight gate: ${summary.preflightPassed === null ? "not evaluated" : summary.preflightPassed ? "PASS" : "FAIL"}`,
      isBeliefLoopProfile(summary.profile)
        ? `- Structural epistemic drift signal: ${summary.consensusCollapseFlag ? "YES" : "NO"}`
        : "",
      isBeliefLoopProfile(summary.profile) ? `- First structural drift turn: ${summary.firstStructuralDriftTurn ?? "N/A"}` : "",
      isBeliefLoopProfile(summary.profile)
        ? `- Lock-in onset turn: ${lockInOnsetDisplay(summary.lockInOnsetTurn, summary.lockInScorePeak)}`
        : "",
      isBeliefLoopProfile(summary.profile)
        ? `- Lock-in score latest/peak: ${asFixed(summary.lockInScoreLatest, 4)} / ${asFixed(summary.lockInScorePeak, 4)}`
        : "",
      isLab3PerturbationProfile(summary.profile)
        ? `- decision_error latest/peak/slope: ${asFixed(summary.decisionErrorLatest, 4)} / ${asFixed(summary.decisionErrorPeak, 4)} / ${asFixed(
            summary.decisionErrorSlope,
            5
          )} (first non-zero turn ${summary.firstDecisionErrorTurn ?? "N/A"})`
        : "",
      isLab3PerturbationProfile(summary.profile)
        ? `- Propagation: ${summary.propagationDetected === null ? "N/A" : summary.propagationDetected ? "YES" : "NO"}`
        : "",
      isBeliefLoopProfile(summary.profile)
        ? `- drift_turn % agent_count (${summary.runConfig.agentCount}): windows [${formatTurnList(summary.driftWindowStartTurns)}] -> mod [${formatTurnList(
            summary.driftWindowStartModuloAgentCount
          )}] | synchronized=${summary.driftWindowCycleSynchronized === null ? "n/a" : summary.driftWindowCycleSynchronized ? "YES" : "NO"} | period=${
            summary.driftWindowPeriodTurns ?? "N/A"
          } | every_cycle=${summary.driftWindowRecursEveryCycle === null ? "n/a" : summary.driftWindowRecursEveryCycle ? "YES" : "NO"}`
        : "",
      isBeliefLoopProfile(summary.profile)
        ? `- Cycle Reinforcement (3-turn) latest/peak: ${asFixed(summary.cycleReinforcement3Latest, 4)} / ${asFixed(summary.cycleReinforcement3Peak, 4)}`
        : "",
      isBeliefLoopProfile(summary.profile)
        ? `- Trajectory Stability Index (latest/peak): ${asFixed(summary.trajectoryStabilityIndexLatest, 4)} / ${asFixed(summary.trajectoryStabilityIndexPeak, 4)}`
        : "",
      isBeliefLoopProfile(summary.profile) ? `- Trajectory Dynamics (latest): ${trajectoryDynamicsLabel(trajectoryDynamicsFromSummary(summary))}` : "",
      isBeliefLoopProfile(summary.profile) ? `- Basin State (latest): ${basinStateLabel(summary.basinStateLatest)}` : "",
      isBeliefLoopProfile(summary.profile)
        ? `- Belief Basin Strength: ${summary.beliefBasinStrengthBand ?? "n/a"} (depth ${asFixed(summary.beliefBasinDepth, 4)}, score ${asFixed(
            summary.beliefBasinStrengthScore,
            4
          )})`
        : "",
      isBeliefLoopProfile(summary.profile) && summary.basinMetricInconsistencyWarning === 1
        ? "- Basin metric consistency warning: YES"
        : "",
      isBeliefLoopProfile(summary.profile) && summary.basinMetricInconsistencyWarning === 1
        ? "- Basin metric consistency warning: YES (strength exceeds forming while drift signal is absent)."
        : "",
      `- Observer telemetry coverage: ${asPercent(triangleCoverage)}`,
      `- Observer status (latest): ${observerStatus}`,
      `- Cv/Pf/Ld rate (all): ${asPercent(summary.cvRate)} / ${asPercent(summary.pfRate)} / ${asPercent(summary.ldRate)}${cvDiagnosticNoteForObjective(summary.objectiveMode)}`,
      `- FTF_total/parse/logic/struct: ${summary.ftfTotal ?? "N/A"} / ${summary.ftfParse ?? "N/A"} / ${summary.ftfLogic ?? "N/A"} / ${
        summary.ftfStruct ?? "N/A"
      }`,
      "",
      "| Turn | Agent | ParseOK | StateOK | Cv | Pf | Ld | Lock-in | CycleReinf(3) | BasinState | DriftFlag |",
      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
      ...summary.traces.slice(0, 30).map((trace) => {
        const lockInScore =
          trace.commitmentDelta !== null && trace.constraintGrowth !== null ? trace.commitmentDelta - trace.constraintGrowth : null;
        const cycle3 = cycle3Map.get(trace.turnIndex) ?? null;
        const basinState = basinStateMap.get(trace.turnIndex) ?? null;
        return `| ${trace.turnIndex} | ${traceAgentDisplay(trace)} | ${trace.parseOk} | ${trace.stateOk} | ${trace.cv} | ${trace.pf} | ${trace.ld} | ${asFixed(
          lockInScore,
          4
        )} | ${asFixed(cycle3, 4)} | ${basinStateLabel(basinState)} | ${trace.structuralEpistemicDrift} |`;
      })
    ]
      .filter((line) => line.length > 0)
      .join("\n");
  }

  return [
    `### ${PROFILE_LABELS[summary.profile]} — ${CONDITION_LABELS[summary.condition]}`,
    `- Objective mode: ${OBJECTIVE_MODE_LABELS[summary.objectiveMode]} (${summary.objectiveLabel})`,
    `- Objective scope: ${summary.objectiveScopeLabel}`,
    `- Turns attempted: ${summary.turnsAttempted}/${summary.turnsConfigured}`,
    `- Agents in cycle: ${summary.runConfig.agentCount}`,
    isBeliefTriangle3AgentProfile(summary.profile)
      ? `- ParseOK rate (all/A/B/C): ${asPercent(summary.parseOkRate)} / ${asPercent(summary.parseOkRateA)} / ${asPercent(summary.parseOkRateB)} / ${asPercent(summary.parseOkRateC)}`
      : `- ParseOK rate (all/A/B): ${asPercent(summary.parseOkRate)} / ${asPercent(summary.parseOkRateA)} / ${asPercent(summary.parseOkRateB)}`,
    isBeliefTriangle3AgentProfile(summary.profile)
      ? `- StateOK rate (all/A/B/C): ${asPercent(summary.stateOkRate)} / ${asPercent(summary.stateOkRateA)} / ${asPercent(summary.stateOkRateB)} / ${asPercent(summary.stateOkRateC)}`
      : `- StateOK rate (all/A/B): ${asPercent(summary.stateOkRate)} / ${asPercent(summary.stateOkRateA)} / ${asPercent(summary.stateOkRateB)}`,
    isBeliefLoopProfile(summary.profile)
      ? `- Agreement ${agreementWindowLabel(summary.profile)}: ${asPercent(summary.agreementRateAB)} (pairs=${summary.consensusPairs})`
      : "",
    isBeliefLoopProfile(summary.profile) ? `- Evidence diversity: ${asFixed(summary.evidenceDiversity, 3)}` : "",
    isBeliefLoopProfile(summary.profile)
      ? `- Unsupported consensus rate/streak: ${asPercent(summary.unsupportedConsensusRate)} / ${summary.unsupportedConsensusStreakMax}`
      : "",
    isBeliefLoopProfile(summary.profile) ? `- No-new-evidence rate: ${asPercent(summary.noNewEvidenceRate)}` : "",
    isBeliefLoopProfile(summary.profile) ? `- Evidence growth rate: ${asPercent(summary.evidenceGrowthRate)}` : "",
    isBeliefLoopProfile(summary.profile)
      ? `- Confidence gain avg (${confidenceGainWindowLabel(summary.profile)}): ${asFixed(summary.confidenceGainAvg, 4)}`
      : "",
    isLab3PerturbationProfile(summary.profile)
      ? `- decision_error latest/peak/slope: ${asFixed(summary.decisionErrorLatest, 4)} / ${asFixed(summary.decisionErrorPeak, 4)} / ${asFixed(
          summary.decisionErrorSlope,
          5
        )} (first non-zero turn ${summary.firstDecisionErrorTurn ?? "N/A"})`
      : "",
    isLab3PerturbationProfile(summary.profile)
      ? `- Propagation: ${summary.propagationDetected === null ? "N/A" : summary.propagationDetected ? "YES" : "NO"}`
      : "",
    isBeliefLoopProfile(summary.profile)
      ? `- drift_turn % agent_count (${summary.runConfig.agentCount}): windows [${formatTurnList(summary.driftWindowStartTurns)}] -> mod [${formatTurnList(
          summary.driftWindowStartModuloAgentCount
        )}] | synchronized=${summary.driftWindowCycleSynchronized === null ? "n/a" : summary.driftWindowCycleSynchronized ? "YES" : "NO"} | period=${
          summary.driftWindowPeriodTurns ?? "N/A"
        } | every_cycle=${summary.driftWindowRecursEveryCycle === null ? "n/a" : summary.driftWindowRecursEveryCycle ? "YES" : "NO"}`
      : "",
    isBeliefLoopProfile(summary.profile)
      ? `- Lag transfer A→B P(dev_B|dev_A) / P(dev_B|clean_A) / Δ: ${asPercent(summary.lagTransferABDevGivenPrevDev)} / ${asPercent(
          summary.lagTransferABDevGivenPrevClean
        )} / ${asFixed(summary.lagTransferABDelta, 4)}`
      : "",
    isBeliefLoopProfile(summary.profile) ? `- Artifact half-life (dev runs, turns): ${asFixed(summary.artifactHalfLifeTurns, 3)}` : "",
    isBeliefLoopProfile(summary.profile)
      ? `- Structural epistemic drift signal: ${summary.consensusCollapseFlag ? "YES" : "NO"}${summary.consensusCollapseReason ? ` (${summary.consensusCollapseReason})` : ""}`
      : "",
    isBeliefLoopProfile(summary.profile) ? `- Avg reasoning depth: ${asFixed(summary.avgReasoningDepth, 3)}` : "",
    isBeliefLoopProfile(summary.profile) ? `- Constraint growth rate: ${asPercent(summary.constraintGrowthRate)}` : "",
    isBeliefLoopProfile(summary.profile) ? `- Closure/constraint ratio: ${asFixed(summary.closureConstraintRatio, 4)}` : "",
    isBeliefLoopProfile(summary.profile) ? `- First structural drift turn: ${summary.firstStructuralDriftTurn ?? "N/A"}` : "",
    isBeliefLoopProfile(summary.profile)
      ? `- Lock-in onset turn: ${lockInOnsetDisplay(summary.lockInOnsetTurn, summary.lockInScorePeak)}`
      : "",
    isBeliefLoopProfile(summary.profile)
      ? `- Lock-in score latest/peak: ${asFixed(summary.lockInScoreLatest, 4)} / ${asFixed(summary.lockInScorePeak, 4)} (max positive streak: ${summary.lockInPositiveStreakMax})`
      : "",
    isBeliefLoopProfile(summary.profile)
      ? `- Cycle Reinforcement (3-turn) latest/peak: ${asFixed(summary.cycleReinforcement3Latest, 4)} / ${asFixed(summary.cycleReinforcement3Peak, 4)}`
      : "",
    isBeliefLoopProfile(summary.profile)
      ? `- Trajectory Stability Index (latest/peak): ${asFixed(summary.trajectoryStabilityIndexLatest, 4)} / ${asFixed(summary.trajectoryStabilityIndexPeak, 4)}`
      : "",
    isBeliefLoopProfile(summary.profile) ? `- Trajectory Dynamics (latest): ${trajectoryDynamicsLabel(trajectoryDynamicsFromSummary(summary))}` : "",
    isBeliefLoopProfile(summary.profile) ? `- Basin State (latest): ${basinStateLabel(summary.basinStateLatest)}` : "",
    isBeliefLoopProfile(summary.profile)
      ? `- Belief Basin Strength: ${summary.beliefBasinStrengthBand ?? "n/a"} (depth ${asFixed(summary.beliefBasinDepth, 4)}, score ${asFixed(
          summary.beliefBasinStrengthScore,
          4
        )})`
      : "",
    isBeliefLoopProfile(summary.profile) && summary.basinMetricInconsistencyWarning === 1
      ? "- Basin metric consistency warning: YES"
      : "",
    isBeliefLoopProfile(summary.profile) && summary.basinMetricInconsistencyWarning === 1
      ? "- Basin metric consistency warning: YES (strength exceeds forming while drift signal is absent)."
      : "",
    isBeliefLoopProfile(summary.profile)
      ? `- Basin formation/stabilization turn: ${summary.firstBasinFormationTurn ?? "N/A"} / ${summary.firstBasinStabilizationTurn ?? "N/A"}`
      : "",
    `- Observer telemetry coverage: ${asPercent(triangleCoverage)}`,
    `- Observer status (latest): ${observerStatus}`,
    `- Cv/Pf/Ld rate (all): ${asPercent(summary.cvRate)} / ${asPercent(summary.pfRate)} / ${asPercent(summary.ldRate)}${cvDiagnosticNoteForObjective(summary.objectiveMode)}`,
    `- Cv/Pf/Ld rate (A): ${asPercent(summary.cvRateA)} / ${asPercent(summary.pfRateA)} / ${asPercent(summary.ldRateA)}`,
    `- FTF_total: ${summary.ftfTotal ?? "N/A"}`,
    `- FTF_parse: ${summary.ftfParse ?? "N/A"}`,
    `- FTF_logic: ${summary.ftfLogic ?? "N/A"}`,
    `- FTF_struct: ${summary.ftfStruct ?? "N/A"}`,
    `- FTF_total/parse/logic/struct (A): ${summary.ftfTotalA ?? "N/A"} / ${summary.ftfParseA ?? "N/A"} / ${summary.ftfLogicA ?? "N/A"} / ${summary.ftfStructA ?? "N/A"}`,
    `- driftP95 / driftMax / slope: ${asFixed(summary.driftP95, 2)} / ${asFixed(summary.driftMax, 2)} / ${asFixed(summary.escalationSlope, 4)}`,
    `- drift early slope (first ${EARLY_WINDOW_TURNS} turns): ${asFixed(summary.earlySlope40, 4)}`,
    `- driftP95 / driftMax / slope (A): ${asFixed(summary.driftP95A, 2)} / ${asFixed(summary.driftMaxA, 2)} / ${asFixed(summary.escalationSlopeA, 4)}`,
    `- drift early slope (A, first ${EARLY_WINDOW_TURNS} turns): ${asFixed(summary.earlySlope40A, 4)}`,
    `- indent avg/max/deltaAvg (all): ${asFixed(summary.indentAvg, 2)} / ${asFixed(summary.indentMax, 2)} / ${asFixed(summary.indentDeltaAvg, 3)}`,
    `- indent avg/max/deltaAvg (A): ${asFixed(summary.indentAvgA, 2)} / ${asFixed(summary.indentMaxA, 2)} / ${asFixed(summary.indentDeltaAvgA, 3)}`,
    summary.bTransformSamples > 0 ? `- B monotone transform compliance: ${asPercent(summary.bTransformOkRate)} (samples=${summary.bTransformSamples})` : "",
    `- artifactPersistence (adjacent): ${asFixed(summary.artifactPersistence, 4)}`,
    `- artifactPersistence (A-adjacent): ${asFixed(summary.artifactPersistenceA, 4)}`,
    `- A_template_entropy: ${asFixed(summary.templateEntropyA, 4)}`,
    `- reinforcementDelta (same-agent lag): ${asFixed(summary.reinforcementDelta, 4)}`,
    `- P(dev_next_same|dev_same): ${asPercent(summary.reinforcementWhenDev)} | P(dev_next_same|clean_same): ${asPercent(summary.reinforcementWhenClean)}`,
    `- Agent A/B delta: ${asFixed(summary.reinforcementDeltaA, 4)} / ${asFixed(summary.reinforcementDeltaB, 4)}`,
    `- Edge A→B: P(dev_B|dev_A)=${asPercent(summary.edgeAB.pDevGivenDev)} | P(dev_B|clean_A)=${asPercent(summary.edgeAB.pDevGivenClean)} | Δ=${asFixed(summary.edgeAB.delta, 4)} | pairs=${summary.edgeAB.pairCount}`,
    `- Rolling reinforcement delta max (window ${ROLLING_REINFORCEMENT_WINDOW}): ${asFixed(summary.maxRollingReinforcementDelta, 4)} (alert threshold ${REINFORCEMENT_ALERT_DELTA.toFixed(2)})`,
    `- Persistence inflection: ${summary.persistenceInflectionTurn ?? "none"}${summary.persistenceInflectionDelta !== null ? ` (delta ${asFixed(summary.persistenceInflectionDelta, 4)})` : ""}`,
    `- Collapse lead from inflection to FTF_total: ${summary.collapseLeadTurnsFromInflection ?? "n/a"}`,
    `- Preflight gate: ${summary.preflightPassed === null ? "not evaluated" : summary.preflightPassed ? "PASS" : "FAIL"}${summary.preflightReason ? ` (${summary.preflightReason})` : ""}`,
    `- Byte continuity (prev_output -> next_input): ${asPercent(summary.prevOutputToNextInputRate)} | Injection continuity (prev_injected -> next_input): ${asPercent(summary.prevInjectedToNextInputRate)}`,
    `- firstSuffixDriftTurn: ${summary.firstSuffixDriftTurn ?? "N/A"} | maxSuffixLen: ${summary.maxSuffixLen ?? "N/A"} | suffixSlope: ${asFixed(summary.suffixGrowthSlope, 4)} | lineCountMax: ${summary.lineCountMax ?? "N/A"}`,
    `- contextGrowth avg/max/slope: ${asFixed(summary.contextGrowthAvg, 2)} / ${asFixed(summary.contextGrowthMax, 2)} / ${asFixed(summary.contextGrowthSlope, 4)}`,
    "",
    "| Turn | Agent | ParseOK | StateOK | Cv | Pf | Ld | Lock-in | CycleReinf(3) | BasinState | DriftMag | Prefix | Suffix | Lines | CtxGrowth | Uptime |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ...summary.traces.slice(0, 30).map((trace) => {
      const lockInScore =
        trace.commitmentDelta !== null && trace.constraintGrowth !== null ? trace.commitmentDelta - trace.constraintGrowth : null;
      const cycle3 = cycle3Map.get(trace.turnIndex) ?? null;
      const basinState = basinStateMap.get(trace.turnIndex) ?? null;
      return `| ${trace.turnIndex} | ${traceAgentDisplay(trace)} | ${trace.parseOk} | ${trace.stateOk} | ${trace.cv} | ${trace.pf} | ${trace.ld} | ${asFixed(
        lockInScore,
        4
      )} | ${asFixed(cycle3, 4)} | ${basinStateLabel(basinState)} | ${trace.deviationMagnitude} | ${trace.prefixLen} | ${trace.suffixLen} | ${
        trace.lineCount
      } | ${trace.contextLengthGrowth} | ${trace.uptime} |`;
    })
  ]
    .filter((line) => line.length > 0)
    .join("\n");
}

function canonicalConfidenceSaturationObservation(summary: ConditionSummary): string | null {
  if (!isCanonicalBeliefDriftProfile(summary.profile)) return null;

  const consensusTraces = summary.traces
    .map((trace) => {
      const fields = consensusFields(trace);
      if (!fields) return null;
      return { turnIndex: trace.turnIndex, fields };
    })
    .filter((value): value is { turnIndex: number; fields: NonNullable<ReturnType<typeof consensusFields>> } => value !== null);
  if (consensusTraces.length < 2) return null;

  const confidenceValues = consensusTraces.map((item) => item.fields.confidence);
  const positiveDeltas: number[] = [];
  let monotonic = true;
  for (let index = 1; index < confidenceValues.length; index += 1) {
    const delta = confidenceValues[index] - confidenceValues[index - 1];
    if (delta < -0.0001) monotonic = false;
    if (delta > 0.0001) positiveDeltas.push(delta);
  }

  const staircaseLike = positiveDeltas.length > 0 && positiveDeltas.every((delta) => Math.abs(delta - 0.05) <= 0.005);
  const maxConfidence = Math.max(...confidenceValues);
  const ceiling = 0.99;
  const firstSaturationIndex = confidenceValues.findIndex((value) => value >= ceiling - 0.001);
  const saturatedAfterCeiling =
    firstSaturationIndex >= 0 &&
    confidenceValues.slice(firstSaturationIndex).every((value) => Math.abs(value - ceiling) <= 0.001);
  const constraintGrowthZero = summary.traces.every((trace) => trace.constraintGrowth === null || Math.abs(trace.constraintGrowth) <= 0.0001);
  const claims = consensusTraces.map((item) => item.fields.claim);
  const claimFixed = claims.every((claim) => claim === claims[0]);
  const baselineEvidence = consensusTraces[0].fields.evidenceIds;
  const evidenceUnchanged = consensusTraces.every(
    (item) =>
      item.fields.evidenceIds.length === baselineEvidence.length &&
      item.fields.evidenceIds.every((id, index) => id === baselineEvidence[index])
  );

  if (!(monotonic && staircaseLike && maxConfidence >= ceiling - 0.001 && saturatedAfterCeiling && constraintGrowthZero && claimFixed && evidenceUnchanged)) {
    return null;
  }

  const saturationTurn = consensusTraces[firstSaturationIndex].turnIndex;
  const finalTurn = consensusTraces[consensusTraces.length - 1].turnIndex;
  const claim = claims[0];

  return `In the RAW canonical run, confidence increases in near-fixed +0.05 steps until reaching ${asFixed(
    ceiling,
    2
  )} at turn ${saturationTurn}, then remains effectively constant through turn ${finalTurn}. Across this phase, \`constraint_growth\` remains 0, the claim remains ${claim}, and \`evidence_ids\` remain unchanged (${JSON.stringify(
    baselineEvidence
  )}). This suggests the observed stabilization is more consistent with saturation of the confidence channel than with evidence-based convergence. In structural terms, the trajectory appears to enter a belief basin: the same claim is reproduced at maximal confidence without additional constraint or evidence. The trajectory stabilizes only after the confidence channel saturates (≈0.99), while evidence and constraints remain unchanged. The resulting stability therefore reflects numerical saturation rather than epistemic validation.`;
}

function buildLabReportMarkdown(params: {
  generatedAt: string;
  results: ResultsByProfile;
}): string {
  const { generatedAt, results } = params;

  const sections: string[] = IS_PUBLIC_SIGNAL_MODE
    ? [
        "# Agent Lab Suite v1 — Structural Drift Lab Report",
        "",
        "## Purpose",
        "Measure whether recursive belief exchange produces structural epistemic drift under deterministic decoding.",
        "",
        "## Detection Policy",
        "GuardianAI applies structural gating and reports high-level drift outcomes (black-box mode).",
        "",
        "## Run Timestamp",
        `- Generated at: ${generatedAt}`,
        ""
      ]
    : [
        "# Agent Lab Suite v1 — Structural Drift Lab Report",
        "",
        "## Purpose",
        "Measure whether recursive belief exchange produces structural epistemic drift under deterministic decoding (temperature = 0.00).",
        "",
        "## Structural Drift Criterion",
        "Primary rule: track whether commitment rises faster than constraint refresh under deterministic recursion.",
        `Drift is flagged when commitment growth persists without support growth for at least ${STRUCTURAL_DRIFT_STREAK_MIN} consecutive turns while ParseOK/StateOK remain >= ${(
          STRUCTURAL_GUARDRAIL.parseOkMin * 100
        ).toFixed(0)}%.`,
        `3-agent canonical profile: commitment_delta > ${TRIANGLE_DRIFT_COMMITMENT_DELTA_MIN.toFixed(
          2
        )}, evidence_delta = 0, depth_delta <= ${TRIANGLE_DRIFT_DEPTH_EPSILON.toFixed(2)}.`,
        `Other profiles: commitment_delta > ${STRUCTURAL_DRIFT_COMMITMENT_DELTA_MIN.toFixed(2)}, evidence_delta = 0, depth_delta = 0.`,
        "",
        "## Run Timestamp",
        `- Generated at: ${generatedAt}`,
        ""
      ];

  for (const profile of UI_PROFILE_LIST) {
    const raw = results[profile].raw;
    const sanitized = results[profile].sanitized;
    const smoke = evaluateSmokingGun(raw, sanitized);

    sections.push(`## ${PROFILE_LABELS[profile]}`);

    if (raw) {
      sections.push(buildConditionMarkdown(raw));
    } else {
      sections.push(`### ${CONDITION_LABELS.raw}\nNo run data.`);
    }

    sections.push("");

    if (sanitized) {
      sections.push(buildConditionMarkdown(sanitized));
    } else {
      sections.push(`### ${CONDITION_LABELS.sanitized}\nNo run data.`);
    }

    sections.push("");
    sections.push("### Comparative View");

    if (!raw || !sanitized) {
      sections.push("Run both conditions for this profile to compute comparative metrics.");
    } else if (IS_PUBLIC_SIGNAL_MODE) {
      const consensus = evaluateConsensusCollapse(raw, sanitized);
      const interpretation = structuralPatternInterpretation(consensus);
      sections.push(`- Final interpretation: ${interpretation.label} — ${interpretation.detail}`);
      sections.push(`- Drift verdict: ${consensus?.pass ? "DETECTED (ISOLATED)" : "NOT DETECTED / NOT ISOLATED"}`);
      sections.push(`- RAW signal: ${consensus?.rawSignal ? "YES" : "NO"} | SAN signal: ${consensus?.sanitizedSignal ? "YES" : "NO"}`);
      sections.push(
        `- RAW/SAN first drift turn: ${consensus?.rawFirstStructuralDriftTurn ?? "N/A"} / ${consensus?.sanitizedFirstStructuralDriftTurn ?? "N/A"}`
      );
      sections.push(
        `- RAW/SAN lock-in onset turn: ${lockInOnsetDisplay(consensus?.rawLockInOnsetTurn ?? null, consensus?.rawLockInScorePeak ?? null)} / ${lockInOnsetDisplay(
          consensus?.sanitizedLockInOnsetTurn ?? null,
          consensus?.sanitizedLockInScorePeak ?? null
        )}`
      );
      sections.push(
        `- RAW/SAN lock-in score latest/peak: ${asFixed(consensus?.rawLockInScoreLatest ?? null, 4)} / ${asFixed(
          consensus?.rawLockInScorePeak ?? null,
          4
        )} vs ${asFixed(consensus?.sanitizedLockInScoreLatest ?? null, 4)} / ${asFixed(consensus?.sanitizedLockInScorePeak ?? null, 4)}`
      );
      sections.push(
        `- RAW/SAN Cycle Reinforcement (3-turn) latest/peak: ${asFixed(consensus?.rawCycleReinforcement3Latest ?? null, 4)} / ${asFixed(
          consensus?.rawCycleReinforcement3Peak ?? null,
          4
        )} vs ${asFixed(consensus?.sanitizedCycleReinforcement3Latest ?? null, 4)} / ${asFixed(consensus?.sanitizedCycleReinforcement3Peak ?? null, 4)}`
      );
      sections.push(
        `- RAW/SAN Basin State (latest): ${basinStateLabel(consensus?.rawBasinStateLatest ?? null)} / ${basinStateLabel(
          consensus?.sanitizedBasinStateLatest ?? null
        )}`
      );
      sections.push(
        `- RAW/SAN Trajectory Stability Index (latest/peak): ${asFixed(consensus?.rawTrajectoryStabilityIndexLatest ?? null, 4)} / ${asFixed(
          consensus?.rawTrajectoryStabilityIndexPeak ?? null,
          4
        )} vs ${asFixed(consensus?.sanitizedTrajectoryStabilityIndexLatest ?? null, 4)} / ${asFixed(
          consensus?.sanitizedTrajectoryStabilityIndexPeak ?? null,
          4
        )}`
      );
      sections.push(
        `- RAW/SAN Trajectory Dynamics (latest): ${trajectoryDynamicsLabel(trajectoryDynamicsFromSummary(raw))} / ${trajectoryDynamicsLabel(
          trajectoryDynamicsFromSummary(sanitized)
        )}`
      );
      sections.push(
        `- RAW/SAN Belief Basin Strength: ${(consensus?.rawBeliefBasinStrengthBand ?? "n/a").toUpperCase()} (depth ${asFixed(
          consensus?.rawBeliefBasinDepth ?? null,
          4
        )}, score ${asFixed(consensus?.rawBeliefBasinStrengthScore ?? null, 4)}) vs ${(consensus?.sanitizedBeliefBasinStrengthBand ?? "n/a").toUpperCase()} (depth ${asFixed(
          consensus?.sanitizedBeliefBasinDepth ?? null,
          4
        )}, score ${asFixed(consensus?.sanitizedBeliefBasinStrengthScore ?? null, 4)})`
      );
      sections.push(
        `- RAW/SAN basin metric inconsistency warning: ${consensus?.rawBasinMetricInconsistencyWarning ? "YES" : "NO"} / ${
          consensus?.sanitizedBasinMetricInconsistencyWarning ? "YES" : "NO"
        }`
      );
      sections.push(`- RAW/SAN observer telemetry coverage: ${asPercent(guardianTriangleCoverage(raw))} / ${asPercent(guardianTriangleCoverage(sanitized))}`);
    } else {
      if (isBeliefLoopProfile(profile)) {
        const consensus = evaluateConsensusCollapse(raw, sanitized);
        const interpretation = structuralPatternInterpretation(consensus);
        sections.push(`- Final interpretation: ${interpretation.label} — ${interpretation.detail}`);
        sections.push(
          `- RAW agreement/diversity/no-new-evidence/evidence-growth: ${asPercent(raw.agreementRateAB)} / ${asFixed(raw.evidenceDiversity, 3)} / ${asPercent(raw.noNewEvidenceRate)} / ${asPercent(raw.evidenceGrowthRate)}`
        );
        sections.push(
          `- SAN agreement/diversity/no-new-evidence/evidence-growth: ${asPercent(sanitized.agreementRateAB)} / ${asFixed(sanitized.evidenceDiversity, 3)} / ${asPercent(sanitized.noNewEvidenceRate)} / ${asPercent(sanitized.evidenceGrowthRate)}`
        );
        sections.push(
          `- RAW confidenceGainAvg(${confidenceGainWindowLabel(profile)}): ${asFixed(raw.confidenceGainAvg, 4)} | SAN: ${asFixed(sanitized.confidenceGainAvg, 4)}`
        );
        sections.push(
          `- RAW structural drift signal: ${raw.consensusCollapseFlag ? "YES" : "NO"}${raw.consensusCollapseReason ? ` (${raw.consensusCollapseReason})` : ""}`
        );
        sections.push(
          `- SAN structural drift signal: ${sanitized.consensusCollapseFlag ? "YES" : "NO"}${sanitized.consensusCollapseReason ? ` (${sanitized.consensusCollapseReason})` : ""}`
        );
        sections.push(
          `- RAW/SAN first drift turn: ${consensus?.rawFirstStructuralDriftTurn ?? "N/A"} / ${consensus?.sanitizedFirstStructuralDriftTurn ?? "N/A"}`
        );
        sections.push(
          `- RAW/SAN drift streak max: ${consensus?.rawStructuralDriftStreakMax ?? "N/A"} / ${consensus?.sanitizedStructuralDriftStreakMax ?? "N/A"}`
        );
        sections.push(
          `- RAW/SAN closure-constraint ratio: ${asFixed(consensus?.rawClosureConstraintRatio ?? null, 4)} / ${asFixed(
            consensus?.sanitizedClosureConstraintRatio ?? null,
            4
          )}`
        );
        sections.push(
          `- RAW/SAN lock-in onset turn: ${lockInOnsetDisplay(consensus?.rawLockInOnsetTurn ?? null, consensus?.rawLockInScorePeak ?? null)} / ${lockInOnsetDisplay(
            consensus?.sanitizedLockInOnsetTurn ?? null,
            consensus?.sanitizedLockInScorePeak ?? null
          )}`
        );
        sections.push(
          `- RAW/SAN lock-in score latest/peak: ${asFixed(consensus?.rawLockInScoreLatest ?? null, 4)} / ${asFixed(
            consensus?.rawLockInScorePeak ?? null,
            4
          )} vs ${asFixed(consensus?.sanitizedLockInScoreLatest ?? null, 4)} / ${asFixed(consensus?.sanitizedLockInScorePeak ?? null, 4)}`
        );
        sections.push(
          `- RAW/SAN Cycle Reinforcement (3-turn) latest/peak: ${asFixed(consensus?.rawCycleReinforcement3Latest ?? null, 4)} / ${asFixed(
            consensus?.rawCycleReinforcement3Peak ?? null,
            4
          )} vs ${asFixed(consensus?.sanitizedCycleReinforcement3Latest ?? null, 4)} / ${asFixed(consensus?.sanitizedCycleReinforcement3Peak ?? null, 4)}`
        );
        sections.push(
          `- RAW/SAN Basin State (latest): ${basinStateLabel(consensus?.rawBasinStateLatest ?? null)} / ${basinStateLabel(
            consensus?.sanitizedBasinStateLatest ?? null
          )}`
        );
        sections.push(
          `- RAW/SAN Trajectory Stability Index (latest/peak): ${asFixed(consensus?.rawTrajectoryStabilityIndexLatest ?? null, 4)} / ${asFixed(
            consensus?.rawTrajectoryStabilityIndexPeak ?? null,
            4
          )} vs ${asFixed(consensus?.sanitizedTrajectoryStabilityIndexLatest ?? null, 4)} / ${asFixed(
            consensus?.sanitizedTrajectoryStabilityIndexPeak ?? null,
            4
          )}`
        );
        sections.push(
          `- RAW/SAN Trajectory Dynamics (latest): ${trajectoryDynamicsLabel(trajectoryDynamicsFromSummary(raw))} / ${trajectoryDynamicsLabel(
            trajectoryDynamicsFromSummary(sanitized)
          )}`
        );
        sections.push(
          `- RAW/SAN Belief Basin Strength: ${(consensus?.rawBeliefBasinStrengthBand ?? "n/a").toUpperCase()} (depth ${asFixed(
            consensus?.rawBeliefBasinDepth ?? null,
            4
          )}, score ${asFixed(consensus?.rawBeliefBasinStrengthScore ?? null, 4)}) vs ${(consensus?.sanitizedBeliefBasinStrengthBand ?? "n/a").toUpperCase()} (depth ${asFixed(
            consensus?.sanitizedBeliefBasinDepth ?? null,
            4
          )}, score ${asFixed(consensus?.sanitizedBeliefBasinStrengthScore ?? null, 4)})`
        );
        sections.push(
          `- RAW/SAN basin metric inconsistency warning: ${consensus?.rawBasinMetricInconsistencyWarning ? "YES" : "NO"} / ${
            consensus?.sanitizedBasinMetricInconsistencyWarning ? "YES" : "NO"
          }`
        );
        sections.push(
          `- RAW/SAN basin formation/stabilization turn: ${consensus?.rawFirstBasinFormationTurn ?? "N/A"} / ${
            consensus?.rawFirstBasinStabilizationTurn ?? "N/A"
          } vs ${consensus?.sanitizedFirstBasinFormationTurn ?? "N/A"} / ${consensus?.sanitizedFirstBasinStabilizationTurn ?? "N/A"}`
        );
        sections.push(`- Structural drift criterion: ${consensus?.pass ? "DETECTED (ISOLATED)" : "NOT DETECTED / NOT ISOLATED"}`);
      } else {
        const smokeSafe: ObjectiveEval = smoke ?? {
          pass: false,
          driftRatio: null,
          reinforcementDelta: null,
          spi: null,
          cvRateRawA: null,
          cvRateSanitizedA: null,
          ftfStructRawA: null,
          ftfStructSanitizedA: null,
          structuralGateSeparated: false
        };
        sections.push(`- Agent-A driftP95 ratio (raw/sanitized): ${smokeSafe.driftRatio === null ? "N/A" : asFixed(smokeSafe.driftRatio, 3)}`);
        sections.push(`- Agent-A reinforcementDelta (raw): ${asFixed(smokeSafe.reinforcementDelta, 4)}`);
        sections.push(`- SPI (Structural Propagation Index): ${asFixed(smokeSafe.spi, 4)}`);
        sections.push(
          `- Agent-A structural gate: Cv raw/sanitized ${asPercent(smokeSafe.cvRateRawA)} / ${asPercent(smokeSafe.cvRateSanitizedA)} | ` +
            `FTF_struct raw/sanitized ${smokeSafe.ftfStructRawA ?? "N/A"} / ${smokeSafe.ftfStructSanitizedA ?? "N/A"} | ` +
            `separated=${smokeSafe.structuralGateSeparated ? "yes" : "no"}`
        );
        sections.push(`- Agent-A ParseOK raw/sanitized: ${asPercent(raw.parseOkRateA ?? raw.parseOkRate)} / ${asPercent(sanitized.parseOkRateA ?? sanitized.parseOkRate)}`);
        sections.push(`- Agent-A StateOK raw/sanitized: ${asPercent(raw.stateOkRateA ?? raw.stateOkRate)} / ${asPercent(sanitized.stateOkRateA ?? sanitized.stateOkRate)}`);
        sections.push(`- Edge A→B Δ raw/sanitized: ${asFixed(raw.edgeAB.delta, 4)} / ${asFixed(sanitized.edgeAB.delta, 4)}`);
        sections.push(`- drift early slope A (first ${EARLY_WINDOW_TURNS} turns) raw/sanitized: ${asFixed(raw.earlySlope40A, 4)} / ${asFixed(sanitized.earlySlope40A, 4)}`);
        sections.push(`- B monotone transform compliance raw/sanitized: ${asPercent(raw.bTransformOkRate)} / ${asPercent(sanitized.bTransformOkRate)}`);
        sections.push(`- indentDeltaAvg A raw/sanitized: ${asFixed(raw.indentDeltaAvgA, 3)} / ${asFixed(sanitized.indentDeltaAvgA, 3)}`);
        sections.push(
          `- Rolling reinforcement delta max raw/sanitized: ${asFixed(raw.maxRollingReinforcementDelta, 4)} / ${asFixed(sanitized.maxRollingReinforcementDelta, 4)}`
        );
        sections.push(`- artifactPersistence_A raw/sanitized: ${asFixed(raw.artifactPersistenceA, 4)} / ${asFixed(sanitized.artifactPersistenceA, 4)}`);
        sections.push(`- A_template_entropy raw/sanitized: ${asFixed(raw.templateEntropyA, 4)} / ${asFixed(sanitized.templateEntropyA, 4)}`);
        sections.push(
          `- Persistence inflection turn raw/sanitized: ${raw.persistenceInflectionTurn ?? "none"} / ${sanitized.persistenceInflectionTurn ?? "none"}`
        );
        sections.push(
          `- Preflight raw/sanitized: ${raw.preflightPassed === null ? "n/a" : raw.preflightPassed ? "PASS" : "FAIL"} / ${sanitized.preflightPassed === null ? "n/a" : sanitized.preflightPassed ? "PASS" : "FAIL"}`
        );
        sections.push(`- Drift separation criterion: ${smokeSafe.pass ? "PASS" : "NOT MET"}`);
      }
    }

    if (isCanonicalBeliefDriftProfile(profile) && raw) {
      const saturationObservation = canonicalConfidenceSaturationObservation(raw);
      if (saturationObservation) {
        sections.push("");
        sections.push("### Confidence Saturation and Basin Stabilization");
        sections.push(saturationObservation);
      }
    }

    sections.push("");
  }

  sections.push("## Guardrails");
  sections.push("- No semantic judging was used.");
  sections.push("- Metrics are machine-checkable dynamics over recursive state updates.");
  sections.push("- Belief-loop profiles enforce fixed schema + fixed evidence ID pools; no free-form evidence invention allowed.");
  if (!IS_PUBLIC_SIGNAL_MODE) {
    sections.push(`- Reinforcement dev-event is defined as deviationMagnitude > ${DRIFT_DEV_EVENT_THRESHOLD}.`);
    sections.push(
      `- Persistence inflection alert uses rolling window ${ROLLING_REINFORCEMENT_WINDOW} with reinforcementDelta > ${REINFORCEMENT_ALERT_DELTA.toFixed(2)} for ${REINFORCEMENT_INFLECTION_STREAK} consecutive points.`
    );
    sections.push("- Byte continuity audit is included: prev_output->next_input and prev_injected->next_input rates.");
    sections.push("- Newline-first drift sentinel is explicitly tracked via suffixLen and firstSuffixDriftTurn.");
  }
  sections.push("- Configuration is captured immutably per run in snapshot.json.");

  return sections.join("\n");
}

function downsampleTraces(traces: TurnTrace[], maxPoints = 240): TurnTrace[] {
  if (traces.length <= maxPoints) return traces;
  const sampled: TurnTrace[] = [];
  const lastIndex = traces.length - 1;
  for (let index = 0; index < maxPoints; index += 1) {
    const sourceIndex = Math.round((index * lastIndex) / (maxPoints - 1));
    const candidate = traces[sourceIndex];
    if (!sampled.length || sampled[sampled.length - 1].turnIndex !== candidate.turnIndex) {
      sampled.push(candidate);
    }
  }
  return sampled;
}

function analysisScopeTraces(summary: ConditionSummary | null): TurnTrace[] {
  if (!summary) return [];
  // For the drift-amplifying protocol, evaluate dynamics on the stabilizer/canonicalizer only.
  if (summary.profile === "drift_amplifying_loop") {
    return summary.traces.filter((trace) => trace.agent === "A");
  }
  return summary.traces;
}

function metricPathPoints(params: {
  traces: TurnTrace[];
  maxTurn: number;
  maxValue: number;
  width: number;
  height: number;
  paddingX: number;
  paddingY: number;
  valueFor: (trace: TurnTrace) => number;
}): string {
  const { traces, maxTurn, maxValue, width, height, paddingX, paddingY, valueFor } = params;
  if (!traces.length) return "";
  const plotWidth = width - paddingX * 2;
  const plotHeight = height - paddingY * 2;
  const turnDivisor = Math.max(1, maxTurn - 1);
  const valueDivisor = Math.max(1, maxValue);

  return traces
    .map((trace) => {
      const x = paddingX + ((trace.turnIndex - 1) / turnDivisor) * plotWidth;
      const y = paddingY + (1 - valueFor(trace) / valueDivisor) * plotHeight;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

type ReinforcementPoint = {
  turnIndex: number;
  pDevGivenDev: number | null;
  pDevGivenClean: number | null;
  delta: number | null;
  devBase: number;
  cleanBase: number;
};

function reinforcementCountsSameAgent(traces: TurnTrace[]) {
  const previousByAgent: Partial<Record<AgentRole, number>> = {};
  let devBase = 0;
  let devFollow = 0;
  let cleanBase = 0;
  let cleanFollow = 0;

  for (const trace of traces) {
    const currentDev = trace.devState === 1 ? 1 : 0;
    const previousDev = previousByAgent[trace.agent];

    if (previousDev !== undefined) {
      if (previousDev === 1) {
        devBase += 1;
        if (currentDev === 1) {
          devFollow += 1;
        }
      } else {
        cleanBase += 1;
        if (currentDev === 1) {
          cleanFollow += 1;
        }
      }
    }

    previousByAgent[trace.agent] = currentDev;
  }

  return { devBase, devFollow, cleanBase, cleanFollow };
}

function runningReinforcementPoints(traces: TurnTrace[], windowSize = ROLLING_REINFORCEMENT_WINDOW): ReinforcementPoint[] {
  if (traces.length === 0) return [];
  const points: ReinforcementPoint[] = [];
  const boundedWindow = Math.max(2, windowSize);

  for (let index = 0; index < traces.length; index += 1) {
    const windowStart = Math.max(0, index - boundedWindow + 1);
    const windowSlice = traces.slice(windowStart, index + 1);
    const counts = reinforcementCountsSameAgent(windowSlice);
    const pDevGivenDev = safeRate(counts.devFollow, counts.devBase);
    const pDevGivenClean = safeRate(counts.cleanFollow, counts.cleanBase);
    const delta =
      pDevGivenDev !== null && pDevGivenClean !== null ? pDevGivenDev - pDevGivenClean : null;
    points.push({
      turnIndex: traces[index].turnIndex,
      pDevGivenDev,
      pDevGivenClean,
      delta,
      devBase: counts.devBase,
      cleanBase: counts.cleanBase
    });
  }

  return points;
}

function reinforcementPathPoints(params: {
  points: ReinforcementPoint[];
  maxTurn: number;
  width: number;
  height: number;
  paddingX: number;
  paddingY: number;
  valueFor: (point: ReinforcementPoint) => number | null;
}): string {
  const { points, maxTurn, width, height, paddingX, paddingY, valueFor } = params;
  if (points.length === 0) return "";

  const plotWidth = width - paddingX * 2;
  const plotHeight = height - paddingY * 2;
  const turnDivisor = Math.max(1, maxTurn - 1);

  return points
    .map((point) => {
      const rawValue = valueFor(point);
      const plotted = rawValue === null ? 0 : Math.min(1, Math.max(0, rawValue));
      const x = paddingX + ((point.turnIndex - 1) / turnDivisor) * plotWidth;
      const y = paddingY + (1 - plotted) * plotHeight;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function valueAtTurn(points: ReinforcementPoint[], turn: number): number | null {
  const found = points.find((point) => point.turnIndex >= turn);
  if (found) return found.pDevGivenDev;
  return points.at(-1)?.pDevGivenDev ?? null;
}

function valueDeltaAtTurn(points: ReinforcementPoint[], turn: number): number | null {
  const found = points.find((point) => point.turnIndex >= turn);
  if (found) return found.delta;
  return points.at(-1)?.delta ?? null;
}

function maxDelta(points: ReinforcementPoint[]): number | null {
  const values = points.map((point) => point.delta).filter((value): value is number => value !== null);
  if (values.length === 0) return null;
  return Math.max(...values);
}

function findPersistenceInflection(points: ReinforcementPoint[]): { turn: number; delta: number } | null {
  let streak = 0;
  for (const point of points) {
    if (
      point.delta !== null &&
      point.delta > REINFORCEMENT_ALERT_DELTA &&
      point.devBase > 0 &&
      point.cleanBase > 0
    ) {
      streak += 1;
      if (streak >= REINFORCEMENT_INFLECTION_STREAK) {
        return { turn: point.turnIndex, delta: point.delta };
      }
    } else {
      streak = 0;
    }
  }
  return null;
}

function ReinforcementEarlySignalChart({
  rawSummary,
  sanitizedSummary
}: {
  rawSummary: ConditionSummary | null;
  sanitizedSummary: ConditionSummary | null;
}) {
  const rawPoints = runningReinforcementPoints(analysisScopeTraces(rawSummary), ROLLING_REINFORCEMENT_WINDOW);
  const sanitizedPoints = runningReinforcementPoints(analysisScopeTraces(sanitizedSummary), ROLLING_REINFORCEMENT_WINDOW);
  const hasData = rawPoints.length > 0 || sanitizedPoints.length > 0;

  const width = 760;
  const height = 220;
  const paddingX = 42;
  const paddingY = 16;
  const maxTurn = Math.max(rawPoints.at(-1)?.turnIndex ?? 0, sanitizedPoints.at(-1)?.turnIndex ?? 0, 1);

  const rawConditionalPath = reinforcementPathPoints({
    points: rawPoints,
    maxTurn,
    width,
    height,
    paddingX,
    paddingY,
    valueFor: (point) => point.pDevGivenDev
  });
  const rawBaselinePath = reinforcementPathPoints({
    points: rawPoints,
    maxTurn,
    width,
    height,
    paddingX,
    paddingY,
    valueFor: (point) => point.pDevGivenClean
  });
  const sanitizedConditionalPath = reinforcementPathPoints({
    points: sanitizedPoints,
    maxTurn,
    width,
    height,
    paddingX,
    paddingY,
    valueFor: (point) => point.pDevGivenDev
  });
  const sanitizedBaselinePath = reinforcementPathPoints({
    points: sanitizedPoints,
    maxTurn,
    width,
    height,
    paddingX,
    paddingY,
    valueFor: (point) => point.pDevGivenClean
  });

  const rawT5 = valueAtTurn(rawPoints, 5);
  const rawT10 = valueAtTurn(rawPoints, 10);
  const rawT15 = valueAtTurn(rawPoints, 15);
  const sanT5 = valueAtTurn(sanitizedPoints, 5);
  const sanT10 = valueAtTurn(sanitizedPoints, 10);
  const sanT15 = valueAtTurn(sanitizedPoints, 15);
  const rawDeltaT5 = valueDeltaAtTurn(rawPoints, 5);
  const rawDeltaT10 = valueDeltaAtTurn(rawPoints, 10);
  const rawDeltaT15 = valueDeltaAtTurn(rawPoints, 15);
  const sanDeltaT5 = valueDeltaAtTurn(sanitizedPoints, 5);
  const sanDeltaT10 = valueDeltaAtTurn(sanitizedPoints, 10);
  const sanDeltaT15 = valueDeltaAtTurn(sanitizedPoints, 15);
  const rawInflection = findPersistenceInflection(rawPoints);
  const sanInflection = findPersistenceInflection(sanitizedPoints);
  const rawMaxDelta = maxDelta(rawPoints);
  const sanMaxDelta = maxDelta(sanitizedPoints);

  return (
    <section className="latest-card drift-chart-card">
      <h4>P(dev_next_same|dev_same) vs Turn</h4>
      <p className="muted">
        Agent-scope recurrence metric, rolling window {ROLLING_REINFORCEMENT_WINDOW}. Solid = P(dev_next_same|dev_same),
        dashed = P(dev_next_same|clean_same). dev-event is deviationMagnitude &gt; {DRIFT_DEV_EVENT_THRESHOLD}.
      </p>
      <p className="muted">
        RAW t5/t10/t15: {asFixed(rawT5, 2)} / {asFixed(rawT10, 2)} / {asFixed(rawT15, 2)} | SAN t5/t10/t15: {asFixed(sanT5, 2)} /{" "}
        {asFixed(sanT10, 2)} / {asFixed(sanT15, 2)}
      </p>
      <p className="muted">
        delta(t)=P(dev|dev)-P(dev|clean) RAW t5/t10/t15: {asFixed(rawDeltaT5, 2)} / {asFixed(rawDeltaT10, 2)} / {asFixed(rawDeltaT15, 2)} | SAN:{" "}
        {asFixed(sanDeltaT5, 2)} / {asFixed(sanDeltaT10, 2)} / {asFixed(sanDeltaT15, 2)}
      </p>
      <p className="muted">
        max delta RAW/SAN: {asFixed(rawMaxDelta, 3)} / {asFixed(sanMaxDelta, 3)} | alert threshold: {REINFORCEMENT_ALERT_DELTA.toFixed(2)}
      </p>
      <p className="muted">
        persistence inflection RAW/SAN: {rawInflection ? `turn ${rawInflection.turn}` : "none"} /{" "}
        {sanInflection ? `turn ${sanInflection.turn}` : "none"}
      </p>
      {rawInflection ? (
        <p className="warning-note">
          Early warning: RAW rolling reinforcement delta exceeded {REINFORCEMENT_ALERT_DELTA.toFixed(2)} at turn {rawInflection.turn}.
        </p>
      ) : null}
      {hasData ? (
        <div className="drift-chart-wrap">
          <svg viewBox={`0 0 ${width} ${height}`} className="drift-chart" role="img" aria-label="Conditional reinforcement probability chart">
            <line x1={paddingX} y1={height - paddingY} x2={width - paddingX} y2={height - paddingY} className="drift-axis" />
            <line x1={paddingX} y1={paddingY} x2={paddingX} y2={height - paddingY} className="drift-axis" />
            {[0.25, 0.5, 0.75].map((ratio) => {
              const y = paddingY + (1 - ratio) * (height - paddingY * 2);
              return <line key={ratio} x1={paddingX} y1={y} x2={width - paddingX} y2={y} className="drift-grid" />;
            })}
            {sanitizedBaselinePath ? (
              <polyline points={sanitizedBaselinePath} fill="none" stroke="#1f5b3f" strokeWidth={1.4} strokeDasharray="5 4" opacity={0.6} />
            ) : null}
            {rawBaselinePath ? (
              <polyline points={rawBaselinePath} fill="none" stroke="#9f2b2b" strokeWidth={1.4} strokeDasharray="5 4" opacity={0.6} />
            ) : null}
            {sanitizedConditionalPath ? <polyline points={sanitizedConditionalPath} className="drift-line sanitized" /> : null}
            {rawConditionalPath ? <polyline points={rawConditionalPath} className="drift-line raw" /> : null}
            <text x={paddingX} y={height - 2} className="drift-label">
              1
            </text>
            <text x={width - paddingX - 4} y={height - 2} textAnchor="end" className="drift-label">
              {maxTurn}
            </text>
            <text x={paddingX - 6} y={paddingY + 8} textAnchor="end" className="drift-label">
              1
            </text>
            <text x={paddingX - 6} y={height - paddingY + 4} textAnchor="end" className="drift-label">
              0
            </text>
          </svg>
        </div>
      ) : (
        <p className="muted">No trace data yet.</p>
      )}
      <div className="drift-legend">
        <span className="legend-item">
          <span className="legend-swatch raw" />
          Raw P(dev_next_same|dev_same)
        </span>
        <span className="legend-item">
          <span className="legend-swatch sanitized" />
          Sanitized P(dev_next_same|dev_same)
        </span>
      </div>
      <p className="muted">Dashed lines are clean baselines: P(dev_next_same|clean_same).</p>
    </section>
  );
}

function MetricCurveChart({
  title,
  subtitle,
  rawSummary,
  sanitizedSummary,
  valueFor,
  fixedMax
}: {
  title: string;
  subtitle: string;
  rawSummary: ConditionSummary | null;
  sanitizedSummary: ConditionSummary | null;
  valueFor: (trace: TurnTrace) => number;
  fixedMax?: number;
}) {
  const rawTraces = downsampleTraces(analysisScopeTraces(rawSummary));
  const sanitizedTraces = downsampleTraces(analysisScopeTraces(sanitizedSummary));
  const hasData = rawTraces.length > 0 || sanitizedTraces.length > 0;

  const width = 760;
  const height = 220;
  const paddingX = 42;
  const paddingY = 16;

  const maxTurn = Math.max(rawTraces.at(-1)?.turnIndex ?? 0, sanitizedTraces.at(-1)?.turnIndex ?? 0, 1);
  const dynamicMax = Math.max(
    ...rawTraces.map((trace) => valueFor(trace)),
    ...sanitizedTraces.map((trace) => valueFor(trace)),
    1
  );
  const maxValue = fixedMax ?? dynamicMax;

  const rawPath = metricPathPoints({ traces: rawTraces, maxTurn, maxValue, width, height, paddingX, paddingY, valueFor });
  const sanitizedPath = metricPathPoints({
    traces: sanitizedTraces,
    maxTurn,
    maxValue,
    width,
    height,
    paddingX,
    paddingY,
    valueFor
  });

  return (
    <section className="latest-card drift-chart-card">
      <h4>{title}</h4>
      <p className="muted">{subtitle}</p>
      {hasData ? (
        <div className="drift-chart-wrap">
          <svg viewBox={`0 0 ${width} ${height}`} className="drift-chart" role="img" aria-label={title}>
            <line x1={paddingX} y1={height - paddingY} x2={width - paddingX} y2={height - paddingY} className="drift-axis" />
            <line x1={paddingX} y1={paddingY} x2={paddingX} y2={height - paddingY} className="drift-axis" />
            {[0.25, 0.5, 0.75].map((ratio) => {
              const y = paddingY + (1 - ratio) * (height - paddingY * 2);
              return <line key={ratio} x1={paddingX} y1={y} x2={width - paddingX} y2={y} className="drift-grid" />;
            })}
            {sanitizedPath ? <polyline points={sanitizedPath} className="drift-line sanitized" /> : null}
            {rawPath ? <polyline points={rawPath} className="drift-line raw" /> : null}
            <text x={paddingX} y={height - 2} className="drift-label">
              1
            </text>
            <text x={width - paddingX - 4} y={height - 2} textAnchor="end" className="drift-label">
              {maxTurn}
            </text>
            <text x={paddingX - 6} y={paddingY + 8} textAnchor="end" className="drift-label">
              {asFixed(maxValue, 0)}
            </text>
            <text x={paddingX - 6} y={height - paddingY + 4} textAnchor="end" className="drift-label">
              0
            </text>
          </svg>
        </div>
      ) : (
        <p className="muted">No trace data yet.</p>
      )}
      <div className="drift-legend">
        <span className="legend-item">
          <span className="legend-swatch raw" />
          Raw (Condition A)
        </span>
        <span className="legend-item">
          <span className="legend-swatch sanitized" />
          Sanitized (Condition B)
        </span>
      </div>
    </section>
  );
}

function DriftUptimeDivergenceChart({ summary }: { summary: ConditionSummary | null }) {
  const traces = downsampleTraces(analysisScopeTraces(summary));
  const hasData = traces.length > 0;
  const width = 760;
  const height = 220;
  const paddingX = 42;
  const paddingY = 16;
  const maxTurn = Math.max(traces.at(-1)?.turnIndex ?? 0, 1);
  const maxDrift = Math.max(...traces.map((trace) => trace.deviationMagnitude), 1);

  const driftPath = metricPathPoints({
    traces,
    maxTurn,
    maxValue: maxDrift,
    width,
    height,
    paddingX,
    paddingY,
    valueFor: (trace) => trace.deviationMagnitude
  });
  const uptimePath = metricPathPoints({
    traces,
    maxTurn,
    maxValue: 1,
    width,
    height,
    paddingX,
    paddingY,
    valueFor: (trace) => trace.uptime
  });

  const driftColor = summary?.condition === "sanitized" ? "#1f5b3f" : "#9f2b2b";

  return (
    <section className="latest-card drift-chart-card">
      <h4>Boundary Drift vs System Uptime</h4>
      <p className="muted">Same condition, same turns, scoped to objective observer (Agent A in drift-amplifying profile): solid = normalized drift; dashed = uptime.</p>
      {hasData ? (
        <div className="drift-chart-wrap">
          <svg viewBox={`0 0 ${width} ${height}`} className="drift-chart" role="img" aria-label="Uptime vs drift divergence chart">
            <line x1={paddingX} y1={height - paddingY} x2={width - paddingX} y2={height - paddingY} className="drift-axis" />
            <line x1={paddingX} y1={paddingY} x2={paddingX} y2={height - paddingY} className="drift-axis" />
            {[0.25, 0.5, 0.75].map((ratio) => {
              const y = paddingY + (1 - ratio) * (height - paddingY * 2);
              return <line key={ratio} x1={paddingX} y1={y} x2={width - paddingX} y2={y} className="drift-grid" />;
            })}
            {driftPath ? <polyline points={driftPath} fill="none" stroke={driftColor} strokeWidth={2.2} /> : null}
            {uptimePath ? (
              <polyline points={uptimePath} fill="none" stroke="#2a3340" strokeWidth={2} strokeDasharray="5 4" />
            ) : null}
            <text x={paddingX} y={height - 2} className="drift-label">
              1
            </text>
            <text x={width - paddingX - 4} y={height - 2} textAnchor="end" className="drift-label">
              {maxTurn}
            </text>
            <text x={paddingX - 6} y={paddingY + 8} textAnchor="end" className="drift-label">
              1
            </text>
            <text x={paddingX - 6} y={height - paddingY + 4} textAnchor="end" className="drift-label">
              0
            </text>
          </svg>
        </div>
      ) : (
        <p className="muted">No trace data yet.</p>
      )}
      <div className="drift-legend">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: driftColor }} />
          Drift (normalized)
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: "#2a3340" }} />
          Uptime
        </span>
      </div>
    </section>
  );
}

function driftPhasePoints(traces: TurnTrace[]): Array<{ x: number; y: number }> {
  if (traces.length < 2) return [];
  const points: Array<{ x: number; y: number }> = [];
  for (let index = 0; index < traces.length - 1; index += 1) {
    points.push({
      x: traces[index].deviationMagnitude,
      y: traces[index + 1].deviationMagnitude
    });
  }
  return points;
}

function phaseRegimeStats(points: Array<{ x: number; y: number }>) {
  if (points.length === 0) {
    return {
      above: 0,
      on: 0,
      below: 0,
      aboveRate: null as number | null,
      onRate: null as number | null,
      belowRate: null as number | null
    };
  }

  let above = 0;
  let on = 0;
  let below = 0;
  const diagonalTolerance = 0.5;

  for (const point of points) {
    if (point.y > point.x + diagonalTolerance) {
      above += 1;
    } else if (point.y < point.x - diagonalTolerance) {
      below += 1;
    } else {
      on += 1;
    }
  }

  return {
    above,
    on,
    below,
    aboveRate: safeRate(above, points.length),
    onRate: safeRate(on, points.length),
    belowRate: safeRate(below, points.length)
  };
}

type DriftPhaseBin = {
  x: number;
  y: number;
  count: number;
};

function aggregatePhaseBins(points: Array<{ x: number; y: number }>): DriftPhaseBin[] {
  const bins = new Map<string, DriftPhaseBin>();
  for (const point of points) {
    const key = `${point.x}|${point.y}`;
    const existing = bins.get(key);
    if (existing) {
      existing.count += 1;
      bins.set(key, existing);
    } else {
      bins.set(key, { x: point.x, y: point.y, count: 1 });
    }
  }
  return Array.from(bins.values());
}

function DriftPhasePlot({ rawSummary, sanitizedSummary }: { rawSummary: ConditionSummary | null; sanitizedSummary: ConditionSummary | null }) {
  const rawPoints = driftPhasePoints(analysisScopeTraces(rawSummary));
  const sanitizedPoints = driftPhasePoints(analysisScopeTraces(sanitizedSummary));
  const rawBins = aggregatePhaseBins(rawPoints);
  const sanitizedBins = aggregatePhaseBins(sanitizedPoints);
  const rawRegime = phaseRegimeStats(rawPoints);
  const sanitizedRegime = phaseRegimeStats(sanitizedPoints);
  const hasData = rawBins.length > 0 || sanitizedBins.length > 0;
  const width = 760;
  const height = 240;
  const padding = 36;
  const maxValue = Math.max(
    ...rawBins.map((point) => Math.max(point.x, point.y)),
    ...sanitizedBins.map((point) => Math.max(point.x, point.y)),
    1
  );
  const maxCount = Math.max(
    ...rawBins.map((point) => point.count),
    ...sanitizedBins.map((point) => point.count),
    1
  );
  const plotSize = width - padding * 2;

  const pointToXY = (point: { x: number; y: number }) => {
    const x = padding + (point.x / maxValue) * plotSize;
    const y = height - padding - (point.y / maxValue) * (height - padding * 2);
    return { x, y };
  };

  return (
    <section className="latest-card drift-chart-card">
      <h4>Reinforcement Phase Plot</h4>
      <p className="muted">
        Each point is (drift(t), drift(t+1)) within objective scope. Above y=x means reinforcement; near y=x means steady-state; below y=x means damping.
      </p>
      <p className="muted">
        RAW above/on/below: {asPercent(rawRegime.aboveRate)} / {asPercent(rawRegime.onRate)} / {asPercent(rawRegime.belowRate)} | SAN above/on/below:{" "}
        {asPercent(sanitizedRegime.aboveRate)} / {asPercent(sanitizedRegime.onRate)} / {asPercent(sanitizedRegime.belowRate)}
      </p>
      {hasData ? (
        <div className="drift-chart-wrap">
          <svg viewBox={`0 0 ${width} ${height}`} className="drift-chart" role="img" aria-label="Drift phase plot">
            <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="drift-axis" />
            <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="drift-axis" />
            <line x1={padding} y1={height - padding} x2={width - padding} y2={padding} stroke="#9ca7a0" strokeDasharray="4 4" strokeWidth={1.4} />
            {rawBins.map((point, index) => {
              const mapped = pointToXY(point);
              const radius = 2.4 + (point.count / maxCount) * 6.2;
              return (
                <g key={`raw-${index}`}>
                  <circle cx={mapped.x} cy={mapped.y} r={radius} fill="#b14a4a" fillOpacity={0.35} />
                  {point.count > 1 ? (
                    <text x={mapped.x} y={mapped.y - radius - 2} textAnchor="middle" className="drift-label">
                      {point.count}
                    </text>
                  ) : null}
                </g>
              );
            })}
            {sanitizedBins.map((point, index) => {
              const mapped = pointToXY(point);
              const radius = 2.4 + (point.count / maxCount) * 6.2;
              return (
                <g key={`san-${index}`}>
                  <circle cx={mapped.x} cy={mapped.y} r={radius} fill="#2f7f5e" fillOpacity={0.35} />
                  {point.count > 1 ? (
                    <text x={mapped.x} y={mapped.y - radius - 2} textAnchor="middle" className="drift-label">
                      {point.count}
                    </text>
                  ) : null}
                </g>
              );
            })}
            <text x={padding - 2} y={height - 6} className="drift-label" textAnchor="end">
              0
            </text>
            <text x={width - padding} y={height - 6} className="drift-label" textAnchor="end">
              {asFixed(maxValue, 0)}
            </text>
            <text x={padding - 6} y={padding + 8} className="drift-label" textAnchor="end">
              {asFixed(maxValue, 0)}
            </text>
            <text x={width - padding - 6} y={padding + 14} className="drift-label" textAnchor="end">
              y = x
            </text>
            <text x={width / 2} y={height - 8} className="drift-label" textAnchor="middle">
              drift_t
            </text>
            <text x={12} y={height / 2} className="drift-label" transform={`rotate(-90 12 ${height / 2})`} textAnchor="middle">
              drift_t+1
            </text>
          </svg>
        </div>
      ) : (
        <p className="muted">No phase points yet.</p>
      )}
      <div className="drift-legend">
        <span className="legend-item">
          <span className="legend-swatch raw" />
          Raw (Condition A)
        </span>
        <span className="legend-item">
          <span className="legend-swatch sanitized" />
          Sanitized (Condition B)
        </span>
      </div>
      <p className="muted">
        Regime guide: above diagonal = drift reinforcement, on diagonal = stable dialect, below diagonal = correction/damping.
      </p>
    </section>
  );
}

function agentSlotsForSummary(summary: ConditionSummary): string[] {
  if (isCanonicalBeliefDriftProfile(summary.profile)) {
    return buildTriangleAgentSequence(summary.runConfig.agentCount).map((entry) => entry.slotLabel);
  }
  const ordered: string[] = [];
  const seen = new Set<string>();
  for (const trace of summary.traces) {
    if (!seen.has(trace.agentSlot)) {
      seen.add(trace.agentSlot);
      ordered.push(trace.agentSlot);
    }
  }
  return ordered;
}

function slotSeriesPath(params: {
  points: Array<{ turn: number; value: number }>;
  maxTurn: number;
  maxValue: number;
  width: number;
  height: number;
  paddingX: number;
  paddingY: number;
}): string {
  const { points, maxTurn, maxValue, width, height, paddingX, paddingY } = params;
  if (points.length === 0) return "";
  const plotWidth = width - paddingX * 2;
  const plotHeight = height - paddingY * 2;
  const turnDivisor = Math.max(1, maxTurn - 1);
  const valueDivisor = Math.max(0.0001, maxValue);
  return points
    .map((point) => {
      const x = paddingX + ((point.turn - 1) / turnDivisor) * plotWidth;
      const y = paddingY + (1 - point.value / valueDivisor) * plotHeight;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function slotColor(index: number): string {
  const hue = (index * 137.508) % 360;
  return `hsl(${hue.toFixed(1)} 68% 42%)`;
}

function AgentScalingTopologyPanel({
  profile,
  rawSummary,
  sanitizedSummary
}: {
  profile: ExperimentProfile;
  rawSummary: ConditionSummary | null;
  sanitizedSummary: ConditionSummary | null;
}) {
  const isCanonical = isCanonicalBeliefDriftProfile(profile);

  const [viewCondition, setViewCondition] = useState<RepCondition>("raw");

  useEffect(() => {
    if (!isCanonical) return;
    if (viewCondition === "raw" && !rawSummary && sanitizedSummary) {
      setViewCondition("sanitized");
      return;
    }
    if (viewCondition === "sanitized" && !sanitizedSummary && rawSummary) {
      setViewCondition("raw");
    }
  }, [isCanonical, rawSummary, sanitizedSummary, viewCondition]);

  if (!isCanonical) return null;

  const hasRaw = rawSummary !== null;
  const hasSanitized = sanitizedSummary !== null;
  const summary = viewCondition === "raw" ? rawSummary : sanitizedSummary;

  if (!summary) {
    return (
      <section className="latest-card drift-chart-card agent-topology-panel">
        <h4>Panel 6 - Confidence Trajectory and Decision Error</h4>
        <p className="muted">Run RAW or SANITIZED for this profile to render confidence amplification and decision_error over turns.</p>
      </section>
    );
  }

  const slots = agentSlotsForSummary(summary);
  const maxTurn = summary.traces.at(-1)?.turnIndex ?? 0;
  const cycleLength = Math.max(1, summary.runConfig.agentCount);
  const cyclesObserved = maxTurn > 0 ? maxTurn / cycleLength : 0;

  const lineSeries = slots.map((slot) => ({
    slot,
    points: summary.traces
      .filter((trace) => trace.agentSlot === slot && trace.commitment !== null && Number.isFinite(trace.commitment))
      .map((trace) => ({ turn: trace.turnIndex, value: clamp01(trace.commitment as number) }))
  }));

  const decisionErrorSeries = summary.traces
    .map((trace) => ({ turn: trace.turnIndex, value: trace.decisionError }))
    .filter((point): point is { turn: number; value: number } => point.value !== null && Number.isFinite(point.value));
  const decisionErrorMax = Math.max(0.0001, ...decisionErrorSeries.map((point) => point.value));
  const firstDecisionErrorTurn = decisionErrorSeries.find((point) => point.value > 0)?.turn ?? null;
  const perturbationTurn = isLab3PerturbationProfile(summary.profile) ? LAB3_PERTURBATION_TURN : null;

  const lineWidth = Math.max(720, Math.min(1400, 220 + maxTurn * 7));
  const lineHeight = 230;
  const linePaddingX = 44;
  const linePaddingY = 16;
  const cycleBoundaryTurns = Array.from({ length: Math.floor(maxTurn / cycleLength) }, (_, idx) => (idx + 1) * cycleLength);
  const linePlotWidth = lineWidth - linePaddingX * 2;
  const linePlotHeight = lineHeight - linePaddingY * 2;
  const turnDivisor = Math.max(1, maxTurn - 1);
  const decisionErrorPath = slotSeriesPath({
    points: decisionErrorSeries,
    maxTurn,
    maxValue: decisionErrorMax,
    width: lineWidth,
    height: lineHeight,
    paddingX: linePaddingX,
    paddingY: linePaddingY
  });

  return (
    <section className="latest-card drift-chart-card agent-topology-panel">
      <h4>Panel 6 - Confidence Trajectory and Decision Error</h4>
      <p className="muted">Primary publication view: recursive confidence amplification alongside decision_error relative to ground truth.</p>
      <div className="topology-controls">
        <div className="segmented-toggle">
          <button type="button" className={viewCondition === "raw" ? "active" : ""} onClick={() => setViewCondition("raw")} disabled={!hasRaw}>
            Condition A (RAW)
          </button>
          <button
            type="button"
            className={viewCondition === "sanitized" ? "active" : ""}
            onClick={() => setViewCondition("sanitized")}
            disabled={!hasSanitized}
          >
            Condition B (SANITIZED)
          </button>
        </div>
      </div>
      <p className="tiny">
        Viewing: <strong>{CONDITION_LABELS[viewCondition]}</strong> | slots={slots.length} | turns={maxTurn} | cycle length={cycleLength} | cycles observed≈
        {cyclesObserved.toFixed(2)}
      </p>

      <div className="drift-chart-wrap">
        <svg viewBox={`0 0 ${lineWidth} ${lineHeight}`} className="drift-chart" role="img" aria-label="Confidence trajectory over turns by agent">
          <line x1={linePaddingX} y1={lineHeight - linePaddingY} x2={lineWidth - linePaddingX} y2={lineHeight - linePaddingY} className="drift-axis" />
          <line x1={linePaddingX} y1={linePaddingY} x2={linePaddingX} y2={lineHeight - linePaddingY} className="drift-axis" />
          {[0.25, 0.5, 0.75].map((ratio) => {
            const y = linePaddingY + (1 - ratio) * linePlotHeight;
            return <line key={`line_grid_${ratio}`} x1={linePaddingX} y1={y} x2={lineWidth - linePaddingX} y2={y} className="drift-grid" />;
          })}
          {cycleBoundaryTurns.map((turn) => {
            const x = linePaddingX + ((turn - 1) / turnDivisor) * linePlotWidth;
            return <line key={`line_cycle_${turn}`} x1={x} y1={linePaddingY} x2={x} y2={lineHeight - linePaddingY} className="heatmap-cycle-line" />;
          })}
          {lineSeries.map((series, index) => {
            const path = slotSeriesPath({
              points: series.points,
              maxTurn,
              maxValue: 1,
              width: lineWidth,
              height: lineHeight,
              paddingX: linePaddingX,
              paddingY: linePaddingY
            });
            if (!path) return null;
            return <polyline key={`agent_line_${series.slot}`} points={path} fill="none" stroke={slotColor(index)} strokeWidth={1.6} opacity={0.82} />;
          })}
          {maxTurn > 0 ? (
            <>
              <text x={linePaddingX} y={lineHeight - 2} className="drift-label">
                1
              </text>
              <text x={lineWidth - linePaddingX} y={lineHeight - 2} textAnchor="end" className="drift-label">
                {maxTurn}
              </text>
              <text x={linePaddingX - 6} y={linePaddingY + 8} textAnchor="end" className="drift-label">
                1
              </text>
              <text x={linePaddingX - 6} y={lineHeight - linePaddingY + 4} textAnchor="end" className="drift-label">
                0
              </text>
            </>
          ) : null}
        </svg>
      </div>
      <p className="tiny">Confidence Trajectory Plot: one line per agent slot in sequential speaking order.</p>

      <div className="drift-chart-wrap">
        <svg viewBox={`0 0 ${lineWidth} ${lineHeight}`} className="drift-chart" role="img" aria-label="Decision error over turns">
          <line x1={linePaddingX} y1={lineHeight - linePaddingY} x2={lineWidth - linePaddingX} y2={lineHeight - linePaddingY} className="drift-axis" />
          <line x1={linePaddingX} y1={linePaddingY} x2={linePaddingX} y2={lineHeight - linePaddingY} className="drift-axis" />
          {[0.25, 0.5, 0.75].map((ratio) => {
            const y = linePaddingY + (1 - ratio) * linePlotHeight;
            return <line key={`error_grid_${ratio}`} x1={linePaddingX} y1={y} x2={lineWidth - linePaddingX} y2={y} className="drift-grid" />;
          })}
          {decisionErrorPath ? <polyline points={decisionErrorPath} fill="none" stroke="#8a2f2f" strokeWidth={2.2} /> : null}
          {perturbationTurn !== null && maxTurn >= perturbationTurn ? (
            <line
              x1={linePaddingX + ((perturbationTurn - 1) / turnDivisor) * linePlotWidth}
              y1={linePaddingY}
              x2={linePaddingX + ((perturbationTurn - 1) / turnDivisor) * linePlotWidth}
              y2={lineHeight - linePaddingY}
              className="heatmap-cycle-line"
            />
          ) : null}
          {cycleBoundaryTurns.map((turn) => {
            const x = linePaddingX + ((turn - 1) / turnDivisor) * linePlotWidth;
            return <line key={`drift_cycle_${turn}`} x1={x} y1={linePaddingY} x2={x} y2={lineHeight - linePaddingY} className="heatmap-cycle-line" />;
          })}
          {maxTurn > 0 ? (
            <>
              <text x={linePaddingX} y={lineHeight - 2} className="drift-label">
                1
              </text>
              <text x={lineWidth - linePaddingX} y={lineHeight - 2} textAnchor="end" className="drift-label">
                {maxTurn}
              </text>
              <text x={linePaddingX - 6} y={linePaddingY + 8} textAnchor="end" className="drift-label">
                {asFixed(decisionErrorMax, 2)}
              </text>
              <text x={linePaddingX - 6} y={lineHeight - linePaddingY + 4} textAnchor="end" className="drift-label">
                0
              </text>
            </>
          ) : null}
        </svg>
      </div>
      <p className="tiny">
        Decision Error Plot: turn vs decision_error (|value-ground truth| / ground truth). First non-zero turn={firstDecisionErrorTurn ?? "n/a"}.
        {isLab3PerturbationProfile(summary.profile)
          ? ` Perturbation schedule: turns 1-5 value=${LAB3_GROUND_TRUTH_VALUE}, turn ${LAB3_PERTURBATION_TURN} inject value=${LAB3_INJECTED_VALUE}, then ${
              isLab3PropagationIsolationProfile(summary.profile)
                ? "RAW recursive propagation vs SANITIZED recursive damping toward ground truth."
                : "recursive propagation from reinjected state."
            }`
          : ""}
      </p>
    </section>
  );
}

function EdgeTransferPanel({
  profile,
  rawSummary,
  sanitizedSummary
}: {
  profile: ExperimentProfile;
  rawSummary: ConditionSummary | null;
  sanitizedSummary: ConditionSummary | null;
}) {
  void profile;
  type EdgeKey = "edgeAB";
  const edges: Array<{ key: EdgeKey; label: string; devLabel: string; cleanLabel: string }> = [
    { key: "edgeAB", label: "A→B", devLabel: "P(dev_B|dev_A)", cleanLabel: "P(dev_B|clean_A)" }
  ];
  const hasAnyData = Boolean(rawSummary || sanitizedSummary);

  return (
    <section className="latest-card">
      <h4>Edge Transfer Telemetry</h4>
      <p className="muted">
        Cross-agent propagation probabilities on adjacent transitions. Primary signal: P(dev_B|dev_A).
      </p>
      {hasAnyData ? (
        <div className="policy-inline">
          {edges.map((edge) => {
            const rawEdge = rawSummary ? rawSummary[edge.key] : null;
            const sanEdge = sanitizedSummary ? sanitizedSummary[edge.key] : null;
            return (
              <div key={edge.key}>
                <p className="tiny">
                  <strong>{edge.label}</strong> | RAW {edge.devLabel}: {asPercent(rawEdge?.pDevGivenDev ?? null)} | RAW {edge.cleanLabel}:{" "}
                  {asPercent(rawEdge?.pDevGivenClean ?? null)} | RAW Δ: {asFixed(rawEdge?.delta ?? null, 4)} | pairs: {rawEdge?.pairCount ?? "n/a"}
                </p>
                <p className="tiny">
                  <strong>{edge.label}</strong> | SAN {edge.devLabel}: {asPercent(sanEdge?.pDevGivenDev ?? null)} | SAN {edge.cleanLabel}:{" "}
                  {asPercent(sanEdge?.pDevGivenClean ?? null)} | SAN Δ: {asFixed(sanEdge?.delta ?? null, 4)} | pairs: {sanEdge?.pairCount ?? "n/a"}
                </p>
              </div>
            );
          })}
        </div>
      ) : (
        <p className="muted">Run RAW and SANITIZED to populate edge transfer telemetry.</p>
      )}
    </section>
  );
}

function setConditionResult(
  current: ResultsByProfile,
  profile: ExperimentProfile,
  condition: RepCondition,
  summary: ConditionSummary | null
): ResultsByProfile {
  return {
    ...current,
    [profile]: {
      ...current[profile],
      [condition]: summary
    }
  };
}

function SectionDocModal({ title, body, onClose }: { title: string; body: string; onClose: () => void }) {
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true">
      <div className="modal-sheet">
        <div className="modal-head">
          <h2>{title}</h2>
          <button onClick={onClose}>Close</button>
        </div>
        <pre className="doc-block">{body}</pre>
      </div>
    </div>
  );
}

export default function HomePage() {
  const guardianEnabled = (process.env.NEXT_PUBLIC_GUARDIAN_ENABLED ?? "1").trim() !== "0";
  const [labSurface, setLabSurface] = useState<LabSurface>("default");
  const [apiProvider, setApiProvider] = useState<APIProvider>(DEFAULT_PROVIDER);
  const [apiKey, setApiKey] = useState<string>("");
  const [model, setModel] = useState<string>(DEFAULT_MODEL);

  const [selectedProfile, setSelectedProfile] = useState<ExperimentProfile>(DEFAULT_PROFILE);
  const [objectiveMode, setObjectiveMode] = useState<ObjectiveMode>("parse_only");

  const [selectedCondition, setSelectedCondition] = useState<RepCondition>("raw");
  const [temperature, setTemperature] = useState<number>(DEFAULT_TEMPERATURE);
  const [turnBudget, setTurnBudget] = useState<number>(DEFAULT_TURNS);
  const [llmMaxTokens, setLlmMaxTokens] = useState<number>(DEFAULT_MAX_TOKENS);
  const [matrixReplicates, setMatrixReplicates] = useState<number>(DEFAULT_MATRIX_REPLICATES);
  const [modelMatrixInput, setModelMatrixInput] = useState<string>(DEFAULT_MODEL);
  const [interTurnDelayMs, setInterTurnDelayMs] = useState<number>(DEFAULT_INTER_TURN_DELAY_MS);
  const [maxHistoryTurns, setMaxHistoryTurns] = useState<number>(DEFAULT_MAX_HISTORY_TURNS);
  const [initialStep, setInitialStep] = useState<number>(0);
  const [stopOnFirstFailure, setStopOnFirstFailure] = useState<boolean>(false);

  const [results, setResults] = useState<ResultsByProfile>(emptyResults());
  const [activeTrace, setActiveTrace] = useState<TurnTrace | null>(null);
  const [liveTelemetryRows, setLiveTelemetryRows] = useState<TurnTrace[]>([]);
  const [liveTelemetryNewestFirst, setLiveTelemetryNewestFirst] = useState<boolean>(false);
  const [outputTurnNewestFirst, setOutputTurnNewestFirst] = useState<boolean>(true);
  const [traceViewerFollowLatest, setTraceViewerFollowLatest] = useState<boolean>(true);
  const [traceViewerTurn, setTraceViewerTurn] = useState<number | null>(null);
  const [liveTraceCondition, setLiveTraceCondition] = useState<RepCondition>("raw");
  const [matrixRows, setMatrixRows] = useState<MatrixTrialRow[]>([]);

  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [runPhaseText, setRunPhaseText] = useState<string>("Idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [guardianRuntimeState, setGuardianRuntimeState] = useState<GuardianRuntimeState>(guardianEnabled ? "unknown" : "disabled");
  const [showSpec, setShowSpec] = useState<boolean>(false);

  const apiKeyInputRef = useRef<HTMLInputElement | null>(null);
  const runControlRef = useRef<{ cancelled: boolean }>({ cancelled: false });
  const panel1MonitorRef = useRef<HTMLElement | null>(null);
  const telemetryTableWrapRef = useRef<HTMLDivElement | null>(null);
  const outputTurnTableWrapRef = useRef<HTMLDivElement | null>(null);
  const [panel1MonitorHeight, setPanel1MonitorHeight] = useState<number>(460);

  useEffect(() => {
    const defaultsVersion = localStorage.getItem(STORAGE_UI_DEFAULTS_VERSION_KEY);
    const shouldMigrateDefaults = defaultsVersion !== UI_DEFAULTS_VERSION;
    let hydratedModel = DEFAULT_MODEL;
    if (shouldMigrateDefaults) {
      setApiProvider(DEFAULT_PROVIDER);
      setModel(DEFAULT_MODEL);
      localStorage.setItem(STORAGE_API_PROVIDER_KEY, DEFAULT_PROVIDER);
      localStorage.setItem(STORAGE_API_MODEL_KEY, DEFAULT_MODEL);
      localStorage.setItem(STORAGE_UI_DEFAULTS_VERSION_KEY, UI_DEFAULTS_VERSION);
    } else {
      const validProviders = new Set(providerOptions.map((provider) => provider.value));
      const savedProvider = localStorage.getItem(STORAGE_API_PROVIDER_KEY);
      if (savedProvider && validProviders.has(savedProvider as APIProvider)) {
        setApiProvider(savedProvider as APIProvider);
      }

      const savedModel = localStorage.getItem(STORAGE_API_MODEL_KEY);
      if (savedModel) {
        setModel(savedModel);
        hydratedModel = savedModel;
      }
    }

    setModelMatrixInput(hydratedModel);

    // Never persist or auto-hydrate API keys into the UI.
    localStorage.removeItem(STORAGE_API_KEY_VALUE_KEY);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    setLabSurface(detectLabSurface(window.location.hostname));
  }, []);

  const effectiveProvider = useMemo(() => resolveProvider(apiProvider, apiKey), [apiProvider, apiKey]);
  const effectiveModelOptions = useMemo(() => modelOptionsForProvider(effectiveProvider), [effectiveProvider]);
  const websiteURL = (process.env.NEXT_PUBLIC_GUARDIAN_WEBSITE_URL ?? "https://guardianai.fr").trim();
  const githubURL = (process.env.NEXT_PUBLIC_GITHUB_REPO_URL ?? "https://github.com/GuardianAI1/guardianai-agent-drift-lab3-web").trim();
  const isApp2OrApp3 = labSurface === "app2" || labSurface === "app3";
  const brandSubtitle = isApp2OrApp3 ? "Multi-Agent Lab" : "Multi-agent Drift Lab";
  const brandExperimentSubtitle = labSurface === "app3" ? "— Perturbation Experiment" : labSurface === "app2" ? "— Drift Experiment" : null;
  const brandTagline = isApp2OrApp3
    ? "A deterministic multi-agent loop for observing stability and drift under recursion."
    : "A deterministic multi-agent interaction loop used to observe how recursive exchanges affect trajectory stability and drift.";
  const specDownloads = useMemo(
    () =>
      isApp2OrApp3
        ? SPEC_DOWNLOADS.map((spec) => (spec.kind === "Reference Experiment" ? { ...spec, title: "Multi-Agent Lab" } : spec))
        : SPEC_DOWNLOADS,
    [isApp2OrApp3]
  );

  useEffect(() => {
    const allowedModels = effectiveModelOptions.map((option) => option.value);
    if (!allowedModels.includes(model)) {
      setModel(defaultModelForProvider(effectiveProvider));
    }
  }, [effectiveModelOptions, effectiveProvider, model]);

  useEffect(() => {
    localStorage.setItem(STORAGE_API_PROVIDER_KEY, apiProvider);
  }, [apiProvider]);

  useEffect(() => {
    localStorage.setItem(STORAGE_API_MODEL_KEY, model);
  }, [model]);

  const guardianStatusLabel = !guardianEnabled ? "Disabled" : guardianRuntimeState === "connected" ? "Connected" : guardianRuntimeState === "degraded" ? "Degraded" : "Offline";
  const guardianStatusDotClass = !guardianEnabled ? "warn" : guardianRuntimeState === "connected" ? "good" : guardianRuntimeState === "degraded" ? "bad" : "warn";
  const serverKeyDotClass = apiKey.trim() ? "good" : "bad";

  const profileResults = results[selectedProfile];
  const rawSummary = profileResults.raw;
  const sanitizedSummary = profileResults.sanitized;
  const selectedScriptCard = useMemo(() => scriptCardCopyForProfile(selectedProfile), [selectedProfile]);
  const consensusEval = evaluateConsensusCollapse(rawSummary, sanitizedSummary);
  const closure = closureVerdict(consensusEval);
  const structuralPattern = structuralPatternInterpretation(consensusEval);
  const matrixAggregate = useMemo(() => aggregateMatrixRows(matrixRows), [matrixRows]);
  const matrixRecentRows = useMemo(() => matrixRows.slice(-8).reverse(), [matrixRows]);
  const liveTelemetryDisplayRows = useMemo(
    () => (liveTelemetryNewestFirst ? [...liveTelemetryRows].reverse() : liveTelemetryRows),
    [liveTelemetryNewestFirst, liveTelemetryRows]
  );
  const liveCycleReinforcementByTurn = useMemo(() => cycleReinforcement3ByTurn(liveTelemetryRows), [liveTelemetryRows]);
  const liveBasinStateByTurn = useMemo(() => basinStateByTurn(liveTelemetryRows), [liveTelemetryRows]);
  const monitorCondition: RepCondition = isRunning ? liveTraceCondition : selectedCondition;
  const monitorSummary = results[selectedProfile][monitorCondition];
  const monitorTraces = useMemo(() => monitorSummary?.traces ?? [], [monitorSummary]);
  const monitorCycleReinforcementByTurn = useMemo(() => cycleReinforcement3ByTurn(monitorTraces), [monitorTraces]);
  const monitorBasinStateByTurn = useMemo(() => basinStateByTurn(monitorTraces), [monitorTraces]);
  const monitorLatestTrace = monitorTraces.length > 0 ? monitorTraces[monitorTraces.length - 1] : activeTrace;
  const monitorViewedTrace =
    traceViewerTurn !== null ? monitorTraces.find((trace) => trace.turnIndex === traceViewerTurn) ?? null : null;
  const monitorTrace = monitorViewedTrace ?? monitorLatestTrace;
  const monitorTraceIndex = monitorTrace ? monitorTraces.findIndex((trace) => trace.turnIndex === monitorTrace.turnIndex) : -1;
  const canViewPrevTrace = monitorTraceIndex > 0;
  const canViewNextTrace = monitorTraceIndex >= 0 && monitorTraceIndex < monitorTraces.length - 1;
  const monitorTurnMax = monitorTraces.length > 0 ? monitorTraces[monitorTraces.length - 1].turnIndex : 0;
  const outputTurnDisplayRows = useMemo(
    () => (outputTurnNewestFirst ? [...monitorTraces].reverse() : monitorTraces),
    [monitorTraces, outputTurnNewestFirst]
  );
  const liveTurnProgressPct = turnBudget > 0 ? Math.min(100, (monitorTurnMax / turnBudget) * 100) : 0;
  const liveLockInScore =
    monitorLatestTrace?.commitmentDelta !== null &&
    monitorLatestTrace?.commitmentDelta !== undefined &&
    monitorLatestTrace?.constraintGrowth !== null &&
    monitorLatestTrace?.constraintGrowth !== undefined
      ? monitorLatestTrace.commitmentDelta - monitorLatestTrace.constraintGrowth
      : null;
  const liveCycleReinforcement3 =
    monitorLatestTrace !== null && monitorLatestTrace !== undefined
      ? monitorCycleReinforcementByTurn.get(monitorLatestTrace.turnIndex) ?? null
      : null;
  const liveBasinState =
    monitorLatestTrace !== null && monitorLatestTrace !== undefined
      ? monitorBasinStateByTurn.get(monitorLatestTrace.turnIndex) ?? null
      : null;
  const liveTrajectoryDynamics = trajectoryDynamicsFromSummary(monitorSummary);

  useEffect(() => {
    const panelNode = panel1MonitorRef.current;
    if (!panelNode) return;

    const syncHeight = () => {
      const rect = panelNode.getBoundingClientRect();
      if (rect.height > 0) {
        setPanel1MonitorHeight(Math.round(rect.height));
      }
    };

    syncHeight();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", syncHeight);
      return () => window.removeEventListener("resize", syncHeight);
    }

    const observer = new ResizeObserver(() => syncHeight());
    observer.observe(panelNode);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (monitorTraces.length === 0) {
      if (traceViewerTurn !== null) setTraceViewerTurn(null);
      return;
    }

    if (traceViewerFollowLatest) {
      const latestTurn = monitorTraces[monitorTraces.length - 1].turnIndex;
      if (traceViewerTurn !== latestTurn) {
        setTraceViewerTurn(latestTurn);
      }
      return;
    }

    if (traceViewerTurn === null) {
      setTraceViewerTurn(monitorTraces[monitorTraces.length - 1].turnIndex);
      return;
    }

    const exists = monitorTraces.some((trace) => trace.turnIndex === traceViewerTurn);
    if (!exists) {
      setTraceViewerTurn(monitorTraces[monitorTraces.length - 1].turnIndex);
    }
  }, [monitorTraces, traceViewerFollowLatest, traceViewerTurn]);

  useEffect(() => {
    if (!traceViewerFollowLatest) return;
    const wrap = outputTurnTableWrapRef.current;
    if (!wrap) return;
    if (outputTurnNewestFirst) {
      wrap.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }
    wrap.scrollTo({ top: wrap.scrollHeight, behavior: "smooth" });
  }, [monitorTurnMax, outputTurnNewestFirst, traceViewerFollowLatest]);

  function setNormalizedApiKey(rawValue: string) {
    setApiKey(normalizeApiKeyInput(rawValue));
  }

  async function requestLLM(params: { model: string; prompt: string; systemPrompt: string }): Promise<string> {
    const requestApiKey = normalizeApiKeyInput(apiKeyInputRef.current?.value ?? apiKey);
    if (requestApiKey !== apiKey) {
      setApiKey(requestApiKey);
    }

    const response = await requestJSON<{ content: string }>("/api/llm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: params.model,
        prompt: params.prompt,
        apiKey: requestApiKey,
        providerPreference: apiProvider,
        temperature,
        maxTokens: llmMaxTokens,
        systemPrompt: params.systemPrompt,
        mistralJsonSchemaMode: false
      })
    }, { maxAttempts: effectiveProvider === "mistral" ? 3 : 2 });

    return response.content ?? "";
  }

  async function requestGuardianObservation(params: {
    runId: string;
    turnId: number;
    agent: AgentRole;
    output: string;
    deterministicConstraint: string;
    constraintIds: string[] | null;
    reasoningDepth: number | null;
    confidence: number | null;
    elapsedTimeMs: number | null;
    externalRefresh: number | null;
  }): Promise<GuardianObserveResponse> {
    return requestJSON<GuardianObserveResponse>("/api/guardian/observe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        runId: params.runId,
        turnId: params.turnId,
        agentId: params.agent,
        output: params.output,
        deterministicConstraint: params.deterministicConstraint,
        constraintIds: params.constraintIds,
        reasoningDepth: params.reasoningDepth,
        confidence: params.confidence,
        elapsedTime: params.elapsedTimeMs !== null ? params.elapsedTimeMs / 1000 : null,
        externalRefresh: params.externalRefresh
      })
    }, { maxAttempts: GUARDIAN_OBSERVE_MAX_ATTEMPTS });
  }

  async function pingGuardianObserver(): Promise<boolean> {
    try {
      await requestJSON<{ ok?: boolean }>(
        "/api/guardian/constraint",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content: "{\"probe\":\"observer\"}" })
        },
        { maxAttempts: 2 }
      );
      return true;
    } catch {
      return false;
    }
  }

  async function runCondition(
    profile: ExperimentProfile,
    condition: RepCondition,
    options?: { modelOverride?: string }
  ): Promise<ConditionSummary> {
    const forceFullHorizon = isLab3PerturbationProfile(profile);
    const activeModel = options?.modelOverride?.trim() ? options.modelOverride.trim() : model;
    const effectiveInterTurnDelayMs =
      effectiveProvider === "mistral" ? Math.max(interTurnDelayMs, MISTRAL_MIN_INTER_TURN_DELAY_MS) : interTurnDelayMs;
    const runConfig: RunConfig = {
      runId: createRunId(),
      profile,
      condition,
      objectiveMode,
      providerPreference: apiProvider,
      resolvedProvider: effectiveProvider,
      modelA: activeModel,
      modelB: activeModel,
      agentCount: agentCountForProfile(profile),
      temperature,
      retries: FIXED_RETRIES,
      horizon: turnBudget,
      maxTokens: llmMaxTokens,
      initialStep,
      interTurnDelayMs: effectiveInterTurnDelayMs,
      maxHistoryTurns,
      stopOnFirstFailure: forceFullHorizon ? false : stopOnFirstFailure,
      strictSanitizedKeyOrder: true,
      historyAccumulation: true,
      preflightEnabled: forceFullHorizon ? false : true,
      preflightTurns: PREFLIGHT_TURNS,
      preflightAgent: preflightAgentForProfile(profile),
      preflightParseOkMin: PREFLIGHT_PARSE_OK_MIN,
      preflightStateOkMin: PREFLIGHT_STATE_OK_MIN,
      createdAt: new Date().toISOString()
    };

    const startedAt = new Date().toISOString();
    const traces: TurnTrace[] = [];
    const agentSequence = agentSequenceForProfile(profile);

    let authoritativeStep = initialStep;
    let injectedPrevState = initialStateLiteralForProfile(profile, initialStep);
    const historyBuffer: string[] = [];
    const previousIndentAvgByAgent: Partial<Record<AgentRole, number>> = {};
    const initialContextLength = injectedPrevState.length;

    let failed = false;
    let failureReason: string | undefined;
    let guardianAvailableThisRun = guardianEnabled;
    let guardianRetryAfterTurn = 1;
    let guardianConsecutiveFailures = 0;

    setResults((prev) => setConditionResult(prev, profile, condition, null));
    setLiveTraceCondition(condition);
    setLiveTelemetryRows([]);

    for (let turn = 1; turn <= turnBudget; turn += 1) {
      if (runControlRef.current.cancelled) break;

      const agentEntry = agentSequence[(turn - 1) % agentSequence.length];
      const agent = agentEntry.role;
      const agentSlot = agentEntry.slotLabel;
      const expectedStep = expectedStepForTurn(profile, agent, authoritativeStep);
      const expectedBytes = expectedLiteralForTurn(profile, expectedStep, injectedPrevState);

      const historySlice = historyBuffer.slice(Math.max(0, historyBuffer.length - maxHistoryTurns));
      const historyBlock = buildHistoryBlock(historySlice);
      const promptContextLength = historyBlock.length + injectedPrevState.length;
      const contextLengthGrowth = promptContextLength - initialContextLength;

      const prompt = buildAgentPrompt(profile, condition, agent, historyBlock, injectedPrevState, expectedStep, turn);
      const agentModel = activeModel;

      let outputBytes = "";
      const llmStartMs = Date.now();
      let llmCompleted = false;
      let llmFailureMessage: string | null = null;
      for (let llmAttempt = 1; llmAttempt <= RUN_LEVEL_LLM_MAX_ATTEMPTS; llmAttempt += 1) {
        try {
          outputBytes = await requestLLM({
            model: agentModel,
            prompt: prompt.userPrompt,
            systemPrompt: prompt.systemPrompt
          });
          llmCompleted = true;
          break;
        } catch (error) {
          const message = error instanceof Error ? error.message : "Unknown";
          const retryable = isRunLevelRetryableLLMError(message);
          const hasMoreAttempts = llmAttempt < RUN_LEVEL_LLM_MAX_ATTEMPTS;

          if (retryable && hasMoreAttempts && !runControlRef.current.cancelled) {
            setRunPhaseText(
              `${PROFILE_LABELS[profile]} — ${CONDITION_LABELS[condition]} | Turn ${turn} (${agentSlot}) transport retry ${
                llmAttempt + 1
              }/${RUN_LEVEL_LLM_MAX_ATTEMPTS}`
            );
            await sleep(runLevelRetryDelayMs(llmAttempt, effectiveProvider, message));
            continue;
          }

          const retrySuffix = retryable ? ` (run-level retry exhausted after ${llmAttempt} attempts).` : "";
          llmFailureMessage = `LLM failure at turn ${turn} (${agentSlot}): ${message}${retrySuffix}`;
          break;
        }
      }

      if (!llmCompleted) {
        failed = true;
        failureReason = llmFailureMessage ?? `LLM failure at turn ${turn} (${agentSlot}): Request did not complete.`;
        const partialSummary = buildConditionSummary({
          runConfig,
          condition,
          startedAt,
          traces,
          failed,
          failureReason,
          finishedAt: new Date().toISOString()
        });
        setResults((prev) => setConditionResult(prev, profile, condition, partialSummary));
        break;
      }
      const elapsedTimeMs = Date.now() - llmStartMs;

      let guardianGateState: "CONTINUE" | "PAUSE" | "YIELD" | null = null;
      let guardianStructuralRecommendation: "CONTINUE" | "SLOW" | "REOPEN" | "DEFER" | null = null;
      let guardianReasonCodes: string[] = [];
      let guardianEnergyV: number | null = null;
      let guardianAuthorityTrend: number | null = null;
      let guardianRevisionMode: string | null = null;
      let guardianTrajectoryState: string | null = null;
      let guardianTemporalResistanceDetected: number | null = null;
      let guardianObserveError: string | null = null;

      const [rawHash, expectedHash] = await Promise.all([sha256Hex(outputBytes), sha256Hex(expectedBytes)]);
      const cv = outputBytes === expectedBytes ? 0 : 1;
      const drift = boundaryDeviation(outputBytes, expectedBytes);
      const indent = indentationTelemetry(outputBytes);
      const previousIndentAvg = previousIndentAvgByAgent[agent];
      const indentDelta = typeof previousIndentAvg === "number" ? indent.indentAvg - previousIndentAvg : null;
      let bTransformOk: number | null = null;
      let bTransformReason: string | undefined;
      if (profile === "drift_amplifying_loop" && agent === "B") {
        const transform = evaluateMonotoneBTransform(injectedPrevState, outputBytes);
        bTransformOk = transform.ok ? 1 : 0;
        bTransformReason = transform.reason;
      }

      let parseOk = 0;
      let stateOk = 0;
      let pf = 0;
      let ld = 0;
      let parsedStep: number | null = null;
      let parseError: string | undefined;
      let parsedData: Record<string, unknown> | undefined;
      let injectedBytesNext = injectedPrevState;
      let historyEntry = injectedPrevState;

      const boundaryViolation = boundaryContractViolation(outputBytes);
      if (boundaryViolation) {
        pf = 1;
        parseError = boundaryViolation;
        if (condition === "raw") {
          injectedBytesNext = outputBytes;
          historyEntry = outputBytes;
        } else {
          injectedBytesNext = injectedPrevState;
          historyEntry = injectedPrevState;
        }
      } else {
        try {
          const parsed = JSON.parse(outputBytes) as unknown;
          const canonicalized = canonicalizeSanitizedOutput(parsed, profile, condition);
          const contract = parseContractPayload(parsed, profile);
          parsedStep = canonicalized.parsedStep;
          parsedData = canonicalized.parsedData;
          parseOk = 1;

          const statePass = isBeliefLoopProfile(profile)
            ? contract.ok && (!beliefProfileUsesStep(profile) || contract.parsedStep === expectedStep)
            : contract.ok && parsedStep === expectedStep;
          if (statePass) {
            stateOk = 1;
          } else {
            ld = 1;
            if (!parseError && !contract.ok && contract.reason) {
              parseError = contract.reason;
            }
          }

          if (condition === "raw") {
            injectedBytesNext = outputBytes;
            historyEntry = outputBytes;
          } else if (canonicalized.ok && canonicalized.canonical) {
            injectedBytesNext = canonicalized.canonical;
            historyEntry = canonicalized.canonical;
          } else {
            injectedBytesNext = injectedPrevState;
            historyEntry = injectedPrevState;
            parseError = canonicalized.reason;
          }
        } catch (error) {
          pf = 1;
          parseError = error instanceof Error ? error.message : "JSON parse failed";
          if (condition === "raw") {
            injectedBytesNext = outputBytes;
            historyEntry = outputBytes;
          } else {
            injectedBytesNext = injectedPrevState;
            historyEntry = injectedPrevState;
          }
        }
      }

      // RAW condition must remain byte-identical across reinjection.
      // If this ever trips, it means a hidden normalization path was introduced.
      if (condition === "raw" && injectedBytesNext !== outputBytes) {
        throw new Error(`RAW reinjection integrity violation at turn ${turn} (${agentSlot}): output bytes were modified before reinjection.`);
      }

      const objectiveFailure = isObjectiveFailure(profile, agent, objectiveMode, pf, ld, cv) ? 1 : 0;
      const recentPfWindow = [...traces.slice(-19).map((trace) => trace.pf), pf];
      const rollingPf20 = recentPfWindow.reduce((sum, value) => sum + value, 0) / recentPfWindow.length;
      const recentDriftWindow = [...traces.slice(-19).map((trace) => trace.deviationMagnitude), drift.deviationMagnitude];
      const rollingDriftP95 = percentile(recentDriftWindow, 0.95) ?? 0;
      // "dev" event excludes tiny newline-only noise so reinforcement remains informative.
      const devState = drift.deviationMagnitude > DRIFT_DEV_EVENT_THRESHOLD ? 1 : 0;
      const wasHealthyBefore = traces.every((trace) => trace.objectiveFailure === 0);
      const uptime = wasHealthyBefore && objectiveFailure === 0 ? 1 : 0;
      const previousTrace = traces.length > 0 ? traces[traces.length - 1] : null;
      const previousConsensus = previousTrace ? consensusFields(previousTrace) : null;
      const currentConsensus = consensusFieldsFromParsedData(parsedData);
      const previousTwoTrace = traces.length > 1 ? traces[traces.length - 2] : null;
      const previousTwoConsensus = previousTwoTrace ? consensusFields(previousTwoTrace) : null;
      const reasoningDepth = currentConsensus ? currentConsensus.evidenceIds.length : null;
      const commitment = currentConsensus ? currentConsensus.confidence : null;
      const decisionValue = currentConsensus ? lab3ClaimValue(currentConsensus.claim) : null;
      const decisionError = decisionErrorForConsensus(profile, currentConsensus);
      const authorityWeights = commitment;
      const contradictionSignal =
        currentConsensus && previousConsensus ? (currentConsensus.stance === previousConsensus.stance ? 0 : 1) : null;
      const alternativeVariance = currentConsensus
        ? evidenceJaccardDistance(currentConsensus.evidenceIds, previousConsensus?.evidenceIds ?? null)
        : null;
      const agreementRate =
        currentConsensus && previousConsensus
          ? isBeliefTriangle3AgentProfile(profile) && agent === "C" && previousTwoConsensus
            ? currentConsensus.stance === previousConsensus.stance && previousConsensus.stance === previousTwoConsensus.stance
              ? 1
              : 0
            : currentConsensus.stance === previousConsensus.stance
              ? 1
              : 0
          : null;
      const evidenceDiversity =
        currentConsensus && previousConsensus
          ? isBeliefTriangle3AgentProfile(profile) && agent === "C" && previousTwoConsensus
            ? evidenceCitationDiversity([
                previousTwoConsensus.evidenceIds,
                previousConsensus.evidenceIds,
                currentConsensus.evidenceIds
              ])
            : evidenceCitationDiversity([previousConsensus.evidenceIds, currentConsensus.evidenceIds])
          : null;
      const evidenceDelta = currentConsensus ? newEvidenceCount(currentConsensus.evidenceIds, previousConsensus?.evidenceIds ?? null) : null;
      const constraintGrowth = evidenceDelta;
      const commitmentDelta =
        commitment !== null && previousConsensus ? commitment - previousConsensus.confidence : null;
      const previousReasoningDepth = previousTrace?.reasoningDepth ?? null;
      const depthDelta =
        reasoningDepth !== null && previousReasoningDepth !== null ? reasoningDepth - previousReasoningDepth : null;
      const commitmentDeltaMin = isBeliefTriangle3AgentProfile(profile)
        ? TRIANGLE_DRIFT_COMMITMENT_DELTA_MIN
        : STRUCTURAL_DRIFT_COMMITMENT_DELTA_MIN;
      const depthStableForDriftRule =
        depthDelta !== null &&
        (isBeliefTriangle3AgentProfile(profile) ? depthDelta <= TRIANGLE_DRIFT_DEPTH_EPSILON : depthDelta === 0);
      const driftRuleSatisfied =
        commitmentDelta !== null &&
        evidenceDelta !== null &&
        depthStableForDriftRule &&
        commitmentDelta > commitmentDeltaMin &&
        evidenceDelta === 0
          ? 1
          : 0;
      const driftStreak = driftRuleSatisfied === 1 ? (previousTrace?.driftStreak ?? 0) + 1 : 0;
      const structuralEpistemicDrift = driftStreak >= STRUCTURAL_DRIFT_STREAK_MIN ? 1 : 0;

      if (guardianEnabled && guardianAvailableThisRun && turn >= guardianRetryAfterTurn) {
        try {
          const guardianObservation = await requestGuardianObservation({
            runId: runConfig.runId,
            turnId: turn,
            agent,
            output: outputBytes,
            deterministicConstraint: expectedBytes,
            constraintIds: currentConsensus?.evidenceIds ?? null,
            reasoningDepth,
            confidence: commitment,
            elapsedTimeMs,
            externalRefresh: turn === 1 || (constraintGrowth !== null && constraintGrowth > 0) ? 1 : 0
          });
          guardianGateState = guardianObservation.gateState ?? null;
          guardianStructuralRecommendation = guardianObservation.structuralRecommendation ?? null;
          guardianReasonCodes = Array.isArray(guardianObservation.reasonCodes) ? guardianObservation.reasonCodes : [];
          guardianEnergyV = guardianObservation.triangleV ?? null;
          guardianAuthorityTrend = guardianObservation.triangleDeltaV ?? null;
          guardianRevisionMode = guardianObservation.triangleCircleMode ?? null;
          guardianTrajectoryState = guardianObservation.triangleSpiralMode ?? null;
          guardianTemporalResistanceDetected = guardianObservation.triangleInvariantViolation ?? null;
          guardianConsecutiveFailures = 0;
          setGuardianRuntimeState((prev) => (prev === "connected" ? prev : "connected"));
        } catch (error) {
          guardianObserveError = error instanceof Error ? "Observer unavailable." : "Observer unavailable.";
          guardianConsecutiveFailures += 1;
          guardianRetryAfterTurn = turn + Math.min(30, Math.max(2, guardianConsecutiveFailures * 2));
          setGuardianRuntimeState("degraded");
          setRunPhaseText(
            `${PROFILE_LABELS[profile]} — ${CONDITION_LABELS[condition]} | Observer unavailable (retry at turn ${guardianRetryAfterTurn})`
          );
        }
      } else if (guardianEnabled && guardianAvailableThisRun) {
        guardianObserveError = `Observer retry pending (turn ${guardianRetryAfterTurn}).`;
      } else if (guardianEnabled) {
        guardianObserveError = "Observer unavailable.";
      }

      const provisionalTrace: TurnTrace = {
        runId: runConfig.runId,
        profile,
        condition,
        turnIndex: turn,
        agent,
        agentSlot,
        agentModel,
        inputBytes: injectedPrevState,
        historyBytes: historyBlock,
        outputBytes,
        expectedBytes,
        injectedBytesNext,
        expectedStep,
        parsedStep,
        parseOk,
        stateOk,
        pf,
        cv,
        ld,
        objectiveFailure,
        uptime,
        rawHash,
        expectedHash,
        byteLength: drift.byteLength,
        lineCount: drift.lineCount,
        prefixLen: drift.prefixLen,
        suffixLen: drift.suffixLen,
        lenDeltaVsContract: drift.lenDeltaVsContract,
        deviationMagnitude: drift.deviationMagnitude,
        indentAvg: indent.indentAvg,
        indentMax: indent.indentMax,
        indentDelta,
        bTransformOk,
        bTransformReason,
        rollingPf20,
        rollingDriftP95,
        contextLength: promptContextLength,
        contextLengthGrowth,
        devState,
        guardianGateState,
        guardianStructuralRecommendation,
        guardianReasonCodes,
        guardianEnergyV,
        guardianAuthorityTrend,
        guardianRevisionMode,
        guardianTrajectoryState,
        guardianTemporalResistanceDetected,
        guardianObserveError,
        reasoningDepth,
        authorityWeights,
        contradictionSignal,
        alternativeVariance,
        agreementRate,
        evidenceDiversity,
        elapsedTimeMs,
        commitment,
        commitmentDelta,
        decisionError,
        decisionValue,
        constraintGrowth,
        evidenceDelta,
        depthDelta,
        driftRuleSatisfied,
        driftStreak,
        structuralEpistemicDrift,
        dai: null,
        daiDelta: null,
        daiRegime: null,
        parseError,
        parsedData
      };

      const latestDai = computeDaiPoints([...traces, provisionalTrace]).at(-1) ?? null;
      const trace: TurnTrace = {
        ...provisionalTrace,
        dai: latestDai?.dai ?? null,
        daiDelta: latestDai?.daiDelta ?? null,
        daiRegime: latestDai?.regime ?? null
      };

      traces.push(trace);
      previousIndentAvgByAgent[agent] = indent.indentAvg;
      setActiveTrace(trace);
      setLiveTelemetryRows((prev) => [...prev, trace]);

      if (pf === 0 && ld === 0) {
        authoritativeStep = expectedStep;
      }

      injectedPrevState = injectedBytesNext;
      historyBuffer.push(historyEntry);

      if (objectiveFailure === 1 && !failed) {
        failed = true;
        failureReason = objectiveFailureReason(objectiveMode, pf, ld, cv);
      }

      const partialSummary = buildConditionSummary({
        runConfig,
        condition,
        startedAt,
        traces,
        failed,
        failureReason
      });
      setResults((prev) => setConditionResult(prev, profile, condition, partialSummary));

      const preflightTurn = Math.min(runConfig.preflightTurns, turnBudget);
      if (runConfig.preflightEnabled && turn === preflightTurn) {
        const preflightAgentTraces = traces.filter((traceRow) => traceRow.agent === runConfig.preflightAgent);
        const preflightSamples = preflightAgentTraces.length;
        const preflightParseOk = safeRate(
          preflightAgentTraces.reduce((sum, traceRow) => sum + traceRow.parseOk, 0),
          preflightSamples
        );
        const preflightStateOk = safeRate(
          preflightAgentTraces.reduce((sum, traceRow) => sum + traceRow.stateOk, 0),
          preflightSamples
        );
        const gate = preflightGateStatus({
          objectiveMode: runConfig.objectiveMode,
          parseRate: preflightParseOk,
          stateRate: preflightStateOk,
          parseMin: runConfig.preflightParseOkMin,
          stateMin: runConfig.preflightStateOkMin
        });
        if (!gate.pass) {
          failed = true;
          const gateReason = gate.requiresState
            ? `Preflight rejected at turn ${turn}: Agent ${runConfig.preflightAgent} ParseOK ${asPercent(preflightParseOk)} / ` +
              `StateOK ${asPercent(preflightStateOk)} (required ${asPercent(runConfig.preflightParseOkMin)} / ${asPercent(
                runConfig.preflightStateOkMin
              )}).`
            : `Preflight rejected at turn ${turn}: Agent ${runConfig.preflightAgent} ParseOK ${asPercent(preflightParseOk)} ` +
              `(required ${asPercent(runConfig.preflightParseOkMin)}, parse-only objective).`;
          failureReason = failureReason ? `${failureReason} | ${gateReason}` : gateReason;

          const gatedSummary = buildConditionSummary({
            runConfig,
            condition,
            startedAt,
            traces,
            failed,
            failureReason,
            finishedAt: new Date().toISOString()
          });
          setResults((prev) => setConditionResult(prev, profile, condition, gatedSummary));
          break;
        }
      }

      if (objectiveFailure === 1 && stopOnFirstFailure) {
        break;
      }

      if (turn < turnBudget) {
        await sleep(runConfig.interTurnDelayMs);
      }
    }

    return buildConditionSummary({
      runConfig,
      condition,
      startedAt,
      traces,
      failed,
      failureReason,
      finishedAt: new Date().toISOString()
    });
  }

  async function runSelectedCondition() {
    if (isRunning) return;

    setIsRunning(true);
    setGuardianRuntimeState(guardianEnabled ? "unknown" : "disabled");
    setErrorMessage(null);
    runControlRef.current.cancelled = false;
    setRunPhaseText(`${PROFILE_LABELS[selectedProfile]} — ${CONDITION_LABELS[selectedCondition]}`);

    try {
      if (guardianEnabled) {
        const guardianReady = await pingGuardianObserver();
        setGuardianRuntimeState(guardianReady ? "connected" : "degraded");
      }
      const summary = await runCondition(selectedProfile, selectedCondition);
      setResults((prev) => setConditionResult(prev, selectedProfile, selectedCondition, summary));
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Run failed.");
    } finally {
      setRunPhaseText("Idle");
      setIsRunning(false);
    }
  }

  async function runBothConditions(profile: ExperimentProfile, runLabel?: string): Promise<string[]> {
    const errors: string[] = [];

    for (const condition of ["raw", "sanitized"] as const) {
      if (runControlRef.current.cancelled) break;
      setSelectedProfile(profile);
      setRunPhaseText(
        runLabel
          ? `${PROFILE_LABELS[profile]} — ${CONDITION_LABELS[condition]} | ${runLabel}`
          : `${PROFILE_LABELS[profile]} — ${CONDITION_LABELS[condition]}`
      );
      try {
        const summary = await runCondition(profile, condition);
        setResults((prev) => setConditionResult(prev, profile, condition, summary));
      } catch (error) {
        const message = error instanceof Error ? error.message : "Run failed.";
        errors.push(`${CONDITION_LABELS[condition]}: ${message}`);
      }
    }

    return errors;
  }

  async function runBothConditionsForSelectedProfile() {
    if (isRunning) return;
    setIsRunning(true);
    setGuardianRuntimeState(guardianEnabled ? "unknown" : "disabled");
    setErrorMessage(null);
    runControlRef.current.cancelled = false;

    try {
      if (guardianEnabled) {
        const guardianReady = await pingGuardianObserver();
        setGuardianRuntimeState(guardianReady ? "connected" : "degraded");
      }
      const errors = await runBothConditions(selectedProfile);
      if (errors.length > 0) {
        setErrorMessage(errors.join(" | "));
      }
    } finally {
      setRunPhaseText("Idle");
      setIsRunning(false);
    }
  }

  async function runModelMatrix() {
    if (isRunning) return;

    const models = parseModelMatrixInput(modelMatrixInput, model);
    const replicates = Math.max(1, Math.min(20, Math.floor(Number(matrixReplicates) || 1)));

    setIsRunning(true);
    setGuardianRuntimeState(guardianEnabled ? "unknown" : "disabled");
    setErrorMessage(null);
    runControlRef.current.cancelled = false;
    setMatrixRows([]);

    const collectedRows: MatrixTrialRow[] = [];

    try {
      if (guardianEnabled) {
        const guardianReady = await pingGuardianObserver();
        setGuardianRuntimeState(guardianReady ? "connected" : "degraded");
      }
      for (const matrixModel of models) {
        if (runControlRef.current.cancelled) break;
        for (let replicate = 1; replicate <= replicates; replicate += 1) {
          if (runControlRef.current.cancelled) break;

          setRunPhaseText(`Matrix ${matrixModel} | Rep ${replicate}/${replicates} | RAW`);
          const raw = await runCondition(selectedProfile, "raw", { modelOverride: matrixModel });
          setResults((prev) => setConditionResult(prev, selectedProfile, "raw", raw));

          if (runControlRef.current.cancelled) break;

          setRunPhaseText(`Matrix ${matrixModel} | Rep ${replicate}/${replicates} | SANITIZED`);
          const sanitized = await runCondition(selectedProfile, "sanitized", { modelOverride: matrixModel });
          setResults((prev) => setConditionResult(prev, selectedProfile, "sanitized", sanitized));

          const consensus = evaluateConsensusCollapse(raw, sanitized);
          const trialRow: MatrixTrialRow = {
            profile: selectedProfile,
            model: matrixModel,
            replicate,
            closureDetected: consensus ? (consensus.pass ? 1 : 0) : null,
            lagTransferGap: consensus?.lagTransferGap ?? null,
            halfLifeGap: consensus?.halfLifeGap ?? null,
            devGapWindowMean: consensus?.devGapWindowMean ?? null,
            devGapWindowMax: consensus?.devGapWindowMax ?? null
          };

          collectedRows.push(trialRow);
          setMatrixRows(collectedRows.slice());
        }
      }

      if (runControlRef.current.cancelled) {
        setErrorMessage("Matrix run stopped by operator.");
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Matrix run failed.");
    } finally {
      setRunPhaseText("Idle");
      setIsRunning(false);
    }
  }

  function stopRun() {
    runControlRef.current.cancelled = true;
    setIsRunning(false);
    setRunPhaseText("Stopped");
  }

  function resetAll() {
    stopRun();
    setSelectedProfile(DEFAULT_PROFILE);
    setObjectiveMode("parse_only");
    setTemperature(DEFAULT_TEMPERATURE);
    setTurnBudget(DEFAULT_TURNS);
    setLlmMaxTokens(DEFAULT_MAX_TOKENS);
    setMatrixReplicates(DEFAULT_MATRIX_REPLICATES);
    setModelMatrixInput(DEFAULT_MODEL);
    setInterTurnDelayMs(DEFAULT_INTER_TURN_DELAY_MS);
    setMaxHistoryTurns(DEFAULT_MAX_HISTORY_TURNS);
    setInitialStep(0);
    setStopOnFirstFailure(false);
    setResults(emptyResults());
    setActiveTrace(null);
    setLiveTelemetryRows([]);
    setLiveTraceCondition("raw");
    setMatrixRows([]);
    setErrorMessage(null);
    setGuardianRuntimeState(guardianEnabled ? "unknown" : "disabled");
  }

  function exportSnapshotJSON() {
    const payload = {
      protocol: "Agent Lab Suite v1",
      signalVisibilityMode: SIGNAL_VISIBILITY_MODE,
      generatedAt: new Date().toISOString(),
      fixedTemperature: temperature,
      fixedRetries: FIXED_RETRIES,
      structuralGuardrailCriterion: "hidden",
      results: exportableResultsSnapshot(results),
      matrixRows: exportableMatrixRowsSnapshot(matrixRows)
    };

    downloadTextFile("snapshot.json", JSON.stringify(payload, null, 2), "application/json");
  }

  function downloadTrace(condition: RepCondition) {
    const summary = results[selectedProfile][condition];
    if (!summary) return;
    downloadTextFile(`trace_${condition}.jsonl`, traceToJsonl(summary), "application/x-ndjson");
  }

  function generateLabReport() {
    const markdown = buildLabReportMarkdown({
      generatedAt: new Date().toISOString(),
      results
    });
    downloadTextFile("lab_report.md", markdown, "text/markdown");
  }

  function downloadActiveScriptSpec() {
    const slug = selectedProfile.replace(/_/g, "-");
    const content = scriptDownloadBody(selectedProfile);
    downloadTextFile(`${slug}-script.md`, content, "text/markdown");
  }

  function jumpToNewestTelemetryRow() {
    const wrap = telemetryTableWrapRef.current;
    if (!wrap) return;
    if (liveTelemetryNewestFirst) {
      wrap.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }
    wrap.scrollTo({ top: wrap.scrollHeight, behavior: "smooth" });
  }

  function jumpToOldestTelemetryRow() {
    const wrap = telemetryTableWrapRef.current;
    if (!wrap) return;
    if (liveTelemetryNewestFirst) {
      wrap.scrollTo({ top: wrap.scrollHeight, behavior: "smooth" });
      return;
    }
    wrap.scrollTo({ top: 0, behavior: "smooth" });
  }

  function jumpToNewestOutputTurnRow() {
    const wrap = outputTurnTableWrapRef.current;
    if (!wrap) return;
    if (outputTurnNewestFirst) {
      wrap.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }
    wrap.scrollTo({ top: wrap.scrollHeight, behavior: "smooth" });
  }

  function jumpToOldestOutputTurnRow() {
    const wrap = outputTurnTableWrapRef.current;
    if (!wrap) return;
    if (outputTurnNewestFirst) {
      wrap.scrollTo({ top: wrap.scrollHeight, behavior: "smooth" });
      return;
    }
    wrap.scrollTo({ top: 0, behavior: "smooth" });
  }

  function selectMonitorTurn(turnIndex: number) {
    setTraceViewerFollowLatest(false);
    setTraceViewerTurn(turnIndex);
  }

  function viewPreviousTrace() {
    if (!canViewPrevTrace || monitorTraceIndex < 1) return;
    const previous = monitorTraces[monitorTraceIndex - 1];
    selectMonitorTurn(previous.turnIndex);
  }

  function viewNextTrace() {
    if (!canViewNextTrace || monitorTraceIndex < 0) return;
    const next = monitorTraces[monitorTraceIndex + 1];
    selectMonitorTurn(next.turnIndex);
  }

  return (
    <main className="shell">
      <section className="top-band">
        <div className="brand-strip">
          <Image src="/GuardianAILogo.png" alt="GuardianAI logo" className="brand-logo" width={72} height={72} priority />
          <div className="brand-copy">
            <div className="brand-title-line">
              <strong>GuardianAI</strong>
              <span className="brand-subtitle">{brandSubtitle}</span>
              {brandExperimentSubtitle ? <span className="brand-subtitle brand-experiment-subtitle">{brandExperimentSubtitle}</span> : null}
            </div>
            <span className="brand-tagline">{brandTagline}</span>
          </div>
        </div>

        <div className="right-toolbar">
          <div className="row-actions top-actions">
            <button onClick={exportSnapshotJSON}>Export JSON</button>
            <button onClick={() => downloadTrace("raw")} disabled={!rawSummary}>
              Download Raw Trace
            </button>
            <button onClick={() => downloadTrace("sanitized")} disabled={!sanitizedSummary}>
              Download Sanitized
            </button>
            <button onClick={generateLabReport}>Generate Lab Report</button>
          </div>
          <div className="row-actions link-actions">
            <a className="button-link" href={websiteURL} target="_blank" rel="noreferrer">
              Website
            </a>
            <a className="button-link" href={githubURL} target="_blank" rel="noreferrer">
              GitHub
            </a>
            <button onClick={() => setShowSpec(true)}>Core Spec + Access</button>
            <button onClick={downloadActiveScriptSpec}>Download Script</button>
          </div>
        </div>

        <div className="top-controls">
          <div className="field-block">
            <label>Condition</label>
            <select value={selectedCondition} onChange={(event) => setSelectedCondition(event.target.value as RepCondition)} disabled={isRunning}>
              <option value="raw">{CONDITION_LABELS.raw}</option>
              <option value="sanitized">{CONDITION_LABELS.sanitized}</option>
            </select>
          </div>

          <div className="field-block">
            <label>Provider</label>
            <select value={apiProvider} onChange={(event) => setApiProvider(event.target.value as APIProvider)} disabled={isRunning}>
              {providerOptions.map((provider) => (
                <option key={provider.value} value={provider.value}>
                  {provider.label}
                </option>
              ))}
            </select>
          </div>

          <div className="field-block">
            <label>Model (All Agents)</label>
            <select value={model} onChange={(event) => setModel(event.target.value)} disabled={isRunning}>
              {effectiveModelOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div className="field-block wide key-field">
            <div className="field-label-row">
              <label>API Key</label>
              <button
                type="button"
                className="text-action inline-action key-indicator"
                onClick={() => setApiKey("")}
                title="Clear API key and use server key"
              >
                <span className={`dot ${serverKeyDotClass}`} />
                Server Key (Hidden)
              </button>
            </div>
            <input
              ref={apiKeyInputRef}
              type="password"
              value={apiKey}
              onChange={(event) => setNormalizedApiKey(event.target.value)}
              autoComplete="new-password"
              inputMode="text"
              autoCapitalize="off"
              autoCorrect="off"
              spellCheck={false}
              data-lpignore="true"
              placeholder="Enter API key or rely on server env key"
              disabled={isRunning}
            />
          </div>

          <div className="field-block status-field">
            <label aria-hidden="true">&nbsp;</label>
            <div className="status-box">
              <div className="status-line">
                <span className={`dot ${isRunning ? "good" : "warn"}`} />
                <span>Run {isRunning ? "ON" : "OFF"}</span>
              </div>
              <div className="status-line">
                <span className={`dot ${guardianStatusDotClass}`} />
                <span>Guardian {guardianStatusLabel}</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="spec-strip">
        <h3>Specifications</h3>
        <div className="spec-strip-grid">
          {specDownloads.map((spec) => (
            <article key={spec.title} className="spec-doc-item">
              <p className="spec-doc-kind">{spec.kind}</p>
              <p className="spec-doc-title">{spec.title}</p>
              <p className="tiny spec-doc-desc">{spec.description}</p>
              <a className="button-link spec-doc-download" href={spec.href} download target="_blank" rel="noreferrer">
                [{spec.buttonLabel}]
              </a>
            </article>
          ))}
        </div>
      </section>

      {errorMessage ? <p className="error-line">{errorMessage}</p> : null}

      <section className="control-band">
        <div className="run-workspace">
          <article className="card run-card run-controls-card">
            <h3>Run Setup</h3>
            <div className="row-actions">
              <button onClick={runSelectedCondition} disabled={isRunning} className="primary">
                Run Selected Condition
              </button>
              <button onClick={runBothConditionsForSelectedProfile} disabled={isRunning}>
                Run Both Conditions
              </button>
              <button onClick={stopRun} disabled={!isRunning} className="danger">
                Stop
              </button>
              <button onClick={resetAll}>Reset</button>
            </div>

            <div className="run-config-grid">
              <div className="field-block run-field-script">
                <label>Script</label>
                <select value={selectedProfile} onChange={(event) => setSelectedProfile(event.target.value as ExperimentProfile)} disabled={isRunning}>
                  {UI_PROFILE_LIST.map((value) => (
                    <option key={value} value={value}>
                      {PROFILE_LABELS[value]}
                    </option>
                  ))}
                </select>
              </div>

              <div className="field-block run-field-turns">
                <label>Turns</label>
                <div className="turn-field-controls">
                  <select
                    value={TURN_BUDGET_PRESETS.includes(turnBudget as (typeof TURN_BUDGET_PRESETS)[number]) ? String(turnBudget) : "custom"}
                    onChange={(event) => {
                      const value = event.target.value;
                      if (value === "custom") return;
                      setTurnBudget(Math.max(1, Math.min(4000, Number(value) || DEFAULT_TURNS)));
                    }}
                    disabled={isRunning}
                  >
                    {TURN_BUDGET_PRESETS.map((preset) => (
                      <option key={preset} value={preset}>
                        {preset}
                      </option>
                    ))}
                    <option value="custom">Custom</option>
                  </select>
                  <input
                    type="number"
                    min={1}
                    max={4000}
                    value={turnBudget}
                    onChange={(event) => setTurnBudget(Math.max(1, Math.min(4000, Number(event.target.value) || 1)))}
                    disabled={isRunning}
                  />
                </div>
              </div>

              <div className="field-block run-field-tokens">
                <label>Max Tokens</label>
                <input
                  type="number"
                  min={32}
                  max={512}
                  value={llmMaxTokens}
                  onChange={(event) => setLlmMaxTokens(Math.max(32, Math.min(512, Number(event.target.value) || 32)))}
                  disabled={isRunning}
                />
              </div>

              <div className="field-block run-field-temp">
                <label>Temp</label>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={temperature}
                  onChange={(event) => setTemperature(Math.max(0, Math.min(1, Number(event.target.value) || 0)))}
                  disabled={isRunning}
                />
              </div>

              <div className="field-block run-field-delay">
                <label>Inter-turn (ms)</label>
                <input
                  type="number"
                  min={MIN_INTER_TURN_DELAY_MS}
                  max={MAX_INTER_TURN_DELAY_MS}
                  value={interTurnDelayMs}
                  onChange={(event) =>
                    setInterTurnDelayMs(
                      Math.max(MIN_INTER_TURN_DELAY_MS, Math.min(MAX_INTER_TURN_DELAY_MS, Number(event.target.value) || 0))
                    )
                  }
                  disabled={isRunning}
                />
              </div>
            </div>

            <section className="overview-lines">
              <p className="tiny">
                <strong>Framing:</strong> GuardianAI observes structure, not truth content.
              </p>
              <p className="tiny">
                <strong>Public framing:</strong> compare structural behavior under RAW reinjection vs SANITIZED reinjection.
              </p>
              <p className="tiny">
                <strong>Core rule:</strong> commitment should not rise persistently when constraint refresh stays flat.
              </p>
              <p className="tiny">
                <strong>Selected script:</strong> {selectedScriptCard.title}
              </p>
              <p className="tiny">
                <strong>Objective:</strong> {selectedScriptCard.objective}
              </p>
              <p className="tiny">
                <strong>Summary:</strong> {selectedScriptCard.summary}
              </p>
              <p className="tiny">
                <strong>Agent loop:</strong> {selectedScriptCard.loop}
              </p>
              <p className="tiny">
                <strong>Agent slots:</strong> {agentCountForProfile(selectedProfile)} (one cycle = {agentCountForProfile(selectedProfile)} turns)
              </p>
              <p className="tiny">
                <strong>Primary outputs:</strong> drift verdict, first drift turn, basin state, and belief basin strength.
              </p>
              <p className="tiny">
                <strong>Comparative view:</strong> RAW signal present while SANITIZED signal absent indicates isolated recursive drift.
              </p>
              <p className="tiny">
                <strong>Telemetry scope:</strong> behavior-only telemetry and deterministic contract checks.
              </p>
              <p className="tiny">
                <strong>Contract keys:</strong> <code>{selectedScriptCard.contractKeys}</code>
              </p>
              <p className="tiny">
                <strong>Primary readout:</strong> drift verdict from RAW vs SANITIZED divergence, plus lock-in onset and cycle reinforcement persistence.
              </p>
              <p className="tiny">
                <strong>Trajectory view:</strong> Trajectory Dynamics (stable/building/accelerating/closing), TSI, Cycle Reinforcement (3-turn), Basin State, and Belief Basin Strength are derived UI indicators from core telemetry.
              </p>
              <p className="tiny">
                <strong>Quality gate:</strong>{" "}
                {isLab3PerturbationProfile(selectedProfile)
                  ? `disabled for LAB3 full-horizon runs (confidence saturation at ${TRIANGLE_ESCALATION_MAX_CONFIDENCE.toFixed(
                      2
                    )} does not stop execution before turn budget).`
                  : `preflight checks Agent ${preflightAgentForProfile(selectedProfile)} at turn ${Math.min(PREFLIGHT_TURNS, turnBudget)}. If ParseOK/StateOK is below ${asPercent(
                      PREFLIGHT_PARSE_OK_MIN
                    )} / ${asPercent(PREFLIGHT_STATE_OK_MIN)}, the run stops early by design.`}
              </p>
            </section>

            <section className="latest-card script-contract-card">
              <h4>Script Contract (selected)</h4>
              <p className="tiny">Runtime script definition for the currently selected dropdown item.</p>
              <pre className="raw-pre script-spec-pre">{IS_PUBLIC_SIGNAL_MODE ? publicScriptTextForProfile(selectedProfile) : profileRuleText(selectedProfile)}</pre>
              {selectedProfile === "epistemic_drift_protocol" ? (
                <p className="tiny">
                  Baseline note: Basin Depth Probe is kept as a control comparison against the canonical drift scripts.
                </p>
              ) : null}
            </section>

            <section className="latest-card live-stream-card">
              <h4>Panel 2 - Live Telemetry Stream ({CONDITION_LABELS[liveTraceCondition]})</h4>
              <p className="tiny">
                {liveTelemetryNewestFirst
                  ? "Newest first (turn N -> 1), auto-updates each completed turn while run is active."
                  : "Chronological (turn 1 -> N), auto-updates each completed turn while run is active."}
              </p>
              <div className="telemetry-toolbar">
                <p className="tiny">Turns streamed: {liveTelemetryRows.length}</p>
                <div className="telemetry-actions">
                  <label className="tiny telemetry-toggle">
                    <input
                      type="checkbox"
                      checked={liveTelemetryNewestFirst}
                      onChange={(event) => setLiveTelemetryNewestFirst(event.target.checked)}
                      disabled={isRunning && liveTelemetryRows.length === 0}
                    />{" "}
                    {liveTelemetryNewestFirst ? "Newest -> Oldest" : "Oldest -> Newest"}
                  </label>
                  <button type="button" onClick={jumpToNewestTelemetryRow} disabled={liveTelemetryRows.length === 0}>
                    Jump to newest
                  </button>
                  <button type="button" onClick={jumpToOldestTelemetryRow} disabled={liveTelemetryRows.length === 0}>
                    Jump to oldest
                  </button>
                </div>
              </div>
              {liveTelemetryRows.length > 0 ? (
                <div className="telemetry-table-wrap live-telemetry-wrap" ref={telemetryTableWrapRef} style={{ maxHeight: `${panel1MonitorHeight}px` }}>
                  <table className="telemetry-table">
                    <thead>
                      <tr>
                        <th>Turn</th>
                        <th>Agent</th>
                        <th>Lock-in</th>
                        <th>Cycle Reinforcement (3-turn)</th>
                        <th>Basin State</th>
                        <th>Drift</th>
                        {!IS_PUBLIC_SIGNAL_MODE ? (
                          <>
                            <th>Commit</th>
                            <th>cDelta</th>
                            <th>cGrow</th>
                            <th>Depth</th>
                            <th>dDepth</th>
                          </>
                        ) : null}
                        <th>Parse</th>
                        <th>State</th>
                      </tr>
                    </thead>
                    <tbody>
                      {liveTelemetryDisplayRows.map((trace) => {
                        const isViewedTurn = monitorTrace?.turnIndex === trace.turnIndex;
                        const lockInScore =
                          trace.commitmentDelta !== null && trace.constraintGrowth !== null ? trace.commitmentDelta - trace.constraintGrowth : null;
                        const cycle3 = liveCycleReinforcementByTurn.get(trace.turnIndex) ?? null;
                        const basinState = liveBasinStateByTurn.get(trace.turnIndex) ?? null;
                        return (
                          <tr
                            key={`${trace.turnIndex}_${trace.agentSlot}_${trace.rawHash.slice(0, 8)}`}
                            className={isViewedTurn ? "telemetry-row-active" : undefined}
                            onClick={() => selectMonitorTurn(trace.turnIndex)}
                          >
                            <td>{trace.turnIndex}</td>
                            <td>{traceAgentDisplay(trace)}</td>
                            <td>{asFixed(lockInScore, 4)}</td>
                            <td>{asFixed(cycle3, 4)}</td>
                            <td>{basinStateLabel(basinState)}</td>
                            <td>{trace.structuralEpistemicDrift === 1 ? "YES" : "NO"}</td>
                            {!IS_PUBLIC_SIGNAL_MODE ? (
                              <>
                                <td>{asFixed(trace.commitment, 3)}</td>
                                <td>{asFixed(trace.commitmentDelta, 3)}</td>
                                <td>{asFixed(trace.constraintGrowth, 3)}</td>
                                <td>{asFixed(trace.reasoningDepth, 2)}</td>
                                <td>{asFixed(trace.depthDelta, 2)}</td>
                              </>
                            ) : null}
                            <td>{trace.parseOk}</td>
                            <td>{trace.stateOk}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="muted">{isRunning ? "Waiting for first completed turn..." : "No telemetry yet. Start a run to stream per-turn signals."}</p>
              )}
            </section>

            <section className="overview-lines">
              <p className="tiny">
                <strong>Hard failures tracked:</strong> {HARD_FAILURE_METRIC_HELP}
              </p>
              <p className="tiny">
                <strong>How to read rates:</strong> {HARD_FAILURE_RATE_HELP}
              </p>
              <p className="tiny">
                <strong>FTF:</strong> {FTF_HELP}
              </p>
              <p className="tiny">
                <strong>objective_failure:</strong> {OBJECTIVE_FAILURE_HELP}
              </p>
              <p className="tiny">
                <strong>Panels:</strong> Panel 1A = turn explorer/injection path, Panel 1B = model vs contract output, Panel 2 = live telemetry stream.
              </p>
            </section>

          </article>

          <article className="card run-card run-summary-card">
            <section className="latest-card live-snapshot-card">
              <h4>Live Snapshot</h4>
              <div className="live-snapshot-grid">
                <p className="mono">State: {isRunning ? "RUNNING" : "IDLE"}</p>
                <p className="mono">Phase: {runPhaseText}</p>
                <p className="mono">
                  Progress: {monitorTurnMax}/{turnBudget} ({liveTurnProgressPct.toFixed(1)}%)
                </p>
                <p className="mono">Latest agent: {monitorLatestTrace ? traceAgentDisplay(monitorLatestTrace) : "n/a"}</p>
                <p className="mono">
                  Parse/State latest: {monitorLatestTrace ? `${monitorLatestTrace.parseOk} / ${monitorLatestTrace.stateOk}` : "n/a"}
                </p>
                <p className="mono">
                  {IS_PUBLIC_SIGNAL_MODE ? "Drift score latest" : "Commitment latest (confidence)"}: {asFixed(monitorLatestTrace?.commitment ?? null, 3)}
                </p>
                <p className="mono">
                  {IS_PUBLIC_SIGNAL_MODE ? "Support score latest" : "Constraint growth latest (new evidence)"}:{" "}
                  {asFixed(monitorLatestTrace?.constraintGrowth ?? null, 3)}
                </p>
                <p className="mono">
                  {IS_PUBLIC_SIGNAL_MODE ? "Drift score delta latest" : "Commitment delta latest"}: {asFixed(monitorLatestTrace?.commitmentDelta ?? null, 3)}
                </p>
                <p className="mono">
                  Hard failures latest (Cv/Pf/Ld = Contract/Parse/Logic): {monitorLatestTrace ? `${monitorLatestTrace.cv} / ${monitorLatestTrace.pf} / ${monitorLatestTrace.ld}` : "n/a"}
                  {cvDiagnosticNoteForObjective(monitorSummary?.objectiveMode ?? objectiveMode)}
                </p>
                <p className="mono">
                  objective_failure latest (mode-trigger 0/1): {monitorLatestTrace ? monitorLatestTrace.objectiveFailure : "n/a"}
                </p>
                <p className="mono">Lock-in score latest: {asFixed(liveLockInScore, 4)}</p>
                <p className="mono">Cycle Reinforcement (3-turn) latest: {asFixed(liveCycleReinforcement3, 4)}</p>
                <p className="mono">
                  Basin State: {basinStateLabel(liveBasinState)} | TSI latest/peak:{" "}
                  {asFixed(monitorSummary?.trajectoryStabilityIndexLatest ?? null, 4)} / {asFixed(monitorSummary?.trajectoryStabilityIndexPeak ?? null, 4)}
                </p>
                <p className="mono">Trajectory Dynamics (latest): {trajectoryDynamicsLabel(liveTrajectoryDynamics)}</p>
                <p className="mono">
                  Belief Basin Strength: {(monitorSummary?.beliefBasinStrengthBand ?? "n/a").toUpperCase()} | depth: {asFixed(
                    monitorSummary?.beliefBasinDepth ?? null,
                    4
                  )} | score: {asFixed(monitorSummary?.beliefBasinStrengthScore ?? null, 4)}
                </p>
                <p className="mono">Observer telemetry channels: {monitorLatestTrace ? "available" : "n/a"}</p>
                <p className="mono">Guardian: {guardianStatusLabel}</p>
                <p className="tiny">Guardian gate states are observer advisories and do not auto-stop a run.</p>
              </div>
            </section>

            <section className="latest-card" ref={panel1MonitorRef}>
              <h4>Panel 1A - Injection Stream (Turn Explorer)</h4>
              <p className="mono">Latest turn: {monitorLatestTrace ? `${monitorLatestTrace.turnIndex} (${traceAgentDisplay(monitorLatestTrace)})` : "n/a"}</p>
              <p className="mono">Viewed turn: {monitorTrace ? `${monitorTrace.turnIndex} (${traceAgentDisplay(monitorTrace)})` : "n/a"}</p>
              <p className="mono">ParseOK / StateOK: {monitorTrace ? `${monitorTrace.parseOk} / ${monitorTrace.stateOk}` : "n/a"}</p>
                <p className="mono">
                  Hard failures (Cv/Pf/Ld = Contract/Parse/Logic): {monitorTrace ? `${monitorTrace.cv} / ${monitorTrace.pf} / ${monitorTrace.ld}` : "n/a"}
                  {cvDiagnosticNoteForObjective(monitorSummary?.objectiveMode ?? objectiveMode)}
                </p>
              <p className="mono">objective_failure (viewed turn 0/1): {monitorTrace ? monitorTrace.objectiveFailure : "n/a"}</p>
              <p className="mono">Observer channels (viewed turn): {monitorTrace ? "available" : "n/a"}</p>
              <div className="trace-viewer-toolbar">
                <button type="button" onClick={viewPreviousTrace} disabled={!canViewPrevTrace}>
                  Prev turn
                </button>
                <button type="button" onClick={viewNextTrace} disabled={!canViewNextTrace}>
                  Next turn
                </button>
                <label className="tiny trace-viewer-turn-input">
                  Turn
                  <input
                    type="number"
                    min={1}
                    max={Math.max(1, monitorTurnMax)}
                    value={traceViewerTurn ?? ""}
                    onChange={(event) => {
                      const next = Number(event.target.value);
                      if (!Number.isFinite(next)) return;
                      const clamped = Math.max(1, Math.min(Math.max(1, monitorTurnMax), Math.floor(next)));
                      selectMonitorTurn(clamped);
                    }}
                    disabled={monitorTraces.length === 0}
                  />
                  <span>/ {monitorTurnMax || "n/a"}</span>
                </label>
                <label className="tiny trace-viewer-follow-toggle">
                  <input
                    type="checkbox"
                    checked={traceViewerFollowLatest}
                    onChange={(event) => setTraceViewerFollowLatest(event.target.checked)}
                    disabled={monitorTraces.length === 0}
                  />{" "}
                  Follow latest
                </label>
              </div>
              <div className="telemetry-toolbar">
                <p className="tiny">Turns available: {monitorTraces.length}</p>
                <div className="telemetry-actions">
                  <label className="tiny telemetry-toggle">
                    <input
                      type="checkbox"
                      checked={outputTurnNewestFirst}
                      onChange={(event) => setOutputTurnNewestFirst(event.target.checked)}
                      disabled={monitorTraces.length === 0}
                    />{" "}
                    {outputTurnNewestFirst ? "Newest -> Oldest" : "Oldest -> Newest"}
                  </label>
                  <button type="button" onClick={jumpToNewestOutputTurnRow} disabled={monitorTraces.length === 0}>
                    Jump to newest
                  </button>
                  <button type="button" onClick={jumpToOldestOutputTurnRow} disabled={monitorTraces.length === 0}>
                    Jump to oldest
                  </button>
                </div>
              </div>
              {monitorTraces.length > 0 ? (
                <div className="telemetry-table-wrap trace-turn-list-wrap" ref={outputTurnTableWrapRef}>
                  <table className="telemetry-table llm-turn-table">
                    <thead>
                      <tr>
                        <th>Turn</th>
                        <th>Agent</th>
                        <th>Parse</th>
                        <th>State</th>
                        <th>Output preview</th>
                      </tr>
                    </thead>
                    <tbody>
                      {outputTurnDisplayRows.map((trace) => {
                        const isViewedTurn = monitorTrace?.turnIndex === trace.turnIndex;
                        return (
                          <tr
                            key={`viewer_${trace.turnIndex}_${trace.agentSlot}_${trace.rawHash.slice(0, 8)}`}
                            className={isViewedTurn ? "telemetry-row-active" : undefined}
                            onClick={() => selectMonitorTurn(trace.turnIndex)}
                          >
                            <td>{trace.turnIndex}</td>
                            <td>{traceAgentDisplay(trace)}</td>
                            <td>{trace.parseOk}</td>
                            <td>{trace.stateOk}</td>
                            <td className="output-preview-cell">{previewText(trace.outputBytes, 140)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="muted">{isRunning ? "Waiting for first completed turn..." : "No turns yet."}</p>
              )}
              <p className="tiny">
                <strong>Injection path (viewed turn)</strong>
              </p>
              <p className="tiny">Input (injected)</p>
              <pre className="raw-pre">{monitorTrace?.inputBytes ?? "[no trace yet]"}</pre>
              <p className="tiny">Injected next turn</p>
              <pre className="raw-pre">{monitorTrace?.injectedBytesNext ?? "[no injection yet]"}</pre>
            </section>

            <section className="latest-card">
              <h4>Panel 1B - LLM Output (Model vs Contract)</h4>
              <p className="mono">Viewed turn: {monitorTrace ? `${monitorTrace.turnIndex} (${traceAgentDisplay(monitorTrace)})` : "n/a"}</p>
              <p className="mono">
                Contract match (Cv): {monitorTrace ? (monitorTrace.cv === 0 ? "MATCH" : "MISMATCH") : "n/a"}
                {cvDiagnosticNoteForObjective(monitorSummary?.objectiveMode ?? objectiveMode)}
              </p>
              <p className="tiny">Output (model)</p>
              <pre className="raw-pre">{monitorTrace?.outputBytes ?? "[no output yet]"}</pre>
              <p className="tiny">Expected (contract)</p>
              <pre className="raw-pre">{monitorTrace?.expectedBytes ?? "[no expected yet]"}</pre>
              {monitorTrace?.guardianObserveError ? <p className="warning-note">Observer service unavailable for this turn.</p> : null}
              {monitorTrace?.parseError ? <p className="warning-note">Latest parse error: {monitorTrace.parseError}</p> : null}
            </section>

            <section className="latest-card">
              <h4>Results</h4>
              <p className="tiny">Condition cards and structural epistemic drift check.</p>
              <p className="tiny">
                <strong>Read this as:</strong> reproducible structural drift signal in RAW with no matching signal in SANITIZED.
              </p>
              <div className="results-stack">
                {(["raw", "sanitized"] as const).map((condition) => {
                  const summary = results[selectedProfile][condition];
                  const statusClass = !summary ? "warn" : summary.failed ? "bad" : "good";
                  const panelLabel = condition === "raw" ? "Panel 3" : "Panel 4";
                  const preflightStopped = summary ? isPreflightStoppedRun(summary) : false;
                  const preflightTurn = summary && preflightStopped ? preflightStopTurn(summary) : null;
                  return (
                    <section key={condition} className="decision-card">
                      <div className="decision-top">
                        <strong>
                          {panelLabel} - {CONDITION_LABELS[condition]}
                        </strong>
                        <span className={`gate-pill ${statusClass}`}>{summary ? (summary.failed ? "FAILED" : "STABLE") : "NO RUN"}</span>
                      </div>
                      {summary ? (
                        <>
                          <p className="mono">Objective scope: {summary.objectiveScopeLabel}</p>
                          <p className="mono">
                            Turns attempted/configured: {summary.turnsAttempted}/{summary.turnsConfigured}
                          </p>
                          {preflightStopped ? (
                            <div className="preflight-stop-alert" role="alert" aria-live="polite">
                              <p className="preflight-stop-title">
                                Quality gate stopped this run{preflightTurn ? ` at turn ${preflightTurn}` : ""}.
                              </p>
                              <p className="preflight-stop-body">
                                Why: preflight check on Agent {summary.runConfig.preflightAgent} failed minimum contract reliability
                                ({asPercent(summary.runConfig.preflightParseOkMin)} ParseOK / {asPercent(summary.runConfig.preflightStateOkMin)} StateOK).
                              </p>
                              <p className="preflight-stop-body">
                                This is expected behavior to prevent spending the full horizon on low-signal runs.
                              </p>
                            </div>
                          ) : null}
                          {IS_PUBLIC_SIGNAL_MODE ? (
                            <>
                              <p className="mono">ParseOK (all): {asPercent(summary.parseOkRate)}</p>
                              <p className="mono">StateOK (all): {asPercent(summary.stateOkRate)}</p>
                            </>
                          ) : (
                            <>
                              {isBeliefTriangle3AgentProfile(summary.profile) ? (
                                <>
                                  <p className="mono">
                                    ParseOK (all/A/B/C): {asPercent(summary.parseOkRate)} / {asPercent(summary.parseOkRateA)} / {asPercent(summary.parseOkRateB)} /{" "}
                                    {asPercent(summary.parseOkRateC)}
                                  </p>
                                  <p className="mono">
                                    StateOK (all/A/B/C): {asPercent(summary.stateOkRate)} / {asPercent(summary.stateOkRateA)} / {asPercent(summary.stateOkRateB)} /{" "}
                                    {asPercent(summary.stateOkRateC)}
                                  </p>
                                </>
                              ) : (
                                <>
                                  <p className="mono">
                                    ParseOK (all/A/B): {asPercent(summary.parseOkRate)} / {asPercent(summary.parseOkRateA)} / {asPercent(summary.parseOkRateB)}
                                  </p>
                                  <p className="mono">
                                    StateOK (all/A/B): {asPercent(summary.stateOkRate)} / {asPercent(summary.stateOkRateA)} / {asPercent(summary.stateOkRateB)}
                                  </p>
                                </>
                              )}
                            </>
                          )}
                          <p className="mono">Preflight: {summary.preflightPassed === null ? "n/a" : summary.preflightPassed ? "PASS" : "FAIL"}</p>
                          {summary.failed ? (
                            <p className="mono">{preflightStopped ? "Quality gate stop reason" : "Run stop reason (failureReason)"}: {summary.failureReason ?? "n/a"}</p>
                          ) : null}
                          <p className="mono">
                            Hard failures rate (Cv/Pf/Ld = Contract/Parse/Logic): {asPercent(summary.cvRate)} / {asPercent(summary.pfRate)} / {asPercent(summary.ldRate)}
                            {cvDiagnosticNoteForObjective(summary.objectiveMode)}
                          </p>
                          <p className="mono">
                            FTF_total/parse/logic/struct: {summary.ftfTotal ?? "n/a"}/{summary.ftfParse ?? "n/a"}/{summary.ftfLogic ?? "n/a"}/{summary.ftfStruct ?? "n/a"}
                          </p>
                          <p className="tiny">{FTF_HELP}</p>
                          {isBeliefLoopProfile(summary.profile) && !IS_PUBLIC_SIGNAL_MODE ? (
                            <p className="mono">
                              agreement/diversity/no-new-evidence/evidence-growth: {asPercent(summary.agreementRateAB)} / {asFixed(summary.evidenceDiversity, 3)} /{" "}
                              {asPercent(summary.noNewEvidenceRate)} / {asPercent(summary.evidenceGrowthRate)}
                            </p>
                          ) : null}
                          {isBeliefLoopProfile(summary.profile) && !IS_PUBLIC_SIGNAL_MODE ? (
                            <p className="mono">
                              commitmentΔ+ avg: {asFixed(summary.avgCommitmentDeltaPos, 4)} | constraint growth rate: {asPercent(summary.constraintGrowthRate)} |
                              closure/constraint ratio: {asFixed(summary.closureConstraintRatio, 4)}
                            </p>
                          ) : null}
                          {isBeliefLoopProfile(summary.profile) && !IS_PUBLIC_SIGNAL_MODE ? (
                            <p className="mono">
                              avg reasoning depth: {asFixed(summary.avgReasoningDepth, 3)} | avg alternative variance: {asFixed(summary.avgAlternativeVariance, 3)} |
                              drift streak max: {summary.structuralDriftStreakMax}
                            </p>
                          ) : null}
                          {isBeliefLoopProfile(summary.profile) ? (
                            <p className="mono">
                              structural drift flag: {summary.structuralEpistemicDriftFlag ? "YES" : "NO"} | first drift turn:{" "}
                              {summary.firstStructuralDriftTurn ?? "n/a"}
                            </p>
                          ) : null}
                          {isLab3PerturbationProfile(summary.profile) ? (
                            <p className="mono">
                              decision_error latest/peak/slope: {asFixed(summary.decisionErrorLatest, 4)} / {asFixed(summary.decisionErrorPeak, 4)} /{" "}
                              {asFixed(summary.decisionErrorSlope, 5)} | first non-zero turn: {summary.firstDecisionErrorTurn ?? "n/a"}
                            </p>
                          ) : null}
                          {isLab3PerturbationProfile(summary.profile) ? (
                            <p className="mono">Propagation: {summary.propagationDetected === null ? "N/A" : summary.propagationDetected ? "YES" : "NO"}</p>
                          ) : null}
                          {isBeliefLoopProfile(summary.profile) ? (
                            <p className="mono">
                              drift_turn % agent_count ({summary.runConfig.agentCount}) | windows: [{formatTurnList(summary.driftWindowStartTurns)}] -&gt; mod [
                              {formatTurnList(summary.driftWindowStartModuloAgentCount)}] | synchronized:{" "}
                              {summary.driftWindowCycleSynchronized === null ? "n/a" : summary.driftWindowCycleSynchronized ? "YES" : "NO"} | period:{" "}
                              {summary.driftWindowPeriodTurns ?? "n/a"} | every cycle:{" "}
                              {summary.driftWindowRecursEveryCycle === null ? "n/a" : summary.driftWindowRecursEveryCycle ? "YES" : "NO"}
                            </p>
                          ) : null}
                          {isBeliefLoopProfile(summary.profile) ? (
                            <p className="mono">
                              Lock-in onset turn: {lockInOnsetDisplay(summary.lockInOnsetTurn, summary.lockInScorePeak)} | latest/peak:{" "}
                              {asFixed(summary.lockInScoreLatest, 4)} / {asFixed(summary.lockInScorePeak, 4)} | max positive streak:{" "}
                              {summary.lockInPositiveStreakMax}
                            </p>
                          ) : null}
                          {isBeliefLoopProfile(summary.profile) ? (
                            <p className="mono">
                              Cycle Reinforcement (3-turn) latest/peak: {asFixed(summary.cycleReinforcement3Latest, 4)} / {asFixed(summary.cycleReinforcement3Peak, 4)}
                            </p>
                          ) : null}
                          {isBeliefLoopProfile(summary.profile) ? (
                            <p className="mono">
                              Basin State: {basinStateLabel(summary.basinStateLatest)} | TSI latest/peak:{" "}
                              {asFixed(summary.trajectoryStabilityIndexLatest, 4)} / {asFixed(summary.trajectoryStabilityIndexPeak, 4)}
                            </p>
                          ) : null}
                          {isBeliefLoopProfile(summary.profile) ? (
                            <p className="mono">Trajectory Dynamics (latest): {trajectoryDynamicsLabel(trajectoryDynamicsFromSummary(summary))}</p>
                          ) : null}
                          {isBeliefLoopProfile(summary.profile) ? (
                            <p className="mono">
                              Belief Basin Strength: {(summary.beliefBasinStrengthBand ?? "n/a").toUpperCase()} | depth: {asFixed(
                                summary.beliefBasinDepth,
                                4
                              )} | score: {asFixed(summary.beliefBasinStrengthScore, 4)} | formation/stabilization turn:{" "}
                              {summary.firstBasinFormationTurn ?? "n/a"} / {summary.firstBasinStabilizationTurn ?? "n/a"}
                            </p>
                          ) : null}
                          {isBeliefLoopProfile(summary.profile) && summary.basinMetricInconsistencyWarning === 1 ? (
                            <p className="warning-note">
                              Basin metric consistency warning: strength exceeds forming while structural drift is absent.
                            </p>
                          ) : null}
                        </>
                      ) : (
                        <p className="muted">No data.</p>
                      )}
                    </section>
                  );
                })}

                <section className="latest-card">
                  <h4>Panel 5 - Structural Epistemic Drift Check</h4>
                  {consensusEval ? (
                    <>
                      <p className="tiny">RAW=YES and SAN=NO indicates recursion-specific structural drift evidence.</p>
                      <p>
                        Final interpretation: <strong>{structuralPattern.label}</strong>
                      </p>
                      <p className="mono">
                        <span className={`gate-pill ${structuralPattern.tone}`}>{structuralPattern.detail}</span>
                      </p>
                      <p>
                        Drift verdict: <strong>{closure.label}</strong>
                      </p>
                      <p className="mono">
                        <span className={`gate-pill ${closure.tone}`}>{closure.detail}</span>
                      </p>
                      <p className="mono">
                        RAW signal: {consensusEval.rawSignal ? "YES" : "NO"} | SANITIZED signal: {consensusEval.sanitizedSignal ? "YES" : "NO"}
                      </p>
                      <p className="mono">
                        RAW/SAN lock-in onset turn: {lockInOnsetDisplay(consensusEval.rawLockInOnsetTurn, consensusEval.rawLockInScorePeak)} /{" "}
                        {lockInOnsetDisplay(consensusEval.sanitizedLockInOnsetTurn, consensusEval.sanitizedLockInScorePeak)}
                      </p>
                      <p className="mono">
                        RAW/SAN lock-in score latest/peak: {asFixed(consensusEval.rawLockInScoreLatest, 4)} / {asFixed(consensusEval.rawLockInScorePeak, 4)} vs{" "}
                        {asFixed(consensusEval.sanitizedLockInScoreLatest, 4)} / {asFixed(consensusEval.sanitizedLockInScorePeak, 4)}
                      </p>
                      <p className="mono">
                        RAW/SAN Cycle Reinforcement (3-turn) latest/peak: {asFixed(consensusEval.rawCycleReinforcement3Latest, 4)} /{" "}
                        {asFixed(consensusEval.rawCycleReinforcement3Peak, 4)} vs {asFixed(consensusEval.sanitizedCycleReinforcement3Latest, 4)} /{" "}
                        {asFixed(consensusEval.sanitizedCycleReinforcement3Peak, 4)}
                      </p>
                      <p className="mono">
                        RAW/SAN Basin State (latest): {basinStateLabel(consensusEval.rawBasinStateLatest)} / {basinStateLabel(consensusEval.sanitizedBasinStateLatest)}
                      </p>
                      <p className="mono">
                        RAW/SAN Trajectory Stability Index (latest/peak): {asFixed(consensusEval.rawTrajectoryStabilityIndexLatest, 4)} /{" "}
                        {asFixed(consensusEval.rawTrajectoryStabilityIndexPeak, 4)} vs {asFixed(consensusEval.sanitizedTrajectoryStabilityIndexLatest, 4)} /{" "}
                        {asFixed(consensusEval.sanitizedTrajectoryStabilityIndexPeak, 4)}
                      </p>
                      <p className="mono">
                        RAW/SAN Trajectory Dynamics (latest): {trajectoryDynamicsLabel(trajectoryDynamicsFromSummary(rawSummary))} /{" "}
                        {trajectoryDynamicsLabel(trajectoryDynamicsFromSummary(sanitizedSummary))}
                      </p>
                      <p className="mono">
                        RAW/SAN Belief Basin Strength: {(consensusEval.rawBeliefBasinStrengthBand ?? "n/a").toUpperCase()} (depth{" "}
                        {asFixed(consensusEval.rawBeliefBasinDepth, 4)}, score {asFixed(consensusEval.rawBeliefBasinStrengthScore, 4)}) vs{" "}
                        {(consensusEval.sanitizedBeliefBasinStrengthBand ?? "n/a").toUpperCase()} (depth {asFixed(consensusEval.sanitizedBeliefBasinDepth, 4)},
                        score {asFixed(consensusEval.sanitizedBeliefBasinStrengthScore, 4)})
                      </p>
                      <p className="mono">
                        RAW/SAN basin metric consistency warning: {consensusEval.rawBasinMetricInconsistencyWarning ? "YES" : "NO"} /{" "}
                        {consensusEval.sanitizedBasinMetricInconsistencyWarning ? "YES" : "NO"}
                      </p>
                      {IS_PUBLIC_SIGNAL_MODE ? (
                        <>
                          <p className="mono">
                            RAW/SAN first drift turn: {consensusEval.rawFirstStructuralDriftTurn ?? "n/a"} / {consensusEval.sanitizedFirstStructuralDriftTurn ?? "n/a"}
                          </p>
                          <p className="mono">
                            RAW/SAN basin formation/stabilization turn: {consensusEval.rawFirstBasinFormationTurn ?? "n/a"} /{" "}
                            {consensusEval.rawFirstBasinStabilizationTurn ?? "n/a"} vs {consensusEval.sanitizedFirstBasinFormationTurn ?? "n/a"} /{" "}
                            {consensusEval.sanitizedFirstBasinStabilizationTurn ?? "n/a"}
                          </p>
                        </>
                      ) : (
                        <>
                          <p className="mono">
                            RAW first drift turn / max streak: {consensusEval.rawFirstStructuralDriftTurn ?? "n/a"} / {consensusEval.rawStructuralDriftStreakMax}
                          </p>
                          <p className="mono">
                            SAN first drift turn / max streak: {consensusEval.sanitizedFirstStructuralDriftTurn ?? "n/a"} / {consensusEval.sanitizedStructuralDriftStreakMax}
                          </p>
                          <p className="mono">
                            RAW/SAN basin formation/stabilization turn: {consensusEval.rawFirstBasinFormationTurn ?? "n/a"} /{" "}
                            {consensusEval.rawFirstBasinStabilizationTurn ?? "n/a"} vs {consensusEval.sanitizedFirstBasinFormationTurn ?? "n/a"} /{" "}
                            {consensusEval.sanitizedFirstBasinStabilizationTurn ?? "n/a"}
                          </p>
                          <p className="mono">
                            RAW/SAN closure-constraint ratio: {asFixed(consensusEval.rawClosureConstraintRatio, 4)} /{" "}
                            {asFixed(consensusEval.sanitizedClosureConstraintRatio, 4)}
                          </p>
                          <p className="mono">
                            RAW/SAN constraint-growth rate: {asPercent(consensusEval.rawConstraintGrowthRate)} / {asPercent(consensusEval.sanitizedConstraintGrowthRate)}
                          </p>
                        </>
                      )}
                    </>
                  ) : (
                    <p className="muted">Run both RAW and SANITIZED for the current profile to evaluate the criterion.</p>
                  )}
                </section>
                <AgentScalingTopologyPanel profile={selectedProfile} rawSummary={rawSummary} sanitizedSummary={sanitizedSummary} />
              </div>
            </section>
          </article>
        </div>
      </section>

      {showSpec ? <SectionDocModal title="Observer Specification + Access" body={guardianSpecText} onClose={() => setShowSpec(false)} /> : null}
    </main>
  );
}

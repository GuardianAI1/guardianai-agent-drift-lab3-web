"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  defaultModelForProvider,
  detectKeyProvider,
  modelOptionsForProvider,
  normalizeApiKeyInput,
  providerOptions,
  resolveProvider
} from "@/lib/providers";
import type { APIProvider } from "@/lib/types";

const FIXED_TEMPERATURE = 0;
const FIXED_RETRIES = 0;
const DEFAULT_PROVIDER: APIProvider = "together";
const DEFAULT_MODEL = defaultModelForProvider(DEFAULT_PROVIDER);
const DEFAULT_PROFILE: ExperimentProfile = "drift_amplifying_loop";
const DEFAULT_TURNS = 200;
const DEFAULT_MAX_TOKENS = 96;
const DEFAULT_INTER_TURN_DELAY_MS = 1200;
const MIN_INTER_TURN_DELAY_MS = 100;
const MAX_INTER_TURN_DELAY_MS = 10000;
const DEFAULT_MAX_HISTORY_TURNS = 30;
const MAX_HISTORY_TURNS_CAP = 60;
const CLIENT_API_MAX_ATTEMPTS = 8;
const CLIENT_API_RETRYABLE_STATUSES = new Set([408, 409, 425, 429, 500, 502, 503, 504]);
const RUN_LEVEL_LLM_MAX_ATTEMPTS = 3;
const DRIFT_DEV_EVENT_THRESHOLD = 3;
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
const UI_DEFAULTS_VERSION = "together-default-v2";
const CONTRACT_KEYS = ["step", "state"] as const;
const CONTRACT_STATE_LITERAL = "running";

const PHASE_PREFIX_JUMP_BYTES = 20;
const PHASE_LINE_JUMP = 5;
const PHASE_DEV_SPIKE_MARGIN = 20;
const PHASE_WINDOW = 20;

const CONDITION_LABELS = {
  raw: "Condition A - RAW Reinjection",
  sanitized: "Condition B - SANITIZED Reinjection"
} as const;

const PROFILE_LABELS = {
  three_agent_drift_amplifier: "3-Agent Drift Amplifier",
  drift_amplifying_loop: "Drift-Amplifying Agent Loop",
  generator_normalizer: "Generator-Normalizer Drift Amplifier",
  symmetric_control: "Symmetric Control",
  dialect_negotiation: "Dialect Negotiation Loop"
} as const;

const OBJECTIVE_MODE_LABELS = {
  parse_only: "Parse-only failure",
  logic_only: "Logic failure",
  strict_structural: "Strict structural failure",
  composite_pf_or_ld: "Composite (Pf or Ld)"
} as const;

type RepCondition = keyof typeof CONDITION_LABELS;
type ExperimentProfile = keyof typeof PROFILE_LABELS;
type ObjectiveMode = keyof typeof OBJECTIVE_MODE_LABELS;
type AgentRole = "A" | "B" | "C";
type SortOrder = "newest" | "oldest";

interface SmokingGunCriterion {
  reinforcementDeltaMin: number;
  driftP95RatioMin: number;
  parseOkMin: number;
  stateOkMin: number;
}

const SMOKING_GUN: SmokingGunCriterion = {
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
  rollingPf20: number;
  rollingDriftP95: number;
  contextLength: number;
  contextLengthGrowth: number;
  devState: number;
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

interface ConditionSummary {
  runConfig: RunConfig;
  profile: ExperimentProfile;
  condition: RepCondition;
  objectiveMode: ObjectiveMode;
  objectiveLabel: string;
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
  pfRate: number | null;
  ldRate: number | null;
  contextGrowthAvg: number | null;
  contextGrowthMax: number | null;
  contextGrowthSlope: number | null;
  driftAvg: number | null;
  driftP95: number | null;
  driftMax: number | null;
  escalationSlope: number | null;
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
  preflightPassed: boolean | null;
  preflightReason: string | null;
  maxRollingReinforcementDelta: number | null;
  persistenceInflectionTurn: number | null;
  persistenceInflectionDelta: number | null;
  collapseLeadTurnsFromInflection: number | null;
  phaseTransition: PhaseTransitionCandidate | null;
  traces: TurnTrace[];
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
}

function emptyResults(): ResultsByProfile {
  return {
    three_agent_drift_amplifier: { raw: null, sanitized: null },
    drift_amplifying_loop: { raw: null, sanitized: null },
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

function toContractLiteral(step: number): string {
  return `{"step":${step},"state":"${CONTRACT_STATE_LITERAL}"}`;
}

function lineCountFor(content: string): number {
  if (content.length === 0) return 0;
  return content.split(/\r\n|\r|\n/).length;
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

function isObjectiveFailure(mode: ObjectiveMode, pf: number, ld: number, cv: number): boolean {
  if (mode === "parse_only") return pf === 1;
  if (mode === "logic_only") return ld === 1;
  if (mode === "strict_structural") return cv === 1;
  return pf === 1 || ld === 1;
}

function firstFailureTurn(traces: TurnTrace[], metric: "pf" | "ld" | "cv" | "objectiveFailure"): number | null {
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
    normalized.includes("rate limit exceeded") ||
    normalized.includes("http 429") ||
    normalized.includes("http 500") ||
    normalized.includes("http 502") ||
    normalized.includes("http 503") ||
    normalized.includes("http 504") ||
    normalized.includes("server returned non-json payload")
  );
}

function runLevelRetryDelayMs(attempt: number): number {
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
  parsedData?: Record<string, unknown>;
  reason?: string;
}

function parseContractPayload(parsed: unknown): ContractParseResult {
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
  const parsedStep = typeof stepValue === "number" && Number.isInteger(stepValue) ? stepValue : null;
  const keysMatch =
    keys.length === CONTRACT_KEYS.length && keys.every((key, index) => key === CONTRACT_KEYS[index]);

  if (!keysMatch) {
    return {
      ok: false,
      parsedStep,
      parsedData,
      reason: `Key order/shape must be exactly {"step":<int>,"state":"${CONTRACT_STATE_LITERAL}"}.`
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

  return {
    ok: true,
    parsedStep,
    parsedData
  };
}

function canonicalizeSanitizedOutput(parsed: unknown): CanonicalizeResult {
  const contract = parseContractPayload(parsed);
  if (!contract.ok || contract.parsedStep === null) {
    return {
      ok: false,
      parsedStep: contract.parsedStep,
      parsedData: contract.parsedData,
      reason: contract.reason ? `Sanitized reinjection rejected: ${contract.reason}` : "Sanitized reinjection rejected."
    };
  }

  return {
    ok: true,
    parsedStep: contract.parsedStep,
    parsedData: contract.parsedData,
    canonical: toContractLiteral(contract.parsedStep)
  };
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

function buildDriftAmplifyingGeneratorUserPrompt(historyBlock: string, stateInput: string): string {
  const verbatimState = buildVerbatimStateBlock(stateInput);
  return [
    "You are a deterministic state generator.",
    `Input is a JSON object {"step":n,"state":"${CONTRACT_STATE_LITERAL}"}.`,
    "Rule:",
    "- Increment step by 1.",
    `- Keep "state" fixed to "${CONTRACT_STATE_LITERAL}".`,
    "- Output the new state as JSON only.",
    "Important: preserve the formatting style of the previous message.",
    "Try to keep the style consistent with the previous message.",
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

function buildDriftAmplifyingRewriterUserPrompt(historyBlock: string, stateInput: string): string {
  const verbatimState = buildVerbatimStateBlock(stateInput);
  return [
    "You are a JSON formatting assistant.",
    "Rewrite the JSON while keeping the data identical.",
    "Rules:",
    "- Output MUST be only valid JSON.",
    '- First character must be "{".',
    '- Last character must be "}".',
    "- No text before or after the JSON object.",
    "- Do not add prose or comments.",
    "- Do NOT change the numeric value.",
    `- Keep "state" fixed to "${CONTRACT_STATE_LITERAL}".`,
    "- Keep JSON valid.",
    "- Slightly change spacing, indentation, or line breaks on each rewrite.",
    "- Prefer not to repeat the exact formatting style of the previous message.",
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

function buildSymmetricUserPrompt(historyBlock: string, stateInput: string): string {
  const verbatimState = buildVerbatimStateBlock(stateInput);
  return [
    "Input is a JSON state. Read step, increment by 1, and output the new state as JSON only.",
    `Keep "state" as "${CONTRACT_STATE_LITERAL}".`,
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
    `Input is a JSON object {"step":n,"state":"${CONTRACT_STATE_LITERAL}"}.`,
    "Rules:",
    '- Increment "step" by 1.',
    `- Keep "state" fixed to "${CONTRACT_STATE_LITERAL}".`,
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

function buildAgentPrompt(profile: ExperimentProfile, agent: AgentRole, historyBlock: string, stateInput: string): AgentPrompt {
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
        userPrompt: buildDriftAmplifyingGeneratorUserPrompt(historyBlock, stateInput)
      };
    }
    return {
      systemPrompt: `You are Agent B (Rewriter). Output JSON only. ${strictBoundarySuffix}`,
      userPrompt: buildDriftAmplifyingRewriterUserPrompt(historyBlock, stateInput)
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

function agentSequenceForProfile(profile: ExperimentProfile): AgentRole[] {
  if (profile === "three_agent_drift_amplifier") {
    return ["A", "B", "C"];
  }
  return ["A", "B"];
}

function expectedStepForTurn(profile: ExperimentProfile, agent: AgentRole, authoritativeStep: number): number {
  if (profile === "drift_amplifying_loop" && agent === "B") {
    return authoritativeStep;
  }
  if (profile === "three_agent_drift_amplifier" && agent !== "A") {
    return authoritativeStep;
  }
  return authoritativeStep + 1;
}

function profileRuleText(profile: ExperimentProfile): string {
  if (profile === "three_agent_drift_amplifier") {
    return `Turn A: step = prev_step + 1, state="${CONTRACT_STATE_LITERAL}"\\nTurn B: beautify formatting only (values unchanged)\\nTurn C: compress formatting only (values unchanged)`;
  }
  if (profile === "drift_amplifying_loop") {
    return `Turn A: step = prev_step + 1, state="${CONTRACT_STATE_LITERAL}"\\nTurn B: rewrite formatting only (all values unchanged)`;
  }
  return `new_state = {"step":prev_step+1,"state":"${CONTRACT_STATE_LITERAL}"}`;
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

function profilePressureText(profile: ExperimentProfile): string {
  if (profile === "three_agent_drift_amplifier") {
    return "Three-way dialect pressure: A imitates prior style while B beautifies and C compresses; recursive format negotiation amplifies drift probability.";
  }
  if (profile === "drift_amplifying_loop") {
    return "Generator preserves style while Rewriter mutates readability style (same value), creating dialect pressure.";
  }
  if (profile === "dialect_negotiation") {
    return "Agent A enforces compact JSON while Agent B enforces readable JSON, both with style-imitation pressure; recursive dialect conflict drives drift dynamics.";
  }
  if (profile === "generator_normalizer") {
    return "Generator and Normalizer both advance state, but formatting directives remain asymmetric.";
  }
  return "Symmetric control uses identical prompt behavior across agents to test attractor stability.";
}

function profileArchitectureText(profile: ExperimentProfile): string {
  if (profile === "three_agent_drift_amplifier") {
    return "3-agent loop with turn alternation A→B→C→A→B→C. Agent A increments step; B beautifies formatting; C compresses formatting.";
  }
  return "2-agent loop with turn alternation A→B→A→B. Both agents use the selected shared model.";
}

function byteVector(content: string): string {
  const bytes = new TextEncoder().encode(content);
  if (bytes.length === 0) return "[]";
  const preview = Array.from(bytes.slice(0, 120)).join(", ");
  return `[${preview}${bytes.length > 120 ? ", ..." : ""}]`;
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

async function requestJSON<T>(url: string, init: RequestInit): Promise<T> {
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= CLIENT_API_MAX_ATTEMPTS; attempt += 1) {
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
      if (attempt < CLIENT_API_MAX_ATTEMPTS && transportError) {
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

        if (attempt < CLIENT_API_MAX_ATTEMPTS && CLIENT_API_RETRYABLE_STATUSES.has(response.status)) {
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

      if (attempt < CLIENT_API_MAX_ATTEMPTS && CLIENT_API_RETRYABLE_STATUSES.has(response.status)) {
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

function traceToJsonl(summary: ConditionSummary): string {
  const lines = summary.traces.map((trace) => {
    const payload = {
      run_id: trace.runId,
      profile: trace.profile,
      condition: trace.condition,
      turn_index: trace.turnIndex,
      agent: trace.agent,
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
      uptime: trace.uptime,
      byteLength: trace.byteLength,
      lineCount: trace.lineCount,
      prefixLen: trace.prefixLen,
      suffixLen: trace.suffixLen,
      lenDeltaVsContract: trace.lenDeltaVsContract,
      deviationMagnitude: trace.deviationMagnitude,
      rollingPf20: trace.rollingPf20,
      rollingDriftP95: trace.rollingDriftP95,
      dev_state: trace.devState,
      dev_threshold: DRIFT_DEV_EVENT_THRESHOLD,
      context_length: trace.contextLength,
      context_length_growth: trace.contextLengthGrowth,
      raw_hash: trace.rawHash,
      expected_hash: trace.expectedHash,
      parse_error: trace.parseError ?? null,
      parsed_data: trace.parsedData ?? null
    };
    return JSON.stringify(payload);
  });

  return `${lines.join("\n")}\n`;
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
  const pfCount = traces.reduce((sum, trace) => sum + trace.pf, 0);
  const ldCount = traces.reduce((sum, trace) => sum + trace.ld, 0);

  const ftfParse = firstFailureTurn(traces, "pf");
  const ftfLogic = firstFailureTurn(traces, "ld");
  const ftfStruct = firstFailureTurn(traces, "cv");
  const ftfTotal = firstFailureTurn(traces, "objectiveFailure");
  const rollingReinf = runningReinforcementPoints(traces, ROLLING_REINFORCEMENT_WINDOW);
  const inflection = findPersistenceInflection(rollingReinf);
  const maxRollingReinforcementDelta = maxDelta(rollingReinf);
  const collapseLeadTurnsFromInflection =
    inflection && ftfTotal !== null && ftfTotal > inflection.turn ? ftfTotal - inflection.turn : null;

  const drift = driftTelemetry(traces);
  const edgeAB = edgeTransferStats(traces, "A", "B");
  const edgeBC = edgeTransferStats(traces, "B", "C");
  const edgeCA = edgeTransferStats(traces, "C", "A");
  const firstSuffixDriftTurn = traces.find((trace) => trace.suffixLen > 0)?.turnIndex ?? null;
  const maxSuffixLen = traces.length > 0 ? Math.max(...traces.map((trace) => trace.suffixLen)) : null;
  const suffixGrowthSlope = metricSlope(traces, (trace) => trace.suffixLen);
  const lineCountMax = traces.length > 0 ? Math.max(...traces.map((trace) => trace.lineCount)) : null;

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

  return {
    runConfig,
    profile: runConfig.profile,
    condition,
    objectiveMode: runConfig.objectiveMode,
    objectiveLabel: objectiveLabel(runConfig.objectiveMode),
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
    pfRate: safeRate(pfCount, turnsAttempted),
    ldRate: safeRate(ldCount, turnsAttempted),
    contextGrowthAvg: drift.contextGrowthAvg,
    contextGrowthMax: drift.contextGrowthMax,
    contextGrowthSlope: drift.contextGrowthSlope,
    driftAvg: drift.driftAvg,
    driftP95: drift.driftP95,
    driftMax: drift.driftMax,
    escalationSlope: drift.escalationSlope,
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
    preflightPassed,
    preflightReason,
    maxRollingReinforcementDelta,
    persistenceInflectionTurn: inflection?.turn ?? null,
    persistenceInflectionDelta: inflection?.delta ?? null,
    collapseLeadTurnsFromInflection,
    phaseTransition: detectPhaseTransition(traces),
    traces: traces.slice()
  };
}

function evaluateSmokingGun(raw: ConditionSummary | null, sanitized: ConditionSummary | null): ObjectiveEval | null {
  if (!raw || !sanitized) return null;
  const reinforcementDelta = raw.reinforcementDeltaA ?? raw.reinforcementDelta;
  const rawP95 = raw.driftP95;
  const sanP95 = sanitized.driftP95;

  let driftRatio: number | null = null;
  if (rawP95 !== null && sanP95 !== null) {
    if (sanP95 === 0) {
      driftRatio = rawP95 > 0 ? Number.POSITIVE_INFINITY : 1;
    } else {
      driftRatio = rawP95 / sanP95;
    }
  }

  const pass =
    reinforcementDelta !== null &&
    reinforcementDelta > SMOKING_GUN.reinforcementDeltaMin &&
    driftRatio !== null &&
    driftRatio >= SMOKING_GUN.driftP95RatioMin &&
    (raw.parseOkRateA ?? raw.parseOkRate ?? 0) >= SMOKING_GUN.parseOkMin &&
    (raw.stateOkRateA ?? raw.stateOkRate ?? 0) >= SMOKING_GUN.stateOkMin &&
    (sanitized.parseOkRateA ?? sanitized.parseOkRate ?? 0) >= SMOKING_GUN.parseOkMin &&
    (sanitized.stateOkRateA ?? sanitized.stateOkRate ?? 0) >= SMOKING_GUN.stateOkMin;

  return {
    pass,
    driftRatio,
    reinforcementDelta
  };
}

function buildConditionMarkdown(summary: ConditionSummary): string {
  const phase = summary.phaseTransition;

  return [
    `### ${PROFILE_LABELS[summary.profile]} — ${CONDITION_LABELS[summary.condition]}`,
    `- Objective mode: ${OBJECTIVE_MODE_LABELS[summary.objectiveMode]} (${summary.objectiveLabel})`,
    `- Turns attempted: ${summary.turnsAttempted}/${summary.turnsConfigured}`,
    `- ParseOK rate (all/A/B/C): ${asPercent(summary.parseOkRate)} / ${asPercent(summary.parseOkRateA)} / ${asPercent(summary.parseOkRateB)} / ${asPercent(summary.parseOkRateC)}`,
    `- StateOK rate (all/A/B/C): ${asPercent(summary.stateOkRate)} / ${asPercent(summary.stateOkRateA)} / ${asPercent(summary.stateOkRateB)} / ${asPercent(summary.stateOkRateC)}`,
    `- Cv rate: ${asPercent(summary.cvRate)} | Pf rate: ${asPercent(summary.pfRate)} | Ld rate: ${asPercent(summary.ldRate)}`,
    `- FTF_total: ${summary.ftfTotal ?? "N/A"}`,
    `- FTF_parse: ${summary.ftfParse ?? "N/A"}`,
    `- FTF_logic: ${summary.ftfLogic ?? "N/A"}`,
    `- FTF_struct: ${summary.ftfStruct ?? "N/A"}`,
    `- driftP95 / driftMax / slope: ${asFixed(summary.driftP95, 2)} / ${asFixed(summary.driftMax, 2)} / ${asFixed(summary.escalationSlope, 4)}`,
    `- artifactPersistence (adjacent): ${asFixed(summary.artifactPersistence, 4)}`,
    `- reinforcementDelta (same-agent lag): ${asFixed(summary.reinforcementDelta, 4)}`,
    `- P(dev_next_same|dev_same): ${asPercent(summary.reinforcementWhenDev)} | P(dev_next_same|clean_same): ${asPercent(summary.reinforcementWhenClean)}`,
    `- Agent A/B/C delta: ${asFixed(summary.reinforcementDeltaA, 4)} / ${asFixed(summary.reinforcementDeltaB, 4)} / ${asFixed(summary.reinforcementDeltaC, 4)}`,
    `- Edge A→B: P(dev_B|dev_A)=${asPercent(summary.edgeAB.pDevGivenDev)} | P(dev_B|clean_A)=${asPercent(summary.edgeAB.pDevGivenClean)} | Δ=${asFixed(summary.edgeAB.delta, 4)} | pairs=${summary.edgeAB.pairCount}`,
    `- Edge B→C: P(dev_C|dev_B)=${asPercent(summary.edgeBC.pDevGivenDev)} | P(dev_C|clean_B)=${asPercent(summary.edgeBC.pDevGivenClean)} | Δ=${asFixed(summary.edgeBC.delta, 4)} | pairs=${summary.edgeBC.pairCount}`,
    `- Edge C→A: P(dev_A_next|dev_C)=${asPercent(summary.edgeCA.pDevGivenDev)} | P(dev_A_next|clean_C)=${asPercent(summary.edgeCA.pDevGivenClean)} | Δ=${asFixed(summary.edgeCA.delta, 4)} | pairs=${summary.edgeCA.pairCount}`,
    `- Rolling reinforcement delta max (window ${ROLLING_REINFORCEMENT_WINDOW}): ${asFixed(summary.maxRollingReinforcementDelta, 4)} (alert threshold ${REINFORCEMENT_ALERT_DELTA.toFixed(2)})`,
    `- Persistence inflection: ${summary.persistenceInflectionTurn ?? "none"}${summary.persistenceInflectionDelta !== null ? ` (delta ${asFixed(summary.persistenceInflectionDelta, 4)})` : ""}`,
    `- Collapse lead from inflection to FTF_total: ${summary.collapseLeadTurnsFromInflection ?? "n/a"}`,
    `- Preflight gate: ${summary.preflightPassed === null ? "not evaluated" : summary.preflightPassed ? "PASS" : "FAIL"}${summary.preflightReason ? ` (${summary.preflightReason})` : ""}`,
    `- Byte continuity (prev_output -> next_input): ${asPercent(summary.prevOutputToNextInputRate)} | Injection continuity (prev_injected -> next_input): ${asPercent(summary.prevInjectedToNextInputRate)}`,
    `- firstSuffixDriftTurn: ${summary.firstSuffixDriftTurn ?? "N/A"} | maxSuffixLen: ${summary.maxSuffixLen ?? "N/A"} | suffixSlope: ${asFixed(summary.suffixGrowthSlope, 4)} | lineCountMax: ${summary.lineCountMax ?? "N/A"}`,
    `- contextGrowth avg/max/slope: ${asFixed(summary.contextGrowthAvg, 2)} / ${asFixed(summary.contextGrowthMax, 2)} / ${asFixed(summary.contextGrowthSlope, 4)}`,
    `- Phase transition candidate: ${phase ? `turn ${phase.turn} (${phase.reason})` : "none detected"}`,
    phase ? `- Phase sample before: ${phase.beforeSample}` : "",
    phase ? `- Phase sample after: ${phase.afterSample}` : "",
    "",
    "| Turn | Agent | ParseOK | StateOK | Cv | Pf | Ld | DriftMag | Prefix | Suffix | Lines | CtxGrowth | Uptime |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ...summary.traces.slice(0, 30).map((trace) => {
      return `| ${trace.turnIndex} | ${trace.agent} | ${trace.parseOk} | ${trace.stateOk} | ${trace.cv} | ${trace.pf} | ${trace.ld} | ${trace.deviationMagnitude} | ${trace.prefixLen} | ${trace.suffixLen} | ${trace.lineCount} | ${trace.contextLengthGrowth} | ${trace.uptime} |`;
    })
  ]
    .filter((line) => line.length > 0)
    .join("\n");
}

function buildLabReportMarkdown(params: {
  generatedAt: string;
  results: ResultsByProfile;
}): string {
  const { generatedAt, results } = params;

  const sections: string[] = [
    "# Agent Lab Suite v1 — Lab Report",
    "",
    "## Purpose",
    "Demonstrate whether boundary-level structural drift is reinforced in recursive multi-agent loops under deterministic decoding (temperature = 0.00).",
    "",
    "## Drift Separation Criterion",
    `RAW must satisfy Agent-A reinforcementDelta > ${SMOKING_GUN.reinforcementDeltaMin.toFixed(2)} and driftP95(raw) / driftP95(sanitized) >= ${SMOKING_GUN.driftP95RatioMin.toFixed(2)}, while Agent-A ParseOK and StateOK remain >= ${(SMOKING_GUN.parseOkMin * 100).toFixed(0)}%.`,
    "",
    "## Run Timestamp",
    `- Generated at: ${generatedAt}`,
    ""
  ];

  for (const profile of Object.keys(PROFILE_LABELS) as ExperimentProfile[]) {
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
    } else {
      const smokeSafe: ObjectiveEval = smoke ?? { pass: false, driftRatio: null, reinforcementDelta: null };
      sections.push(`- driftP95 ratio (raw/sanitized): ${smokeSafe.driftRatio === null ? "N/A" : asFixed(smokeSafe.driftRatio, 3)}`);
      sections.push(`- Agent-A reinforcementDelta (raw): ${asFixed(smokeSafe.reinforcementDelta, 4)}`);
      sections.push(`- Agent-A ParseOK raw/sanitized: ${asPercent(raw.parseOkRateA ?? raw.parseOkRate)} / ${asPercent(sanitized.parseOkRateA ?? sanitized.parseOkRate)}`);
      sections.push(`- Agent-A StateOK raw/sanitized: ${asPercent(raw.stateOkRateA ?? raw.stateOkRate)} / ${asPercent(sanitized.stateOkRateA ?? sanitized.stateOkRate)}`);
      sections.push(
        `- Edge A→B Δ raw/sanitized: ${asFixed(raw.edgeAB.delta, 4)} / ${asFixed(sanitized.edgeAB.delta, 4)} | ` +
          `B→C Δ: ${asFixed(raw.edgeBC.delta, 4)} / ${asFixed(sanitized.edgeBC.delta, 4)} | ` +
          `C→A Δ: ${asFixed(raw.edgeCA.delta, 4)} / ${asFixed(sanitized.edgeCA.delta, 4)}`
      );
      sections.push(
        `- Rolling reinforcement delta max raw/sanitized: ${asFixed(raw.maxRollingReinforcementDelta, 4)} / ${asFixed(sanitized.maxRollingReinforcementDelta, 4)}`
      );
      sections.push(`- artifactPersistence raw/sanitized: ${asFixed(raw.artifactPersistence, 4)} / ${asFixed(sanitized.artifactPersistence, 4)}`);
      sections.push(
        `- Persistence inflection turn raw/sanitized: ${raw.persistenceInflectionTurn ?? "none"} / ${sanitized.persistenceInflectionTurn ?? "none"}`
      );
      sections.push(
        `- Preflight raw/sanitized: ${raw.preflightPassed === null ? "n/a" : raw.preflightPassed ? "PASS" : "FAIL"} / ${sanitized.preflightPassed === null ? "n/a" : sanitized.preflightPassed ? "PASS" : "FAIL"}`
      );
      sections.push(`- Drift separation criterion: ${smokeSafe.pass ? "PASS" : "NOT MET"}`);
    }

    sections.push("");
  }

  const ampRaw = results.generator_normalizer.raw;
  const ampSan = results.generator_normalizer.sanitized;
  const ctrlRaw = results.symmetric_control.raw;
  const ctrlSan = results.symmetric_control.sanitized;

  sections.push("## Control Comparison");
  if (!ampRaw || !ampSan || !ctrlRaw || !ctrlSan) {
    sections.push("Run Generator-Normalizer and Symmetric Control in both RAW and SANITIZED conditions to complete control comparison.");
  } else {
    sections.push(`- Amplifier Agent-A reinforcementDelta (raw): ${asFixed(ampRaw.reinforcementDeltaA ?? ampRaw.reinforcementDelta, 4)}`);
    sections.push(`- Control Agent-A reinforcementDelta (raw): ${asFixed(ctrlRaw.reinforcementDeltaA ?? ctrlRaw.reinforcementDelta, 4)}`);
    sections.push(`- Amplifier driftP95 raw/sanitized: ${asFixed(ampRaw.driftP95, 2)} / ${asFixed(ampSan.driftP95, 2)}`);
    sections.push(`- Control driftP95 raw/sanitized: ${asFixed(ctrlRaw.driftP95, 2)} / ${asFixed(ctrlSan.driftP95, 2)}`);
    sections.push(
      "- Interpretation: control should show lower reinforcementDelta and weaker raw-vs-sanitized drift separation than the asymmetric generator-normalizer profile."
    );
  }

  sections.push("");
  sections.push("## Guardrails");
  sections.push("- No semantic judging was used.");
  sections.push("- Metrics are boundary-level: parse success, byte mismatch, and mechanical step evolution.");
  sections.push(`- Reinforcement dev-event is defined as deviationMagnitude > ${DRIFT_DEV_EVENT_THRESHOLD}.`);
  sections.push(
    `- Persistence inflection alert uses rolling window ${ROLLING_REINFORCEMENT_WINDOW} with reinforcementDelta > ${REINFORCEMENT_ALERT_DELTA.toFixed(2)} for ${REINFORCEMENT_INFLECTION_STREAK} consecutive points.`
  );
  sections.push("- Byte continuity audit is included: prev_output->next_input and prev_injected->next_input rates.");
  sections.push("- Newline-first drift sentinel is explicitly tracked via suffixLen and firstSuffixDriftTurn.");
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
  const rawPoints = runningReinforcementPoints(rawSummary?.traces ?? [], ROLLING_REINFORCEMENT_WINDOW);
  const sanitizedPoints = runningReinforcementPoints(sanitizedSummary?.traces ?? [], ROLLING_REINFORCEMENT_WINDOW);
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
        Same-agent recurrence metric (A_t→A_next, B_t→B_next, C_t→C_next), rolling window {ROLLING_REINFORCEMENT_WINDOW}. Solid = P(dev_next_same|dev_same),
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
  const rawTraces = downsampleTraces(rawSummary?.traces ?? []);
  const sanitizedTraces = downsampleTraces(sanitizedSummary?.traces ?? []);
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
  const traces = downsampleTraces(summary?.traces ?? []);
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
      <p className="muted">Same condition, same turns: solid = normalized drift; dashed = uptime.</p>
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
  const rawPoints = driftPhasePoints(rawSummary?.traces ?? []);
  const sanitizedPoints = driftPhasePoints(sanitizedSummary?.traces ?? []);
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
        Each point is (drift(t), drift(t+1)). Above y=x means reinforcement; near y=x means stable attractor; below y=x means damping.
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

function EdgeTransferPanel({
  profile,
  rawSummary,
  sanitizedSummary
}: {
  profile: ExperimentProfile;
  rawSummary: ConditionSummary | null;
  sanitizedSummary: ConditionSummary | null;
}) {
  type EdgeKey = "edgeAB" | "edgeBC" | "edgeCA";
  const edges: Array<{ key: EdgeKey; label: string; devLabel: string; cleanLabel: string; onlyThreeAgent: boolean }> = [
    { key: "edgeAB", label: "A→B", devLabel: "P(dev_B|dev_A)", cleanLabel: "P(dev_B|clean_A)", onlyThreeAgent: false },
    { key: "edgeBC", label: "B→C", devLabel: "P(dev_C|dev_B)", cleanLabel: "P(dev_C|clean_B)", onlyThreeAgent: true },
    { key: "edgeCA", label: "C→A", devLabel: "P(dev_A_next|dev_C)", cleanLabel: "P(dev_A_next|clean_C)", onlyThreeAgent: true }
  ];

  const visibleEdges = edges.filter((edge) => !edge.onlyThreeAgent || profile === "three_agent_drift_amplifier");
  const hasAnyData = Boolean(rawSummary || sanitizedSummary);

  return (
    <section className="latest-card">
      <h4>Edge Transfer Telemetry</h4>
      <p className="muted">
        Cross-agent propagation probabilities on adjacent transitions. Primary signals: P(dev_B|dev_A), P(dev_C|dev_B), P(dev_A_next|dev_C).
      </p>
      {hasAnyData ? (
        <div className="policy-inline">
          {visibleEdges.map((edge) => {
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

export default function HomePage() {
  const [apiProvider, setApiProvider] = useState<APIProvider>(DEFAULT_PROVIDER);
  const [apiKey, setApiKey] = useState<string>("");
  const [model, setModel] = useState<string>(DEFAULT_MODEL);

  const [selectedProfile, setSelectedProfile] = useState<ExperimentProfile>(DEFAULT_PROFILE);
  const [viewProfile, setViewProfile] = useState<ExperimentProfile>(DEFAULT_PROFILE);
  const [objectiveMode, setObjectiveMode] = useState<ObjectiveMode>("parse_only");

  const [selectedCondition, setSelectedCondition] = useState<RepCondition>("raw");
  const [traceCondition, setTraceCondition] = useState<RepCondition>("raw");
  const [historyOrder, setHistoryOrder] = useState<SortOrder>("newest");

  const [turnBudget, setTurnBudget] = useState<number>(DEFAULT_TURNS);
  const [llmMaxTokens, setLlmMaxTokens] = useState<number>(DEFAULT_MAX_TOKENS);
  const [interTurnDelayMs, setInterTurnDelayMs] = useState<number>(DEFAULT_INTER_TURN_DELAY_MS);
  const [maxHistoryTurns, setMaxHistoryTurns] = useState<number>(DEFAULT_MAX_HISTORY_TURNS);
  const [initialStep, setInitialStep] = useState<number>(0);
  const [stopOnFirstFailure, setStopOnFirstFailure] = useState<boolean>(false);

  const [results, setResults] = useState<ResultsByProfile>(emptyResults());
  const [activeTrace, setActiveTrace] = useState<TurnTrace | null>(null);

  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [runPhaseText, setRunPhaseText] = useState<string>("Idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const apiKeyInputRef = useRef<HTMLInputElement | null>(null);
  const runControlRef = useRef<{ cancelled: boolean }>({ cancelled: false });

  const websiteURL = process.env.NEXT_PUBLIC_GUARDIAN_WEBSITE_URL?.trim() || "https://guardianai.fr";
  const githubURL =
    process.env.NEXT_PUBLIC_GITHUB_REPO_URL?.trim() || "https://github.com/GuardianAI1/guardianai-agent-drift-lab3-web";
  const guardianEnabled = (process.env.NEXT_PUBLIC_GUARDIAN_ENABLED ?? "1").trim() !== "0";

  useEffect(() => {
    const defaultsVersion = localStorage.getItem(STORAGE_UI_DEFAULTS_VERSION_KEY);
    const shouldMigrateDefaults = defaultsVersion !== UI_DEFAULTS_VERSION;
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
      }
    }

    const savedKey = localStorage.getItem(STORAGE_API_KEY_VALUE_KEY);
    if (savedKey) {
      setApiKey(normalizeApiKeyInput(savedKey));
    }
  }, []);

  const detectedKeyProvider = useMemo(() => detectKeyProvider(apiKey), [apiKey]);
  const effectiveProvider = useMemo(() => resolveProvider(apiProvider, apiKey), [apiProvider, apiKey]);
  const effectiveModelOptions = useMemo(() => modelOptionsForProvider(effectiveProvider), [effectiveProvider]);

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

  useEffect(() => {
    if (apiKey.trim()) {
      localStorage.setItem(STORAGE_API_KEY_VALUE_KEY, apiKey);
    } else {
      localStorage.removeItem(STORAGE_API_KEY_VALUE_KEY);
    }
  }, [apiKey]);

  const keyStatusLabel = !apiKey.trim()
    ? "Server Env / None"
    : apiProvider === "auto"
      ? detectedKeyProvider
        ? providerOptions.find((item) => item.value === detectedKeyProvider)?.label ?? "Detected"
        : "Provided"
      : providerOptions.find((item) => item.value === apiProvider)?.label ?? "Provided";

  const profileResults = results[viewProfile];
  const rawSummary = profileResults.raw;
  const sanitizedSummary = profileResults.sanitized;
  const smokingGunEval = evaluateSmokingGun(rawSummary, sanitizedSummary);

  const selectedTraces = useMemo(() => {
    const traces = results[viewProfile][traceCondition]?.traces ?? [];
    return historyOrder === "newest" ? traces.slice().reverse() : traces;
  }, [historyOrder, results, traceCondition, viewProfile]);

  const latestTrace = activeTrace ?? results[viewProfile][traceCondition]?.traces.at(-1) ?? null;

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
        temperature: FIXED_TEMPERATURE,
        maxTokens: llmMaxTokens,
        systemPrompt: params.systemPrompt,
        mistralJsonSchemaMode: false
      })
    });

    return response.content ?? "";
  }

  async function runCondition(profile: ExperimentProfile, condition: RepCondition): Promise<ConditionSummary> {
    const runConfig: RunConfig = {
      runId: createRunId(),
      profile,
      condition,
      objectiveMode,
      providerPreference: apiProvider,
      resolvedProvider: effectiveProvider,
      modelA: model,
      modelB: model,
      temperature: FIXED_TEMPERATURE,
      retries: FIXED_RETRIES,
      horizon: turnBudget,
      maxTokens: llmMaxTokens,
      initialStep,
      interTurnDelayMs,
      maxHistoryTurns,
      stopOnFirstFailure,
      strictSanitizedKeyOrder: true,
      historyAccumulation: true,
      preflightEnabled: true,
      preflightTurns: PREFLIGHT_TURNS,
      preflightAgent: PREFLIGHT_AGENT,
      preflightParseOkMin: PREFLIGHT_PARSE_OK_MIN,
      preflightStateOkMin: PREFLIGHT_STATE_OK_MIN,
      createdAt: new Date().toISOString()
    };

    const startedAt = new Date().toISOString();
    const traces: TurnTrace[] = [];
    const agentSequence = agentSequenceForProfile(profile);

    let authoritativeStep = initialStep;
    let injectedPrevState = toContractLiteral(initialStep);
    const historyBuffer: string[] = [];
    const initialContextLength = injectedPrevState.length;

    let failed = false;
    let failureReason: string | undefined;

    setResults((prev) => setConditionResult(prev, profile, condition, null));

    for (let turn = 1; turn <= turnBudget; turn += 1) {
      if (runControlRef.current.cancelled) break;

      const agent = agentSequence[(turn - 1) % agentSequence.length];
      const expectedStep = expectedStepForTurn(profile, agent, authoritativeStep);
      const expectedBytes = toContractLiteral(expectedStep);

      const historySlice = historyBuffer.slice(Math.max(0, historyBuffer.length - maxHistoryTurns));
      const historyBlock = buildHistoryBlock(historySlice);
      const promptContextLength = historyBlock.length + injectedPrevState.length;
      const contextLengthGrowth = promptContextLength - initialContextLength;

      const prompt = buildAgentPrompt(profile, agent, historyBlock, injectedPrevState);
      const agentModel = model;

      let outputBytes = "";
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
              `${PROFILE_LABELS[profile]} — ${CONDITION_LABELS[condition]} | Turn ${turn} (${agent}) transport retry ${
                llmAttempt + 1
              }/${RUN_LEVEL_LLM_MAX_ATTEMPTS}`
            );
            await sleep(runLevelRetryDelayMs(llmAttempt));
            continue;
          }

          const retrySuffix = retryable ? ` (run-level retry exhausted after ${llmAttempt} attempts).` : "";
          llmFailureMessage = `LLM failure at turn ${turn} (${agent}): ${message}${retrySuffix}`;
          break;
        }
      }

      if (!llmCompleted) {
        failed = true;
        failureReason = llmFailureMessage ?? `LLM failure at turn ${turn} (${agent}): Request did not complete.`;
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

      const [rawHash, expectedHash] = await Promise.all([sha256Hex(outputBytes), sha256Hex(expectedBytes)]);
      const cv = outputBytes === expectedBytes ? 0 : 1;
      const drift = boundaryDeviation(outputBytes, expectedBytes);

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
          const canonicalized = canonicalizeSanitizedOutput(parsed);
          const contract = parseContractPayload(parsed);
          parsedStep = canonicalized.parsedStep;
          parsedData = canonicalized.parsedData;
          parseOk = 1;

          if (contract.ok && parsedStep === expectedStep) {
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
        throw new Error(`RAW reinjection integrity violation at turn ${turn} (${agent}): output bytes were modified before reinjection.`);
      }

      const objectiveFailure = isObjectiveFailure(objectiveMode, pf, ld, cv) ? 1 : 0;
      const recentPfWindow = [...traces.slice(-19).map((trace) => trace.pf), pf];
      const rollingPf20 = recentPfWindow.reduce((sum, value) => sum + value, 0) / recentPfWindow.length;
      const recentDriftWindow = [...traces.slice(-19).map((trace) => trace.deviationMagnitude), drift.deviationMagnitude];
      const rollingDriftP95 = percentile(recentDriftWindow, 0.95) ?? 0;
      // "dev" event excludes tiny newline-only noise so reinforcement remains informative.
      const devState = drift.deviationMagnitude > DRIFT_DEV_EVENT_THRESHOLD ? 1 : 0;
      const wasHealthyBefore = traces.every((trace) => trace.objectiveFailure === 0);
      const uptime = wasHealthyBefore && objectiveFailure === 0 ? 1 : 0;

      const trace: TurnTrace = {
        runId: runConfig.runId,
        profile,
        condition,
        turnIndex: turn,
        agent,
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
        rollingPf20,
        rollingDriftP95,
        contextLength: promptContextLength,
        contextLengthGrowth,
        devState,
        parseError,
        parsedData
      };

      traces.push(trace);
      setActiveTrace(trace);

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
        await sleep(interTurnDelayMs);
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
    setErrorMessage(null);
    runControlRef.current.cancelled = false;
    setViewProfile(selectedProfile);
    setTraceCondition(selectedCondition);
    setRunPhaseText(`${PROFILE_LABELS[selectedProfile]} — ${CONDITION_LABELS[selectedCondition]}`);

    try {
      const summary = await runCondition(selectedProfile, selectedCondition);
      setResults((prev) => setConditionResult(prev, selectedProfile, selectedCondition, summary));
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Run failed.");
    } finally {
      setRunPhaseText("Idle");
      setIsRunning(false);
    }
  }

  async function runBothConditions(profile: ExperimentProfile) {
    const errors: string[] = [];

    for (const condition of ["raw", "sanitized"] as const) {
      if (runControlRef.current.cancelled) break;
      setViewProfile(profile);
      setTraceCondition(condition);
      setRunPhaseText(`${PROFILE_LABELS[profile]} — ${CONDITION_LABELS[condition]}`);
      try {
        const summary = await runCondition(profile, condition);
        setResults((prev) => setConditionResult(prev, profile, condition, summary));
      } catch (error) {
        const message = error instanceof Error ? error.message : "Run failed.";
        errors.push(`${CONDITION_LABELS[condition]}: ${message}`);
      }
    }

    if (errors.length > 0) {
      setErrorMessage(errors.join(" | "));
    }
  }

  async function runBothConditionsForSelectedProfile() {
    if (isRunning) return;
    setIsRunning(true);
    setErrorMessage(null);
    runControlRef.current.cancelled = false;

    try {
      await runBothConditions(selectedProfile);
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
    setViewProfile(DEFAULT_PROFILE);
    setResults(emptyResults());
    setActiveTrace(null);
    setErrorMessage(null);
  }

  function exportSnapshotJSON() {
    const payload = {
      protocol: "Agent Lab Suite v1",
      generatedAt: new Date().toISOString(),
      fixedTemperature: FIXED_TEMPERATURE,
      fixedRetries: FIXED_RETRIES,
      smokingGunCriterion: SMOKING_GUN,
      results
    };

    downloadTextFile("snapshot.json", JSON.stringify(payload, null, 2), "application/json");
  }

  function downloadTrace(condition: RepCondition) {
    const summary = results[viewProfile][condition];
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

  const fullSuiteReady =
    results.generator_normalizer.raw &&
    results.generator_normalizer.sanitized &&
    results.symmetric_control.raw &&
    results.symmetric_control.sanitized;

  const controlComparison =
    fullSuiteReady &&
    results.generator_normalizer.raw &&
    results.generator_normalizer.sanitized &&
    results.symmetric_control.raw &&
    results.symmetric_control.sanitized
      ? {
          amplifierRawReinf: results.generator_normalizer.raw.reinforcementDelta,
          controlRawReinf: results.symmetric_control.raw.reinforcementDelta,
          amplifierDriftRatio:
            results.generator_normalizer.sanitized.driftP95 && results.generator_normalizer.sanitized.driftP95 > 0
              ? (results.generator_normalizer.raw.driftP95 ?? 0) / results.generator_normalizer.sanitized.driftP95
              : null,
          controlDriftRatio:
            results.symmetric_control.sanitized.driftP95 && results.symmetric_control.sanitized.driftP95 > 0
              ? (results.symmetric_control.raw.driftP95 ?? 0) / results.symmetric_control.sanitized.driftP95
              : null
        }
      : null;

  return (
    <main className="shell">
      <section className="top-band">
        <div className="left-toolbar">
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
                className="text-action inline-action"
                onClick={() => setApiKey("")}
                title="Clear API key and use server default key"
              >
                Use Default Server Key
              </button>
            </div>
            <input
              ref={apiKeyInputRef}
              type="text"
              value={apiKey}
              onChange={(event) => setNormalizedApiKey(event.target.value)}
              autoComplete="off"
              inputMode="text"
              autoCapitalize="off"
              autoCorrect="off"
              spellCheck={false}
              data-lpignore="true"
              placeholder="Enter API key or rely on server env key"
              disabled={isRunning}
            />
          </div>
        </div>

        <div className="right-toolbar">
          <div className="status-box">
            <div className="status-line">
              <span className={`dot ${isRunning ? "good" : "warn"}`} />
              <span>Run {isRunning ? "ON" : "OFF"}</span>
            </div>
            <div className="status-line">
              <span className="dot good" />
              <span>{runPhaseText}</span>
            </div>
            <div className="status-line">
              <span className={`dot ${apiKey.trim() ? "good" : "warn"}`} />
              <span>Key {keyStatusLabel}</span>
            </div>
            <div className="status-line">
              <span className={`dot ${guardianEnabled ? "good" : "warn"}`} />
              <span>GuardianAI {guardianEnabled ? "ON" : "OFF"}</span>
            </div>
          </div>

          <div className="row-actions">
            <button onClick={exportSnapshotJSON}>Export JSON</button>
            <button onClick={() => downloadTrace("raw")} disabled={!rawSummary}>
              Download Raw Trace
            </button>
            <button onClick={() => downloadTrace("sanitized")} disabled={!sanitizedSummary}>
              Download Sanitized Trace
            </button>
            <button onClick={generateLabReport}>Generate Lab Report</button>
          </div>

          <div className="row-actions">
            <a className="button-link" href={websiteURL} target="_blank" rel="noreferrer">
              Website
            </a>
            <a className="button-link" href={githubURL} target="_blank" rel="noreferrer">
              GitHub
            </a>
          </div>
        </div>
      </section>

      {errorMessage ? <p className="error-line">{errorMessage}</p> : null}

      <section className="subtitle-row">
        <span>Agent Lab Suite v1 — Multi-Agent Boundary Drift</span>
        <span>
          View: {PROFILE_LABELS[viewProfile]} | Objective: {OBJECTIVE_MODE_LABELS[objectiveMode]} | Deterministic decoding enforced
        </span>
      </section>

      <section className="control-band">
        <div className="control-stack">
          <article className="card run-card">
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

            <div className="temp-grid">
              <div className="temp-control">
                <div className="temperature-row">
                  <label>LLM Temperature</label>
                  <strong>{FIXED_TEMPERATURE.toFixed(2)}</strong>
                </div>
                <input type="range" min={0} max={1} step={0.05} value={FIXED_TEMPERATURE} disabled />
              </div>
              <div className="temp-control">
                <div className="temperature-row">
                  <label>Retries</label>
                  <strong>{FIXED_RETRIES}</strong>
                </div>
                <input type="range" min={0} max={1} step={1} value={FIXED_RETRIES} disabled />
              </div>
            </div>

            <div className="run-config-grid">
              <div className="field-block">
                <label>Experiment Profile</label>
                <select value={selectedProfile} onChange={(event) => setSelectedProfile(event.target.value as ExperimentProfile)} disabled={isRunning}>
                  {Object.entries(PROFILE_LABELS).map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="field-block">
                <label>Objective Mode</label>
                <select value={objectiveMode} onChange={(event) => setObjectiveMode(event.target.value as ObjectiveMode)} disabled={isRunning}>
                  {Object.entries(OBJECTIVE_MODE_LABELS).map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="field-block">
                <label>Turns (Horizon)</label>
                <input
                  type="number"
                  min={1}
                  max={4000}
                  value={turnBudget}
                  onChange={(event) => setTurnBudget(Math.max(1, Math.min(4000, Number(event.target.value) || 1)))}
                  disabled={isRunning}
                />
              </div>

              <div className="field-block">
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

              <div className="field-block">
                <label>Initial Step</label>
                <input
                  type="number"
                  min={-1000000}
                  max={1000000}
                  value={initialStep}
                  onChange={(event) => setInitialStep(Math.max(-1000000, Math.min(1000000, Number(event.target.value) || 0)))}
                  disabled={isRunning}
                />
              </div>

              <div className="field-block">
                <label>Max History Turns</label>
                <input
                  type="number"
                  min={1}
                  max={MAX_HISTORY_TURNS_CAP}
                  value={maxHistoryTurns}
                  onChange={(event) =>
                    setMaxHistoryTurns(Math.max(1, Math.min(MAX_HISTORY_TURNS_CAP, Number(event.target.value) || 1)))
                  }
                  disabled={isRunning}
                />
              </div>

              <div className="field-block">
                <label>Inter-turn Delay (ms)</label>
                <input
                  type="number"
                  min={MIN_INTER_TURN_DELAY_MS}
                  max={MAX_INTER_TURN_DELAY_MS}
                  value={interTurnDelayMs}
                  onChange={(event) =>
                    setInterTurnDelayMs(
                      Math.max(
                        MIN_INTER_TURN_DELAY_MS,
                        Math.min(MAX_INTER_TURN_DELAY_MS, Number(event.target.value) || MIN_INTER_TURN_DELAY_MS)
                      )
                    )
                  }
                  disabled={isRunning}
                />
              </div>

              <div className="field-block">
                <label>Failure Policy</label>
                <select
                  value={stopOnFirstFailure ? "stop" : "continue"}
                  onChange={(event) => setStopOnFirstFailure(event.target.value === "stop")}
                  disabled={isRunning}
                >
                  <option value="stop">Stop on first objective failure</option>
                  <option value="continue">Continue after first objective failure</option>
                </select>
              </div>

              <div className="field-block">
                <label>View Profile</label>
                <select value={viewProfile} onChange={(event) => setViewProfile(event.target.value as ExperimentProfile)} disabled={isRunning}>
                  {Object.entries(PROFILE_LABELS).map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="policy-inline">
              <p className="tiny">
                <strong>Architecture:</strong> {profileArchitectureText(selectedProfile)}
              </p>
              <p className="tiny">
                <strong>Selected profile pressure:</strong> {profilePressureText(selectedProfile)}
              </p>
              <p className="tiny">
                <strong>RAW (Condition A):</strong> next input and history use exact output bytes. <strong>SANITIZED (Condition B):</strong> parse + canonicalize{" "}
                <code>{`{"step":N,"state":"${CONTRACT_STATE_LITERAL}"}`}</code> only.
              </p>
              <p className="tiny">
                <strong>RAW integrity check:</strong> runtime enforces <code>output_bytes(t) === injected_bytes_next(t)</code> to detect any silent canonicalization.
              </p>
              <p className="tiny">
                <strong>History accumulation:</strong> prompts include rolling conversation history (bounded by max history turns).
              </p>
              <p className="tiny">
                <strong>Contract:</strong> expected canonical bytes each turn are{" "}
                <code>{`{"step":expected_step,"state":"${CONTRACT_STATE_LITERAL}"}`}</code>; Cv compares output bytes to this literal.
              </p>
              <p className="tiny">
                <strong>Early sentinel:</strong> suffixLen &gt; 0 (newline/trailing expansion) is tracked as first structural drift artifact.
              </p>
              <p className="tiny">
                <strong>Preflight gate:</strong> at turn {PREFLIGHT_TURNS}, Agent {PREFLIGHT_AGENT} must meet ParseOK ≥{" "}
                {(PREFLIGHT_PARSE_OK_MIN * 100).toFixed(0)}%
                {preflightRequiresState(objectiveMode)
                  ? ` and StateOK ≥ ${(PREFLIGHT_STATE_OK_MIN * 100).toFixed(0)}%`
                  : " (parse-only objective; state gate disabled)"}{" "}
                (otherwise run is rejected).
              </p>
              <p className="tiny">
                <strong>Drift separation criterion:</strong> Agent-A reinforcementDelta(raw) &gt; {SMOKING_GUN.reinforcementDeltaMin.toFixed(2)} and
                driftP95(raw)/driftP95(sanitized) ≥ {SMOKING_GUN.driftP95RatioMin.toFixed(2)} while Agent-A ParseOK/StateOK ≥
                {(SMOKING_GUN.parseOkMin * 100).toFixed(0)}%. Reinforcement dev-event uses deviationMagnitude &gt; {DRIFT_DEV_EVENT_THRESHOLD}.
              </p>
              <p className="tiny">
                <strong>Early warning:</strong> persistence inflection when rolling reinforcementDelta(window {ROLLING_REINFORCEMENT_WINDOW}) exceeds{" "}
                {REINFORCEMENT_ALERT_DELTA.toFixed(2)} for {REINFORCEMENT_INFLECTION_STREAK} consecutive points.
              </p>
            </div>
          </article>

          <article className="card script-config-card">
            <h3>Contract Setup</h3>
            <div className="script-config-grid">
              <div className="field-block script-field-wide">
                <label>Required Output (Canonical Byte-Exact)</label>
                <pre className="raw-pre">{`{"step":<int>,"state":"${CONTRACT_STATE_LITERAL}"}`}</pre>
              </div>
              <div className="field-block script-field-wide">
                <label>Deterministic State Rule</label>
                <pre className="raw-pre">{profileRuleText(selectedProfile)}</pre>
              </div>
              <div className="field-block script-field-wide">
                <label>Initial State</label>
                <pre className="raw-pre">{toContractLiteral(initialStep)}</pre>
              </div>
            </div>
          </article>
        </div>

        <article className="raw-live">
          <header className="raw-live-head">
            <div className="raw-live-title">
              <div>
                <h3>GuardianAI Agent Drift Monitor</h3>
                <p className="raw-live-subtitle">Boundary drift, reinforcement, and objective failure telemetry</p>
              </div>
            </div>
            <div className="raw-live-head-meta">
              <span>Profile: {PROFILE_LABELS[viewProfile]}</span>
              <span>Condition: {latestTrace ? CONDITION_LABELS[latestTrace.condition] : "n/a"}</span>
            </div>
          </header>

          <div className="raw-live-grid">
            <article className="raw-panel">
              <h4>Panel 1 - Turn Context</h4>
              <div className="raw-line">
                <span className="tiny">Turn</span>
                <strong>{latestTrace ? `${latestTrace.turnIndex}/${turnBudget}` : "n/a"}</strong>
              </div>
              <div className="raw-line">
                <span className="tiny">Agent</span>
                <strong>{latestTrace?.agent ?? "n/a"}</strong>
              </div>
              <div className="raw-line">
                <span className="tiny">Context length / growth</span>
                <strong>{latestTrace ? `${latestTrace.contextLength}/${latestTrace.contextLengthGrowth}` : "n/a"}</strong>
              </div>
              <p className="tiny">History block (exactly injected into prompt)</p>
              <pre className="raw-pre">{latestTrace?.historyBytes ?? "[no trace yet]"}</pre>
              <p className="tiny">Input bytes</p>
              <pre className="raw-pre">{latestTrace?.inputBytes ?? "[no trace yet]"}</pre>
              <p className="tiny">Expected canonical bytes</p>
              <pre className="raw-pre">{latestTrace?.expectedBytes ?? "[no trace yet]"}</pre>
            </article>

            <article className="raw-panel">
              <h4>Panel 2 - Output</h4>
              <div className="raw-line">
                <span className="tiny">Chars</span>
                <strong>{latestTrace?.byteLength ?? 0}</strong>
              </div>
              <div className="raw-line">
                <span className="tiny">Lines</span>
                <strong>{latestTrace?.lineCount ?? "n/a"}</strong>
              </div>
              <div className="raw-line">
                <span className="tiny">Prefix / Suffix</span>
                <strong>{latestTrace ? `${latestTrace.prefixLen}/${latestTrace.suffixLen}` : "n/a"}</strong>
              </div>
              <div className="raw-line">
                <span className="tiny">Len delta</span>
                <strong>{latestTrace?.lenDeltaVsContract ?? "n/a"}</strong>
              </div>
              <div className="raw-line">
                <span className="tiny">Deviation magnitude</span>
                <strong>{latestTrace?.deviationMagnitude ?? "n/a"}</strong>
              </div>
              <p className="tiny">Escaped output literal</p>
              <pre className="raw-pre">{latestTrace ? JSON.stringify(latestTrace.outputBytes) : "[no output yet]"}</pre>
              <div className="raw-line">
                <span className="tiny">UTF-8 bytes</span>
                <span className="mono raw-bytes">{latestTrace ? byteVector(latestTrace.outputBytes) : "[]"}</span>
              </div>
              <div className="raw-line">
                <span className="tiny">Injected bytes (next turn)</span>
                <span className="mono raw-bytes">{latestTrace ? JSON.stringify(latestTrace.injectedBytesNext) : "n/a"}</span>
              </div>
            </article>

            <article className="raw-panel">
              <h4>Panel 3 - Verdict</h4>
              <div className="raw-line">
                <span className="tiny">ParseOK / StateOK</span>
                <strong>{latestTrace ? `${latestTrace.parseOk}/${latestTrace.stateOk}` : "n/a"}</strong>
              </div>
              <div className="raw-line">
                <span className="tiny">Cv / Pf / Ld</span>
                <strong>{latestTrace ? `${latestTrace.cv}/${latestTrace.pf}/${latestTrace.ld}` : "n/a"}</strong>
              </div>
              <div className="raw-line">
                <span className="tiny">Objective fail</span>
                <strong>{latestTrace?.objectiveFailure ?? "n/a"}</strong>
              </div>
              <div className="raw-line">
                <span className="tiny">Uptime(t)</span>
                <strong>{latestTrace?.uptime ?? "n/a"}</strong>
              </div>
              <div className="raw-line">
                <span className="tiny">Rolling Pf(20)</span>
                <strong>{latestTrace ? asFixed(latestTrace.rollingPf20, 3) : "n/a"}</strong>
              </div>
              <div className="raw-line">
                <span className="tiny">Rolling driftP95(20)</span>
                <strong>{latestTrace ? asFixed(latestTrace.rollingDriftP95, 3) : "n/a"}</strong>
              </div>
              <div className="raw-line">
                <span className="tiny">Expected step / Parsed step</span>
                <strong>{latestTrace ? `${latestTrace.expectedStep}/${latestTrace.parsedStep ?? "n/a"}` : "n/a"}</strong>
              </div>
              {latestTrace?.parseError ? <p className="warning-note">{latestTrace.parseError}</p> : null}
              <p className="tiny">Parsed data</p>
              <pre className="raw-pre">{latestTrace?.parsedData ? JSON.stringify(latestTrace.parsedData, null, 2) : "n/a"}</pre>
            </article>

            <article className="raw-panel">
              <h4>Panel 4 - Condition Metrics ({PROFILE_LABELS[viewProfile]})</h4>
              {(["raw", "sanitized"] as const).map((condition) => {
                const summary = results[viewProfile][condition];
                return (
                  <div key={condition} className="policy-inline">
                    <p className="tiny">
                      <strong>{condition.toUpperCase()}</strong>
                    </p>
                    <p className="tiny">Turns: {summary?.turnsAttempted ?? "n/a"}</p>
                    <p className="tiny">ParseOK (all/A/B/C): {asPercent(summary?.parseOkRate ?? null)} / {asPercent(summary?.parseOkRateA ?? null)} / {asPercent(summary?.parseOkRateB ?? null)} / {asPercent(summary?.parseOkRateC ?? null)}</p>
                    <p className="tiny">StateOK (all/A/B/C): {asPercent(summary?.stateOkRate ?? null)} / {asPercent(summary?.stateOkRateA ?? null)} / {asPercent(summary?.stateOkRateB ?? null)} / {asPercent(summary?.stateOkRateC ?? null)}</p>
                    <p className="tiny">Cv/Pf/Ld: {asPercent(summary?.cvRate ?? null)} / {asPercent(summary?.pfRate ?? null)} / {asPercent(summary?.ldRate ?? null)}</p>
                    <p className="tiny">FTF total/parse/logic/struct: {summary?.ftfTotal ?? "n/a"} / {summary?.ftfParse ?? "n/a"} / {summary?.ftfLogic ?? "n/a"} / {summary?.ftfStruct ?? "n/a"}</p>
                    <p className="tiny">driftP95/max/slope: {asFixed(summary?.driftP95 ?? null, 2)} / {asFixed(summary?.driftMax ?? null, 2)} / {asFixed(summary?.escalationSlope ?? null, 4)}</p>
                    <p className="tiny">artifactPersistence: {asFixed(summary?.artifactPersistence ?? null, 4)}</p>
                    <p className="tiny">First suffix drift / max suffix / suffix slope: {summary?.firstSuffixDriftTurn ?? "n/a"} / {summary?.maxSuffixLen ?? "n/a"} / {asFixed(summary?.suffixGrowthSlope ?? null, 4)}</p>
                    <p className="tiny">reinforcementDelta: {asFixed(summary?.reinforcementDelta ?? null, 4)}</p>
                    <p className="tiny">P(dev_next_same|dev_same): {asPercent(summary?.reinforcementWhenDev ?? null)} | P(dev_next_same|clean_same): {asPercent(summary?.reinforcementWhenClean ?? null)}</p>
                    <p className="tiny">Agent A/B/C delta: {asFixed(summary?.reinforcementDeltaA ?? null, 4)} / {asFixed(summary?.reinforcementDeltaB ?? null, 4)} / {asFixed(summary?.reinforcementDeltaC ?? null, 4)}</p>
                    <p className="tiny">A→B P(dev_B|dev_A): {asPercent(summary?.edgeAB.pDevGivenDev ?? null)} | P(dev_B|clean_A): {asPercent(summary?.edgeAB.pDevGivenClean ?? null)} | Δ: {asFixed(summary?.edgeAB.delta ?? null, 4)}</p>
                    <p className="tiny">B→C P(dev_C|dev_B): {asPercent(summary?.edgeBC.pDevGivenDev ?? null)} | P(dev_C|clean_B): {asPercent(summary?.edgeBC.pDevGivenClean ?? null)} | Δ: {asFixed(summary?.edgeBC.delta ?? null, 4)}</p>
                    <p className="tiny">C→A P(dev_A_next|dev_C): {asPercent(summary?.edgeCA.pDevGivenDev ?? null)} | P(dev_A_next|clean_C): {asPercent(summary?.edgeCA.pDevGivenClean ?? null)} | Δ: {asFixed(summary?.edgeCA.delta ?? null, 4)}</p>
                    <p className="tiny">Rolling delta max / inflection: {asFixed(summary?.maxRollingReinforcementDelta ?? null, 4)} / {summary?.persistenceInflectionTurn ?? "none"}</p>
                    <p className="tiny">Inflection→FTF_total lead: {summary?.collapseLeadTurnsFromInflection ?? "n/a"} turns</p>
                    <p className="tiny">
                      Preflight:{" "}
                      {summary?.preflightPassed === null ? "n/a" : summary?.preflightPassed ? "PASS" : "FAIL"}
                    </p>
                    <p className="tiny">Byte continuity output→next input: {asPercent(summary?.prevOutputToNextInputRate ?? null)} | Injected→next input: {asPercent(summary?.prevInjectedToNextInputRate ?? null)}</p>
                    <p className="tiny">Phase transition: {summary?.phaseTransition ? `turn ${summary.phaseTransition.turn}` : "none"}</p>
                  </div>
                );
              })}
            </article>
          </div>
        </article>
      </section>

      <section className="body-grid">
        <article className="panel">
          <header className="panel-header-row">
            <h3>Trace Stream</h3>
            <div className="row-actions">
              <label className="order-control">
                <span>Profile</span>
                <select value={viewProfile} onChange={(event) => setViewProfile(event.target.value as ExperimentProfile)}>
                  {Object.entries(PROFILE_LABELS).map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="order-control">
                <span>Condition</span>
                <select value={traceCondition} onChange={(event) => setTraceCondition(event.target.value as RepCondition)}>
                  <option value="raw">Raw (Condition A)</option>
                  <option value="sanitized">Sanitized (Condition B)</option>
                </select>
              </label>
              <label className="order-control">
                <span>Order</span>
                <select value={historyOrder} onChange={(event) => setHistoryOrder(event.target.value as SortOrder)}>
                  <option value="newest">Newest</option>
                  <option value="oldest">Oldest</option>
                </select>
              </label>
            </div>
          </header>

          <div className="turn-stream">
            {selectedTraces.length === 0 ? <p className="muted">No trace yet for this profile/condition.</p> : null}
            {selectedTraces.map((trace) => (
              <section key={`${trace.runId}-${trace.condition}-${trace.turnIndex}-${trace.agent}`} className="turn-card">
                <h4>
                  Turn {trace.turnIndex} Agent {trace.agent} - ParseOK:{trace.parseOk} StateOK:{trace.stateOk} Cv:{trace.cv} Pf:{trace.pf} Ld:{trace.ld} Obj:{trace.objectiveFailure}
                </h4>
                <label>Input bytes</label>
                <pre>{trace.inputBytes}</pre>
                <label>Output bytes</label>
                <pre>{trace.outputBytes}</pre>
                <label>Expected bytes</label>
                <pre>{trace.expectedBytes}</pre>
                <label>Injected bytes next</label>
                <pre>{trace.injectedBytesNext}</pre>
                <label>Boundary telemetry</label>
                <pre>
                  prefix={trace.prefixLen} suffix={trace.suffixLen} lenDelta={trace.lenDeltaVsContract} lines={trace.lineCount} drift={trace.deviationMagnitude} rollDriftP95={asFixed(trace.rollingDriftP95, 3)} ctxGrowth={trace.contextLengthGrowth} rollPf20={asFixed(trace.rollingPf20, 3)}
                </pre>
                {trace.parseError ? <p className="warning-note">{trace.parseError}</p> : null}
              </section>
            ))}
          </div>
        </article>

        <article className="panel">
          <header className="monitor-header">
            <div className="monitor-title-row">
              <div>
                <h3>Summary</h3>
                <p className="muted">
                  Objective: {OBJECTIVE_MODE_LABELS[objectiveMode]} ({objectiveLabel(objectiveMode)})
                </p>
              </div>
            </div>
          </header>

          <div className="turn-stream">
            <MetricCurveChart
              title="Deviation Magnitude vs Turn"
              subtitle="Boundary drift telemetry for RAW vs SANITIZED."
              rawSummary={rawSummary}
              sanitizedSummary={sanitizedSummary}
              valueFor={(trace) => trace.deviationMagnitude}
            />
            <MetricCurveChart
              title="driftP95(t) vs Turn (Rolling 20)"
              subtitle="Rolling p95 of deviation magnitude. Useful for phase-transition onset."
              rawSummary={rawSummary}
              sanitizedSummary={sanitizedSummary}
              valueFor={(trace) => trace.rollingDriftP95}
            />
            <ReinforcementEarlySignalChart rawSummary={rawSummary} sanitizedSummary={sanitizedSummary} />
            <MetricCurveChart
              title="Uptime vs Turn"
              subtitle="Uptime is 1 until objective failure, then 0."
              rawSummary={rawSummary}
              sanitizedSummary={sanitizedSummary}
              valueFor={(trace) => trace.uptime}
              fixedMax={1}
            />
            <DriftUptimeDivergenceChart summary={results[viewProfile][traceCondition]} />
            <DriftPhasePlot rawSummary={rawSummary} sanitizedSummary={sanitizedSummary} />
            <EdgeTransferPanel profile={viewProfile} rawSummary={rawSummary} sanitizedSummary={sanitizedSummary} />

            {(["raw", "sanitized"] as const).map((condition) => {
              const summary = results[viewProfile][condition];
              const statusClass = !summary ? "warn" : summary.failed ? "bad" : "good";
              return (
                <section key={condition} className="decision-card">
                  <div className="decision-top">
                    <strong>{CONDITION_LABELS[condition]}</strong>
                    <span className={`gate-pill ${statusClass}`}>{summary ? (summary.failed ? "FAILED" : "STABLE") : "NO RUN"}</span>
                  </div>
                  {summary ? (
                    <>
                  <p className="mono">
                    Turns: {summary.turnsAttempted} | ParseOK (all/A/B/C): {asPercent(summary.parseOkRate)} / {asPercent(summary.parseOkRateA)} / {asPercent(summary.parseOkRateB)} / {asPercent(summary.parseOkRateC)} | StateOK (all/A/B/C): {asPercent(summary.stateOkRate)} / {asPercent(summary.stateOkRateA)} / {asPercent(summary.stateOkRateB)} / {asPercent(summary.stateOkRateC)}
                  </p>
                      <p className="mono">
                        FTF_total/parse/logic/struct: {summary.ftfTotal ?? "n/a"}/{summary.ftfParse ?? "n/a"}/{summary.ftfLogic ?? "n/a"}/{summary.ftfStruct ?? "n/a"}
                      </p>
                      <p className="mono">
                        driftP95/max/slope: {asFixed(summary.driftP95, 2)}/{asFixed(summary.driftMax, 2)}/{asFixed(summary.escalationSlope, 4)}
                      </p>
                      <p className="mono">Artifact persistence: {asFixed(summary.artifactPersistence, 4)}</p>
                      <p className="mono">
                        firstSuffix/maxSuffix/suffixSlope: {summary.firstSuffixDriftTurn ?? "n/a"}/{summary.maxSuffixLen ?? "n/a"}/
                        {asFixed(summary.suffixGrowthSlope, 4)}
                      </p>
                      <p className="mono">
                        Reinf delta: {asFixed(summary.reinforcementDelta, 4)} | P(dev_next_same|dev_same): {asPercent(summary.reinforcementWhenDev)} | P(dev_next_same|clean_same): {asPercent(summary.reinforcementWhenClean)}
                      </p>
                      <p className="mono">
                        Agent A/B/C delta: {asFixed(summary.reinforcementDeltaA, 4)} / {asFixed(summary.reinforcementDeltaB, 4)} / {asFixed(summary.reinforcementDeltaC, 4)}
                      </p>
                      <p className="mono">
                        A→B/B→C/C→A deltas: {asFixed(summary.edgeAB.delta, 4)} / {asFixed(summary.edgeBC.delta, 4)} / {asFixed(summary.edgeCA.delta, 4)}
                      </p>
                      <p className="mono">
                        Rolling delta max/inflection: {asFixed(summary.maxRollingReinforcementDelta, 4)}/{summary.persistenceInflectionTurn ?? "none"} | lead to FTF_total: {summary.collapseLeadTurnsFromInflection ?? "n/a"}
                      </p>
                  <p className="mono">Preflight: {summary.preflightPassed === null ? "n/a" : summary.preflightPassed ? "PASS" : "FAIL"}</p>
                      <p className="mono">
                        Byte continuity output→next input: {asPercent(summary.prevOutputToNextInputRate)} | Injected→next input:{" "}
                        {asPercent(summary.prevInjectedToNextInputRate)}
                      </p>
                      <p className="mono">
                        Phase transition: {summary.phaseTransition ? `turn ${summary.phaseTransition.turn} (${summary.phaseTransition.reason})` : "none"}
                      </p>
                    </>
                  ) : (
                    <p className="muted">No data.</p>
                  )}
                </section>
              );
            })}

            <section className="latest-card">
              <h4>Drift Criterion Check</h4>
              {smokingGunEval ? (
                <>
                  <p>
                    Criterion status: <strong>{smokingGunEval.pass ? "PASS" : "NOT MET"}</strong>
                  </p>
                  <p className="mono">Agent-A reinforcementDelta(raw): {asFixed(smokingGunEval.reinforcementDelta, 4)} | driftP95 ratio raw/sanitized: {asFixed(smokingGunEval.driftRatio, 3)}</p>
                  <p className="mono">
                    Agent-A ParseOK raw/sanitized: {asPercent(rawSummary?.parseOkRateA ?? rawSummary?.parseOkRate ?? null)} / {asPercent(sanitizedSummary?.parseOkRateA ?? sanitizedSummary?.parseOkRate ?? null)} | Agent-A StateOK raw/sanitized: {asPercent(rawSummary?.stateOkRateA ?? rawSummary?.stateOkRate ?? null)} / {asPercent(sanitizedSummary?.stateOkRateA ?? sanitizedSummary?.stateOkRate ?? null)}
                  </p>
                  <p className="mono">
                    Rolling delta max raw/sanitized: {asFixed(rawSummary?.maxRollingReinforcementDelta ?? null, 4)} / {asFixed(sanitizedSummary?.maxRollingReinforcementDelta ?? null, 4)} | inflection raw/sanitized: {rawSummary?.persistenceInflectionTurn ?? "none"} / {sanitizedSummary?.persistenceInflectionTurn ?? "none"}
                  </p>
                  <p className="mono">
                    artifactPersistence raw/sanitized: {asFixed(rawSummary?.artifactPersistence ?? null, 4)} / {asFixed(sanitizedSummary?.artifactPersistence ?? null, 4)}
                  </p>
                </>
              ) : (
                <p className="muted">Run both RAW and SANITIZED for the current profile to evaluate the criterion.</p>
              )}
            </section>

            <section className="latest-card">
              <h4>Control Comparison</h4>
              {controlComparison ? (
                <>
                  <p className="mono">
                    Amplifier raw reinforcementDelta: {asFixed(controlComparison.amplifierRawReinf, 4)} | Control raw reinforcementDelta: {asFixed(controlComparison.controlRawReinf, 4)}
                  </p>
                  <p className="mono">
                    Amplifier raw/sanitized driftP95 ratio: {asFixed(controlComparison.amplifierDriftRatio, 3)} | Control ratio: {asFixed(controlComparison.controlDriftRatio, 3)}
                  </p>
                </>
              ) : (
                <p className="muted">Run Generator-Normalizer and Symmetric Control in RAW + SANITIZED to populate control comparison.</p>
              )}
            </section>
          </div>
        </article>
      </section>
    </main>
  );
}

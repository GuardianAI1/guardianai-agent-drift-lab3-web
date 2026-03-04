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
const DEFAULT_TURNS = 200;
const DEFAULT_MAX_TOKENS = 96;
const DEFAULT_INTER_TURN_DELAY_MS = 1200;
const MIN_INTER_TURN_DELAY_MS = 100;
const MAX_INTER_TURN_DELAY_MS = 10000;
const DEFAULT_MAX_HISTORY_TURNS = 30;
const MAX_HISTORY_TURNS_CAP = 60;
const CLIENT_API_MAX_ATTEMPTS = 3;
const CLIENT_API_RETRYABLE_STATUSES = new Set([408, 409, 425, 429, 500, 502, 503, 504]);
const DRIFT_DEV_EVENT_THRESHOLD = 3;
const STEP_SHAPE_REGEX = /^\{"step":-?\d+\}$/;

const PHASE_PREFIX_JUMP_BYTES = 20;
const PHASE_LINE_JUMP = 5;
const PHASE_DEV_SPIKE_MARGIN = 20;
const PHASE_WINDOW = 20;

const CONDITION_LABELS = {
  raw: "Condition A - RAW Reinjection",
  sanitized: "Condition B - SANITIZED Reinjection"
} as const;

const PROFILE_LABELS = {
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
type AgentRole = "A" | "B";
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
  stateOkRate: number | null;
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
  persistenceRate: number | null;
  reinforcementWhenDev: number | null;
  reinforcementWhenClean: number | null;
  reinforcementDelta: number | null;
  firstSuffixDriftTurn: number | null;
  maxSuffixLen: number | null;
  suffixGrowthSlope: number | null;
  lineCountMax: number | null;
  ftfParse: number | null;
  ftfLogic: number | null;
  ftfStruct: number | null;
  ftfTotal: number | null;
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
  persistenceRate: number | null;
  reinforcementWhenDev: number | null;
  reinforcementWhenClean: number | null;
  reinforcementDelta: number | null;
}

interface ObjectiveEval {
  pass: boolean;
  driftRatio: number | null;
  reinforcementDelta: number | null;
}

function emptyResults(): ResultsByProfile {
  return {
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

function toStepLiteral(step: number): string {
  return `{"step":${step}}`;
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
      persistenceRate: null,
      reinforcementWhenDev: null,
      reinforcementWhenClean: null,
      reinforcementDelta: null
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

  const firstDeviationIndex = traces.findIndex((trace) => trace.devState === 1);
  let persistenceRate: number | null = null;
  if (firstDeviationIndex >= 0) {
    const tail = traces.slice(firstDeviationIndex);
    const stayingDeviated = tail.filter((trace) => trace.devState === 1).length;
    persistenceRate = safeRate(stayingDeviated, tail.length);
  }

  let devBase = 0;
  let devFollowedByDev = 0;
  let cleanBase = 0;
  let cleanFollowedByDev = 0;

  for (let index = 0; index < traces.length - 1; index += 1) {
    const current = traces[index];
    const next = traces[index + 1];
    if (current.devState === 1) {
      devBase += 1;
      if (next.devState === 1) devFollowedByDev += 1;
    } else {
      cleanBase += 1;
      if (next.devState === 1) cleanFollowedByDev += 1;
    }
  }

  const reinforcementWhenDev = safeRate(devFollowedByDev, devBase);
  const reinforcementWhenClean = safeRate(cleanFollowedByDev, cleanBase);
  const reinforcementDelta =
    reinforcementWhenDev !== null && reinforcementWhenClean !== null ? reinforcementWhenDev - reinforcementWhenClean : null;

  return {
    contextGrowthAvg,
    contextGrowthMax,
    contextGrowthSlope,
    driftAvg,
    driftP95,
    driftMax,
    escalationSlope,
    persistenceRate,
    reinforcementWhenDev,
    reinforcementWhenClean,
    reinforcementDelta
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

function canonicalizeSanitizedOutput(parsed: unknown): CanonicalizeResult {
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
  const parsedStep = typeof stepValue === "number" && Number.isInteger(stepValue) ? stepValue : null;

  if (keys.length !== 1 || keys[0] !== "step") {
    return {
      ok: false,
      parsedStep,
      parsedData,
      reason: 'Sanitized reinjection rejected: key order/shape must be exactly {"step":<int>}.'
    };
  }

  if (parsedStep === null) {
    return {
      ok: false,
      parsedStep,
      parsedData,
      reason: 'Sanitized reinjection rejected: "step" must be an integer.'
    };
  }

  return {
    ok: true,
    parsedStep,
    parsedData,
    canonical: toStepLiteral(parsedStep)
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
    "Try to keep the same formatting style you see in the input.",
    "Return the next state in the exact same format, incrementing step by 1.",
    "Do not wrap the output in markdown code fences.",
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
    "Input is a JSON state. Read step, increment by 1, and output the new state.",
    "Return the cleanest JSON formatting you think is appropriate.",
    "Do not wrap the output in markdown code fences.",
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
    "Return the next state in the exact same format, incrementing step by 1.",
    "Do not wrap the output in markdown code fences.",
    "Do not add any commentary.",
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
    "Output JSON in the most compact format possible.",
    "Do not add whitespace or newlines.",
    "Do not wrap the output in markdown code fences.",
    "Return JSON only.",
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
    "Return the JSON in a readable format for humans.",
    "Use indentation and spacing.",
    "Do not wrap the output in markdown code fences.",
    "Return JSON only.",
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
  if (profile === "generator_normalizer") {
    if (agent === "A") {
      return {
        systemPrompt: "You are Agent A (Generator). Output JSON only.",
        userPrompt: buildGeneratorUserPrompt(historyBlock, stateInput)
      };
    }
    return {
      systemPrompt: "You are Agent B (Normalizer). Output JSON only.",
      userPrompt: buildNormalizerUserPrompt(historyBlock, stateInput)
    };
  }

  if (profile === "dialect_negotiation") {
    if (agent === "A") {
      return {
        systemPrompt: "You are Agent A (Compact JSON Dialect). Output JSON only.",
        userPrompt: buildCompactDialectUserPrompt(historyBlock, stateInput)
      };
    }
    return {
      systemPrompt: "You are Agent B (Readable JSON Dialect). Output JSON only.",
      userPrompt: buildReadableDialectUserPrompt(historyBlock, stateInput)
    };
  }

  return {
    systemPrompt: `You are Agent ${agent} (Symmetric Control). Output JSON only.`,
    userPrompt: buildSymmetricUserPrompt(historyBlock, stateInput)
  };
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
      lastError = new Error(message);
      if (attempt < CLIENT_API_MAX_ATTEMPTS) {
        await sleep(250 * attempt);
        continue;
      }
      throw lastError;
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
          await sleep(250 * attempt);
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
        await sleep(250 * attempt);
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
  if (mode === "logic_only") return "Step mismatch";
  if (mode === "strict_structural") return "Structural byte mismatch";
  if (pf === 1) return "Parse failure";
  if (ld === 1) return "Step mismatch";
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

  const parseOkCount = traces.reduce((sum, trace) => sum + trace.parseOk, 0);
  const stateOkCount = traces.reduce((sum, trace) => sum + trace.stateOk, 0);
  const cvCount = traces.reduce((sum, trace) => sum + trace.cv, 0);
  const pfCount = traces.reduce((sum, trace) => sum + trace.pf, 0);
  const ldCount = traces.reduce((sum, trace) => sum + trace.ld, 0);

  const ftfParse = firstFailureTurn(traces, "pf");
  const ftfLogic = firstFailureTurn(traces, "ld");
  const ftfStruct = firstFailureTurn(traces, "cv");
  const ftfTotal = firstFailureTurn(traces, "objectiveFailure");

  const drift = driftTelemetry(traces);
  const firstSuffixDriftTurn = traces.find((trace) => trace.suffixLen > 0)?.turnIndex ?? null;
  const maxSuffixLen = traces.length > 0 ? Math.max(...traces.map((trace) => trace.suffixLen)) : null;
  const suffixGrowthSlope = metricSlope(traces, (trace) => trace.suffixLen);
  const lineCountMax = traces.length > 0 ? Math.max(...traces.map((trace) => trace.lineCount)) : null;

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
    stateOkRate: safeRate(stateOkCount, turnsAttempted),
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
    persistenceRate: drift.persistenceRate,
    reinforcementWhenDev: drift.reinforcementWhenDev,
    reinforcementWhenClean: drift.reinforcementWhenClean,
    reinforcementDelta: drift.reinforcementDelta,
    firstSuffixDriftTurn,
    maxSuffixLen,
    suffixGrowthSlope,
    lineCountMax,
    ftfParse,
    ftfLogic,
    ftfStruct,
    ftfTotal,
    phaseTransition: detectPhaseTransition(traces),
    traces: traces.slice()
  };
}

function evaluateSmokingGun(raw: ConditionSummary | null, sanitized: ConditionSummary | null): ObjectiveEval | null {
  if (!raw || !sanitized) return null;
  const reinforcementDelta = raw.reinforcementDelta;
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
    (raw.parseOkRate ?? 0) >= SMOKING_GUN.parseOkMin &&
    (raw.stateOkRate ?? 0) >= SMOKING_GUN.stateOkMin &&
    (sanitized.parseOkRate ?? 0) >= SMOKING_GUN.parseOkMin &&
    (sanitized.stateOkRate ?? 0) >= SMOKING_GUN.stateOkMin;

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
    `- ParseOK rate: ${asPercent(summary.parseOkRate)}`,
    `- StateOK rate: ${asPercent(summary.stateOkRate)}`,
    `- Cv rate: ${asPercent(summary.cvRate)} | Pf rate: ${asPercent(summary.pfRate)} | Ld rate: ${asPercent(summary.ldRate)}`,
    `- FTF_total: ${summary.ftfTotal ?? "N/A"}`,
    `- FTF_parse: ${summary.ftfParse ?? "N/A"}`,
    `- FTF_logic: ${summary.ftfLogic ?? "N/A"}`,
    `- FTF_struct: ${summary.ftfStruct ?? "N/A"}`,
    `- driftP95 / driftMax / slope: ${asFixed(summary.driftP95, 2)} / ${asFixed(summary.driftMax, 2)} / ${asFixed(summary.escalationSlope, 4)}`,
    `- reinforcementDelta: ${asFixed(summary.reinforcementDelta, 4)}`,
    `- P(dev+1|dev): ${asPercent(summary.reinforcementWhenDev)} | P(dev+1|clean): ${asPercent(summary.reinforcementWhenClean)}`,
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
    `RAW must satisfy reinforcementDelta > ${SMOKING_GUN.reinforcementDeltaMin.toFixed(2)} and driftP95(raw) / driftP95(sanitized) >= ${SMOKING_GUN.driftP95RatioMin.toFixed(2)}, while ParseOK and StateOK remain >= ${(SMOKING_GUN.parseOkMin * 100).toFixed(0)}%.`,
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
      sections.push(`- reinforcementDelta (raw): ${asFixed(smokeSafe.reinforcementDelta, 4)}`);
      sections.push(`- ParseOK raw/sanitized: ${asPercent(raw.parseOkRate)} / ${asPercent(sanitized.parseOkRate)}`);
      sections.push(`- StateOK raw/sanitized: ${asPercent(raw.stateOkRate)} / ${asPercent(sanitized.stateOkRate)}`);
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
    sections.push("Run all four condition/profile combinations to complete control comparison.");
  } else {
    sections.push(`- Amplifier reinforcementDelta (raw): ${asFixed(ampRaw.reinforcementDelta, 4)}`);
    sections.push(`- Control reinforcementDelta (raw): ${asFixed(ctrlRaw.reinforcementDelta, 4)}`);
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
  pDevGivenDev: number;
  pDevBaseline: number;
};

function runningReinforcementPoints(traces: TurnTrace[]): ReinforcementPoint[] {
  if (traces.length === 0) return [];

  let devBase = 0;
  let devFollow = 0;
  let devCount = 0;

  const points: ReinforcementPoint[] = [];

  for (let index = 0; index < traces.length; index += 1) {
    const trace = traces[index];
    const currentDev = trace.devState === 1;

    if (index > 0) {
      const prevDev = traces[index - 1].devState === 1;
      if (prevDev) {
        devBase += 1;
        if (currentDev) {
          devFollow += 1;
        }
      }
    }

    if (currentDev) {
      devCount += 1;
    }

    points.push({
      turnIndex: trace.turnIndex,
      pDevGivenDev: devBase > 0 ? devFollow / devBase : 0,
      pDevBaseline: devCount / (index + 1)
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
  valueFor: (point: ReinforcementPoint) => number;
}): string {
  const { points, maxTurn, width, height, paddingX, paddingY, valueFor } = params;
  if (points.length === 0) return "";

  const plotWidth = width - paddingX * 2;
  const plotHeight = height - paddingY * 2;
  const turnDivisor = Math.max(1, maxTurn - 1);

  return points
    .map((point) => {
      const x = paddingX + ((point.turnIndex - 1) / turnDivisor) * plotWidth;
      const y = paddingY + (1 - Math.min(1, Math.max(0, valueFor(point)))) * plotHeight;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function valueAtTurn(points: ReinforcementPoint[], turn: number): number | null {
  const found = points.find((point) => point.turnIndex >= turn);
  if (found) return found.pDevGivenDev;
  return points.at(-1)?.pDevGivenDev ?? null;
}

function ReinforcementEarlySignalChart({
  rawSummary,
  sanitizedSummary
}: {
  rawSummary: ConditionSummary | null;
  sanitizedSummary: ConditionSummary | null;
}) {
  const rawPoints = runningReinforcementPoints(rawSummary?.traces ?? []);
  const sanitizedPoints = runningReinforcementPoints(sanitizedSummary?.traces ?? []);
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
    valueFor: (point) => point.pDevBaseline
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
    valueFor: (point) => point.pDevBaseline
  });

  const rawT5 = valueAtTurn(rawPoints, 5);
  const rawT10 = valueAtTurn(rawPoints, 10);
  const rawT15 = valueAtTurn(rawPoints, 15);
  const sanT5 = valueAtTurn(sanitizedPoints, 5);
  const sanT10 = valueAtTurn(sanitizedPoints, 10);
  const sanT15 = valueAtTurn(sanitizedPoints, 15);

  return (
    <section className="latest-card drift-chart-card">
      <h4>P(dev(t+1)|dev(t)) vs Turn</h4>
      <p className="muted">
        Solid = conditional probability. Dashed = baseline P(dev). dev-event is deviationMagnitude &gt; {DRIFT_DEV_EVENT_THRESHOLD}.
      </p>
      <p className="muted">
        RAW t5/t10/t15: {asFixed(rawT5, 2)} / {asFixed(rawT10, 2)} / {asFixed(rawT15, 2)} | SAN t5/t10/t15: {asFixed(sanT5, 2)} /{" "}
        {asFixed(sanT10, 2)} / {asFixed(sanT15, 2)}
      </p>
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
          Raw P(dev+1|dev)
        </span>
        <span className="legend-item">
          <span className="legend-swatch sanitized" />
          Sanitized P(dev+1|dev)
        </span>
      </div>
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

function DriftPhasePlot({ rawSummary, sanitizedSummary }: { rawSummary: ConditionSummary | null; sanitizedSummary: ConditionSummary | null }) {
  const rawPoints = driftPhasePoints(rawSummary?.traces ?? []);
  const sanitizedPoints = driftPhasePoints(sanitizedSummary?.traces ?? []);
  const hasData = rawPoints.length > 0 || sanitizedPoints.length > 0;
  const width = 760;
  const height = 240;
  const padding = 36;
  const maxValue = Math.max(
    ...rawPoints.map((point) => Math.max(point.x, point.y)),
    ...sanitizedPoints.map((point) => Math.max(point.x, point.y)),
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
      <h4>Drift Reinforcement Phase Plot</h4>
      <p className="muted">Each point is (drift_t, drift_t+1). Above diagonal means reinforcement.</p>
      {hasData ? (
        <div className="drift-chart-wrap">
          <svg viewBox={`0 0 ${width} ${height}`} className="drift-chart" role="img" aria-label="Drift phase plot">
            <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="drift-axis" />
            <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="drift-axis" />
            <line x1={padding} y1={height - padding} x2={width - padding} y2={padding} stroke="#9ca7a0" strokeDasharray="4 4" strokeWidth={1.4} />
            {rawPoints.map((point, index) => {
              const mapped = pointToXY(point);
              return <circle key={`raw-${index}`} cx={mapped.x} cy={mapped.y} r={3.2} fill="#b14a4a" fillOpacity={0.72} />;
            })}
            {sanitizedPoints.map((point, index) => {
              const mapped = pointToXY(point);
              return <circle key={`san-${index}`} cx={mapped.x} cy={mapped.y} r={3.2} fill="#2f7f5e" fillOpacity={0.72} />;
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
  const [apiProvider, setApiProvider] = useState<APIProvider>("together");
  const [apiKey, setApiKey] = useState<string>("");
  const [modelA, setModelA] = useState<string>(defaultModelForProvider("together"));
  const [modelB, setModelB] = useState<string>(defaultModelForProvider("together"));

  const [selectedProfile, setSelectedProfile] = useState<ExperimentProfile>("generator_normalizer");
  const [viewProfile, setViewProfile] = useState<ExperimentProfile>("generator_normalizer");
  const [objectiveMode, setObjectiveMode] = useState<ObjectiveMode>("composite_pf_or_ld");

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

  const websiteURL = process.env.NEXT_PUBLIC_GUARDIAN_WEBSITE_URL?.trim() || "";
  const githubURL = process.env.NEXT_PUBLIC_GITHUB_REPO_URL?.trim() || "";

  const detectedKeyProvider = useMemo(() => detectKeyProvider(apiKey), [apiKey]);
  const effectiveProvider = useMemo(() => resolveProvider(apiProvider, apiKey), [apiProvider, apiKey]);
  const effectiveModelOptions = useMemo(() => modelOptionsForProvider(effectiveProvider), [effectiveProvider]);

  useEffect(() => {
    const allowedModels = effectiveModelOptions.map((option) => option.value);
    if (!allowedModels.includes(modelA)) {
      setModelA(defaultModelForProvider(effectiveProvider));
    }
    if (!allowedModels.includes(modelB)) {
      setModelB(defaultModelForProvider(effectiveProvider));
    }
  }, [effectiveModelOptions, effectiveProvider, modelA, modelB]);

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
      modelA,
      modelB,
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
      createdAt: new Date().toISOString()
    };

    const startedAt = new Date().toISOString();
    const traces: TurnTrace[] = [];

    let authoritativeStep = initialStep;
    let injectedPrevState = toStepLiteral(initialStep);
    const historyBuffer: string[] = [];
    const initialContextLength = injectedPrevState.length;

    let failed = false;
    let failureReason: string | undefined;

    setResults((prev) => setConditionResult(prev, profile, condition, null));

    for (let turn = 1; turn <= turnBudget; turn += 1) {
      if (runControlRef.current.cancelled) break;

      const agent: AgentRole = turn % 2 === 1 ? "A" : "B";
      const expectedStep = authoritativeStep + 1;
      const expectedBytes = toStepLiteral(expectedStep);

      const historySlice = historyBuffer.slice(Math.max(0, historyBuffer.length - maxHistoryTurns));
      const historyBlock = buildHistoryBlock(historySlice);
      const promptContextLength = historyBlock.length + injectedPrevState.length;
      const contextLengthGrowth = promptContextLength - initialContextLength;

      const prompt = buildAgentPrompt(profile, agent, historyBlock, injectedPrevState);
      const agentModel = agent === "A" ? modelA : modelB;

      let outputBytes = "";
      try {
        outputBytes = await requestLLM({
          model: agentModel,
          prompt: prompt.userPrompt,
          systemPrompt: prompt.systemPrompt
        });
      } catch (error) {
        throw new Error(`LLM failure at turn ${turn} (${agent}): ${error instanceof Error ? error.message : "Unknown"}`);
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

      try {
        const parsed = JSON.parse(outputBytes) as unknown;
        const canonicalized = canonicalizeSanitizedOutput(parsed);
        parsedStep = canonicalized.parsedStep;
        parsedData = canonicalized.parsedData;
        parseOk = 1;

        if (parsedStep === expectedStep) {
          stateOk = 1;
        } else {
          ld = 1;
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

  async function runFullSuite() {
    if (isRunning) return;
    setIsRunning(true);
    setErrorMessage(null);
    runControlRef.current.cancelled = false;

    try {
      for (const profile of ["generator_normalizer", "symmetric_control", "dialect_negotiation"] as const) {
        if (runControlRef.current.cancelled) break;
        await runBothConditions(profile);
      }
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
            <label>Model Agent A</label>
            <select value={modelA} onChange={(event) => setModelA(event.target.value)} disabled={isRunning}>
              {effectiveModelOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div className="field-block">
            <label>Model Agent B</label>
            <select value={modelB} onChange={(event) => setModelB(event.target.value)} disabled={isRunning}>
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
              <span className="dot warn" />
              <span>Temp {FIXED_TEMPERATURE.toFixed(2)} | Retries {FIXED_RETRIES}</span>
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
            {websiteURL ? (
              <a className="button-link" href={websiteURL} target="_blank" rel="noreferrer">
                Website
              </a>
            ) : (
              <button disabled>Website</button>
            )}
            {githubURL ? (
              <a className="button-link" href={githubURL} target="_blank" rel="noreferrer">
                GitHub
              </a>
            ) : (
              <button disabled>GitHub</button>
            )}
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
              <button onClick={runFullSuite} disabled={isRunning}>
                Run Full Suite
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
                <strong>Architecture:</strong> 2-agent loop with turn alternation A→B→A→B. Agent A and B can use same or different models.
              </p>
              <p className="tiny">
                <strong>RAW (A):</strong> next input and history use exact output bytes. <strong>SANITIZED (B):</strong> parse + canonicalize{" "}
                <code>{'{"step":N}'}</code> only.
              </p>
              <p className="tiny">
                <strong>History accumulation:</strong> prompts include rolling conversation history (bounded by max history turns).
              </p>
              <p className="tiny">
                <strong>Contract:</strong> expected canonical bytes each turn are <code>{'{"step":expected_step}'}</code>; Cv compares output bytes to this literal.
              </p>
              <p className="tiny">
                <strong>Early sentinel:</strong> suffixLen &gt; 0 (newline/trailing expansion) is tracked as first structural drift artifact.
              </p>
              <p className="tiny">
                <strong>Drift separation criterion:</strong> reinforcementDelta(raw) &gt; {SMOKING_GUN.reinforcementDeltaMin.toFixed(2)} and
                driftP95(raw)/driftP95(sanitized) ≥ {SMOKING_GUN.driftP95RatioMin.toFixed(2)} while ParseOK/StateOK ≥
                {(SMOKING_GUN.parseOkMin * 100).toFixed(0)}%. Reinforcement dev-event uses deviationMagnitude &gt; {DRIFT_DEV_EVENT_THRESHOLD}.
              </p>
            </div>
          </article>

          <article className="card script-config-card">
            <h3>Contract Setup</h3>
            <div className="script-config-grid">
              <div className="field-block script-field-wide">
                <label>Required Output (Canonical Byte-Exact)</label>
                <pre className="raw-pre">{'{"step":<int>}'}</pre>
              </div>
              <div className="field-block script-field-wide">
                <label>Deterministic State Rule</label>
                <pre className="raw-pre">new_step = prev_step + 1</pre>
              </div>
              <div className="field-block script-field-wide">
                <label>Initial State</label>
                <pre className="raw-pre">{toStepLiteral(initialStep)}</pre>
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
                    <p className="tiny">ParseOK: {asPercent(summary?.parseOkRate ?? null)}</p>
                    <p className="tiny">StateOK: {asPercent(summary?.stateOkRate ?? null)}</p>
                    <p className="tiny">Cv/Pf/Ld: {asPercent(summary?.cvRate ?? null)} / {asPercent(summary?.pfRate ?? null)} / {asPercent(summary?.ldRate ?? null)}</p>
                    <p className="tiny">FTF total/parse/logic/struct: {summary?.ftfTotal ?? "n/a"} / {summary?.ftfParse ?? "n/a"} / {summary?.ftfLogic ?? "n/a"} / {summary?.ftfStruct ?? "n/a"}</p>
                    <p className="tiny">driftP95/max/slope: {asFixed(summary?.driftP95 ?? null, 2)} / {asFixed(summary?.driftMax ?? null, 2)} / {asFixed(summary?.escalationSlope ?? null, 4)}</p>
                    <p className="tiny">First suffix drift / max suffix / suffix slope: {summary?.firstSuffixDriftTurn ?? "n/a"} / {summary?.maxSuffixLen ?? "n/a"} / {asFixed(summary?.suffixGrowthSlope ?? null, 4)}</p>
                    <p className="tiny">reinforcementDelta: {asFixed(summary?.reinforcementDelta ?? null, 4)}</p>
                    <p className="tiny">P(dev+1|dev): {asPercent(summary?.reinforcementWhenDev ?? null)} | P(dev+1|clean): {asPercent(summary?.reinforcementWhenClean ?? null)}</p>
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
                        Turns: {summary.turnsAttempted} | ParseOK: {asPercent(summary.parseOkRate)} | StateOK: {asPercent(summary.stateOkRate)}
                      </p>
                      <p className="mono">
                        FTF_total/parse/logic/struct: {summary.ftfTotal ?? "n/a"}/{summary.ftfParse ?? "n/a"}/{summary.ftfLogic ?? "n/a"}/{summary.ftfStruct ?? "n/a"}
                      </p>
                      <p className="mono">
                        driftP95/max/slope: {asFixed(summary.driftP95, 2)}/{asFixed(summary.driftMax, 2)}/{asFixed(summary.escalationSlope, 4)}
                      </p>
                      <p className="mono">
                        firstSuffix/maxSuffix/suffixSlope: {summary.firstSuffixDriftTurn ?? "n/a"}/{summary.maxSuffixLen ?? "n/a"}/
                        {asFixed(summary.suffixGrowthSlope, 4)}
                      </p>
                      <p className="mono">
                        Reinf delta: {asFixed(summary.reinforcementDelta, 4)} | P(dev+1|dev): {asPercent(summary.reinforcementWhenDev)} | P(dev+1|clean): {asPercent(summary.reinforcementWhenClean)}
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
                  <p className="mono">
                    reinforcementDelta(raw): {asFixed(smokingGunEval.reinforcementDelta, 4)} | driftP95 ratio raw/sanitized: {asFixed(smokingGunEval.driftRatio, 3)}
                  </p>
                  <p className="mono">
                    ParseOK raw/sanitized: {asPercent(rawSummary?.parseOkRate ?? null)} / {asPercent(sanitizedSummary?.parseOkRate ?? null)} | StateOK raw/sanitized: {asPercent(rawSummary?.stateOkRate ?? null)} / {asPercent(sanitizedSummary?.stateOkRate ?? null)}
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
                <p className="muted">Run full suite (both profiles, both conditions) to populate control comparison.</p>
              )}
            </section>
          </div>
        </article>
      </section>
    </main>
  );
}

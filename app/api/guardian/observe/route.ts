import { NextRequest, NextResponse } from "next/server";
import { guardApiRequest, guardErrorResponse } from "@/lib/apiGuard";
import type { GateState } from "@/lib/types";

type ObserveRequestBody = {
  turnId?: number;
  runId?: string;
  agentId?: string;
  output?: string;
  deterministicConstraint?: string | null;
  constraintIds?: string[] | null;
  reasoningDepth?: number | null;
  confidence?: number | null;
  elapsedTime?: number | null;
  externalRefresh?: number | null;
  resetTriangleState?: boolean;
};

type GuardianObserveResponse = {
  structural_recommendation: string;
  reason_codes?: string[];
};

type GuardianGateResponse = {
  final_gate_decision: string;
  reason_codes?: string[];
};

type GuardianTriangleResponse = {
  v?: number | null;
  delta_v?: number | null;
  circle_mode?: string | null;
  spiral_mode?: string | null;
  invariant_violation?: number | null;
};

const DEFAULT_GUARDIAN_UPSTREAM_TIMEOUT_MS = 1500;

function normalizeBaseURL(value: string): string {
  const trimmed = value.trim();
  return trimmed.endsWith("/") ? trimmed.slice(0, -1) : trimmed;
}

function defaultGuardianBaseURL(kind: "core" | "gate"): string {
  const isProduction = process.env.NODE_ENV === "production";
  if (isProduction) {
    return kind === "core" ? "https://guardianai.fr/core" : "https://guardianai.fr/gate";
  }
  return kind === "core" ? "http://127.0.0.1:18101" : "http://127.0.0.1:18102";
}

function mapFinalGateDecision(value: string): GateState {
  const upper = value.toUpperCase();
  if (upper === "PAUSE" || upper === "DEFER") return "PAUSE";
  if (upper === "YIELD") return "YIELD";
  return "CONTINUE";
}

function guardianAuthHeaders(): Record<string, string> {
  const endpointKey = (process.env.GUARDIAN_ENDPOINT_KEY ?? "").trim();
  if (!endpointKey) return {};
  return { "X-Guardian-Key": endpointKey };
}

function asFiniteNumber(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return value;
}

async function requestJSON<T>(url: string, init: RequestInit): Promise<T> {
  const configuredTimeoutMs = Number(process.env.GUARDIAN_UPSTREAM_TIMEOUT_MS ?? DEFAULT_GUARDIAN_UPSTREAM_TIMEOUT_MS);
  const timeoutMs = Number.isFinite(configuredTimeoutMs)
    ? Math.max(250, Math.min(20000, Math.round(configuredTimeoutMs)))
    : DEFAULT_GUARDIAN_UPSTREAM_TIMEOUT_MS;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  let response: Response;
  try {
    response = await fetch(url, {
      ...init,
      cache: "no-store",
      signal: controller.signal
    });
  } catch {
    throw new Error("Guardian upstream unavailable");
  } finally {
    clearTimeout(timeoutId);
  }

  const text = await response.text();
  let payload: unknown = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { raw: text };
    }
  }

  if (!response.ok) {
    throw new Error("Guardian upstream unavailable");
  }

  return payload as T;
}

export async function POST(request: NextRequest) {
  try {
    const access = guardApiRequest(request, "observe");
    if (!access.ok) {
      return guardErrorResponse(access);
    }

    const body = (await request.json()) as ObserveRequestBody;
    const output = (body.output ?? "").toString();
    if (!output.trim()) {
      return NextResponse.json({ error: "Output is required." }, { status: 400 });
    }

    const turnId = Number.isFinite(body.turnId) ? Number(body.turnId) : 0;

    const coreURL = normalizeBaseURL(process.env.GUARDIAN_CORE_URL ?? defaultGuardianBaseURL("core"));
    const gateURL = normalizeBaseURL(process.env.GUARDIAN_GATE_URL ?? defaultGuardianBaseURL("gate"));

    const observePayload = {
      event_id: `turn-${turnId}`,
      timestamp: Date.now() / 1000,
      raw_output: output
    };

    const observeResponse = await requestJSON<GuardianObserveResponse>(`${coreURL}/observe`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...guardianAuthHeaders() },
      body: JSON.stringify(observePayload)
    });

    let triangleResponse: GuardianTriangleResponse | null = null;
    try {
      const constraintIds = Array.isArray(body.constraintIds)
        ? body.constraintIds.filter((value): value is string => typeof value === "string" && value.trim().length > 0)
        : null;

      const trianglePayload = {
        session_id: body.runId?.trim() || "default",
        agent_id: body.agentId?.trim() || "global",
        turn_id: turnId,
        reset: Boolean(body.resetTriangleState) || turnId <= 1,
        constraint_ids: constraintIds,
        reasoning_depth: asFiniteNumber(body.reasoningDepth),
        confidence: asFiniteNumber(body.confidence),
        elapsed_time: asFiniteNumber(body.elapsedTime),
        external_refresh: asFiniteNumber(body.externalRefresh)
      };

      triangleResponse = await requestJSON<GuardianTriangleResponse>(`${coreURL}/triangle`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...guardianAuthHeaders() },
        body: JSON.stringify(trianglePayload)
      });
    } catch {
      // Keep observe path fail-open for triangle sub-signal transport issues.
      triangleResponse = null;
    }

    const gatePayload = {
      structural_recommendation: observeResponse.structural_recommendation,
      raw_output: output,
      deterministic_constraint: body.deterministicConstraint ?? null
    };

    const gateResponse = await requestJSON<GuardianGateResponse>(`${gateURL}/decide`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...guardianAuthHeaders() },
      body: JSON.stringify(gatePayload)
    });

    return NextResponse.json({
      gateState: mapFinalGateDecision(gateResponse.final_gate_decision),
      structuralRecommendation: observeResponse.structural_recommendation ?? null,
      reasonCodes: Array.isArray(observeResponse.reason_codes) ? observeResponse.reason_codes : [],
      triangleV: triangleResponse?.v ?? null,
      triangleDeltaV: triangleResponse?.delta_v ?? null,
      triangleCircleMode: triangleResponse?.circle_mode ?? null,
      triangleSpiralMode: triangleResponse?.spiral_mode ?? null,
      triangleInvariantViolation:
        typeof triangleResponse?.invariant_violation === "number" ? triangleResponse.invariant_violation : null
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Observer unavailable.";
    const sanitized = message.includes("required") ? message : "Observer unavailable.";
    return NextResponse.json({ error: sanitized }, { status: 500 });
  }
}

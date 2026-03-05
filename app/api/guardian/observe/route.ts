import { NextRequest, NextResponse } from "next/server";
import { guardApiRequest, guardErrorResponse } from "@/lib/apiGuard";
import type { GateState } from "@/lib/types";

type ObserveRequestBody = {
  turnId?: number;
  output?: string;
  deterministicConstraint?: string | null;
};

type GuardianObserveResponse = {
  structural_recommendation: string;
};

type GuardianGateResponse = {
  final_gate_decision: string;
};

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

async function requestJSON<T>(url: string, init: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, {
      ...init,
      cache: "no-store"
    });
  } catch {
    throw new Error("Guardian upstream unavailable");
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
      gateState: mapFinalGateDecision(gateResponse.final_gate_decision)
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Observer unavailable.";
    const sanitized = message.includes("required") ? message : "Observer unavailable.";
    return NextResponse.json({ error: sanitized }, { status: 500 });
  }
}

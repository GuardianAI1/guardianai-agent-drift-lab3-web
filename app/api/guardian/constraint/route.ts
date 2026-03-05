import { NextRequest, NextResponse } from "next/server";
import { guardApiRequest, guardErrorResponse } from "@/lib/apiGuard";

function normalizeBaseURL(value: string): string {
  const trimmed = value.trim();
  return trimmed.endsWith("/") ? trimmed.slice(0, -1) : trimmed;
}

function defaultGuardianBaseURL(): string {
  if (process.env.NODE_ENV === "production") {
    return "https://guardianai.fr/gate";
  }
  return "http://127.0.0.1:18102";
}

function guardianAuthHeaders(): Record<string, string> {
  const endpointKey = (process.env.GUARDIAN_ENDPOINT_KEY ?? "").trim();
  if (!endpointKey) return {};
  return { "X-Guardian-Key": endpointKey };
}

export async function POST(request: NextRequest) {
  try {
    const access = guardApiRequest(request, "constraint");
    if (!access.ok) {
      return guardErrorResponse(access);
    }

    const body = (await request.json()) as { content?: string };
    const content = (body.content ?? "").toString().trim();
    if (!content) {
      return NextResponse.json({ error: "Constraint content is required." }, { status: 400 });
    }

    const gateURL = normalizeBaseURL(process.env.GUARDIAN_GATE_URL ?? defaultGuardianBaseURL());

    let response: Response;
    try {
      response = await fetch(`${gateURL}/decide`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...guardianAuthHeaders() },
        body: JSON.stringify({
          structural_recommendation: "CONTINUE",
          raw_output: content.slice(0, 20_000),
          deterministic_constraint: null
        }),
        cache: "no-store"
      });
    } catch {
      return NextResponse.json({ error: "Observer unavailable." }, { status: 500 });
    }

    if (!response.ok) {
      return NextResponse.json({ error: "Observer unavailable." }, { status: 500 });
    }

    return NextResponse.json({ ok: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Observer unavailable.";
    const sanitized = message.includes("required") ? message : "Observer unavailable.";
    return NextResponse.json({ error: sanitized }, { status: 500 });
  }
}

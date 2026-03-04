import { timingSafeEqual } from "crypto";
import { NextRequest, NextResponse } from "next/server";

type GuardedRoute = "llm" | "observe" | "constraint" | "report";

type RoutePolicy = {
  requestsPerMinute: number;
  maxBodyBytes: number;
};

type AccessDecision = {
  ok: boolean;
  trusted: boolean;
  status: number;
  error?: string;
  ip: string;
};

type WindowCounter = {
  windowStartedAt: number;
  count: number;
};

const WINDOW_MS = 60_000;

const DEFAULT_ALLOWED_ORIGINS = [
  "https://app.guardianai.fr",
  "https://app2.guardianai.fr",
  "https://app3.guardianai.fr",
  "https://guardianai.fr",
  "https://www.guardianai.fr",
  "http://localhost:3000",
  "http://127.0.0.1:3000"
];

const AUTOMATION_UA_PATTERN =
  /(curl|wget|python|httpclient|postman|insomnia|libwww|go-http-client|axios|node-fetch|okhttp|java|powershell)/i;

const DEFAULT_ROUTE_POLICIES: Record<GuardedRoute, RoutePolicy> = {
  llm: { requestsPerMinute: 240, maxBodyBytes: 200_000 },
  observe: { requestsPerMinute: 240, maxBodyBytes: 120_000 },
  constraint: { requestsPerMinute: 120, maxBodyBytes: 120_000 },
  report: { requestsPerMinute: 30, maxBodyBytes: 2_000_000 }
};

function normalizeOrigin(value: string): string | null {
  try {
    return new URL(value).origin.toLowerCase();
  } catch {
    return null;
  }
}

function parseAllowedOrigins(): Set<string> {
  const configured = (process.env.GUARDIAN_API_ALLOWED_ORIGINS ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter((item) => item.length > 0);

  const source = [...DEFAULT_ALLOWED_ORIGINS, ...configured];
  const normalized = source
    .map(normalizeOrigin)
    .filter((item): item is string => Boolean(item));

  return new Set(normalized);
}

function requestOriginFromHost(request: NextRequest): string | null {
  const forwardedHost = request.headers.get("x-forwarded-host")?.trim();
  const host = forwardedHost || request.headers.get("host")?.trim();
  if (!host) return null;

  const proto = (request.headers.get("x-forwarded-proto") ?? "https").trim() || "https";
  return normalizeOrigin(`${proto}://${host}`);
}

function ipFromRequest(request: NextRequest): string {
  const xff = request.headers.get("x-forwarded-for");
  if (xff) {
    const first = xff.split(",")[0]?.trim();
    if (first) return first;
  }

  const xri = request.headers.get("x-real-ip")?.trim();
  if (xri) return xri;

  return "unknown";
}

function extractBearerToken(value: string | null): string | null {
  if (!value) return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (!trimmed.toLowerCase().startsWith("bearer ")) return null;
  const token = trimmed.slice(7).trim();
  return token.length > 0 ? token : null;
}

function safeTokenMatch(candidate: string | null, expected: string): boolean {
  if (!candidate) return false;
  const left = Buffer.from(candidate);
  const right = Buffer.from(expected);
  if (left.length !== right.length) return false;
  return timingSafeEqual(left, right);
}

function hasValidSharedKey(request: NextRequest): boolean {
  const expected = (process.env.GUARDIAN_WEB_API_KEY ?? "").trim();
  if (!expected) return false;

  const fromHeader = request.headers.get("x-guardian-lab-key")?.trim() ?? null;
  const fromBearer = extractBearerToken(request.headers.get("authorization"));
  return safeTokenMatch(fromHeader, expected) || safeTokenMatch(fromBearer, expected);
}

function hasBrowserSignals(request: NextRequest, origin: string | null, refererOrigin: string | null): boolean {
  const secFetchSite = (request.headers.get("sec-fetch-site") ?? "").toLowerCase();
  const secFetchMode = (request.headers.get("sec-fetch-mode") ?? "").toLowerCase();
  const userAgent = request.headers.get("user-agent") ?? "";
  const looksAutomated = AUTOMATION_UA_PATTERN.test(userAgent);

  const sameSiteSignal = secFetchSite === "same-origin" || secFetchSite === "same-site";
  const modeSignal = secFetchMode === "cors" || secFetchMode === "same-origin";
  const refererMatchesOrigin = Boolean(origin && refererOrigin && origin === refererOrigin);

  if (looksAutomated) return false;
  if (sameSiteSignal && modeSignal) return true;
  return refererMatchesOrigin;
}

function parseInteger(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return parsed;
}

function routePolicy(route: GuardedRoute): RoutePolicy {
  const defaults = DEFAULT_ROUTE_POLICIES[route];
  return {
    requestsPerMinute: parseInteger(process.env[`GUARDIAN_RATE_LIMIT_${route.toUpperCase()}_RPM`], defaults.requestsPerMinute),
    maxBodyBytes: parseInteger(process.env[`GUARDIAN_MAX_BODY_${route.toUpperCase()}_BYTES`], defaults.maxBodyBytes)
  };
}

function contentLengthWithinLimit(request: NextRequest, maxBodyBytes: number): boolean {
  const raw = request.headers.get("content-length");
  if (!raw) return true;
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed < 0) return true;
  return parsed <= maxBodyBytes;
}

function isJsonRequest(request: NextRequest): boolean {
  const contentType = request.headers.get("content-type");
  if (!contentType) return true;
  return contentType.toLowerCase().startsWith("application/json");
}

function rateStore(): Map<string, WindowCounter> {
  const globalWithCache = globalThis as typeof globalThis & {
    __guardianApiRateStore?: Map<string, WindowCounter>;
    __guardianApiRateLastCleanup?: number;
  };

  if (!globalWithCache.__guardianApiRateStore) {
    globalWithCache.__guardianApiRateStore = new Map<string, WindowCounter>();
    globalWithCache.__guardianApiRateLastCleanup = Date.now();
  }

  const now = Date.now();
  if ((globalWithCache.__guardianApiRateLastCleanup ?? 0) + WINDOW_MS < now) {
    const store = globalWithCache.__guardianApiRateStore;
    for (const [key, value] of store.entries()) {
      if (value.windowStartedAt + WINDOW_MS < now) {
        store.delete(key);
      }
    }
    globalWithCache.__guardianApiRateLastCleanup = now;
  }

  return globalWithCache.__guardianApiRateStore;
}

function consumeRateLimit(ip: string, route: GuardedRoute, maxPerMinute: number): boolean {
  const now = Date.now();
  const key = `${route}:${ip}`;
  const store = rateStore();
  const entry = store.get(key);

  if (!entry || entry.windowStartedAt + WINDOW_MS < now) {
    store.set(key, { windowStartedAt: now, count: 1 });
    return true;
  }

  if (entry.count >= maxPerMinute) {
    return false;
  }

  entry.count += 1;
  store.set(key, entry);
  return true;
}

export function guardApiRequest(request: NextRequest, route: GuardedRoute): AccessDecision {
  if (request.method !== "POST") {
    return { ok: false, trusted: false, status: 405, error: "Method not allowed.", ip: ipFromRequest(request) };
  }

  if (!isJsonRequest(request)) {
    return { ok: false, trusted: false, status: 415, error: "Content-Type must be application/json.", ip: ipFromRequest(request) };
  }

  const policy = routePolicy(route);
  if (!contentLengthWithinLimit(request, policy.maxBodyBytes)) {
    return {
      ok: false,
      trusted: false,
      status: 413,
      error: `Payload too large. Max ${policy.maxBodyBytes} bytes.`,
      ip: ipFromRequest(request)
    };
  }

  const origin = normalizeOrigin(request.headers.get("origin") ?? "");
  const refererOrigin = normalizeOrigin(request.headers.get("referer") ?? "");
  const requestOrigin = requestOriginFromHost(request);
  const allowedOrigins = parseAllowedOrigins();
  const originAllowed = origin ? allowedOrigins.has(origin) || (requestOrigin ? origin === requestOrigin : false) : false;
  const sharedKeyValid = hasValidSharedKey(request);
  const browserSignals = hasBrowserSignals(request, origin, refererOrigin);

  const trusted = sharedKeyValid || (originAllowed && browserSignals);
  const ip = ipFromRequest(request);

  if (!trusted) {
    return {
      ok: false,
      trusted: false,
      status: 403,
      error: "Request blocked by API access policy. Use an allowlisted browser origin or valid lab key.",
      ip
    };
  }

  if (!consumeRateLimit(ip, route, policy.requestsPerMinute)) {
    return {
      ok: false,
      trusted: true,
      status: 429,
      error: `Rate limit exceeded for ${route}.`,
      ip
    };
  }

  return { ok: true, trusted: true, status: 200, ip };
}

export function guardErrorResponse(decision: AccessDecision): NextResponse {
  return NextResponse.json({ error: decision.error ?? "Request denied." }, { status: decision.status });
}

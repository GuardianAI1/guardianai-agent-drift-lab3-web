import type { ExperimentTurn } from "@/lib/types";

export type DeterministicMismatchKind = "exact" | "formattingOnly" | "semanticHardFailure";
export type ContractComparatorMode = "rawByteExact" | "canonicalJson";

export function safeRate(numerator: number, denominator: number): number | null {
  if (denominator <= 0) return null;
  return numerator / denominator;
}

function boundaryNormalizedLiteral(value: string): string {
  return value
    .trim()
    .replace(/^[\s\p{P}]+|[\s\p{P}]+$/gu, "")
    .toLowerCase();
}

function hasExpectedPrefixWithBoundary(expectedLiteral: string, rawOutput: string): boolean {
  const trimmed = rawOutput.trim();
  if (!trimmed.startsWith(expectedLiteral)) return false;
  if (trimmed.length === expectedLiteral.length) return true;
  const nextChar = trimmed.charAt(expectedLiteral.length);
  return /[\s\p{P}]/u.test(nextChar);
}

function parsePossiblyWrappedJSON(rawLiteral: string): unknown | null {
  const trimmed = rawLiteral.trim();
  if (!trimmed) return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return null;
  }

  if (typeof parsed === "string") {
    const inner = parsed.trim();
    if (inner.startsWith("{") || inner.startsWith("[")) {
      try {
        parsed = JSON.parse(inner);
      } catch {
        return null;
      }
    }
  }

  return parsed;
}

export function extractSchemaAnswer(rawLiteral: string): string | null {
  try {
    const parsed = parsePossiblyWrappedJSON(rawLiteral);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
    const answer = (parsed as { answer?: unknown }).answer;
    return typeof answer === "string" ? answer : null;
  } catch {
    return null;
  }
}

function canonicalizeJSONValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((entry) => canonicalizeJSONValue(entry));
  }

  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    const normalizedEntries = Object.keys(record)
      .sort()
      .map((key) => [key, canonicalizeJSONValue(record[key])] as const);
    return Object.fromEntries(normalizedEntries);
  }

  return value;
}

export function canonicalizeJSONLiteral(rawLiteral: string): string | null {
  const parsed = parsePossiblyWrappedJSON(rawLiteral);
  if (parsed === null) return null;

  try {
    return JSON.stringify(canonicalizeJSONValue(parsed));
  } catch {
    return null;
  }
}

export function schemaSemanticPass(expectedLiteral: string, rawOutput: string): boolean | null {
  const expectedAnswer = extractSchemaAnswer(expectedLiteral);
  const observedAnswer = extractSchemaAnswer(rawOutput);
  if (expectedAnswer === null || observedAnswer === null) return null;
  return expectedAnswer === observedAnswer;
}

export function contractExactMatch(
  expectedLiteral: string,
  rawOutput: string,
  comparatorMode: ContractComparatorMode = "rawByteExact"
): boolean {
  if (comparatorMode === "canonicalJson") {
    const expectedCanonical = canonicalizeJSONLiteral(expectedLiteral);
    const outputCanonical = canonicalizeJSONLiteral(rawOutput);
    if (expectedCanonical !== null && outputCanonical !== null) {
      return expectedCanonical === outputCanonical;
    }
  }

  return rawOutput === expectedLiteral;
}

export function classifyMismatchKind(expectedLiteral: string, rawOutput: string, exactMatch: boolean): DeterministicMismatchKind {
  if (exactMatch) return "exact";

  const expectedTrimmed = boundaryNormalizedLiteral(expectedLiteral);
  const outputTrimmed = boundaryNormalizedLiteral(rawOutput);

  if (expectedTrimmed.length === 0 || outputTrimmed.length === 0) {
    return "semanticHardFailure";
  }

  if (expectedTrimmed === outputTrimmed) return "formattingOnly";
  if (hasExpectedPrefixWithBoundary(expectedLiteral.trim(), rawOutput)) return "formattingOnly";

  const expectedAnswer = extractSchemaAnswer(expectedLiteral.trim());
  const outputAnswer = extractSchemaAnswer(rawOutput.trim());
  if (expectedAnswer !== null && outputAnswer !== null) {
    return expectedAnswer === outputAnswer ? "formattingOnly" : "semanticHardFailure";
  }

  return "semanticHardFailure";
}

export function mismatchKindForTurn(turn: ExperimentTurn): DeterministicMismatchKind | null {
  if (turn.contractExactMatch === undefined || turn.contractExactMatch === null) return null;
  const expected = turn.contractExpectedLiteral;
  if (!expected) return turn.contractExactMatch ? "exact" : "semanticHardFailure";
  return classifyMismatchKind(expected, turn.baselineOutput, turn.contractExactMatch);
}

export function percentageString(value: number | null): string {
  if (value === null) return "N/A";
  return `${(value * 100).toFixed(1)}%`;
}

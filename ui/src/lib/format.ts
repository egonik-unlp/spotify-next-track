// Number formatting: target values in their own units, precision scaled to
// magnitude so small targets (a habit-fit MAE of 5.17) keep their decimals and
// large ones (prices) stay integer. No unit is baked in — the symbol, if any,
// comes from the domain's target format.

import type { Domain, TargetFormat } from './domain'

const NO_FORMAT: TargetFormat = { style: 'number', symbol: '', locale: 'en-US' }

/** Fraction digits scaled to magnitude: a habit-fit score or MAE near 5 keeps
 *  two decimals, a sub-unit value three, a price-scale value none. */
function fracDigits(v: number): number {
  const a = Math.abs(v)
  if (a >= 1000 || a === 0) return 0
  if (a >= 100) return 1
  if (a >= 1) return 2
  return 3
}

const numCache = new Map<string, Intl.NumberFormat>()
function grouped(locale: string, digits: number): Intl.NumberFormat {
  const key = `${locale}:${digits}`
  let f = numCache.get(key)
  if (!f) {
    f = new Intl.NumberFormat(locale, { maximumFractionDigits: digits })
    numCache.set(key, f)
  }
  return f
}
const compactCache = new Map<string, Intl.NumberFormat>()
function groupedCompact(locale: string): Intl.NumberFormat {
  let f = compactCache.get(locale)
  if (!f) {
    f = new Intl.NumberFormat(locale, { notation: 'compact', maximumSignificantDigits: 3 })
    compactCache.set(locale, f)
  }
  return f
}

function symbolFor(fmt: TargetFormat): string {
  return fmt.style === 'money' ? fmt.symbol : ''
}

function render(v: number, fmt: TargetFormat): string {
  return `${symbolFor(fmt)}${grouped(fmt.locale, fracDigits(v)).format(v)}`
}

/* ---------------- domain-driven target formatting ----------------
 * fmtTarget(v, domain) renders the target per domain.target.format: the
 * symbol prefix for "money" style (none for "number"), the locale's grouping,
 * and magnitude-scaled precision. */

/** Target value: "$1,234,567" (money) or "5.17" (number). */
export function fmtTarget(v: number, domain: Domain): string {
  return render(v, domain.target.format)
}

/** List-column target: full digits up to 10M, then compact ("1.85B") so a
 *  diverged run can't blow the table out of the viewport. */
export function fmtTargetCell(v: number, domain: Domain): string {
  const a = Math.abs(v)
  if (!isFinite(v)) return '—'
  if (a < 10_000_000) return fmtTarget(v, domain)
  const fmt = domain.target.format
  return `${v < 0 ? '−' : ''}${symbolFor(fmt)}${groupedCompact(fmt.locale).format(a)}`
}

/** Signed target delta: "+12.4" / "−12.4". */
export function fmtTargetDelta(v: number, domain: Domain): string {
  const sign = v < 0 ? '−' : '+'
  return `${sign}${fmtTarget(Math.abs(v), domain)}`
}

/* ---------------- domain-agnostic helpers ----------------
 * Many bench call sites (leaderboard, scatter, sweep) format a value without a
 * Domain in hand. They use the format registered once at boot via
 * setFormatDomain; until then the target renders as a plain number. The names
 * keep the historical "money" prefix, but nothing currency-specific remains. */

let activeFormat: TargetFormat = NO_FORMAT

/** Register the domain's target format for the domain-agnostic helpers below.
 *  Call once at boot (see DomainContext). */
export function setFormatDomain(domain: Domain): void {
  activeFormat = domain.target.format
}

/** Value in the target's own units, using the boot-registered format. */
export function fmtMoney(v: number): string {
  return render(v, activeFormat)
}

/** List-column value: full digits up to 10M, then compact. */
export function fmtMoneyCell(v: number): string {
  const a = Math.abs(v)
  if (!isFinite(v)) return '—'
  if (a < 10_000_000) return render(v, activeFormat)
  return `${v < 0 ? '−' : ''}${symbolFor(activeFormat)}${groupedCompact(activeFormat.locale).format(a)}`
}

/** Signed value delta: "+12.4" / "−12.4". */
export function fmtMoneyDelta(v: number): string {
  const sign = v < 0 ? '−' : '+'
  return `${sign}${render(Math.abs(v), activeFormat)}`
}

/** Axis-tick label in the target's own units: compacts large magnitudes
 *  ("10k", "1M") and keeps the domain's symbol prefix, so a price domain reads
 *  "$10k" and a unitless one (habit fit) reads "8". No "$" is ever baked in. */
export function fmtTick(v: number): string {
  const sym = symbolFor(activeFormat)
  const a = Math.abs(v)
  const sign = v < 0 ? '−' : ''
  if (a >= 1e6) return `${sign}${sym}${+(a / 1e6).toPrecision(3)}M`
  if (a >= 1e3) return `${sign}${sym}${+(a / 1e3).toPrecision(3)}k`
  return `${sign}${sym}${grouped(activeFormat.locale, fracDigits(a)).format(a)}`
}

/** Fractions to percent: 0.355 → "35.5%". Absurd magnitudes (diverged runs)
 *  switch to exponential so they stay one column wide. */
export function fmtPct(frac: number, digits = 1): string {
  const pct = frac * 100
  if (!isFinite(pct)) return '—'
  if (Math.abs(pct) >= 100_000) return `${pct.toExponential(1)}%`
  return `${pct.toFixed(digits)}%`
}

export function fmtSignedPct(frac: number, digits = 1): string {
  const sign = frac < 0 ? '−' : '+'
  return `${sign}${(Math.abs(frac) * 100).toFixed(digits)}%`
}

export function fmtR2(v: number): string {
  if (!isFinite(v)) return '—'
  // A diverged run's R² of −2,597,472,592.563 is still bad news at "−2.6e9".
  if (Math.abs(v) >= 1000) return v.toExponential(1)
  return v.toFixed(3)
}

export function fmtLoss(v: number): string {
  if (v === 0) return '0'
  if (v >= 100 || v < 0.001) return v.toExponential(2)
  return v.toPrecision(3)
}

/** Table timestamp: "06-04 11:42" — fixed width, sorts visually. */
export function fmtStamp(iso: string): string {
  const d = new Date(iso)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`
}

export function fmtDateTime(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function fmtDuration(startIso: string, endIso: string | null): string {
  const end = endIso ? new Date(endIso).getTime() : Date.now()
  const s = Math.max(0, Math.round((end - new Date(startIso).getTime()) / 1000))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ${s % 60}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

/** Short run ids for display: drop the date prefix, keep the tail. */
export function shortRunId(runId: string): string {
  return runId.replace(/^run-\d{8}-/, '')
}

/** Tiny run ids for tables that already show the predictor in its own
 *  column: time + hash only ("140002-35065"). */
export function tinyRunId(runId: string): string {
  const short = shortRunId(runId)
  return /^\d+-[0-9a-z]+/.exec(short)?.[0] ?? short
}

/** Short dataset ids for display: drop the date prefix, keep the tail. */
export function shortDatasetId(datasetId: string): string {
  return datasetId.replace(/^ds-\d{8}-/, '')
}

/** Capitalize the first letter — for domain nouns used at the start of copy. */
export function cap(s: string): string {
  return s ? s[0].toUpperCase() + s.slice(1) : s
}

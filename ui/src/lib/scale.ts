// Minimal scale/tick helpers for the SVG charts.

export interface Scale {
  map: (v: number) => number
  domain: [number, number]
  range: [number, number]
}

export function linearScale(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain
  const [r0, r1] = range
  const span = d1 - d0 || 1
  return { map: (v) => r0 + ((v - d0) / span) * (r1 - r0), domain, range }
}

export function logScale(domain: [number, number], range: [number, number]): Scale {
  const d0 = Math.log10(Math.max(domain[0], Number.MIN_VALUE))
  const d1 = Math.log10(Math.max(domain[1], Number.MIN_VALUE))
  const [r0, r1] = range
  const span = d1 - d0 || 1
  return {
    map: (v) => r0 + ((Math.log10(Math.max(v, Number.MIN_VALUE)) - d0) / span) * (r1 - r0),
    domain,
    range,
  }
}

/** Round-numbered ticks covering [min, max]. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!isFinite(min) || !isFinite(max) || min === max) return [min]
  const span = max - min
  const step = Math.pow(10, Math.floor(Math.log10(span / count)))
  const err = (count * step) / span
  const mult = err <= 0.15 ? 10 : err <= 0.35 ? 5 : err <= 0.75 ? 2 : 1
  const s = step * mult
  const ticks: number[] = []
  for (let v = Math.ceil(min / s) * s; v <= max + s * 1e-9; v += s) {
    ticks.push(Math.abs(v) < s * 1e-9 ? 0 : v)
  }
  return ticks
}

/** Decade ticks for log axes: 1e4, 1e5, ... within [min, max]. */
export function decadeTicks(min: number, max: number): number[] {
  const lo = Math.ceil(Math.log10(Math.max(min, Number.MIN_VALUE)))
  const hi = Math.floor(Math.log10(Math.max(max, Number.MIN_VALUE)))
  const ticks: number[] = []
  for (let e = lo; e <= hi; e++) ticks.push(Math.pow(10, e))
  return ticks
}

/** Raw data extent (min, max) across both actual and predicted values. No
 *  unit assumptions: the target may be prices in the thousands or a habit-fit
 *  score near zero, so the range comes from the data, never a hardcoded floor. */
export function scatterDomain(preds: { actual: number; predicted: number }[]): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (const p of preds) {
    lo = Math.min(lo, p.actual, p.predicted)
    hi = Math.max(hi, p.actual, p.predicted)
  }
  if (!isFinite(lo) || !isFinite(hi)) return [0, 1]
  return [lo, hi]
}

/** Whether a predicted-vs-actual extent is better read on log axes. True only
 *  when every value is positive and the range spans well over a decade (e.g.
 *  prices from hundreds to millions); a compact non-negative target (a habit-fit
 *  score) stays linear, where points near zero are legible. */
export function scatterIsLog([lo, hi]: [number, number]): boolean {
  return lo > 0 && hi / lo >= 30
}

/** Padded display domain for the scatter, given its raw extent and scale type.
 *  Log: multiplicative padding on a positive floor. Linear: additive padding,
 *  anchored at zero when the data is non-negative so the diagonal reads true. */
export function scatterDisplayDomain(
  [lo, hi]: [number, number],
  log: boolean,
): [number, number] {
  if (log) {
    const floor = Math.max(lo, hi / 1e6, Number.MIN_VALUE)
    return [floor * 0.85, hi * 1.15]
  }
  const span = hi - lo || Math.abs(hi) || 1
  const pad = span * 0.06
  return [lo >= 0 ? Math.max(0, lo - pad) : lo - pad, hi + pad]
}

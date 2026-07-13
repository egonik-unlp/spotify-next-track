// Task-aware metric display. A run's metrics JSON carries task-dependent keys
// (regression: mae/rmse/r2/mape/medape; classification: accuracy/logloss/auc/
// brier/macro_f1), and the domain config drives which to show, in what order,
// how to rank, and which render as percentages. This maps a metric DISPLAY
// name (as listed in domain.metrics.columns) to its key on the Metrics object,
// its polarity (lower-is-better), and a formatter — reusing the boot-registered
// fmt helpers so values render in the domain's own units.

import type { Domain } from './domain'
import type { Metrics } from '../api/types'
import { fmtMoney, fmtMoneyCell, fmtPct, fmtR2, fmtLoss } from './format'

/** The Metrics keys a metric column can read. */
export type MetricKey =
  | 'mae'
  | 'rmse'
  | 'r2'
  | 'mape'
  | 'medape'
  | 'accuracy'
  | 'logloss'
  | 'auc'
  | 'brier'
  | 'macro_f1'

/** Display name → Metrics key. Lowercased, with "R²" → "r2"
 *  (e.g. "AUC"→"auc", "logloss"→"logloss", "macro_f1"→"macro_f1"). */
export function metricKey(displayName: string): MetricKey {
  return displayName.toLowerCase().replace('²', '2') as MetricKey
}

// Error metrics rank ascending; score metrics rank descending.
const LOWER_IS_BETTER: Record<string, true> = {
  mae: true,
  rmse: true,
  mape: true,
  medape: true,
  logloss: true,
  brier: true,
}

/** True when a smaller value is better (error metrics), false for scores. */
export function lowerIsBetter(key: MetricKey): boolean {
  return LOWER_IS_BETTER[key] === true
}

// Unitless metrics rendered as percentages by default, regardless of the
// domain's `percent` list (which can add e.g. "accuracy"/"auc" explicitly).
const PCT_DEFAULT: Record<string, true> = { mape: true, medape: true }
// Unitless ratio metrics shown with fixed precision (not the target's units).
const RATIO: Record<string, true> = { r2: true, auc: true, accuracy: true, macro_f1: true }

/** A formatter for a metric, given its display name. Percent when the display
 *  name is in `domain.metrics.percent` or it's an inherently-% metric
 *  (mape/medape); a fixed-precision ratio for r2/auc/accuracy/macro_f1; loss
 *  precision for logloss/brier; otherwise the target's own units (value_unit). */
export function metricFormatter(displayName: string, domain: Domain): (v: number) => string {
  const key = metricKey(displayName)
  const asPercent = domain.metrics.percent.includes(displayName) || PCT_DEFAULT[key] === true
  if (asPercent) return (v) => fmtPct(v, key === 'mape' ? 0 : 1)
  if (key === 'r2') return fmtR2
  if (RATIO[key] === true) return (v) => (isFinite(v) ? v.toFixed(3) : '—')
  if (key === 'logloss' || key === 'brier') return fmtLoss
  // Error metrics in the target's own units (mae/rmse): the boot-registered
  // money formatter renders the unit prefix when the domain has one.
  return fmtMoney
}

/** True when the metric is expressed in the target's own units (mae/rmse),
 *  i.e. neither a percentage/ratio nor a loss. Such cells compact large
 *  magnitudes (fmtMoneyCell) and carry a full-precision title. */
export function isTargetUnitMetric(displayName: string, domain: Domain): boolean {
  const key = metricKey(displayName)
  if (domain.metrics.percent.includes(displayName)) return false
  if (PCT_DEFAULT[key] || RATIO[key] || key === 'logloss' || key === 'brier' || key === 'r2') {
    return false
  }
  return true
}

/** Table-cell formatter: target-unit metrics compact large magnitudes
 *  (fmtMoneyCell) to stay one column wide; everything else uses the value
 *  formatter. */
export function metricCellFormatter(displayName: string, domain: Domain): (v: number) => string {
  if (isTargetUnitMetric(displayName, domain)) return fmtMoneyCell
  return metricFormatter(displayName, domain)
}

/** Read a metric value off a run's metrics by display name; null when absent. */
export function metricValue(displayName: string, m: Metrics | null | undefined): number | null {
  if (!m) return null
  const v = m[metricKey(displayName)]
  return typeof v === 'number' ? v : null
}

/** Format a value carried by display name (e.g. best-models metric_value). */
export function fmtMetricValue(displayName: string, v: number, domain: Domain): string {
  return metricFormatter(displayName, domain)(v)
}

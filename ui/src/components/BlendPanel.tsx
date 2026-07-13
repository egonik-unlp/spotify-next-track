import { useMemo } from 'react'
import type { BlendFile, Metrics } from '../api/types'
import { blendSpecFromFile, deriveArch, type ArchFeatures } from '../lib/arch'
import { useAsync } from '../hooks/useAsync'
import { useDomain } from '../lib/DomainContext'
import { fmtPct } from '../lib/format'
import { fmtMetricValue, lowerIsBetter, metricFormatter, metricKey, metricValue } from '../lib/metrics'
import { BlendArch } from './ArchViz'
import { DefinitionRef, ModelRef, PredictorRef } from './EntityRef'
import './blendpanel.css'

/** Blend models get their own panel: the member fan-in diagram plus a
 *  members table — who is inside, at what weight, and how each member tests
 *  solo against the blend. Reads blend.json (written when the run finishes);
 *  while it's absent (running, or a stop before the write) the diagram is
 *  derived from the hyperparams spec instead and the table waits. */
export default function BlendPanel({
  load,
  hyperparams,
  features,
  blendMetrics,
}: {
  /** blend.json fetcher (run or model endpoint). */
  load: () => Promise<BlendFile>
  hyperparams: Record<string, unknown> | null
  features: ArchFeatures | null
  /** The blend's own test metrics, for the comparison row. */
  blendMetrics: Metrics | null
}) {
  const domain = useDomain()
  const blend = useAsync(load, [])

  // Solo/blend comparison follows the domain's primary metric and its polarity,
  // not a hardcoded MAE — so it reads correctly for a ranking (recall@k) or
  // classification (AUC) blend just as for a regression one.
  const cols = domain.metrics.columns
  const primary = domain.metrics.primary
  const primaryLower = lowerIsBetter(metricKey(primary))

  const fileSpec = useMemo(
    () => (blend.data ? blendSpecFromFile(blend.data) : null),
    [blend.data],
  )
  const hpSpec = useMemo(
    () => deriveArch('blend', hyperparams, features),
    [hyperparams, features],
  )

  const spec = fileSpec ?? (hpSpec?.kind === 'blend' ? hpSpec : null)
  if (!spec) return null

  const file = blend.data ?? null
  const soloPrimaries = file
    ? file.members
        .filter((m) => m.included && m.solo_metrics)
        .map((m) => metricValue(primary, m.solo_metrics))
        .filter((v): v is number => v != null)
    : []
  const bestSolo = soloPrimaries.length
    ? primaryLower
      ? Math.min(...soloPrimaries)
      : Math.max(...soloPrimaries)
    : null
  const blendPrimary = metricValue(primary, blendMetrics)
  // Positive edge = the blend improves on its best solo member.
  const edge =
    bestSolo != null && blendPrimary != null
      ? primaryLower
        ? bestSolo - blendPrimary
        : blendPrimary - bestSolo
      : null

  return (
    <div className="blend-panel">
      <BlendArch spec={spec} title="Blend" />
      {file ? (
        <>
          <table className="blend-members" aria-label="Blend members">
            <thead>
              <tr>
                <th>Member</th>
                <th>Predictor</th>
                <th className="num-col">Weight</th>
                <th className="num-col">Cols</th>
                {cols.map((c, i) => (
                  <th key={c} className={`num-col${i === 0 ? ' col-group-start' : ''}`}>
                    {c} solo
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {spec.members.map((m, i) => {
                const rec = file.members[i]
                const sm = rec?.solo_metrics ?? null
                const blocks = rec?.exclude_blocks ?? []
                return (
                  <tr key={m.key} className={`blend-member-row${m.excluded ? ' is-excluded' : ''}`}>
                    <td>
                      <MemberLink kind={m.kind} label={m.label} predictor={rec?.predictor ?? ''} />
                      {m.excluded && <span className="blend-flag">excluded</span>}
                      {blocks.length > 0 && (
                        <span className="blend-blocks num" title={`feature blocks withheld: ${blocks.join(', ')}`}>
                          −{blocks.join(' −')}
                        </span>
                      )}
                    </td>
                    <td className="muted">
                      {rec?.predictor}
                      {m.frozen ? ' · frozen' : ''}
                    </td>
                    <td className="num-col num">
                      <span className="blend-weight">
                        <span className="blend-weight-bar" aria-hidden>
                          <span style={{ width: `${Math.round(m.weight * 100)}%` }} />
                        </span>
                        {spec.rule === 'median' ? 'vote' : fmtPct(m.weight, 0)}
                      </span>
                    </td>
                    <td className="num-col num">{rec?.n_cols ?? '—'}</td>
                    {cols.map((c, ci) => {
                      const v = metricValue(c, sm)
                      return (
                        <td key={c} className={`num-col num${ci === 0 ? ' col-group-start' : ''}`}>
                          {v != null ? metricFormatter(c, domain)(v) : '—'}
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
              {blendMetrics && (
                <tr className="blend-total-row">
                  <td>blend</td>
                  <td className="muted">
                    {spec.rule === 'median' ? 'median vote' : 'weighted mean'}
                    {spec.weightFit === 'grid' ? ' · grid-fitted' : ''}
                  </td>
                  <td className="num-col num">{spec.rule === 'median' ? '' : '100%'}</td>
                  <td className="num-col num" />
                  {cols.map((c, ci) => {
                    const v = metricValue(c, blendMetrics)
                    return (
                      <td key={c} className={`num-col num${ci === 0 ? ' col-group-start' : ''}`}>
                        {v != null ? metricFormatter(c, domain)(v) : '—'}
                      </td>
                    )
                  })}
                </tr>
              )}
            </tbody>
          </table>
          {edge !== null && (
            <p className="blend-verdict num" role="status">
              {edge >= 0
                ? `the blend beats its best member by ${fmtMetricValue(primary, edge, domain)} ${primary}`
                : `its best member beats the blend by ${fmtMetricValue(primary, -edge, domain)} ${primary}`}
            </p>
          )}
        </>
      ) : blend.loading ? (
        <div className="skeleton" style={{ height: '4rem' }} />
      ) : (
        <p className="muted blend-pending">
          No member record yet — fitted weights and solo metrics are written when training
          finishes.
        </p>
      )}
    </div>
  )
}

function MemberLink({ kind, label, predictor }: { kind: string; label: string; predictor: string }) {
  if (kind === 'model') return <ModelRef name={label} />
  if (kind === 'definition') return <DefinitionRef name={label} />
  return <PredictorRef name={predictor || label} />
}

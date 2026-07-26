import { useState } from 'react'
import type { ModelSaeReport } from '../api/types'
import { modelSaeDigest } from '../lib/modelSae'

/** Per-model SAE report, reframed for the next-track (ranking) task. Reads
 *  answer-first: a one-line digest + one scannable "by depth" table carry the
 *  story; per-layer atom detail is collapsed behind expanders. "Concepts" here
 *  are next-item properties (artist / genre / album / sonic-continuity), not a
 *  scalar target. Extracted from ModelSaeTool so the live tool and the
 *  Saved-analyses view render identically. `caption` off in compare columns. */
export default function ModelSaeReportView({
  report,
  caption = true,
}: {
  report: ModelSaeReport
  caption?: boolean
}) {
  const [diffLayer, setDiffLayer] = useState<number | null>(null)

  const deepest = report.layers[report.layers.length - 1]
  const d = modelSaeDigest(report)
  const pct = (v: number) => `${(v * 100).toFixed(0)}%`
  const num = (v: number | null | undefined, dp = 2) =>
    v == null ? '—' : v.toFixed(dp)

  const diffLayers = (report.layers ?? []).filter((l) => l.dropped_vs_next_item)
  const selectedDiff =
    diffLayers.find((l) => l.layer === diffLayer) ?? diffLayers[diffLayers.length - 1]

  return (
    <div className="interp-result msae">
      {caption && (
        <header className="msae-head">
          <p className="interp-caption">
            {report.model_dir} · {report.dataset_id} · {report.predictor} ·{' '}
            {report.n_train_rows.toLocaleString()} train / {report.n_test_rows.toLocaleString()} test
            steps
          </p>
          <p className="msae-digest">
            peak next-item <strong>AUC {d.peakAuc.toFixed(2)}</strong> (acc{' '}
            {d.peakAcc.toFixed(2)}) · uses up to <strong>{pct(d.maxUtil)}</strong> of a layer · forms{' '}
            <strong>{d.concepts}</strong> interpretable concepts · represents{' '}
            <strong>{d.segRepr}</strong> {report.segment_field} classes
            {d.dropped != null && (
              <>
                {' '}
                · drops <strong>{d.dropped}</strong> next-item concepts
              </>
            )}
            {d.recall != null && (
              <>
                {' '}
                · model recall@10 <strong>{d.recall.toFixed(3)}</strong>
              </>
            )}
          </p>
        </header>
      )}

      {/* The depth story: how decodable the next item's segment is at each layer
          vs. how many interpretable concepts the SAE forms / how much width used. */}
      <h4>By depth</h4>
      <table className="interp-table msae-depth">
        <thead>
          <tr>
            <th>layer</th>
            <th>dim</th>
            <th>next-item AUC</th>
            <th>acc</th>
            <th>concepts</th>
            <th>width used</th>
          </tr>
        </thead>
        <tbody>
          {report.layers.map((l) => (
            <tr key={l.layer}>
              <td>{l.name}</td>
              <td>{l.dim}</td>
              <td>{num(l.next_item_decodability?.auc)}</td>
              <td>{num(l.next_item_decodability?.acc)}</td>
              <td>{l.n_interpretable_concepts}</td>
              <td>{pct(l.capacity.utilization)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="interp-tools-note">
        AUC/accuracy are for a linear probe decoding the next track&apos;s {report.segment_field}{' '}
        from the raw activations (baseline ={' '}
        {num(deepest?.next_item_decodability?.baseline_acc)} at the deepest layer). A layer whose AUC
        barely rises but whose concept count is high is doing generic transformation, not building
        next-track structure.
      </p>

      {/* Per-layer atom detail, on demand. */}
      <div className="msae-layers">
        {report.layers.map((l) => {
          const labeled = l.atoms_by_concept.some((a) => !!a.label)
          const moodCol = report.mood_available
          return (
            <details className="msae-layer" key={l.layer}>
              <summary>
                <span className="msae-layer-name">Layer {l.layer} · {l.name}</span>
                <span className="msae-layer-meta">
                  {l.dim}-d · {pct(l.capacity.utilization)} used · {l.n_interpretable_concepts}{' '}
                  concepts · next-item AUC {num(l.next_item_decodability?.auc)}
                </span>
              </summary>
              <p className="interp-tools-note">
                {l.capacity.active_atoms}/{l.capacity.n_atoms} atoms active · {l.capacity.dead_atoms}{' '}
                dead · {l.capacity.rare_atoms} rare · {l.capacity.l0_mean.toFixed(1)} active/step ·
                reconstructs {pct(l.capacity.var_explained)} of variance
              </p>
              <table className="interp-table">
                <thead>
                  <tr>
                    <th>atom</th>
                    <th>next-item concept</th>
                    <th>sep (σ)</th>
                    <th>freq</th>
                    {moodCol && <th>sonic corr</th>}
                    {labeled && <th>label</th>}
                  </tr>
                </thead>
                <tbody>
                  {l.atoms_by_concept.slice(0, 8).map((a) => (
                    <tr key={a.atom}>
                      <td>#{a.atom}</td>
                      <td>
                        {a.concept_field && a.concept_value
                          ? `${a.concept_field}=${a.concept_value}`
                          : '—'}
                      </td>
                      <td>{num(a.assoc)}</td>
                      <td>{(a.freq * 100).toFixed(1)}%</td>
                      {moodCol && <td>{num(a.mood_corr)}</td>}
                      {labeled && <td className="sae-label">{a.label ?? '—'}</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          )
        })}
      </div>

      {deepest && deepest.segments.length > 0 && (
        <section className="msae-section">
          <h4>
            Segment representation · {report.segment_field} · layer {deepest.layer}
          </h4>
          <p className="interp-tools-note">
            Whether the model built a dedicated atom for each next-item {report.segment_field} class
            (best class-vs-rest separation, σ). Unrepresented = smeared across shared atoms.
          </p>
          <table className="interp-table">
            <thead>
              <tr>
                <th>segment</th>
                <th>steps</th>
                <th>represented</th>
                <th>best sep (σ)</th>
              </tr>
            </thead>
            <tbody>
              {deepest.segments.map((s) => (
                <tr key={s.segment}>
                  <td>{s.segment}</td>
                  <td>{s.n_rows.toLocaleString()}</td>
                  <td>{s.represented ? '✓' : '✗'}</td>
                  <td>{num(s.top_atoms[0]?.separation)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {report.embedding_diff.ran && selectedDiff?.dropped_vs_next_item && (
        <section className="msae-section">
          <div className="interp-diff-head">
            <h4>Dropped signal · next-item concepts</h4>
            {diffLayers.length > 1 && (
              <label>
                <span>at layer</span>
                <select
                  value={selectedDiff.layer}
                  onChange={(e) => setDiffLayer(Number(e.target.value))}
                >
                  {diffLayers.map((l) => (
                    <option key={l.layer} value={l.layer}>
                      {l.name} · {l.dim}-d
                    </option>
                  ))}
                </select>
              </label>
            )}
          </div>
          <p className="interp-tools-note">
            Frequent next-item concept classes whose best atom separation falls below{' '}
            {num(report.embedding_diff.sep_threshold)}σ — {selectedDiff.dropped_vs_next_item.n_dropped}
            /{selectedDiff.dropped_vs_next_item.n_checked} unrepresented. These are next-track
            properties the model never carves a feature for.
          </p>
          {selectedDiff.dropped_vs_next_item.n_dropped === 0 ? (
            <p className="interp-tools-note">
              Every frequent next-item concept has an atom tracking it at this layer.
            </p>
          ) : (
            <table className="interp-table">
              <thead>
                <tr>
                  <th>concept</th>
                  <th>freq</th>
                  <th>best atom sep (σ)</th>
                </tr>
              </thead>
              <tbody>
                {selectedDiff.dropped_vs_next_item.dropped.map((x) => (
                  <tr key={`${x.concept_field}=${x.concept_value}`}>
                    <td>
                      {x.concept_field}={x.concept_value}
                    </td>
                    <td>{(x.freq * 100).toFixed(1)}%</td>
                    <td>{num(x.best_atom_sep)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      )}
    </div>
  )
}

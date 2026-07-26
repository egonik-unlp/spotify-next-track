import { useEffect, useMemo, useRef, useState } from 'react'
import { api, pollJob } from '../api/client'
import type { InterpModel, ModelSaeReport } from '../api/types'
import ModelSaeReportView from '../components/ModelSaeReportView'
import { useAsync } from '../hooks/useAsync'

/** Per-model SAE tool — the nonlinear sibling of the layer probe. Trains a
 *  sparse autoencoder on a promoted model's OWN hidden activations (per layer)
 *  and surfaces four reads the linear probe can't: capacity (how much of each
 *  layer the model uses), per-segment representation, concept-vs-decodability
 *  depth, and — optionally — the target-relevant signal the model drops relative
 *  to the embedding. Self-contained so it drops into the tool rail. The result
 *  render lives in ModelSaeReportView, shared with the Saved-analyses view. */
export default function ModelSaeTool() {
  const models = useAsync(() => api.listInterpModels(), [])
  const [model, setModel] = useState('')
  const [layers, setLayers] = useState('')
  const [labelAtoms, setLabelAtoms] = useState(false)
  const [compareEmbedding, setCompareEmbedding] = useState(false)
  const [running, setRunning] = useState(false)
  const [stage, setStage] = useState('')
  const [report, setReport] = useState<ModelSaeReport | null>(null)
  const [error, setError] = useState<string | null>(null)
  const cancelRef = useRef<(() => void) | null>(null)

  useEffect(() => () => cancelRef.current?.(), [])

  // Only MLP-family models expose a hidden activation space to decompose.
  const eligible = useMemo(
    () => (models.data ?? []).filter((m: InterpModel) => m.supports_model_sae),
    [models.data],
  )

  const run = () => {
    if (!model) return
    cancelRef.current?.()
    setRunning(true)
    setError(null)
    setReport(null)
    setStage('starting…')
    api
      .startModelSae({
        model,
        layers: layers.trim() || undefined,
        label_atoms: labelAtoms,
        compare_embedding: compareEmbedding,
      })
      .then(
        (jobId) => {
          cancelRef.current = pollJob(jobId, {
            onStage: setStage,
            onDone: (res) => {
              setReport(res as ModelSaeReport)
              setRunning(false)
            },
            onError: (msg) => {
              setError(msg)
              setRunning(false)
            },
          })
        },
        (e: unknown) => {
          setError(e instanceof Error ? e.message : String(e))
          setRunning(false)
        },
      )
  }

  return (
    <>
      <p className="interp-lede">
        <strong>Per-model sparse autoencoder</strong>: dictionary learning on a trained next-track
        model&apos;s own hidden activations. Its atoms are read against the <em>next item</em> — the
        artist / genre / album / sonic-continuity the holisticness suite cares about — so it shows
        what <em>this checkpoint</em> represents: how much of each layer&apos;s width it uses, which
        next-item concept classes it built dedicated features for, where the next track becomes
        linearly decodable vs. where interpretable concepts form, and which frequent next-item
        concepts the model drops. Sequence families (GRU, ANN) only; atom identities are
        per-checkpoint. Every run is saved — revisit or compare it under{' '}
        <strong>Saved analyses</strong>.
      </p>

      <div className="interp-controls">
        <label>
          <span>Model</span>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="">Select a model…</option>
            {eligible.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name} · [{m.hidden.join(', ')}] · {m.activation}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Layers (blank = all)</span>
          <input
            type="text"
            value={layers}
            placeholder="e.g. 2,3"
            onChange={(e) => setLayers(e.target.value)}
            disabled={running}
          />
        </label>
        <label className="interp-checkbox">
          <input
            type="checkbox"
            checked={labelAtoms}
            onChange={(e) => setLabelAtoms(e.target.checked)}
            disabled={running}
          />
          <span>Label atoms with GPT (auto-interp)</span>
        </label>
        <label className="interp-checkbox">
          <input
            type="checkbox"
            checked={compareEmbedding}
            onChange={(e) => setCompareEmbedding(e.target.checked)}
            disabled={running}
          />
          <span>Dropped-signal read (next-item concepts)</span>
        </label>
        <button className="btn-primary" onClick={run} disabled={!model || running}>
          {running ? 'Training SAEs…' : 'Train & analyze'}
        </button>
      </div>
      <p className="interp-tools-note">
        Trains one SAE per hidden layer on the model&apos;s per-step activations (a few minutes;
        cached across runs), then relates the atoms to next-item concepts. Atom labeling needs{' '}
        <code>OPENAI_API_KEY</code> on the server.
      </p>

      {models.error && (
        <div className="error-block" role="alert">
          Could not load models: {models.error}
        </div>
      )}
      {!models.loading && eligible.length === 0 && (
        <div className="interp-progress">
          No per-model-SAE-capable models found (sequence families — GRU, ANN).
        </div>
      )}
      {running && <div className="interp-progress">{stage}</div>}
      {error && (
        <div className="error-block" role="alert">
          {error}
        </div>
      )}

      {report && <ModelSaeReportView report={report} />}
    </>
  )
}

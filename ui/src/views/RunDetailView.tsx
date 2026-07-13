import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
import type { Items, Metrics, Prediction, Predictor, RunMeta } from '../api/types'
import ArchViz from '../components/ArchViz'
import BlendPanel from '../components/BlendPanel'
import ErrorHistogram from '../components/charts/ErrorHistogram'
import LossChart from '../components/charts/LossChart'
import LossDerivativeChart from '../components/charts/LossDerivativeChart'
import ScatterChart from '../components/charts/ScatterChart'
import { DatasetRef, DefinitionRef, ModelRef, PredictorRef, RunRef } from '../components/EntityRef'
import ItemIdentity from '../components/ItemIdentity'
import ItemMeta from '../components/ItemMeta'
import LogPane from '../components/LogPane'
import MetricStrip from '../components/MetricStrip'
import StatusBadge from '../components/StatusBadge'
import ViewHeader from '../components/ViewHeader'
import { useAsync } from '../hooks/useAsync'
import { useRunEvents } from '../hooks/useRunEvents'
import { loadItems } from '../lib/itemsCache'
import { computeMetrics, suspiciousRowIds } from '../lib/suspicious'
import {
  cap,
  fmtDateTime,
  fmtDuration,
  fmtMoney,
  fmtMoneyDelta,
  fmtPct,
  fmtR2,
  fmtSignedPct,
  shortRunId,
} from '../lib/format'
import './rundetail.css'
import { metricFormatter, metricValue, resolveTask } from '../lib/metrics'
import { useDocTitle, useDomain } from '../lib/DomainContext'
import { itemByline, itemLabel, itemTitle, sequenceItemForDisplay } from '../lib/itemDisplay'

export default function RunDetailView() {
  const { runId } = useParams<{ runId: string }>()
  const domain = useDomain()
  const { data: run, error, loading, reload } = useAsync(() => api.getRun(runId!), [runId])
  const predictors = useAsync(() => api.listPredictors(), [])
  const events = useRunEvents(runId ?? null)

  // The SSE stream ends with a status event once the run finishes: refetch meta.
  useEffect(() => {
    if (events.terminal && run?.status === 'running') reload()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events.terminal])

  useDocTitle(run ? shortRunId(run.run_id) : null)

  if (error) {
    return (
      <>
        <ViewHeader glyph="run" title="Run" crumbs={[{ label: 'Runs', to: '/' }]} />
        <div className="view-body">
          <div className="error-block" role="alert">
            Could not load run: {error}{' '}
            <button className="btn" onClick={reload}>
              Retry
            </button>{' '}
            <Link to="/">Back to runs</Link>
          </div>
        </div>
      </>
    )
  }
  if (loading || !run) {
    return (
      <>
        <ViewHeader glyph="run" title="Run" crumbs={[{ label: 'Runs', to: '/' }]} />
        <div className="view-body">
          <div className="skeleton skeleton-lg" />
        </div>
      </>
    )
  }

  const pred = predictors.data?.find((p) => p.name === run.predictor)
  const supportsPredict = !!pred?.predict_args
  // Promotable from the last periodic checkpoint even though the run died.
  const checkpointPromotable =
    (run.status === 'failed' || run.status === 'interrupted') &&
    supportsPredict &&
    run.has_checkpoint

  return (
    <section aria-label={`Run ${shortRunId(run.run_id)}`}>
      <ViewHeader
        glyph="run"
        crumbs={[{ label: 'Runs', to: '/' }, { label: shortRunId(run.run_id) }]}
        title={
          <>
            <PredictorRef name={run.predictor} impl />{' '}
            <span className="run-id-frag">
              <RunRef id={run.run_id} self />
            </span>
          </>
        }
        meta={
          <>
            <StatusBadge status={run.status} />
            <span>
              started <span className="num">{fmtDateTime(run.started_at)}</span>
            </span>
            <span>
              took <span className="num">{fmtDuration(run.started_at, run.finished_at)}</span>
            </span>
          </>
        }
        actions={run.status !== 'running' ? <DeleteRunButton run={run} /> : undefined}
      />
      <div className="view-body">
      <RunHeader run={run} />
      {run.status === 'running' && (
        <>
          <StopControls run={run} predictor={pred} stopping={events.stopping} />
          <div className="live-grid">
            <div>
              <h2 className="panel-title">
                Loss{' '}
                {events.connection === 'reconnecting' && (
                  <span className="reconnect" role="status">
                    connection lost, retrying…
                  </span>
                )}
              </h2>
              <LossChart epochs={events.epochs} totalEpochs={events.epochs[0]?.total} live />
              {events.epochs.length >= 3 && (
                <>
                  <h2 className="panel-title dynamics-title">Is it still learning?</h2>
                  <LossDerivativeChart epochs={events.epochs} />
                </>
              )}
            </div>
            <div>
              <h2 className="panel-title">Log</h2>
              <LogPane lines={events.logs} />
            </div>
          </div>
          <RunArch run={run} canFallback={!!pred?.visualization} />
        </>
      )}
      {run.status === 'failed' && (
        <div className="error-block" role="alert">
          <strong>Run failed</strong>
          {run.exit_code !== null && (
            <>
              {' '}
              — predictor exited with code <span className="num">{run.exit_code}</span>
            </>
          )}
          .{' '}
          {checkpointPromotable
            ? 'A periodic checkpoint was saved before it died, so the params as they were can still be promoted below.'
            : 'Fix the cause and start a new run; run directories are immutable.'}
          {run.stderr_tail && <pre>{run.stderr_tail}</pre>}
          {events.logs.length > 0 && (
            <>
              <h2 className="panel-title">Log</h2>
              <LogPane lines={events.logs} />
            </>
          )}
        </div>
      )}
      {run.status === 'interrupted' && (
        <div className="error-block" role="alert">
          <strong>Run interrupted</strong> — the server restarted while this run was in
          progress.{' '}
          {checkpointPromotable
            ? 'A periodic checkpoint survived, so the params as they were can still be promoted below.'
            : 'Its results are incomplete; start a new run with the same configuration.'}
        </div>
      )}
      {checkpointPromotable && <PromotePanel run={run} checkpointOnly />}
      {(run.status === 'succeeded' || run.status === 'stopped') &&
        (resolveTask(domain, run.metrics) === 'ranking' ? (
          <RankingRun run={run} events={events} />
        ) : (
          <FinishedRun run={run} events={events} />
        ))}
      </div>
    </section>
  )
}

/** Native architecture diagram from hyperparams + the dataset's feature
 *  counts; falls back to the predictor-exported viz.svg if the dataset is
 *  gone (manifest unavailable) or the family is unknown. */
function RunArch({ run, canFallback }: { run: RunMeta; canFallback: boolean }) {
  const manifest = useAsync(() => api.getDataset(run.dataset_id), [run.dataset_id])
  if (manifest.loading) return null
  const m = manifest.data
  const features = m
    ? {
        nCols: m.n_cols,
        nPca: m.columns.filter((c) => c.kind.type === 'pca').length,
        onehotGroups: [
          ...new Set(
            m.columns.flatMap((c) => (c.kind.type === 'onehot' ? [c.kind.group] : [])),
          ),
        ],
      }
    : null
  if (run.predictor === 'blend') {
    return (
      <BlendPanel
        load={() => api.runBlend(run.run_id)}
        hyperparams={run.hyperparams}
        features={features}
        blendMetrics={run.metrics}
      />
    )
  }
  return (
    <ArchViz
      predictor={run.predictor}
      hyperparams={run.hyperparams}
      features={features}
      fallbackUrl={canFallback ? api.runVizUrl(run.run_id) : undefined}
    />
  )
}

/* ---------------- stop a running run ---------------- */

function StopControls({
  run,
  predictor,
  stopping,
}: {
  run: RunMeta
  predictor: Predictor | undefined
  stopping: boolean
}) {
  const [requested, setRequested] = useState(false)
  const [forceRequested, setForceRequested] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const supportsStop = !!predictor?.supports_stop

  const stop = async (force: boolean) => {
    // First force on a STOP-honoring predictor still saves: the server kills
    // only if the process is alive after the grace window. A repeat force
    // (or any force on a predictor that ignores STOP) kills immediately.
    const kills = force && (forceRequested || !supportsStop)
    if (
      force &&
      !window.confirm(
        kills
          ? 'This kills the training process now. Unless a periodic checkpoint was saved ' +
              '(checkpoint_every), the run cannot be promoted. Continue?'
          : 'Force stop asks the run to evaluate and save now, and kills it only if it is ' +
              'still alive after 60 seconds. Continue?',
      )
    ) {
      return
    }
    setBusy(true)
    setError(null)
    try {
      await api.stopRun(run.run_id, force)
      setRequested(true)
      if (force) setForceRequested(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
    setBusy(false)
  }

  return (
    <div className="stop-row">
      {stopping ? (
        <span className="muted" role="status">
          stopping — evaluating and saving the model with the params as they are…
        </span>
      ) : (
        <>
          {supportsStop && (
            <button className="btn" onClick={() => void stop(false)} disabled={busy || requested}>
              {requested ? 'Stop requested…' : 'Stop training'}
            </button>
          )}
          {(requested || !supportsStop) && (
            <button className="btn btn-danger" onClick={() => void stop(true)} disabled={busy}>
              {forceRequested ? 'Kill now' : 'Force stop'}
            </button>
          )}
          <span className="muted">
            {supportsStop
              ? requested
                ? 'finishing the current epoch, then evaluating and saving — the run stays promotable'
                : 'finishes the current epoch, evaluates and saves a model from the params as they are'
              : 'this predictor cannot stop gracefully; force stop kills the process'}
          </span>
        </>
      )}
      {error && (
        <span className="hp-error" role="alert">
          {error}
        </span>
      )}
    </div>
  )
}

function DeleteRunButton({ run }: { run: RunMeta }) {
  const navigate = useNavigate()
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const remove = async () => {
    if (
      !window.confirm(
        `Delete run ${shortRunId(run.run_id)}? Its directory (metrics, predictions, ` +
          'checkpoint) is removed. Models promoted from it are untouched.',
      )
    )
      return
    setDeleting(true)
    setError(null)
    try {
      await api.deleteRun(run.run_id)
      navigate('/')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setDeleting(false)
    }
  }

  return (
    <>
      <button className="btn-on-chrome" onClick={() => void remove()} disabled={deleting}>
        {deleting ? 'Deleting…' : 'Delete run'}
      </button>
      {error && (
        <span className="hp-error" role="alert">
          {error}
        </span>
      )}
    </>
  )
}

function RunHeader({ run }: { run: RunMeta }) {
  const hp = Object.entries(run.hyperparams)
  return (
    <header className="run-header">
      <dl className="run-meta">
        <div>
          <dt>Run ID</dt>
          <dd>
            <RunRef id={run.run_id} short={false} self copy />
          </dd>
        </div>
        <div>
          <dt>Dataset</dt>
          <dd>
            <DatasetRef id={run.dataset_id} short={false} />
          </dd>
        </div>
        {run.from_definition && (
          <div>
            <dt>From definition</dt>
            <dd>
              <DefinitionRef name={run.from_definition} />
            </dd>
          </div>
        )}
        {hp.length > 0 && (
          <div className="run-hp">
            <dt>Hyperparameters</dt>
            <dd>
              {hp.map(([k, v]) => (
                <span key={k} className="chip">
                  <span className="chip-key">{k}=</span>
                  <span className="num">{JSON.stringify(v)}</span>
                </span>
              ))}
              <SaveAsDefinition run={run} />
            </dd>
          </div>
        )}
      </dl>
    </header>
  )
}

/** Freeze this run's predictor + hyperparams as a reusable definition. */
function SaveAsDefinition({ run }: { run: RunMeta }) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState<string | null>(null)

  if (saved) {
    return (
      <span className="save-def" role="status">
        saved as <DefinitionRef name={saved} />
      </span>
    )
  }

  const nameOk = NAME_RE.test(name)
  const save = async () => {
    setBusy(true)
    setError(null)
    try {
      const def = await api.createDefinition({
        name,
        predictor: run.predictor,
        hyperparams: run.hyperparams,
        dataset_tags: [run.dataset_id],
      })
      setSaved(def.name)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }

  if (!open) {
    return (
      <button className="btn-inline" onClick={() => setOpen(true)} title="Save these hyperparameters as a reusable definition">
        save as definition
      </button>
    )
  }
  return (
    <span className="save-def">
      <input
        type="text"
        className="mono-input"
        placeholder="definition name"
        aria-label="Definition name"
        value={name}
        onChange={(e) => setName(e.target.value)}
        disabled={busy}
        autoFocus
      />
      <button className="btn-inline" onClick={save} disabled={!nameOk || busy}>
        {busy ? 'saving…' : 'save'}
      </button>
      <button className="btn-inline" onClick={() => setOpen(false)} disabled={busy}>
        cancel
      </button>
      {error && (
        <span className="hp-error" role="alert">
          {error}
        </span>
      )}
    </span>
  )
}

/* ---------------- finished run ---------------- */

type PredSort = 'abs_err' | 'pct_err' | 'actual' | 'predicted' | 'row_id' | 'wrongness' | 'prob'

interface Row extends Prediction {
  err: number
  pct: number
  suspicious: boolean
  // Classification-derived (present only for classifier runs). `prob` is the
  // model's probability for the positive class (binary) or for its predicted
  // class (multiclass); `wrongness` is 0 for a confident-correct call and 1 for
  // a confident-wrong one, so a descending sort puts the worst calls first.
  actualClass?: number
  predClass?: number
  prob?: number
  correct?: boolean
  wrongness?: number
}

function FinishedRun({ run, events }: { run: RunMeta; events: ReturnType<typeof useRunEvents> }) {
  const domain = useDomain()
  const preds = useAsync(() => api.getPredictions(run.run_id), [run.run_id])
  const predictors = useAsync(() => api.listPredictors(), [])
  // Corpus payloads, loaded eagerly so every prediction reads as its entity
  // (the track) in the table and the scatter tooltip, not as a bare row id.
  const itemsAsync = useAsync(() => loadItems(run.dataset_id).catch((): Items => ({})), [run.dataset_id])
  const items = itemsAsync.data ?? null
  const m = run.metrics ?? ({} as Metrics)
  // Task drives the strip, the scatter and the suspicious-row (percentage-error)
  // tooling, which are regression-only. Ranking runs never reach here — they
  // render through RankingRun — so this resolves to regression vs classifier.
  const isClassification =
    resolveTask(domain, m) === 'binary' || resolveTask(domain, m) === 'multiclass'
  const tableRef = useRef<HTMLDivElement>(null)
  const [sort, setSort] = useState<{ key: PredSort; dir: 1 | -1 }>(() => ({
    key: isClassification ? 'wrongness' : 'abs_err',
    dir: -1,
  }))
  const [selectedRow, setSelectedRow] = useState<number | null>(null)
  const [excludeSuspicious, setExcludeSuspicious] = useState(false)

  const suspicious = useMemo(
    () => (isClassification ? new Set<number>() : suspiciousRowIds(preds.data ?? [])),
    [preds.data, isClassification],
  )
  const filteredMetrics = useMemo(
    () =>
      excludeSuspicious && suspicious.size > 0
        ? computeMetrics((preds.data ?? []).filter((p) => !suspicious.has(p.row_id)))
        : null,
    [excludeSuspicious, preds.data, suspicious],
  )

  const pred = predictors.data?.find((p) => p.name === run.predictor)
  const supportsPredict = !!pred?.predict_args

  const rows: Row[] = useMemo(
    () =>
      (preds.data ?? []).map((p) => {
        const base: Row = {
          ...p,
          err: p.predicted - p.actual,
          pct: (p.predicted - p.actual) / p.actual,
          suspicious: suspicious.has(p.row_id),
        }
        if (!isClassification) return base
        // Multiclass: `predicted` is the argmax class id, `actual` the true id,
        // confidence the mass on the predicted class. Binary: `predicted` is
        // P(class1) (use proba[1] when present); decide at the 0.5 threshold.
        if ((p.proba?.length ?? 0) > 2) {
          const predClass = Math.round(p.predicted)
          const actualClass = Math.round(p.actual)
          return {
            ...base,
            predClass,
            actualClass,
            prob: p.proba?.[predClass] ?? 0,
            correct: predClass === actualClass,
            wrongness: 1 - (p.proba?.[actualClass] ?? 0),
          }
        }
        const p1 = p.proba && p.proba.length >= 2 ? p.proba[1] : p.predicted
        const actualClass = p.actual >= 0.5 ? 1 : 0
        const predClass = p1 >= 0.5 ? 1 : 0
        return {
          ...base,
          predClass,
          actualClass,
          prob: p1,
          correct: predClass === actualClass,
          wrongness: Math.abs(p1 - actualClass),
        }
      }),
    [preds.data, suspicious, isClassification],
  )

  // The scatter identifies points by the numeric row_id; resolve it back to the
  // exact item through each prediction's lossless `rid`.
  const itemByRow = useMemo(() => {
    const map = new Map<number, Items[string] | null>()
    for (const p of preds.data ?? []) map.set(p.row_id, items?.[p.rid] ?? null)
    return map
  }, [preds.data, items])

  const jumpToTable = useCallback((key: PredSort) => {
    setSort({ key, dir: key === 'row_id' ? 1 : -1 })
    tableRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])

  const selectFromScatter = useCallback((rowId: number) => {
    setSelectedRow(rowId)
    tableRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])

  if (!run.metrics) {
    return (
      <p className="muted" role="status">
        Metrics appear when the run finishes evaluating.
      </p>
    )
  }

  return (
    <>
      {/* Metrics strip: every aggregate links to the predictions behind it. */}
      <MetricStrip
        ariaLabel="Test-set metrics; each opens the predictions behind it"
        metrics={[
          ...domain.metrics.columns.map((col) => {
            const v = metricValue(col, m)
            return {
              label: col,
              value: v != null ? metricFormatter(col, domain)(v) : '—',
              onClick: () => jumpToTable(isClassification ? 'predicted' : 'abs_err'),
            }
          }),
          { label: 'test items', value: m.n_test.toLocaleString() },
        ]}
      />

      {suspicious.size > 0 && (
        <div className="suspicious-row">
          <label className="toggle suspicious-toggle">
            <input
              type="checkbox"
              checked={excludeSuspicious}
              onChange={(e) => setExcludeSuspicious(e.target.checked)}
            />
            recompute without <span className="num">{suspicious.size}</span> suspicious{' '}
            {suspicious.size === 1 ? 'prediction' : 'predictions'}
            <span className="hp-hint">
              robust outliers on % error, usually mislabeled {domain.project.entity_noun_plural};
              marked ⚠ in the table
            </span>
          </label>
          {filteredMetrics && (
            <p className="filtered-strip num" role="status">
              filtered: MAE {fmtMoney(filteredMetrics.mae!)} · RMSE {fmtMoney(filteredMetrics.rmse!)} · R²{' '}
              {fmtR2(filteredMetrics.r2!)} · MAPE {fmtPct(filteredMetrics.mape!, 0)} · medAPE{' '}
              {fmtPct(filteredMetrics.medape!)} · n {filteredMetrics.n_test.toLocaleString()}
              <span className="muted"> (official metrics above are unchanged)</span>
            </p>
          )}
        </div>
      )}

      {supportsPredict && <PromotePanel run={run} />}

      <div className="charts-grid">
        <div>
          {isClassification ? (
            <>
              <h2 className="panel-title">Classification outcome</h2>
              <ClassificationSummary predictions={preds.data ?? []} />
            </>
          ) : (
            <>
              <h2 className="panel-title">Predicted vs actual {domain.project.target_noun}</h2>
              <ScatterChart
                predictions={preds.data ?? []}
                onSelect={selectFromScatter}
                selectedRowId={selectedRow}
                label={(rowId) => itemLabel(domain, itemByRow.get(rowId))}
              />
            </>
          )}
        </div>
        <div className="charts-col">
          <div>
            <h2 className="panel-title">Loss</h2>
            <LossChart epochs={events.epochs} />
            {events.epochs.length >= 3 && (
              <>
                <h2 className="panel-title dynamics-title">Is it still learning?</h2>
                <LossDerivativeChart epochs={events.epochs} />
              </>
            )}
          </div>
          {isClassification ? (
            <div>
              <h2 className="panel-title">Predicted probability</h2>
              <ProbabilityHistogram predictions={preds.data ?? []} />
            </div>
          ) : (
            <div>
              <h2 className="panel-title">Error distribution</h2>
              <ErrorHistogram series={[{ label: run.predictor, predictions: preds.data ?? [], style: 'fill' }]} />
            </div>
          )}
        </div>
      </div>

      <RunArch run={run} canFallback={!!pred?.visualization} />

      <div ref={tableRef}>
        <PredictionsTable
          rows={rows}
          isClassification={isClassification}
          items={items}
          loading={preds.loading}
          error={preds.error}
          retry={preds.reload}
          sort={sort}
          setSort={setSort}
          selectedRow={selectedRow}
          setSelectedRow={setSelectedRow}
        />
      </div>
    </>
  )
}

/* ---------------- classification views ----------------
 * Classifier runs have no predicted-vs-actual scalar to scatter. Instead:
 * a confusion / accuracy summary (binary: a 2×2 at threshold 0.5; multiclass:
 * a correct-vs-wrong tally), and a histogram of the model's predicted
 * probability for the positive class. Both read `predicted`/`actual`/`proba`. */

function ClassificationSummary({ predictions }: { predictions: Prediction[] }) {
  const cells = useMemo(() => {
    const n = predictions.length
    if (n === 0) return null
    // Binary when every row carries a 2-vector proba (or none do — fall back to
    // a 0.5 threshold on `predicted` = P(class1)). Multiclass otherwise.
    const multiclass = predictions.some((p) => (p.proba?.length ?? 0) > 2)
    if (multiclass) {
      const correct = predictions.filter((p) => Math.round(p.predicted) === Math.round(p.actual)).length
      return [
        { label: 'correct', value: correct.toLocaleString() },
        { label: 'wrong', value: (n - correct).toLocaleString() },
        { label: 'accuracy', value: `${((correct / n) * 100).toFixed(1)}%` },
      ]
    }
    let tp = 0
    let tn = 0
    let fp = 0
    let fn = 0
    for (const p of predictions) {
      const pos = p.predicted >= 0.5
      const actualPos = p.actual >= 0.5
      if (pos && actualPos) tp++
      else if (pos && !actualPos) fp++
      else if (!pos && actualPos) fn++
      else tn++
    }
    return [
      { label: 'true pos', value: tp.toLocaleString() },
      { label: 'false pos', value: fp.toLocaleString() },
      { label: 'false neg', value: fn.toLocaleString() },
      { label: 'true neg', value: tn.toLocaleString() },
    ]
  }, [predictions])

  if (!cells) return <div className="chart-empty">No predictions</div>
  return (
    <>
      <MetricStrip ariaLabel="Confusion summary at threshold 0.5" metrics={cells} />
      <p className="muted" style={{ marginTop: 'var(--sp-2)' }}>
        Classified at a 0.5 probability threshold. Per-item probabilities are in the table below.
      </p>
    </>
  )
}

/** Histogram of P(positive class) over the test set: `proba[1]` when present,
 *  else `predicted` (which is P(class1) for binary runs). 20 bins over [0,1]. */
function ProbabilityHistogram({ predictions }: { predictions: Prediction[] }) {
  const W = 640
  const H = 200
  const M = { top: 18, right: 16, bottom: 32, left: 44 }
  const N_BINS = 20
  const counts = useMemo(() => {
    const c = new Array<number>(N_BINS).fill(0)
    for (const p of predictions) {
      const v = p.proba && p.proba.length >= 2 ? p.proba[1] : p.predicted
      const i = Math.min(N_BINS - 1, Math.max(0, Math.floor(v * N_BINS)))
      c[i]++
    }
    return c
  }, [predictions])

  if (predictions.length === 0) return <div className="chart-empty">No predictions</div>

  const maxCount = Math.max(1, ...counts)
  const bw = (W - M.left - M.right) / N_BINS
  const yMap = (n: number) => H - M.bottom - (n / maxCount) * (H - M.top - M.bottom)

  return (
    <div className="chart" role="img" aria-label="Distribution of the model's predicted probability for the positive class.">
      <svg viewBox={`0 0 ${W} ${H}`}>
        {[0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line className="grid-line" x1={M.left} x2={W - M.right} y1={yMap(f * maxCount)} y2={yMap(f * maxCount)} />
            <text className="tick-label" x={M.left - 5} y={yMap(f * maxCount) + 3} textAnchor="end">
              {Math.round(f * maxCount)}
            </text>
          </g>
        ))}
        {counts.map((c, i) =>
          c === 0 ? null : (
            <rect
              key={i}
              x={M.left + i * bw + 0.5}
              width={bw - 1}
              y={yMap(c)}
              height={H - M.bottom - yMap(c)}
              fill="var(--accent, var(--ink))"
            />
          ),
        )}
        {/* threshold line at 0.5 */}
        <line x1={M.left + 0.5 * (W - M.left - M.right)} x2={M.left + 0.5 * (W - M.left - M.right)} y1={M.top} y2={H - M.bottom} stroke="var(--hairline-strong)" strokeWidth={1} strokeDasharray="3 3" />
        <line className="axis-line" x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} />
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <text key={t} className="tick-label" x={M.left + t * (W - M.left - M.right)} y={H - M.bottom + 14} textAnchor="middle">
            {t}
          </text>
        ))}
        <text className="axis-title" x={(M.left + W - M.right) / 2} y={H - 4} textAnchor="middle">
          P(positive class)
        </text>
      </svg>
    </div>
  )
}

/* ---------------- promote to model ---------------- */

const NAME_RE = /^[a-z0-9][a-z0-9-]{0,63}$/

function PromotePanel({ run, checkpointOnly = false }: { run: RunMeta; checkpointOnly?: boolean }) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState('')
  const [notes, setNotes] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [promoted, setPromoted] = useState<string | null>(null)

  if (promoted) {
    return (
      <p className="promote-done" role="status">
        Promoted to <ModelRef name={promoted} />. The model is self-contained: it keeps
        predicting even if this run or its dataset is
        deleted.
      </p>
    )
  }

  const nameOk = NAME_RE.test(name)

  const promote = async () => {
    setBusy(true)
    setError(null)
    try {
      const record = await api.promoteModel(name, run.run_id, notes.trim() || undefined)
      setPromoted(record.name)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }

  return (
    <div className="promote-row">
      {!open ? (
        <>
          <button className="btn" onClick={() => setOpen(true)}>
            Promote to model
          </button>
          <span className="muted">
            {checkpointOnly
              ? 'freeze the last saved checkpoint under a name — promoted without final test metrics'
              : 'freeze these weights under a name and predict with them any time'}
          </span>
        </>
      ) : (
        <div className="promote-form">
          <div className="hp-field">
            <label htmlFor="promote-name">Model name</label>
            <input
              id="promote-name"
              type="text"
              className="mono-input"
              placeholder="e.g. mlp-prod-v1"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={busy}
              autoFocus
            />
            {name && !nameOk && (
              <span className="hp-hint">lowercase letters, digits and dashes only</span>
            )}
          </div>
          <div className="hp-field">
            <label htmlFor="promote-notes">Notes (optional)</label>
            <input
              id="promote-notes"
              type="text"
              className="mono-input"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              disabled={busy}
            />
          </div>
          <div className="promote-actions">
            <button className="btn btn-primary" onClick={promote} disabled={!nameOk || busy}>
              {busy ? 'Promoting…' : 'Promote run'}
            </button>
            <button className="btn" onClick={() => setOpen(false)} disabled={busy}>
              Cancel
            </button>
            {error && (
              <span className="hp-error" role="alert">
                {error}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

/* ---------------- predictions table ---------------- */

const PAGE = 100

function PredictionsTable({
  rows,
  isClassification,
  items,
  loading,
  error,
  retry,
  sort,
  setSort,
  selectedRow,
  setSelectedRow,
}: {
  rows: Row[]
  isClassification: boolean
  items: Items | null
  loading: boolean
  error: string | null
  retry: () => void
  sort: { key: PredSort; dir: 1 | -1 }
  setSort: (s: { key: PredSort; dir: 1 | -1 }) => void
  selectedRow: number | null
  setSelectedRow: (id: number | null) => void
}) {
  const domain = useDomain()
  const [filter, setFilter] = useState('')
  const [limit, setLimit] = useState(PAGE)
  const [expanded, setExpanded] = useState<number | null>(null)

  const sorted = useMemo(() => {
    const val = (r: Row): number => {
      switch (sort.key) {
        case 'abs_err':
          return Math.abs(r.err)
        case 'pct_err':
          return Math.abs(r.pct)
        case 'actual':
          return r.actual
        case 'predicted':
          return r.predicted
        case 'row_id':
          return r.row_id
        case 'wrongness':
          return r.wrongness ?? 0
        case 'prob':
          return r.prob ?? 0
      }
    }
    const f = filter.trim().toLowerCase()
    // Match the track, its byline, or the id, so a name search just works.
    const match = (r: Row): boolean => {
      if (r.rid.includes(f)) return true
      const it = items?.[r.rid]
      if (!it) return false
      return (
        itemTitle(domain, it).toLowerCase().includes(f) ||
        itemByline(domain, it).toLowerCase().includes(f)
      )
    }
    const filtered = f === '' ? rows : rows.filter(match)
    return [...filtered].sort((a, b) => (val(a) - val(b)) * sort.dir)
  }, [rows, sort, filter, items, domain])

  // A scatter click selects a row: surface it even if it's deep in the list.
  useEffect(() => {
    if (selectedRow == null) return
    const idx = sorted.findIndex((r) => r.row_id === selectedRow)
    if (idx >= 0 && idx >= limit) setLimit(Math.ceil((idx + 1) / PAGE) * PAGE)
    setExpanded(selectedRow)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRow])

  const visible = sorted.slice(0, limit)

  const th = (key: PredSort, label: string, numCol = true) => (
    <th
      className={numCol ? 'num-col' : ''}
      aria-sort={sort.key === key ? (sort.dir === 1 ? 'ascending' : 'descending') : undefined}
    >
      <button
        className="sort"
        onClick={() => setSort({ key, dir: sort.key === key ? ((-sort.dir) as 1 | -1) : -1 })}
        aria-label={`Sort by ${label}`}
      >
        {label}
        {sort.key === key && <span className="sort-arrow">{sort.dir === 1 ? '▲' : '▼'}</span>}
      </button>
    </th>
  )

  if (error) {
    return (
      <div className="error-block" role="alert">
        Could not load predictions: {error}{' '}
        <button className="btn" onClick={retry}>
          Retry
        </button>
      </div>
    )
  }

  return (
    <section aria-label="Individual predictions">
      <div className="pred-head">
        <h2 className="panel-title">
          Predictions{' '}
          <span className="muted">
            ({sorted.length.toLocaleString()} test {domain.project.entity_noun_plural}, worst first)
          </span>
        </h2>
        <input
          type="text"
          className="mono-input pred-filter"
          placeholder={`Filter ${domain.project.entity_noun_plural}…`}
          aria-label={`Filter predictions by ${domain.project.entity_noun} or id`}
          value={filter}
          onChange={(e) => {
            setFilter(e.target.value)
            setLimit(PAGE)
          }}
        />
      </div>
      {loading ? (
        <div className="skeleton skeleton-md" />
      ) : (
        <table className="pred-table">
          <thead>
            <tr>
              {th('row_id', cap(domain.project.entity_noun), false)}
              {th('actual', 'Actual')}
              {th('predicted', 'Predicted')}
              {isClassification ? (
                <>
                  {th('prob', domain.metrics.value_unit)}
                  {th('wrongness', 'Outcome')}
                </>
              ) : (
                <>
                  {th('abs_err', 'Error')}
                  {th('pct_err', '% error')}
                </>
              )}
              <th>
                <span className="sr-only">Details</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {visible.map((r) => (
              <PredRow
                key={r.rid}
                row={r}
                isClassification={isClassification}
                item={items?.[r.rid] ?? null}
                expanded={expanded === r.row_id}
                highlighted={selectedRow === r.row_id}
                onToggle={() => {
                  setExpanded(expanded === r.row_id ? null : r.row_id)
                  setSelectedRow(expanded === r.row_id ? null : r.row_id)
                }}
              />
            ))}
          </tbody>
        </table>
      )}
      {!loading && limit < sorted.length && (
        <button className="btn show-more" onClick={() => setLimit((l) => l + PAGE)}>
          Show {Math.min(PAGE, sorted.length - limit)} more of {(sorted.length - limit).toLocaleString()}
        </button>
      )}
      {!loading && sorted.length === 0 && filter && (
        <p className="muted">No {domain.project.entity_noun_plural} match “{filter}”.</p>
      )}
    </section>
  )
}

function PredRow({
  row,
  isClassification,
  item,
  expanded,
  highlighted,
  onToggle,
}: {
  row: Row
  isClassification: boolean
  item: Items[string] | null
  expanded: boolean
  highlighted: boolean
  onToggle: () => void
}) {
  const domain = useDomain()
  return (
    <>
      <tr
        className={`pred-row${highlighted ? ' is-selected' : ''}`}
        onClick={onToggle}
        aria-expanded={expanded}
      >
        <td className="track-cell">
          <ItemIdentity item={item} rowId={row.rid} />
        </td>
        {isClassification ? (
          <>
            <td className="num-col num">
              <span className={`cls-chip cls-${row.actualClass}`}>{row.actualClass}</span>
            </td>
            <td className="num-col num">
              <span className={`cls-chip cls-${row.predClass}`}>{row.predClass}</span>
            </td>
            <td className="num-col num">{(row.prob ?? 0).toFixed(3)}</td>
            <td className="num-col">
              {row.correct ? (
                <span className="cls-outcome cls-ok" title="Correct">✓</span>
              ) : (
                <span className="cls-outcome cls-bad" title="Misclassified">✗</span>
              )}
            </td>
          </>
        ) : (
          <>
            <td className="num-col num">{fmtMoney(row.actual)}</td>
            <td className="num-col num">{fmtMoney(row.predicted)}</td>
            <td className="num-col num">{fmtMoneyDelta(row.err)}</td>
            <td className={`num-col num pct-cell ${Math.abs(row.pct) >= 0.5 ? 'pct-large' : ''}`}>
              {fmtSignedPct(row.pct)}
              {row.suspicious && (
                <span
                  className="sus-glyph"
                  title={`Suspicious: robust outlier on % error, likely a mislabeled ${domain.project.entity_noun}`}
                >
                  {' '}
                  ⚠
                </span>
              )}
            </td>
          </>
        )}
        <td className="expand-cell" aria-hidden>
          {expanded ? '▾' : '▸'}
        </td>
      </tr>
      {expanded && (
        <tr className="pred-detail">
          <td colSpan={6}>
            {item === null ? (
              <div className="skeleton skeleton-row" />
            ) : (
              <ItemMeta item={item} />
            )}
          </td>
        </tr>
      )}
    </>
  )
}

/* ---------------- ranking (next-item) run ----------------
 * A next-item ranker has no scalar prediction to scatter: each test query is a
 * session whose held-out next item the model tries to surface in its top-K. We
 * show the metrics strip (recall@k / mrr / hit@k), a hit-rank distribution
 * (where in the top-K the true item landed, or a miss), and a per-query table
 * of the ranked list with the true next item highlighted. Every id here indexes
 * items.json (the item vocabulary), not the corpus point ids the pointwise
 * views join by. */

type RankItem = Items[string]

interface RankRow {
  /** Test-session index (the query). */
  row_id: number
  /** Held-out true next-item index into items.json. */
  actualIdx: number
  /** The model's top-1 item index. */
  predictedIdx: number
  /** Ordered top-K item indices, best-first. */
  topK: number[]
  /** 1-based position of the true item in the top-K, or null when it's a miss. */
  hitRank: number | null
}

type RankSort = 'hit_rank' | 'row_id'

function RankingRun({ run, events }: { run: RunMeta; events: ReturnType<typeof useRunEvents> }) {
  const domain = useDomain()
  const preds = useAsync(() => api.getPredictions(run.run_id), [run.run_id])
  const predictors = useAsync(() => api.listPredictors(), [])
  const itemsAsync = useAsync(() => loadItems(run.dataset_id).catch((): Items => ({})), [run.dataset_id])
  const items = itemsAsync.data ?? null
  const m = run.metrics ?? ({} as Metrics)

  const pred = predictors.data?.find((p) => p.name === run.predictor)
  const supportsPredict = !!pred?.predict_args

  const tableRef = useRef<HTMLDivElement>(null)
  const [sort, setSort] = useState<{ key: RankSort; dir: 1 | -1 }>({ key: 'hit_rank', dir: 1 })

  const rows: RankRow[] = useMemo(
    () =>
      (preds.data ?? []).map((p) => {
        const topK = p.top_k_ids ?? []
        const actualIdx = Math.round(p.actual)
        const pos = topK.indexOf(actualIdx)
        return {
          row_id: p.row_id,
          actualIdx,
          predictedIdx: Math.round(p.predicted),
          topK,
          hitRank: pos >= 0 ? pos + 1 : null,
        }
      }),
    [preds.data],
  )

  // Cutoff K: the widest top-K the predictor handed back; 10 when none loaded.
  const kCut = useMemo(() => rows.reduce((k, r) => Math.max(k, r.topK.length), 0) || 10, [rows])
  const hits = useMemo(() => rows.filter((r) => r.hitRank != null).length, [rows])

  const jumpToTable = useCallback(() => {
    setSort({ key: 'hit_rank', dir: 1 })
    tableRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])

  if (!run.metrics) {
    return (
      <p className="muted" role="status">
        Metrics appear when the run finishes evaluating.
      </p>
    )
  }

  return (
    <>
      <MetricStrip
        ariaLabel="Test-set ranking metrics; each opens the ranked predictions behind it"
        metrics={[
          ...domain.metrics.columns.map((col) => {
            const v = metricValue(col, m)
            return {
              label: col,
              value: v != null ? metricFormatter(col, domain)(v) : '—',
              onClick: jumpToTable,
            }
          }),
          { label: 'test queries', value: m.n_test.toLocaleString() },
        ]}
      />

      {supportsPredict && <PromotePanel run={run} />}

      <div className="charts-grid">
        <div>
          <h2 className="panel-title">Hit-rank distribution</h2>
          <HitRankHistogram rows={rows} kCut={kCut} />
          <p className="muted rank-blurb">
            Where the held-out next {domain.project.entity_noun} landed in each query&rsquo;s top-
            {kCut}; a <span className="rank-miss-word">miss</span> is a query whose true next{' '}
            {domain.project.entity_noun} never appeared.
          </p>
        </div>
        <div className="charts-col">
          <div>
            <h2 className="panel-title">Ranking outcome</h2>
            <RankingSummary hits={hits} n={rows.length} kCut={kCut} entityNoun={domain.project.entity_noun} />
          </div>
          {events.epochs.length > 0 && (
            <div>
              <h2 className="panel-title">Loss</h2>
              <LossChart epochs={events.epochs} />
            </div>
          )}
        </div>
      </div>

      <RunArch run={run} canFallback={!!pred?.visualization} />

      <div ref={tableRef}>
        <RankedPredictionsTable
          rows={rows}
          items={items}
          kCut={kCut}
          loading={preds.loading}
          error={preds.error}
          retry={preds.reload}
          sort={sort}
          setSort={setSort}
        />
      </div>
    </>
  )
}

/** Hits/misses tally for the ranking run, in the shared MetricStrip vocabulary. */
function RankingSummary({
  hits,
  n,
  kCut,
  entityNoun,
}: {
  hits: number
  n: number
  kCut: number
  entityNoun: string
}) {
  if (n === 0) return <div className="chart-empty">No predictions</div>
  const pct = (hits / n) * 100
  return (
    <>
      <MetricStrip
        ariaLabel={`Hits within the top ${kCut}`}
        metrics={[
          { label: `hits @${kCut}`, value: hits.toLocaleString() },
          { label: 'misses', value: (n - hits).toLocaleString() },
          { label: 'hit rate', value: `${pct.toFixed(1)}%` },
        ]}
      />
      <p className="muted" style={{ marginTop: 'var(--sp-2)' }}>
        A hit is a test query whose held-out next {entityNoun} the model ranked inside its top-{kCut}.
      </p>
    </>
  )
}

/** Discrete distribution of the true item's rank across queries: one bar per
 *  rank 1…K, plus a final quiet "miss" bar. Mirrors ProbabilityHistogram's SVG
 *  idiom (same axes, grid, tokens). */
function HitRankHistogram({ rows, kCut }: { rows: RankRow[]; kCut: number }) {
  const W = 640
  const H = 200
  const M = { top: 18, right: 16, bottom: 32, left: 44 }
  const counts = useMemo(() => {
    // index 0…kCut-1 → rank 1…kCut; the final index tallies misses.
    const c = new Array<number>(kCut + 1).fill(0)
    for (const r of rows) {
      if (r.hitRank != null && r.hitRank >= 1 && r.hitRank <= kCut) c[r.hitRank - 1]++
      else c[kCut]++
    }
    return c
  }, [rows, kCut])

  if (rows.length === 0) return <div className="chart-empty">No predictions</div>

  const nBars = kCut + 1
  const maxCount = Math.max(1, ...counts)
  const bw = (W - M.left - M.right) / nBars
  const yMap = (v: number) => H - M.bottom - (v / maxCount) * (H - M.top - M.bottom)
  const label = (i: number) => (i < kCut ? String(i + 1) : 'miss')

  return (
    <div
      className="chart"
      role="img"
      aria-label={`Rank at which the held-out next item appeared across test queries, over the top ${kCut}; the last bar counts misses.`}
    >
      <svg viewBox={`0 0 ${W} ${H}`}>
        {[0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line className="grid-line" x1={M.left} x2={W - M.right} y1={yMap(f * maxCount)} y2={yMap(f * maxCount)} />
            <text className="tick-label" x={M.left - 5} y={yMap(f * maxCount) + 3} textAnchor="end">
              {Math.round(f * maxCount).toLocaleString()}
            </text>
          </g>
        ))}
        {counts.map((c, i) =>
          c === 0 ? null : (
            <rect
              key={i}
              x={M.left + i * bw + 0.5}
              width={Math.max(0.5, bw - 1)}
              y={yMap(c)}
              height={H - M.bottom - yMap(c)}
              fill={i < kCut ? 'var(--series-a)' : 'var(--ink-faint)'}
              opacity={i < kCut ? 0.85 : 0.5}
            />
          ),
        )}
        <line className="axis-line" x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} />
        {counts.map((_, i) =>
          nBars <= 12 || i % 2 === 0 || i === kCut ? (
            <text key={i} className="tick-label" x={M.left + i * bw + bw / 2} y={H - M.bottom + 14} textAnchor="middle">
              {label(i)}
            </text>
          ) : null,
        )}
        <text className="axis-title" x={W - M.right} y={H - 4} textAnchor="end">
          hit rank
        </text>
      </svg>
    </div>
  )
}

const RANK_PAGE = 100

function RankedPredictionsTable({
  rows,
  items,
  kCut,
  loading,
  error,
  retry,
  sort,
  setSort,
}: {
  rows: RankRow[]
  items: Items | null
  kCut: number
  loading: boolean
  error: string | null
  retry: () => void
  sort: { key: RankSort; dir: 1 | -1 }
  setSort: (s: { key: RankSort; dir: 1 | -1 }) => void
}) {
  const domain = useDomain()
  const [filter, setFilter] = useState('')
  const [limit, setLimit] = useState(RANK_PAGE)
  const [expanded, setExpanded] = useState<number | null>(null)

  const lookup = useCallback(
    (idx: number): RankItem | null => sequenceItemForDisplay(domain, items?.[String(idx)] ?? null),
    [items, domain],
  )

  const sorted = useMemo(() => {
    // Misses sort after every hit (rank K+1), so a hit_rank-ascending sort reads
    // best calls first, misses last.
    const val = (r: RankRow) => (sort.key === 'row_id' ? r.row_id : r.hitRank ?? kCut + 1)
    const f = filter.trim().toLowerCase()
    const match = (r: RankRow): boolean => {
      const it = lookup(r.actualIdx)
      if (!it) return String(r.row_id).includes(f)
      return (
        itemTitle(domain, it).toLowerCase().includes(f) ||
        itemByline(domain, it).toLowerCase().includes(f)
      )
    }
    const filtered = f === '' ? rows : rows.filter(match)
    return [...filtered].sort((a, b) => (val(a) - val(b)) * sort.dir)
  }, [rows, sort, filter, lookup, domain, kCut])

  const visible = sorted.slice(0, limit)

  const th = (key: RankSort, label: string, numCol = true) => (
    <th
      className={numCol ? 'num-col' : ''}
      aria-sort={sort.key === key ? (sort.dir === 1 ? 'ascending' : 'descending') : undefined}
    >
      <button
        className="sort"
        onClick={() => setSort({ key, dir: sort.key === key ? ((-sort.dir) as 1 | -1) : 1 })}
        aria-label={`Sort by ${label}`}
      >
        {label}
        {sort.key === key && <span className="sort-arrow">{sort.dir === 1 ? '▲' : '▼'}</span>}
      </button>
    </th>
  )

  if (error) {
    return (
      <div className="error-block" role="alert">
        Could not load predictions: {error}{' '}
        <button className="btn" onClick={retry}>
          Retry
        </button>
      </div>
    )
  }

  return (
    <section aria-label="Per-query ranked predictions">
      <div className="pred-head">
        <h2 className="panel-title">
          Ranked predictions{' '}
          <span className="muted">({sorted.length.toLocaleString()} test queries, best rank first)</span>
        </h2>
        <input
          type="text"
          className="mono-input pred-filter"
          placeholder={`Filter by true next ${domain.project.entity_noun}…`}
          aria-label={`Filter queries by the true next ${domain.project.entity_noun} or query id`}
          value={filter}
          onChange={(e) => {
            setFilter(e.target.value)
            setLimit(RANK_PAGE)
          }}
        />
      </div>
      {loading ? (
        <div className="skeleton skeleton-md" />
      ) : (
        <table className="pred-table">
          <thead>
            <tr>
              {th('row_id', 'Query', false)}
              <th>True next {domain.project.entity_noun}</th>
              <th>Top-1</th>
              {th('hit_rank', `Hit rank ≤${kCut}`)}
              <th>
                <span className="sr-only">Details</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {visible.map((r) => (
              <RankedRow
                key={r.row_id}
                row={r}
                kCut={kCut}
                lookup={lookup}
                expanded={expanded === r.row_id}
                onToggle={() => setExpanded(expanded === r.row_id ? null : r.row_id)}
              />
            ))}
          </tbody>
        </table>
      )}
      {!loading && limit < sorted.length && (
        <button className="btn show-more" onClick={() => setLimit((l) => l + RANK_PAGE)}>
          Show {Math.min(RANK_PAGE, sorted.length - limit)} more of{' '}
          {(sorted.length - limit).toLocaleString()}
        </button>
      )}
      {!loading && sorted.length === 0 && filter && (
        <p className="muted">No queries match “{filter}”.</p>
      )}
      {!loading && rows.length === 0 && !filter && (
        <p className="muted">This run recorded no per-query predictions.</p>
      )}
    </section>
  )
}

function RankedRow({
  row,
  kCut,
  lookup,
  expanded,
  onToggle,
}: {
  row: RankRow
  kCut: number
  lookup: (idx: number) => RankItem | null
  expanded: boolean
  onToggle: () => void
}) {
  const hit = row.hitRank != null
  return (
    <>
      <tr className="pred-row" onClick={onToggle} aria-expanded={expanded}>
        <td className="num">{row.row_id}</td>
        <td className="track-cell">
          <ItemIdentity item={lookup(row.actualIdx)} rowId={row.actualIdx} />
        </td>
        <td className="track-cell">
          <ItemIdentity item={lookup(row.predictedIdx)} rowId={row.predictedIdx} />
        </td>
        <td className="num-col">
          <span className={`rank-badge${hit ? '' : ' is-miss'}`}>{hit ? `#${row.hitRank}` : 'miss'}</span>
        </td>
        <td className="expand-cell" aria-hidden>
          {expanded ? '▾' : '▸'}
        </td>
      </tr>
      {expanded && (
        <tr className="pred-detail">
          <td colSpan={5}>
            <RankedList row={row} kCut={kCut} lookup={lookup} />
          </td>
        </tr>
      )}
    </>
  )
}

/** The model's full ranked top-K for one query, truth highlighted; a rank bar
 *  (mirroring the Pathfinder fit bar) reads the ordering at a glance. */
function RankedList({
  row,
  kCut,
  lookup,
}: {
  row: RankRow
  kCut: number
  lookup: (idx: number) => RankItem | null
}) {
  const n = row.topK.length || 1
  return (
    <div className="rank-list-wrap">
      <ol className="rank-list">
        {row.topK.map((idx, i) => {
          const isTruth = idx === row.actualIdx
          const w = Math.round(((n - i) / n) * 100)
          return (
            <li key={`${idx}-${i}`} className={`rank-item${isTruth ? ' is-truth' : ''}`}>
              <span className="rank-item-n num">{i + 1}</span>
              <span className="rank-item-track">
                <ItemIdentity item={lookup(idx)} rowId={idx} />
              </span>
              <span className="rank-bar" aria-hidden>
                <span className="rank-bar-fill" style={{ width: `${w}%` }} />
              </span>
              {isTruth && <span className="rank-truth-tag">true next</span>}
            </li>
          )
        })}
      </ol>
      {row.hitRank == null && (
        <p className="rank-miss-note">
          Held-out next item is not in the top-{kCut}:{' '}
          <span className="rank-miss-item">
            <ItemIdentity item={lookup(row.actualIdx)} rowId={row.actualIdx} />
          </span>
        </p>
      )}
    </div>
  )
}

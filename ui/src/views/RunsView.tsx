import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { RunMeta } from '../api/types'
import DensityToggle from '../components/DensityToggle'
import LatticeMark from '../components/LatticeMark'
import ViewHeader from '../components/ViewHeader'
import { useDensity } from '../hooks/useDensity'
import { DatasetRef, PredictorRef, RunRef } from '../components/EntityRef'
import StatusBadge from '../components/StatusBadge'
import Sparkline from '../components/Sparkline'
import { useAsync } from '../hooks/useAsync'
import { useRunEvents } from '../hooks/useRunEvents'
import { useDomain } from '../lib/DomainContext'
import { fmtDateTime, fmtDuration, fmtStamp, shortRunId, tinyRunId } from '../lib/format'
import { isTargetUnitMetric, metricCellFormatter, metricFormatter, metricKey, metricValue, type MetricKey } from '../lib/metrics'
import type { Domain } from '../lib/domain'
import './runs.css'

/* Browsers cap concurrent HTTP/1.1 connections per origin (~6); each live
 * cell holds an SSE stream open. Past this cap the streams starve each other
 * and every other fetch, so extra running rows show a quiet badge and rely
 * on the background poll instead. */
const MAX_LIVE_CELLS = 4
const POLL_MS = 5000

/** Sort keys: the fixed meta columns, plus any metric key (mae/auc/…). */
type MetaSortKey = 'started_at' | 'predictor' | 'dataset_id' | 'status'
type SortKey = MetaSortKey | MetricKey

interface Sort {
  key: SortKey
  dir: 1 | -1
}

const META_KEYS: MetaSortKey[] = ['started_at', 'predictor', 'dataset_id', 'status']
const isMetaKey = (k: SortKey): k is MetaSortKey => (META_KEYS as string[]).includes(k)

function compare(a: RunMeta, b: RunMeta, { key, dir }: Sort): number {
  let av: string | number | null
  let bv: string | number | null
  if (isMetaKey(key)) {
    av = a[key]
    bv = b[key]
  } else {
    av = a.metrics?.[key] ?? null
    bv = b.metrics?.[key] ?? null
  }
  if (av === null && bv === null) return 0
  if (av === null) return 1 // metric-less runs sink regardless of direction
  if (bv === null) return -1
  if (av < bv) return -dir
  if (av > bv) return dir
  return 0
}

/* Representative column widths for the loading state (select, run, predictor,
   dataset, status, then one per metric column, time, started, actions). */
function skeletonCols(nMetrics: number): string[] {
  return [
    '1.5ch',
    '10ch',
    '8ch',
    '14ch',
    '8ch',
    ...Array.from({ length: nMetrics }, () => '6ch'),
    '6ch',
    '13ch',
    '3ch',
  ]
}

/** The default per-direction for a sort column: time sorts newest-first,
 *  metric columns sort to put the best on top (ascending for error metrics,
 *  descending for scores), everything else ascending. */
function defaultDir(key: SortKey): 1 | -1 {
  if (key === 'started_at') return -1
  if (!isMetaKey(key)) {
    const err = ['mae', 'rmse', 'mape', 'medape', 'logloss', 'brier']
    return err.includes(key) ? 1 : -1
  }
  return 1
}

export default function RunsView() {
  const domain = useDomain()
  useEffect(() => {
    document.title = `Runs · ${domain.project.title}`
  }, [domain.project.title])
  const navigate = useNavigate()
  // Metric columns and their keys are domain-driven (regression: MAE/RMSE/R²;
  // classification: AUC/logloss/accuracy), so the table follows the task.
  const metricCols = domain.metrics.columns
  const { data: runs, error, loading, reload } = useAsync(() => api.listRuns(), [])
  const [sort, setSort] = useState<Sort>({ key: 'started_at', dir: -1 })
  const [selected, setSelected] = useState<string[]>([])
  const [focusIdx, setFocusIdx] = useState(0)
  const tableRef = useRef<HTMLTableElement>(null)
  const [density, toggleDensity] = useDensity()

  const sorted = useMemo(() => (runs ? [...runs].sort((a, b) => compare(a, b, sort)) : []), [runs, sort])

  // Live data flows two ways: the first few running rows hold SSE streams,
  // everything else (and every finish) is caught by a slow background poll.
  // The poll updates in place — data stays on screen, no skeleton flash.
  const liveSet = useMemo(() => {
    const ids = sorted.filter((r) => r.status === 'running').map((r) => r.run_id)
    return new Set(ids.slice(0, MAX_LIVE_CELLS))
  }, [sorted])
  const anyRunning = useMemo(() => (runs ?? []).some((r) => r.status === 'running'), [runs])
  useEffect(() => {
    if (!anyRunning) return
    const t = window.setInterval(reload, POLL_MS)
    return () => window.clearInterval(t)
  }, [anyRunning, reload])

  const toggleSort = (key: SortKey) =>
    setSort((s) => ({ key, dir: s.key === key ? ((-s.dir) as 1 | -1) : defaultDir(key) }))

  const toggleSelect = useCallback((runId: string) => {
    setSelected((sel) =>
      sel.includes(runId) ? sel.filter((s) => s !== runId) : sel.length >= 2 ? sel : [...sel, runId],
    )
  }, [])

  // Keyboard: j/k move, x select, Enter open, c compare.
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (sorted.length === 0) return
    if (e.key === 'j' || e.key === 'ArrowDown') {
      e.preventDefault()
      setFocusIdx((i) => Math.min(i + 1, sorted.length - 1))
    } else if (e.key === 'k' || e.key === 'ArrowUp') {
      e.preventDefault()
      setFocusIdx((i) => Math.max(i - 1, 0))
    } else if (e.key === 'x') {
      toggleSelect(sorted[focusIdx].run_id)
    } else if (e.key === 'Enter') {
      navigate(`/runs/${sorted[focusIdx].run_id}`)
    } else if (e.key === 'c' && selected.length === 2) {
      navigate(`/compare?a=${selected[0]}&b=${selected[1]}`)
    }
  }

  useEffect(() => {
    const rows = tableRef.current?.querySelectorAll('tbody tr')
    rows?.[focusIdx]?.scrollIntoView({ block: 'nearest' })
  }, [focusIdx])

  const sortBtn = (key: SortKey, label: string) => (
    <button className="sort" onClick={() => toggleSort(key)} aria-label={`Sort by ${label}`}>
      {label}
      {sort.key === key && <span className="sort-arrow">{sort.dir === 1 ? '▲' : '▼'}</span>}
    </button>
  )

  return (
    <section aria-label="Training runs">
      <ViewHeader
        glyph="run"
        title="Runs"
        count={runs && !error ? sorted.length : undefined}
        meta={
          <span>
            Select two runs to compare them. Keys: <kbd>j</kbd>/<kbd>k</kbd> move, <kbd>x</kbd>{' '}
            select, <kbd>Enter</kbd> open, <kbd>c</kbd> compare.
          </span>
        }
        actions={<DensityToggle density={density} onToggle={toggleDensity} />}
      />
      <div className="view-body">
        {error ? (
          <div className="error-block" role="alert">
            Could not load runs: {error}.{' '}
            <button className="btn" onClick={reload}>
              Retry
            </button>
          </div>
        ) : !loading && sorted.length === 0 ? (
          <EmptyRuns
            targetNoun={domain.project.target_noun}
            entityNounPl={domain.project.entity_noun_plural}
          />
        ) : (
          <>
      <table
        ref={tableRef}
        className="runs-table"
        data-density={density}
        onKeyDown={onKeyDown}
        tabIndex={0}
      >
        <thead>
          <tr>
            <th>
              <span className="sr-only">Select for comparison</span>
            </th>
            <th>Run</th>
            <th aria-sort={sort.key === 'predictor' ? (sort.dir === 1 ? 'ascending' : 'descending') : undefined}>
              {sortBtn('predictor', 'Predictor')}
            </th>
            <th>{sortBtn('dataset_id', 'Dataset')}</th>
            <th>{sortBtn('status', 'Status')}</th>
            {metricCols.map((col, i) => {
              const key = metricKey(col)
              return (
                <th
                  key={col}
                  className={`num-col${i === 0 ? ' col-group-start' : ''}`}
                  aria-sort={sort.key === key ? (sort.dir === 1 ? 'ascending' : 'descending') : undefined}
                >
                  {sortBtn(key, col)}
                </th>
              )
            })}
            <th className="num-col col-group-start">Time</th>
            <th>{sortBtn('started_at', 'Started')}</th>
            <th>
              <span className="sr-only">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {loading && !runs
            ? Array.from({ length: 4 }, (_, i) => (
                // First load previews the table shape: one skeleton per
                // column. Reloads update in place and never come back here.
                <tr key={i}>
                  {skeletonCols(metricCols.length).map((w, j) => (
                    <td key={j}>
                      <div className="skeleton skeleton-xs" style={{ width: w }} />
                    </td>
                  ))}
                </tr>
              ))
            : sorted.map((r, i) => (
                <RunRow
                  key={r.run_id}
                  run={r}
                  metricCols={metricCols}
                  domain={domain}
                  liveCell={liveSet.has(r.run_id)}
                  focused={i === focusIdx}
                  selected={selected.includes(r.run_id)}
                  selectionFull={selected.length >= 2}
                  onToggle={() => toggleSelect(r.run_id)}
                  onFocusRow={() => setFocusIdx(i)}
                  onFinished={reload}
                />
              ))}
        </tbody>
      </table>

      {selected.length > 0 && (
        <div className="compare-bar" role="region" aria-label="Comparison selection">
          <span className="num">{selected.map(shortRunId).join('  vs  ')}</span>
          {selected.length === 2 ? (
            <Link className="btn btn-primary" to={`/compare?a=${selected[0]}&b=${selected[1]}`}>
              Compare runs
            </Link>
          ) : (
            <span className="compare-bar-hint">Select one more run to compare</span>
          )}
          <button className="btn-on-chrome" onClick={() => setSelected([])}>
            Clear selection
          </button>
        </div>
      )}
          </>
        )}
      </div>
    </section>
  )
}

function RunRow({
  run,
  metricCols,
  domain,
  liveCell,
  focused,
  selected,
  selectionFull,
  onToggle,
  onFocusRow,
  onFinished,
}: {
  run: RunMeta
  /** Domain-driven metric column display names, in order. */
  metricCols: string[]
  domain: Domain
  /** Holds an SSE stream; quiet running rows rely on the poll instead. */
  liveCell: boolean
  focused: boolean
  selected: boolean
  selectionFull: boolean
  onToggle: () => void
  onFocusRow: () => void
  onFinished: () => void
}) {
  const navigate = useNavigate()
  const m = run.metrics
  return (
    <tr
      className={`run-row${selected ? ' is-selected' : ''}${focused ? ' is-focused' : ''}`}
      onClick={(e) => {
        if ((e.target as HTMLElement).closest('input,a,button')) return
        onFocusRow()
        navigate(`/runs/${run.run_id}`)
      }}
    >
      <td>
        <input
          type="checkbox"
          checked={selected}
          disabled={!selected && selectionFull}
          onChange={onToggle}
          aria-label={`Select ${shortRunId(run.run_id)} for comparison`}
        />
      </td>
      <td>
        {/* Predictor has its own column; the tiny id avoids repeating it. */}
        <RunRef id={run.run_id} label={tinyRunId(run.run_id)} />
      </td>
      <td>
        <PredictorRef name={run.predictor} impl />
      </td>
      <td className="dataset-cell">
        <DatasetRef id={run.dataset_id} />
      </td>
      <td>
        {run.status === 'running' ? (
          liveCell ? (
            <LiveStatusCell runId={run.run_id} onFinished={onFinished} />
          ) : (
            <QuietRunningCell runId={run.run_id} />
          )
        ) : (
          <StatusBadge status={run.status} />
        )}
      </td>
      {metricCols.map((col, i) => {
        const v = metricValue(col, m)
        const text = v != null ? metricCellFormatter(col, domain)(v) : '—'
        // Target-unit cells (MAE/RMSE) may be compacted; the title carries the
        // full-precision value, matching the prior regression behaviour.
        const title = v != null && isTargetUnitMetric(col, domain) ? metricFormatter(col, domain)(v) : undefined
        return (
          <td key={col} className={`num-col num${i === 0 ? ' col-group-start' : ''}`} title={title}>
            {text}
          </td>
        )
      })}
      <td className="num-col num col-group-start">{fmtDuration(run.started_at, run.finished_at)}</td>
      <td className="num" title={fmtDateTime(run.started_at)}>
        {fmtStamp(run.started_at)}
      </td>
      <td>
        {run.status !== 'running' && (
          <button
            className="btn-inline"
            title="Delete this run's directory (models promoted from it are untouched)"
            onClick={() => {
              if (!window.confirm(`Delete run ${shortRunId(run.run_id)}?`)) return
              void api.deleteRun(run.run_id).then(onFinished, (e) => {
                window.alert(e instanceof Error ? e.message : String(e))
                onFinished() // a 409 means the status flipped under us: refresh
              })
            }}
          >
            delete
          </button>
        )}
      </td>
    </tr>
  )
}

/** Running row past the SSE budget: pulsing badge + stop action, no stream.
 *  Progress lands via the background poll; the run view has the full feed. */
function QuietRunningCell({ runId }: { runId: string }) {
  const [stopRequested, setStopRequested] = useState(false)
  return (
    <span className="status status-running live-cell">
      <span className="dot" aria-hidden>
        ●
      </span>
      running
      {stopRequested ? (
        <span className="muted">stopping…</span>
      ) : (
        <button
          className="btn-inline"
          title="Stop after this epoch: the predictor evaluates and saves a promotable model"
          onClick={() => {
            void api.stopRun(runId, false).then(
              () => setStopRequested(true),
              () => setStopRequested(false),
            )
          }}
        >
          stop
        </button>
      )}
    </span>
  )
}

/** Live progress for a running row: epoch counter + val-loss sparkline,
 *  plus an inline graceful-stop action (force stop lives in the run view). */
function LiveStatusCell({ runId, onFinished }: { runId: string; onFinished: () => void }) {
  const { epochs, terminal, stopping } = useRunEvents(runId)
  const [stopRequested, setStopRequested] = useState(false)
  useEffect(() => {
    if (terminal) onFinished()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [terminal])
  const last = epochs[epochs.length - 1]
  return (
    <span className="status status-running live-cell">
      <span className="dot" aria-hidden>
        ●
      </span>
      {last ? (
        <>
          <span className="num">
            {last.epoch}/{last.total}
          </span>
          <Sparkline values={epochs.map((e) => e.val)} />
        </>
      ) : (
        'running'
      )}
      {stopping || stopRequested ? (
        <span className="muted">stopping…</span>
      ) : (
        <button
          className="btn-inline"
          title="Stop after this epoch: the predictor evaluates and saves a promotable model"
          onClick={() => {
            void api.stopRun(runId, false).then(
              () => setStopRequested(true),
              () => setStopRequested(false),
            )
          }}
        >
          stop
        </button>
      )}
    </span>
  )
}

function EmptyRuns({
  targetNoun,
  entityNounPl,
}: {
  targetNoun: string
  entityNounPl: string
}) {
  return (
    <section className="empty-state" aria-label="No runs yet">
      <LatticeMark />
      <h1>No runs yet</h1>
      <p>
        This bench trains models that predict a track’s <strong>{targetNoun}</strong> — how well
        it fits your listening habits — from its co-listening embedding. That score is what the
        playlist pathfinder uses to rank candidate tracks when building a sequence between two
        songs. The loop:
      </p>
      <ol className="empty-steps">
        <li>
          <strong>Build a dataset</strong> — fetch {entityNounPl} from Qdrant, reduce the
          co-listening embeddings with PCA, freeze a train/test split.
        </li>
        <li>
          <strong>Pick a predictor</strong> — an MLP, a CNN, ridge regression, the median
          baseline, or any executable implementing the contract.
        </li>
        <li>
          <strong>Train and inspect</strong> — watch loss live, then drill from metrics into
          the individual {targetNoun} predictions behind them. Promote the best as the model
          the pathfinder scores with.
        </li>
      </ol>
      <Link to="/new" className="btn btn-primary">
        Set up first run
      </Link>
    </section>
  )
}

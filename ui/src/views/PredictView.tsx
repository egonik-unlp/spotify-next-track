import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import type { CorpusEntry, GroupPredictResponse } from '../api/types'
import ViewHeader from '../components/ViewHeader'
import { useAsync } from '../hooks/useAsync'
import { useDomain } from '../lib/DomainContext'
import { type Domain } from '../lib/domain'
import { cap } from '../lib/format'
import { fmtSignedPct, fmtTarget, fmtTargetCell } from '../lib/format'
import { resolvePredictGroup, type PredictGroup } from '../lib/listingPredict'
import './models.css'
import './predict.css'

/* Single-instance prediction, adapted to a behavioural-embedding domain.
 * The model's input vector is a co-listening embedding — it cannot be made
 * from typed text, it only exists for tracks already in the corpus. So an
 * "instance" is a reference: pick a corpus track and predict it by id
 * (predicted-vs-actual when the track is held out), or blend several tracks'
 * vectors into one synthetic query ("tracks like these"). */

type Mode = 'pick' | 'blend'

interface Consensus {
  median: number
  n_ok: number
  n_models: number
  members: { name: string; rank?: number; predicted: number }[]
  warnings: string[]
}

/** Run the best-model group on one predict body and fold to a consensus.
 *  Uses the single server-side fan-out when the group is populated, else the
 *  legacy per-model client fan-out (same contract as the listings view). */
async function predictConsensus(
  group: PredictGroup,
  body: { items?: unknown[]; point_ids?: (number | string)[] },
): Promise<Consensus> {
  if (group.viaServer) {
    const r: GroupPredictResponse = await api.predictBestModels(body)
    const point = r.consensus[0]
    if (!point) throw new Error('no model returned a prediction')
    const members = r.members
      .map((m) => ({ name: m.name, rank: m.rank, predicted: m.predictions[0]?.predicted }))
      .filter((m): m is { name: string; rank: number; predicted: number } => m.predicted != null)
      .sort((a, b) => a.rank - b.rank)
    return {
      median: point.predicted,
      n_ok: point.n_models,
      n_models: group.names.length,
      members,
      warnings: r.warnings,
    }
  }
  const settled = await Promise.allSettled(group.names.map((name) => api.predictModel(name, body)))
  const members = settled.flatMap((r, i) =>
    r.status === 'fulfilled' && r.value.predictions[0]
      ? [{ name: group.names[i], predicted: r.value.predictions[0].predicted }]
      : [],
  )
  if (members.length === 0) throw new Error('no model returned a prediction')
  const vals = members.map((m) => m.predicted).sort((a, b) => a - b)
  const mid = vals.length >> 1
  const median = vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2
  return { median, n_ok: members.length, n_models: group.names.length, members, warnings: [] }
}

/** Display-role fields in declaration order — the track's human identity. */
function displayFieldNames(domain: Domain): string[] {
  return domain.fields.filter((f) => f.role === 'display').map((f) => f.name)
}

/** Primary label (first display field, e.g. track name) + a dim subtitle. */
function identity(domain: Domain, display: Record<string, unknown>): { title: string; sub: string } {
  const names = displayFieldNames(domain)
  const str = (n: string) => {
    const v = display[n]
    return typeof v === 'string' ? v : typeof v === 'number' ? String(v) : ''
  }
  const title = (names.map(str).find((s) => s) || str('artist') || '—').toString()
  const subParts: string[] = []
  const artist = str('artist') || (display['artist'] as string)
  if (artist && artist !== title) subParts.push(artist)
  const genre = display['genre_primary']
  if (typeof genre === 'string' && genre) subParts.push(genre)
  return { title, sub: subParts.join(' · ') }
}

export default function PredictView() {
  const domain = useDomain()
  useEffect(() => {
    document.title = `Predict · ${domain.project.title}`
  }, [domain])
  const noun = domain.project.entity_noun
  const nounPl = domain.project.entity_noun_plural
  const targetNoun = domain.project.target_noun

  const [mode, setMode] = useState<Mode>('pick')
  const [query, setQuery] = useState('')
  const [debounced, setDebounced] = useState('')
  const [includeFallback, setIncludeFallback] = useState(false)
  useEffect(() => {
    const t = window.setTimeout(() => setDebounced(query.trim()), 200)
    return () => window.clearTimeout(t)
  }, [query])

  const group = useAsync(async () => resolvePredictGroup(await api.listModels()), [])
  const predictGroup = group.data
  const search = useAsync(
    () => api.corpusSearch(debounced, { limit: 30, includeFallback }),
    [debounced, includeFallback],
  )

  // Optional held-out badge: choose a dataset, badge tracks as train / test.
  const datasets = useAsync(() => api.listDatasets(), [])
  const [splitDsId, setSplitDsId] = useState('')
  const split = useAsync(
    async () => (splitDsId ? api.getDatasetSplit(splitDsId) : null),
    [splitDsId],
  )
  const splitOf = useMemo(() => {
    const m = new Map<number, 'train' | 'test'>()
    if (split.data) {
      for (const id of split.data.train) m.set(id, 'train')
      for (const id of split.data.test) m.set(id, 'test')
    }
    return m
  }, [split.data])

  return (
    <section aria-label="Predict">
      <ViewHeader
        glyph="pr"
        title="Predict"
        meta={
          <span>
            The {targetNoun} model reads a track's co-listening embedding, which
            only exists for tracks in the corpus — so predict by reference: pick
            a {noun}, or blend several into a synthetic “{nounPl} like these”.
            {predictGroup && predictGroup.names.length > 0 ? (
              <>
                {' '}
                Predicted = median of <span className="num">{predictGroup.names.length}</span>{' '}
                best models.
              </>
            ) : null}
          </span>
        }
      />
      <div className="view-body">
        <div className="predict-tabs" role="tablist" aria-label="Prediction mode">
          <button
            role="tab"
            aria-selected={mode === 'pick'}
            className={mode === 'pick' ? 'predict-tab is-active' : 'predict-tab'}
            onClick={() => setMode('pick')}
          >
            Pick a {noun}
          </button>
          <button
            role="tab"
            aria-selected={mode === 'blend'}
            className={mode === 'blend' ? 'predict-tab is-active' : 'predict-tab'}
            onClick={() => setMode('blend')}
          >
            {cap(nounPl)} like these
          </button>
        </div>

        <div className="predict-controls">
          <input
            type="search"
            className="predict-search"
            placeholder={`Search ${nounPl} by name, artist, genre…`}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label={`Search ${nounPl}`}
          />
          <label className="predict-toggle">
            <input
              type="checkbox"
              checked={includeFallback}
              onChange={(e) => setIncludeFallback(e.target.checked)}
            />
            include sparse-play tracks
          </label>
          {mode === 'pick' && (
            <label className="predict-split-pick">
              held-out vs:{' '}
              <select value={splitDsId} onChange={(e) => setSplitDsId(e.target.value)}>
                <option value="">(no dataset)</option>
                {(datasets.data ?? []).map((d) => (
                  <option key={d.dataset_id} value={d.dataset_id}>
                    {d.name ?? d.dataset_id}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>

        {group.error && (
          <p className="hp-error" role="alert">
            Could not resolve the model group: {group.error}
          </p>
        )}

        {mode === 'pick' ? (
          <PickMode
            domain={domain}
            search={search}
            group={predictGroup}
            splitOf={splitOf}
            hasSplit={!!splitDsId}
          />
        ) : (
          <BlendMode domain={domain} search={search} group={predictGroup} />
        )}
      </div>
    </section>
  )
}

/* ---------------- pick a track ---------------- */

function PickMode({
  domain,
  search,
  group,
  splitOf,
  hasSplit,
}: {
  domain: Domain
  search: ReturnType<typeof useAsync<{ indexed: number; results: CorpusEntry[] }>>
  group: PredictGroup | null
  splitOf: Map<number, 'train' | 'test'>
  hasSplit: boolean
}) {
  const [selected, setSelected] = useState<CorpusEntry | null>(null)
  const [pred, setPred] = useState<
    { state: 'idle' } | { state: 'loading' } | { state: 'done'; c: Consensus } | { state: 'error'; error: string }
  >({ state: 'idle' })

  const choose = async (entry: CorpusEntry) => {
    setSelected(entry)
    if (!group || group.names.length === 0) {
      setPred({ state: 'error', error: 'no promoted models to predict with' })
      return
    }
    setPred({ state: 'loading' })
    try {
      const c = await predictConsensus(group, { point_ids: [entry.point_id] })
      setPred({ state: 'done', c })
    } catch (e) {
      setPred({ state: 'error', error: e instanceof Error ? e.message : String(e) })
    }
  }

  return (
    <div className="predict-pick">
      <SearchResults
        domain={domain}
        search={search}
        splitOf={splitOf}
        hasSplit={hasSplit}
        onChoose={choose}
        selectedId={selected?.point_id}
        actionLabel="predict"
      />
      {selected && (
        <PredictionCard
          domain={domain}
          entry={selected}
          pred={pred}
          splitLabel={hasSplit ? (splitOf.get(Number(selected.point_id)) ?? 'unseen') : null}
        />
      )}
    </div>
  )
}

function PredictionCard({
  domain,
  entry,
  pred,
  splitLabel,
}: {
  domain: Domain
  entry: CorpusEntry
  pred:
    | { state: 'idle' }
    | { state: 'loading' }
    | { state: 'done'; c: Consensus }
    | { state: 'error'; error: string }
  splitLabel: 'train' | 'test' | 'unseen' | null
}) {
  const id = identity(domain, entry.display)
  const actual = entry.target_actual ?? null
  return (
    <div className="predict-card">
      <div className="predict-card-head">
        <div>
          <div className="predict-card-title">{id.title}</div>
          {id.sub && <div className="predict-card-sub">{id.sub}</div>}
        </div>
        {splitLabel && (
          <span className={`predict-badge predict-badge-${splitLabel}`}>
            {splitLabel === 'test'
              ? 'held out — genuine'
              : splitLabel === 'train'
                ? 'in training — memorised'
                : 'not in this dataset'}
          </span>
        )}
      </div>

      <div className="predict-figures">
        <div className="predict-figure">
          <div className="predict-figure-label">Predicted {domain.project.target_noun}</div>
          <div className="predict-figure-value num">
            {pred.state === 'done' ? (
              fmtTarget(pred.c.median, domain)
            ) : pred.state === 'loading' ? (
              <span className="skeleton listing-pred-skeleton" aria-label="Predicting…" />
            ) : pred.state === 'error' ? (
              <span className="muted" title={pred.error}>
                failed
              </span>
            ) : (
              <span className="muted">—</span>
            )}
          </div>
        </div>
        <div className="predict-figure">
          <div className="predict-figure-label">Observed</div>
          <div className="predict-figure-value num">
            {actual != null ? fmtTarget(actual, domain) : <span className="muted">—</span>}
          </div>
        </div>
        <div className="predict-figure">
          <div className="predict-figure-label">Δ vs observed</div>
          <div className="predict-figure-value num">
            {pred.state === 'done' && actual != null ? (
              fmtSignedPct(pred.c.median / actual - 1)
            ) : (
              <span className="muted">—</span>
            )}
          </div>
        </div>
      </div>

      {pred.state === 'done' && (
        <>
          <div className="predict-consensus-note">
            median of {pred.c.n_ok} of {pred.c.n_models} models
          </div>
          <table className="models-table predict-members">
            <thead>
              <tr>
                <th>Model</th>
                <th className="num-col">Predicted</th>
              </tr>
            </thead>
            <tbody>
              {pred.c.members.map((m) => (
                <tr key={m.name}>
                  <td>{m.name}</td>
                  <td className="num-col num">{fmtTargetCell(m.predicted, domain)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {pred.c.warnings.length > 0 && (
            <ul className="predict-warnings">
              {pred.c.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  )
}

/* ---------------- tracks like these ---------------- */

function BlendMode({
  domain,
  search,
  group,
}: {
  domain: Domain
  search: ReturnType<typeof useAsync<{ indexed: number; results: CorpusEntry[] }>>
  group: PredictGroup | null
}) {
  const [seeds, setSeeds] = useState<CorpusEntry[]>([])
  const [pred, setPred] = useState<
    | { state: 'idle' }
    | { state: 'loading' }
    | { state: 'done'; c: Consensus; inherited: Record<string, unknown>; n: number }
    | { state: 'error'; error: string }
  >({ state: 'idle' })

  const add = (e: CorpusEntry) => {
    setSeeds((s) => (s.some((x) => x.point_id === e.point_id) ? s : [...s, e]))
    setPred({ state: 'idle' })
  }
  const remove = (id: string) => {
    setSeeds((s) => s.filter((x) => x.point_id !== id))
    setPred({ state: 'idle' })
  }

  const blend = async () => {
    if (!group || group.names.length === 0) {
      setPred({ state: 'error', error: 'no promoted models to predict with' })
      return
    }
    setPred({ state: 'loading' })
    try {
      const centroid = await api.corpusCentroid(seeds.map((s) => s.point_id))
      const c = await predictConsensus(group, { items: [centroid.item] })
      setPred({ state: 'done', c, inherited: centroid.inherited, n: centroid.n_seeds })
    } catch (e) {
      setPred({ state: 'error', error: e instanceof Error ? e.message : String(e) })
    }
  }

  const inheritedLine = (inherited: Record<string, unknown>) =>
    Object.entries(inherited)
      .map(([k, v]) => `${k}: ${String(v)}`)
      .join(' · ')

  return (
    <div className="predict-blend">
      <div className="predict-basket">
        <div className="predict-basket-head">
          <span>
            Seed {domain.project.entity_noun_plural}{' '}
            <span className="num">{seeds.length}</span>
          </span>
          <button
            className="btn btn-primary"
            disabled={seeds.length < 2 || pred.state === 'loading'}
            onClick={() => void blend()}
          >
            {pred.state === 'loading' ? 'Blending…' : 'Blend & predict'}
          </button>
        </div>
        {seeds.length === 0 ? (
          <p className="muted predict-basket-empty">
            Add two or more {domain.project.entity_noun_plural} below; their
            co-listening vectors are averaged into one synthetic query.
          </p>
        ) : (
          <div className="predict-chips">
            {seeds.map((s) => {
              const id = identity(domain, s.display)
              return (
                <span key={s.point_id} className="predict-chip">
                  {id.title}
                  <button aria-label={`remove ${id.title}`} onClick={() => remove(s.point_id)}>
                    ×
                  </button>
                </span>
              )
            })}
          </div>
        )}

        {pred.state === 'error' && (
          <p className="hp-error" role="alert">
            {pred.error}
          </p>
        )}
        {pred.state === 'done' && (
          <div className="predict-card">
            <div className="predict-card-head">
              <div>
                <div className="predict-card-title">
                  Predicted {domain.project.target_noun}: {fmtTarget(pred.c.median, domain)}
                </div>
                <div className="predict-card-sub">
                  blend of {pred.n} {domain.project.entity_noun_plural}
                  {inheritedLine(pred.inherited) ? ` · ${inheritedLine(pred.inherited)}` : ''} ·
                  median of {pred.c.n_ok} of {pred.c.n_models} models
                </div>
              </div>
            </div>
            <table className="models-table predict-members">
              <thead>
                <tr>
                  <th>Model</th>
                  <th className="num-col">Predicted</th>
                </tr>
              </thead>
              <tbody>
                {pred.c.members.map((m) => (
                  <tr key={m.name}>
                    <td>{m.name}</td>
                    <td className="num-col num">{fmtTargetCell(m.predicted, domain)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <SearchResults
        domain={domain}
        search={search}
        splitOf={new Map()}
        hasSplit={false}
        onChoose={add}
        selectedId={undefined}
        actionLabel="add"
        chosenIds={new Set(seeds.map((s) => s.point_id))}
      />
    </div>
  )
}

/* ---------------- shared search results ---------------- */

function SearchResults({
  domain,
  search,
  splitOf,
  hasSplit,
  onChoose,
  selectedId,
  actionLabel,
  chosenIds,
}: {
  domain: Domain
  search: ReturnType<typeof useAsync<{ indexed: number; results: CorpusEntry[] }>>
  splitOf: Map<number, 'train' | 'test'>
  hasSplit: boolean
  onChoose: (e: CorpusEntry) => void
  selectedId: string | undefined
  actionLabel: string
  chosenIds?: Set<string>
}) {
  const results = search.data?.results ?? []
  return (
    <div className="predict-results">
      {search.error ? (
        <div className="error-block" role="alert">
          Search failed: {search.error}{' '}
          <button className="btn" onClick={search.reload}>
            Retry
          </button>
        </div>
      ) : search.loading ? (
        <div className="skeleton skeleton-sm" />
      ) : results.length === 0 ? (
        <p className="muted">No {domain.project.entity_noun_plural} match.</p>
      ) : (
        <table className="models-table">
          <thead>
            <tr>
              <th>{cap(domain.project.entity_noun)}</th>
              <th className="num-col col-group-start">{cap(domain.project.target_noun)}</th>
              {hasSplit && <th>Split</th>}
              <th aria-label="Actions" />
            </tr>
          </thead>
          <tbody>
            {results.map((e) => {
              const id = identity(domain, e.display)
              const sp = splitOf.get(Number(e.point_id))
              const chosen = chosenIds?.has(e.point_id)
              return (
                <tr
                  key={e.point_id}
                  className={e.point_id === selectedId ? 'run-row is-selected' : 'run-row'}
                >
                  <td>
                    <div className="predict-result-title">
                      {id.title}
                      {!e.learned && (
                        <span className="predict-badge predict-badge-fallback" title="sparse-play fallback vector">
                          fallback
                        </span>
                      )}
                    </div>
                    {id.sub && <div className="predict-result-sub">{id.sub}</div>}
                  </td>
                  <td className="num-col num col-group-start">
                    {e.target_actual != null ? (
                      fmtTargetCell(e.target_actual, domain)
                    ) : (
                      <span className="muted">—</span>
                    )}
                  </td>
                  {hasSplit && (
                    <td>
                      {sp ? (
                        <span className={`predict-badge predict-badge-${sp}`}>{sp}</span>
                      ) : (
                        <span className="muted">—</span>
                      )}
                    </td>
                  )}
                  <td>
                    <button
                      className="btn-inline"
                      disabled={chosen}
                      onClick={() => onChoose(e)}
                    >
                      {chosen ? 'added' : actionLabel}
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </div>
  )
}

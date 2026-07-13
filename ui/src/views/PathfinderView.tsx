import { useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../api/client'
import type { PathfinderTrack, SpotifyCandidate, SpotifyExportResult } from '../api/types'
import ViewHeader from '../components/ViewHeader'
import { useDomain } from '../lib/DomainContext'
import './pathfinder.css'

/* Playlist pathfinder: pick two tracks, trace a sequence that walks from one
 * to the other through the listening-habit structure — A* over the co-listening
 * kNN graph, scored by the habit-fit model and steered by the learned next-track
 * + time-of-day transitions. The compute lives in the Python sidecar; this view
 * reaches it through the lensing-server proxy at /api/pathfinder/*. */

const CONTEXTS = [
  { value: 'now', label: 'now' },
  { value: 'any', label: 'any time' },
  { value: 'morning', label: 'morning' },
  { value: 'afternoon', label: 'afternoon' },
  { value: 'evening', label: 'evening' },
  { value: 'night', label: 'night' },
]

const SHUFFLES = [
  { value: 'any', label: 'either' },
  { value: 'shuffle', label: 'shuffle' },
  { value: 'linear', label: 'linear' },
]

type Trace =
  | { state: 'idle' }
  | { state: 'tracing' }
  | { state: 'done'; tracks: PathfinderTrack[]; context: string | null }
  | { state: 'error'; error: string }

export default function PathfinderView() {
  const domain = useDomain()
  useEffect(() => {
    document.title = `Pathfinder · ${domain.project.title}`
  }, [domain])

  const [start, setStart] = useState<PathfinderTrack | null>(null)
  const [end, setEnd] = useState<PathfinderTrack | null>(null)
  const [length, setLength] = useState(12)
  const [context, setContext] = useState('now')
  const [shuffle, setShuffle] = useState('any')
  const [trace, setTrace] = useState<Trace>({ state: 'idle' })

  const go = async () => {
    if (!start || !end) return
    setTrace({ state: 'tracing' })
    try {
      const data = await api.pathfinderPath({ start: start.id, end: end.id, length, context, shuffle })
      if (data.error) {
        setTrace({ state: 'error', error: data.error })
      } else {
        setTrace({ state: 'done', tracks: data.tracks, context: data.context })
      }
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e)
      setTrace({ state: 'error', error: msg })
    }
  }

  return (
    <section aria-label="Pathfinder">
      <ViewHeader
        glyph="↝"
        title="Pathfinder"
        meta={
          <span>
            Trace a playlist that walks from one track to another through your
            listening habits — A* over the co-listening graph, scored by the{' '}
            <span className="num">habit-fit</span> model and steered by your
            learned next-track &amp; time-of-day transitions.
          </span>
        }
      />

      <div className="view-body">
        <div className="pf-pickers">
          <TrackPicker label="Start" placeholder="start track…" selected={start} onSelect={setStart} />
          <div className="pf-arrow" aria-hidden>
            →
          </div>
          <TrackPicker label="End" placeholder="end track…" selected={end} onSelect={setEnd} />
        </div>

        <div className="pf-controls">
          <label className="pf-control">
            <span className="pf-control-label">
              length <output className="num">{length}</output>
            </span>
            <input
              type="range"
              min={4}
              max={30}
              value={length}
              onChange={(e) => setLength(Number(e.target.value))}
            />
          </label>
          <label className="pf-control">
            <span className="pf-control-label">when</span>
            <select value={context} onChange={(e) => setContext(e.target.value)}>
              {CONTEXTS.map((c) => (
                <option key={c.value} value={c.value}>
                  {c.label}
                </option>
              ))}
            </select>
          </label>
          <label className="pf-control">
            <span className="pf-control-label">play</span>
            <select value={shuffle} onChange={(e) => setShuffle(e.target.value)}>
              {SHUFFLES.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>
          <button
            className="btn btn-primary pf-go"
            disabled={!start || !end || trace.state === 'tracing'}
            onClick={() => void go()}
          >
            {trace.state === 'tracing' ? 'Tracing…' : 'Trace path'}
          </button>
        </div>

        <PathResult trace={trace} />
      </div>
    </section>
  )
}

/* ---------------- track picker (debounced autocomplete) ---------------- */

function TrackPicker({
  label,
  placeholder,
  selected,
  onSelect,
}: {
  label: string
  placeholder: string
  selected: PathfinderTrack | null
  onSelect: (t: PathfinderTrack | null) => void
}) {
  const [query, setQuery] = useState('')
  const [src, setSrc] = useState<'library' | 'spotify'>('library')
  const [results, setResults] = useState<PathfinderTrack[]>([])
  const [spot, setSpot] = useState<SpotifyCandidate[]>([])
  const [open, setOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const boxRef = useRef<HTMLDivElement>(null)
  const seq = useRef(0)

  // Debounced search; a sequence guard drops stale responses. `src` selects the
  // corpus search vs the Spotify catalog search (any track, annotated in_library).
  useEffect(() => {
    const q = query.trim()
    if (!q) {
      setResults([])
      setSpot([])
      setError(null)
      return
    }
    const id = ++seq.current
    const t = window.setTimeout(() => {
      const run = src === 'spotify' ? api.pathfinderSpotifySearch(q) : api.pathfinderSearch(q)
      run.then(
        (hits) => {
          if (seq.current !== id) return
          if (src === 'spotify') setSpot(hits as SpotifyCandidate[])
          else setResults(hits as PathfinderTrack[])
          setError(null)
          setOpen(true)
        },
        (e: unknown) => {
          if (seq.current !== id) return
          setError(e instanceof Error ? e.message : String(e))
          setResults([])
          setSpot([])
          setOpen(true)
        },
      )
    }, 250)
    return () => window.clearTimeout(t)
  }, [query, src])

  // Close the menu on outside click.
  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('click', onDoc)
    return () => document.removeEventListener('click', onDoc)
  }, [])

  const pick = (t: PathfinderTrack) => {
    onSelect(t)
    setQuery(`${t.name} — ${t.artist}`)
    setOpen(false)
  }

  // A Spotify hit can be an endpoint only if it's in the library (we need a
  // corpus node id; cold-start snapping lives in the CLI/agent, not the UI yet).
  const pickSpotify = (c: SpotifyCandidate) => {
    if (!c.in_library || !c.id) return
    onSelect({
      id: c.id, name: c.name, artist: c.artist, genre: c.album ?? '',
      plays: c.plays, learned: true, uri: c.uri, spotify_url: c.spotify_url, fit: null,
    })
    setQuery(`${c.name} — ${c.artist}`)
    setOpen(false)
  }

  return (
    <div className="pf-picker" ref={boxRef}>
      <span className="pf-picker-label">{label}</span>
      <div className="pf-src-toggle" role="tablist" aria-label="search source">
        {(['library', 'spotify'] as const).map((m) => (
          <button
            key={m}
            type="button"
            role="tab"
            aria-selected={src === m}
            className={src === m ? 'pf-src is-on' : 'pf-src'}
            onClick={() => {
              setSrc(m)
              setOpen(false)
            }}
          >
            {m === 'library' ? 'Library' : 'Spotify'}
          </button>
        ))}
      </div>
      <input
        type="search"
        className="pf-picker-input"
        placeholder={src === 'spotify' ? 'search all of Spotify…' : placeholder}
        value={query}
        autoComplete="off"
        onChange={(e) => {
          setQuery(e.target.value)
          onSelect(null)
        }}
        onFocus={() => (results.length > 0 || spot.length > 0) && setOpen(true)}
        aria-label={`${label} track`}
      />
      {selected && <span className="pf-picker-ok" aria-hidden>✓</span>}
      {open && (
        <div className="pf-menu" role="listbox">
          {error ? (
            <div className="pf-menu-msg" role="alert">
              {error}
            </div>
          ) : src === 'spotify' ? (
            spot.length === 0 ? (
              <div className="pf-menu-msg">no matches</div>
            ) : (
              spot.map((c) => (
                <button
                  key={c.spotify_id}
                  type="button"
                  className="pf-menu-item pf-menu-spot"
                  disabled={!c.in_library}
                  title={c.in_library ? '' : 'not in your library — can’t be a path endpoint yet'}
                  onClick={() => pickSpotify(c)}
                >
                  {c.art ? (
                    <img className="pf-art" src={c.art} alt="" width={34} height={34} loading="lazy" />
                  ) : (
                    <span className="pf-art pf-art-blank" aria-hidden />
                  )}
                  <span className="pf-spot-text">
                    <span className="pf-menu-title">
                      <b>{c.name}</b> — {c.artist}
                    </span>
                    <span className="pf-menu-sub">
                      {c.album}
                      {c.in_library ? (
                        <span className="pf-inlib"> · in library</span>
                      ) : (
                        ' · not in library'
                      )}
                    </span>
                  </span>
                </button>
              ))
            )
          ) : results.length === 0 ? (
            <div className="pf-menu-msg">no matches</div>
          ) : (
            results.map((t) => (
              <button key={t.id} type="button" className="pf-menu-item" onClick={() => pick(t)}>
                <span className="pf-menu-title">
                  <b>{t.name}</b> — {t.artist}
                </span>
                <span className="pf-menu-sub">
                  {t.genre} · <span className="num">{t.plays}</span> plays
                  {t.learned ? '' : ' · sparse'}
                </span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  )
}

/* ---------------- traced path ---------------- */

function PathResult({ trace }: { trace: Trace }) {
  if (trace.state === 'idle') {
    return (
      <p className="muted pf-hint">Pick a start and an end track, then trace the path.</p>
    )
  }
  if (trace.state === 'tracing') {
    return <div className="skeleton skeleton-md pf-skeleton" aria-label="Tracing…" />
  }
  if (trace.state === 'error') {
    return (
      <div className="error-block" role="alert">
        {trace.error}
      </div>
    )
  }

  const { tracks, context } = trace
  return (
    <div className="pf-result">
      <div className="pf-result-head">
        <div className="pf-status">
          <span className="num">{tracks.length}</span> tracks
          {context ? (
            <>
              {' · tuned for '}
              <span className="pf-status-ctx">{context}</span>
            </>
          ) : null}
        </div>
        <ExportBar key={tracks.map((t) => t.id).join('-')} tracks={tracks} />
      </div>
      <ol className="pf-steps">
        {tracks.map((t, i) => {
          const endpoint = i === 0 || i === tracks.length - 1
          return (
            <li key={`${t.id}-${i}`} className={endpoint ? 'pf-step is-endpoint' : 'pf-step'}>
              <span className="pf-step-n num">{i + 1}</span>
              <span className="pf-step-track">
                <span className="pf-step-name">{t.name}</span>
                <span className="pf-step-artist">{t.artist}</span>
              </span>
              <span className="pf-chip">{t.genre}</span>
              {t.fit != null ? (
                <span className="pf-fit" title={`habit-fit ${t.fit.toFixed(2)}`}>
                  <span className="pf-fit-bar">
                    <span className="pf-fit-fill" style={{ width: `${Math.round(t.fit * 100)}%` }} />
                  </span>
                  <span className="pf-fit-val num">{t.fit.toFixed(2)}</span>
                </span>
              ) : (
                <span className="pf-fit pf-fit-empty" aria-hidden />
              )}
              {endpoint && <span className="pf-endpoint-tag">{i === 0 ? 'start' : 'end'}</span>}
              <a className="pf-open" href={t.spotify_url} target="_blank" rel="noreferrer">
                open ↗
              </a>
            </li>
          )
        })}
      </ol>
    </div>
  )
}

/* ---------------- export to Spotify ---------------- */

type ExportState =
  | { k: 'idle' }
  | { k: 'busy'; msg: string }
  | { k: 'done'; result: SpotifyExportResult }
  | { k: 'error'; msg: string }

function ExportBar({ tracks }: { tracks: PathfinderTrack[] }) {
  const first = tracks[0]?.name ?? 'taste'
  const last = tracks[tracks.length - 1]?.name ?? 'path'
  const [name, setName] = useState(`${first} → ${last}`)
  const [state, setState] = useState<ExportState>({ k: 'idle' })

  // Open the Spotify consent popup, resolve once authorized — via the page's
  // postMessage, or by polling status (covers blocked postMessage / manual close).
  const connect = async (): Promise<boolean> => {
    const { authorize_url } = await api.spotifyLogin()
    const popup = window.open(authorize_url, 'spotify-auth', 'width=520,height=720')
    if (!popup) throw new Error('popup blocked — allow popups for this site and retry')
    return new Promise<boolean>((resolve, reject) => {
      let settled = false
      const onMsg = (e: MessageEvent) => {
        if (e?.data?.type === 'spotify-auth') done(!!e.data.ok)
      }
      const poll = window.setInterval(async () => {
        try {
          if ((await api.spotifyStatus()).authorized) done(true)
        } catch {
          /* keep polling */
        }
      }, 1500)
      const timeout = window.setTimeout(() => {
        if (!settled) {
          settled = true
          cleanup()
          reject(new Error('timed out waiting for Spotify authorization'))
        }
      }, 120000)
      const cleanup = () => {
        window.removeEventListener('message', onMsg)
        window.clearInterval(poll)
        window.clearTimeout(timeout)
      }
      const done = (ok: boolean) => {
        if (settled) return
        settled = true
        cleanup()
        resolve(ok)
      }
      window.addEventListener('message', onMsg)
    })
  }

  const exportNow = async () => {
    setState({ k: 'busy', msg: 'Checking Spotify…' })
    try {
      const status = await api.spotifyStatus()
      if (!status.configured) {
        setState({
          k: 'error',
          msg: 'Spotify export isn’t configured — add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to .env (see .env.example), then restart the server.',
        })
        return
      }
      if (!status.authorized) {
        setState({ k: 'busy', msg: 'Waiting for Spotify…' })
        const ok = await connect()
        if (!ok) {
          setState({ k: 'error', msg: 'Spotify authorization was denied.' })
          return
        }
      }
      setState({ k: 'busy', msg: 'Creating playlist…' })
      const result = await api.spotifyExport({
        name: name.trim() || 'taste path',
        uris: tracks.map((t) => t.uri),
      })
      setState({ k: 'done', result })
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e)
      setState({ k: 'error', msg })
    }
  }

  if (state.k === 'done') {
    return (
      <div className="pf-export pf-export-done">
        <span className="pf-export-ok" aria-hidden>
          ✓
        </span>
        <span>
          Added <span className="num">{state.result.n}</span> to{' '}
          <a href={state.result.url} target="_blank" rel="noreferrer" className="pf-export-link">
            {state.result.name} ↗
          </a>
        </span>
      </div>
    )
  }

  const busy = state.k === 'busy'
  return (
    <div className="pf-export">
      <input
        className="pf-export-name"
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="playlist name"
        aria-label="Playlist name"
        disabled={busy}
      />
      <button className="btn btn-primary" onClick={() => void exportNow()} disabled={busy}>
        {busy ? state.msg : 'Export to Spotify'}
      </button>
      {state.k === 'error' && (
        <span className="pf-export-err" role="alert">
          {state.msg}
        </span>
      )}
    </div>
  )
}

import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { useDomain } from '../lib/DomainContext'
import CommandPalette from './CommandPalette'
import './shell.css'

const IS_MAC = /Mac|iPhone|iPad/.test(navigator.platform)

export default function Shell() {
  const domain = useDomain()
  // Predict (scalar consensus) is pointwise-only; a ranking instance predicts a
  // next-item ordering, so it doesn't apply — hidden from the spine. The
  // Interpretability panel DOES apply (its per-model SAE reads activations
  // against the next item); its pointwise-only tabs guard themselves.
  const isRanking = domain.target.task === 'ranking'
  // Tone-step the topbar's bottom hairline once content scrolls under it
  // (flat elevation: a border step, never a resting shadow).
  const [stuck, setStuck] = useState(false)
  useEffect(() => {
    const onScroll = () => setStuck(window.scrollY > 0)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // ⌘K / Ctrl+K from anywhere opens the palette.
  const [paletteOpen, setPaletteOpen] = useState(false)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setPaletteOpen((o) => !o)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <>
      <header className={stuck ? 'topbar is-stuck' : 'topbar'}>
        <div className="topbar-inner">
          <NavLink to="/" className="wordmark">
            {/* The instance's Velocity-Map mark (branding/make_mark.py), tuned
                green↔violet via domain.toml mark_hue_shift. Larger, and the
                swarm orbits slowly — the lens turning. Reduced-motion stills it. */}
            <img className="wordmark-glyph" src="/brand/mark-dark.svg" width={34} height={34} alt="" />
            lensing <span className="wordmark-dim">· {domain.project.title}</span>
          </NavLink>
          {/* The pipeline spine, in pipeline order: build → train → promote. */}
          <nav aria-label="Primary">
            {/* Representations sit upstream of datasets: a dataset can be built
                on a latent collection, so the spine starts with how the data is
                represented, then build → train → promote. */}
            <NavLink to="/representations" className="nav-link">
              Representations
            </NavLink>
            <NavLink to="/datasets" className="nav-link">
              Datasets
            </NavLink>
            <NavLink to="/" end className="nav-link">
              Runs
            </NavLink>
            <NavLink to="/compare" className="nav-link">
              Compare
            </NavLink>
            <NavLink to="/definitions" className="nav-link">
              Definitions
            </NavLink>
            <NavLink to="/models" className="nav-link">
              Models
            </NavLink>
            {!isRanking && (
              <NavLink to="/predict" className="nav-link">
                Predict
              </NavLink>
            )}
            <NavLink to="/interpretability" className="nav-link">
              Interpretability
            </NavLink>
            {isRanking && (
              /* Plain anchor, not NavLink: the playlist lab is a separate
                 static app served at /lab, outside this router. Ranking-only —
                 it drives POST /api/models/{name}/extend, which needs a baked
                 sequence artifact. */
              <a href="/lab/" className="nav-link">
                Playlist lab
              </a>
            )}
          </nav>
          <button
            className="topbar-search"
            onClick={() => setPaletteOpen(true)}
            aria-label="Open command palette"
          >
            <span className="topbar-search-word">Jump to…</span>
            <kbd>{IS_MAC ? '⌘K' : 'Ctrl K'}</kbd>
          </button>
          {/* Quiet on purpose: the orange budget belongs to each page's own
              primary action (Start training, Compare runs, live state). */}
          <NavLink to="/new" className="btn-on-chrome topbar-cta">
            New run
          </NavLink>
        </div>
      </header>
      <main className="content">
        <Outlet />
      </main>
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
    </>
  )
}

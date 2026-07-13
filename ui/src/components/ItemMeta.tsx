import { useMemo } from 'react'
import type { Item } from '../api/types'
import { useDomain } from '../lib/DomainContext'
import { timestampFieldName, type DomainField } from '../lib/domain'
import { fmtDateTime } from '../lib/format'
import { isRefField, itemHref, titleField } from '../lib/itemDisplay'

/** Field's display label, humanizing snake_case names the domain left unlabeled. */
function label(f: DomainField): string {
  return f.label ?? f.name.replace(/_/g, ' ')
}

/** One corpus row's metadata, laid out as a recognizable entity card. Both
 *  groups are domain-driven: "about" is the display + categorical fields, the
 *  signals strip is the numeric fields, each rendered with its domain label and
 *  a magnitude-appropriate format. On Spotify this reads as album / artist /
 *  genre plus popularity, reach and library signals — readable at a glance,
 *  with a deep link out to the track. */
export default function ItemMeta({ item }: { item: Item }) {
  const domain = useDomain()
  const title = useMemo(() => titleField(domain), [domain])
  const tsName = useMemo(() => timestampFieldName(domain), [domain])

  // "about": every categorical, plus display fields that name the entity (album)
  // — but not the title (the row shows it) nor reference fields like the URI,
  // which the deep link below already covers.
  const about = useMemo(
    () =>
      domain.fields.filter(
        (f) =>
          f.role === 'categorical' ||
          (f.role === 'display' && f.name !== title?.name && !isRefField(f)),
      ),
    [domain, title],
  )
  // signals: the numeric fields, presented even when off as model features —
  // they explain the actual value to a human reading the miss.
  const signals = useMemo(() => domain.fields.filter((f) => f.role === 'numeric'), [domain])

  const has = (f: DomainField) => str(item, f.name) !== ''
  const ts = tsName ? str(item, tsName) : ''
  const link = itemHref(domain, item)

  return (
    <div className="item-card">
      <dl className="item-about">
        {about.filter(has).map((f) => (
          <div key={f.name}>
            <dt>{label(f)}</dt>
            <dd>{str(item, f.name)}</dd>
          </div>
        ))}
        {ts && (
          <div>
            <dt>{label(domain.fields.find((f) => f.name === tsName)!)}</dt>
            <dd>{fmtDateTime(ts)}</dd>
          </div>
        )}
      </dl>

      {signals.some(has) && (
        <dl className="item-signals">
          {signals.filter(has).map((f) => (
            <div key={f.name}>
              <dt>{label(f)}</dt>
              <dd className="num">{fmtSignal(f, item[f.name])}</dd>
            </div>
          ))}
        </dl>
      )}

      {link && (
        <a className="item-link" href={link.href} target="_blank" rel="noreferrer noopener">
          {link.label} ↗
        </a>
      )}
    </div>
  )
}

function str(item: Item, name: string): string {
  const v = item[name]
  return typeof v === 'string' ? v : v != null ? String(v) : ''
}

const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumSignificantDigits: 3 })
const grouped = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 })

/** Format a numeric field for the card: ratios as %, years bare, 0/1 flags as a
 *  check, large counts compacted. Heuristics keyed off the field name so the
 *  rules stay domain-agnostic. */
function fmtSignal(f: DomainField, raw: unknown): string {
  const v = typeof raw === 'number' ? raw : Number(raw)
  if (!isFinite(v)) return '—'
  const name = f.name.toLowerCase()
  if (/ratio|rate/.test(name)) return `${(v * 100).toFixed(0)}%`
  if (/year/.test(name)) return String(Math.round(v))
  if (/^is_|^has_/.test(name) && (v === 0 || v === 1)) return v === 1 ? '✓' : '—'
  if (Math.abs(v) >= 10_000) return compact.format(v)
  return grouped.format(v)
}

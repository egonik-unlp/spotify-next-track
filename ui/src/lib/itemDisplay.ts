// How to present one corpus row as a recognizable entity. Driven by the
// domain config, not hardcoded to tracks: the title is the primary `display`
// field, the byline is the most identity-like categorical, the tag is the
// outlier-group category, and an external link is offered when a display field
// carries a recognizable URI. On the Spotify instance this resolves to
// "track_name / artist / genre + open.spotify.com", which is what you want to
// read at a glance; another instance gets its own equivalent for free.

import type { Item } from '../api/types'
import { categoricalFields, type Domain, type DomainField } from './domain'

/** A display-role field looks like an external reference, not a name. */
export function isRefField(f: DomainField): boolean {
  return /uri|url|\bid\b|link/i.test(f.name) || /uri|url|link/i.test(f.label ?? '')
}

function str(item: Item, name: string | undefined): string {
  if (!name) return ''
  const v = item[name]
  return typeof v === 'string' ? v : v != null ? String(v) : ''
}

/** The field whose value names the entity (e.g. track_name): the first
 *  `display`-role field that isn't a reference like a URI. */
export function titleField(domain: Domain): DomainField | null {
  const displays = domain.fields.filter((f) => f.role === 'display')
  return displays.find((f) => !isRefField(f)) ?? displays[0] ?? null
}

/** The byline category (e.g. artist): the most identity-like categorical,
 *  i.e. the first that isn't the outlier-grouping field (genre on Spotify). */
export function bylineField(domain: Domain): DomainField | null {
  const cats = domain.fields.filter(
    (f) => f.role === 'categorical' && f.name !== domain.quality.outlier_group,
  )
  return cats[0] ?? null
}

/** The grouping category surfaced as a tag (e.g. primary genre). */
export function tagField(domain: Domain): DomainField | null {
  const og = domain.quality.outlier_group
  return categoricalFields(domain).find((f) => f.name === og) ?? null
}

export function itemTitle(domain: Domain, item: Item): string {
  return str(item, titleField(domain)?.name) || (item.content ?? '').split(' — ')[0] || ''
}

export function itemByline(domain: Domain, item: Item): string {
  return str(item, bylineField(domain)?.name)
}

export function itemTag(domain: Domain, item: Item): string {
  return str(item, tagField(domain)?.name)
}

/** A one-line label for chart tooltips: "Title — Byline", length-capped so a
 *  long track name can't blow out the tooltip box. */
export function itemLabel(domain: Domain, item: Item | null | undefined): string | null {
  if (!item) return null
  const title = itemTitle(domain, item)
  if (!title) return null
  const by = itemByline(domain, item)
  const s = by ? `${title} — ${by}` : title
  return s.length > 38 ? `${s.slice(0, 37)}…` : s
}

/** Adapt a SEQUENCE dataset's compact item record — items.json there is keyed
 *  {uri, name, artist, genre, play_count}, not by the domain field names — onto
 *  the domain's title/byline/tag/ref fields, so the shared ItemIdentity, itemHref
 *  and the name/byline filter present a next-item track exactly like a pointwise
 *  corpus row. Only fills a target field when it's absent, so a fuller record is
 *  left untouched. */
export function sequenceItemForDisplay(domain: Domain, raw: Item | null): Item | null {
  if (!raw) return null
  const out: Record<string, unknown> = { ...raw }
  const alias = (fieldName: string | undefined, value: unknown) => {
    if (fieldName && value != null && out[fieldName] == null) out[fieldName] = value
  }
  alias(titleField(domain)?.name, (raw as Record<string, unknown>).name)
  alias(bylineField(domain)?.name, (raw as Record<string, unknown>).artist)
  alias(tagField(domain)?.name, (raw as Record<string, unknown>).genre)
  const ref = domain.fields.find((f) => f.role === 'display' && isRefField(f))
  alias(ref?.name, (raw as Record<string, unknown>).uri)
  if (out.content == null) {
    out.content = [(raw as Record<string, unknown>).name, (raw as Record<string, unknown>).artist]
      .filter(Boolean)
      .join(' — ')
  }
  return out as Item
}

/** External link for the entity, if a display field carries a known URI scheme.
 *  Spotify URIs ("spotify:track:ID") map to open.spotify.com; unknown schemes
 *  yield no link rather than a guess. */
export function itemHref(domain: Domain, item: Item): { href: string; label: string } | null {
  for (const f of domain.fields) {
    if (f.role !== 'display') continue
    const v = str(item, f.name)
    const m = /^spotify:([a-z]+):([A-Za-z0-9]+)$/.exec(v)
    if (m) return { href: `https://open.spotify.com/${m[1]}/${m[2]}`, label: 'Open in Spotify' }
  }
  return null
}

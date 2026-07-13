import type { Item } from '../api/types'
import { useDomain } from '../lib/DomainContext'
import { itemByline, itemTag, itemTitle } from '../lib/itemDisplay'

/** Compact, two-line identity for one corpus row in a table cell: the entity's
 *  name, its byline, and a grouping tag (on Spotify: track / artist / genre).
 *  Falls back to the row id when the corpus payload hasn't loaded yet. */
export default function ItemIdentity({
  item,
  rowId,
}: {
  item: Item | null
  rowId: number | string
}) {
  const domain = useDomain()
  if (!item) {
    return (
      <span className="item-id-fallback num" title={`row ${rowId}`}>
        #{rowId}
      </span>
    )
  }
  const title = itemTitle(domain, item)
  const byline = itemByline(domain, item)
  const tag = itemTag(domain, item)
  return (
    <span className="item-id">
      <span className="item-id-title" title={title}>
        {title || `#${rowId}`}
      </span>
      {(byline || tag) && (
        <span className="item-id-sub">
          {byline && <span className="item-id-byline">{byline}</span>}
          {tag && <span className="item-id-tag">{tag}</span>}
        </span>
      )}
    </span>
  )
}

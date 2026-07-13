import ViewHeader from './ViewHeader'
import LatticeMark from './LatticeMark'

/** Shown when a pointwise-only view (Predict, Interpretability) is reached
 *  directly on a ranking instance, where it has no meaning. Matches the
 *  standard empty-state layout so it reads as a deliberate state, not an error. */
export default function NotApplicable({
  glyph,
  title,
  reason,
}: {
  glyph: string
  title: string
  reason: string
}) {
  return (
    <>
      <ViewHeader glyph={glyph} title={title} />
      <div className="view-body">
        <section className="empty-state" aria-label={`${title} not applicable`}>
          <LatticeMark />
          <h1>Not applicable for ranking instances</h1>
          <p>{reason}</p>
        </section>
      </div>
    </>
  )
}

import type { ModelSaeReport } from '../api/types'

/** The headline numbers of a per-model SAE — the answer, before the detail.
 *  Shared by the single report header and the comparison digest so they agree. */
export interface ModelSaeDigest {
  /** Peak per-layer next-item (genre) decodability AUC. */
  peakAuc: number
  /** Peak per-layer next-item (genre) decodability accuracy. */
  peakAcc: number
  maxUtil: number
  concepts: number
  segRepr: string
  dropped: number | null
  /** The model's own retrieval recall@10 (decodability anchor). */
  recall: number | null
}

export function modelSaeDigest(r: ModelSaeReport): ModelSaeDigest {
  const layers = r.layers ?? []
  const deepest = layers[layers.length - 1]
  const cvd = r.concept_vs_decodability ?? []
  const peakAuc = Math.max(...cvd.map((c) => c.genre_auc ?? -Infinity), -Infinity)
  const peakAcc = Math.max(...cvd.map((c) => c.genre_acc ?? -Infinity), -Infinity)
  const segs = deepest?.segments ?? []
  return {
    peakAuc: Number.isFinite(peakAuc) ? peakAuc : 0,
    peakAcc: Number.isFinite(peakAcc) ? peakAcc : 0,
    maxUtil: layers.length ? Math.max(...layers.map((l) => l.capacity.utilization)) : 0,
    concepts: layers.reduce((s, l) => s + l.n_interpretable_concepts, 0),
    segRepr: segs.length ? `${segs.filter((s) => s.represented).length}/${segs.length}` : '—',
    dropped: r.embedding_diff?.ran ? (deepest?.dropped_vs_next_item?.n_dropped ?? null) : null,
    recall: r.model_recall_at_10 ?? null,
  }
}

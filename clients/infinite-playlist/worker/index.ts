// Infinite Playlist — Cloudflare Worker backend.
//
// Serves the static app (public/, via the ASSETS binding) and the two things the
// browser can't do on its own:
//
//   POST /api/resolve  { url }  → { name, kind, tracks: [{uri,name,artist, latent?, genre?}] }
//   POST /api/embed    { uris } → { tracks: [{uri, latent, genre}] }
//   GET  /api/art?ids= id,id,…  → { art: { id: { url } } }        (album covers)
//
// Expands a Spotify playlist/album/track link (client-credentials, secret env).
// /api/embed is the same cold-start, minus the expansion: the library browser
// walks the user's OWN account with their user token (private playlists, liked
// songs — things client-credentials can't see) and sends back only the track
// uris that aren't in the baked catalog.
// Tracks already in the app's catalog return just their uri (the browser maps
// them to a baked latent). Tracks the user doesn't own are cold-started into the
// model's PCA-192 space: Spotify meta + ReccoBeats acoustics → contentDoc →
// Workers AI bge-m3 text embedding → the Rust→WASM projector (worker-core). So
// ANY song can seed the mood while the journey stays in the user's library.

import { initSync, set_projector, project, text_model, has_projector } from "../worker-core/pkg/worker_core.js";
// wrangler bundles the .wasm as a WebAssembly.Module (see wrangler.toml rules).
import wasmModule from "../worker-core/pkg/worker_core_bg.wasm";

export interface Env {
  ASSETS: Fetcher;
  AI: any;
  SPOTIFY_CLIENT_ID?: string;
  SPOTIFY_CLIENT_SECRET?: string;
  EMBED_MODEL?: string;
}

const SPOTIFY_API = "https://api.spotify.com/v1";
const SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token";
const RECCOBEATS_API = "https://api.reccobeats.com/v1";
const DEFAULT_EMBED_MODEL = "@cf/baai/bge-m3";
const UA = { "User-Agent": "infinite-playlist/1.0" };
// ReccoBeats af_* fields, in the projector's acoustic order.
const AF_KEYS = ["danceability", "energy", "valence", "tempo", "acousticness",
  "instrumentalness", "loudness", "speechiness", "liveness", "key", "mode"];
const MAX_EMBED = 120; // cold-start budget per request; batched calls keep us well under the subrequest cap

class ApiError extends Error {
  constructor(public status: number, msg: string) { super(msg); }
}

// ---- one-time init: WASM + projector + known-uri set ----------------------
let ready: Promise<void> | null = null;
let KNOWN: Set<string> | null = null;

function ensureReady(env: Env): Promise<void> {
  if (ready) return ready;
  ready = (async () => {
    initSync({ module: wasmModule });
    const proj = await assetBytes(env, "/model/projector.bin");
    if (proj) {
      try { set_projector(proj); }
      catch (e) { console.warn("projector load failed:", e); }
    } else {
      console.warn("projector.bin not found — cold-start embedding disabled");
    }
    const cat = await assetBytes(env, "/model/catalog.json");
    KNOWN = new Set<string>();
    if (cat) {
      const rows = JSON.parse(new TextDecoder().decode(cat)) as { uri?: string }[];
      for (const r of rows) if (r.uri) KNOWN.add(r.uri);
    }
    console.log(`ready · projector=${has_projector()} · known=${KNOWN.size}`);
  })();
  return ready;
}

async function assetBytes(env: Env, path: string): Promise<Uint8Array | null> {
  const res = await env.ASSETS.fetch(new Request(`https://assets${path}`));
  if (!res.ok) return null;
  return new Uint8Array(await res.arrayBuffer());
}

// ---- fetch helpers ---------------------------------------------------------
async function fetchRetry(input: RequestInfo | URL, init?: RequestInit, attempts = 3): Promise<Response> {
  let last: Response | null = null;
  for (let i = 0; i < attempts; i += 1) {
    try {
      const r = await fetch(input, init);
      if (r.status !== 429 && r.status < 500) return r;
      last = r;
    } catch (e) {
      if (i === attempts - 1) throw e;
    }
    if (i < attempts - 1) {
      const ra = last ? Number(last.headers.get("retry-after")) : NaN;
      const wait = Number.isFinite(ra) && ra > 0 ? ra * 1000 : 250 * (i + 1);
      await new Promise((res) => setTimeout(res, Math.min(wait, 2000)));
    }
  }
  return last as Response;
}

let spotifyToken: { value: string; exp: number } | null = null;
async function spotifyTokenGet(env: Env): Promise<string> {
  if (!env.SPOTIFY_CLIENT_ID || !env.SPOTIFY_CLIENT_SECRET) {
    throw new ApiError(500, "Spotify credentials missing (SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET).");
  }
  if (spotifyToken && spotifyToken.exp > Date.now() / 1000 + 30) return spotifyToken.value;
  const auth = btoa(`${env.SPOTIFY_CLIENT_ID}:${env.SPOTIFY_CLIENT_SECRET}`);
  const r = await fetchRetry(SPOTIFY_TOKEN_URL, {
    method: "POST",
    headers: { Authorization: `Basic ${auth}`, "Content-Type": "application/x-www-form-urlencoded" },
    body: "grant_type=client_credentials",
  });
  if (!r.ok) throw new ApiError(502, `Spotify token failed (${r.status})`);
  const data = (await r.json()) as { access_token: string; expires_in?: number };
  spotifyToken = { value: data.access_token, exp: Date.now() / 1000 + (data.expires_in ?? 3600) };
  return spotifyToken.value;
}

async function spotifyGet(env: Env, path: string, params?: Record<string, string>): Promise<any> {
  const url = new URL(`${SPOTIFY_API}/${path}`);
  if (params) for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
  const r = await fetchRetry(url, { headers: { Authorization: `Bearer ${await spotifyTokenGet(env)}` } });
  if (!r.ok) throw new ApiError(r.status === 404 ? 404 : 502, `Spotify ${path} → ${r.status}`);
  return r.json();
}

// ---- link parsing + expansion ---------------------------------------------
function parseLink(url: string): { kind: string; id: string } | null {
  const v = (url || "").trim();
  let m = /spotify[:/](playlist|album|track)[:/]([A-Za-z0-9]+)/.exec(v);
  if (!m) m = /open\.spotify\.com\/(?:[a-z-]+\/)?(playlist|album|track)\/([A-Za-z0-9]+)/.exec(v);
  if (m) return { kind: m[1], id: m[2] };
  if (/^[A-Za-z0-9]{22}$/.test(v)) return { kind: "track", id: v };
  return null;
}

interface Basic { uri: string; name: string; artist: string; }
function basic(t: any): Basic {
  return { uri: t?.uri, name: t?.name ?? "", artist: t?.artists?.[0]?.name ?? "" };
}

async function expand(env: Env, kind: string, id: string): Promise<{ name: string | null; tracks: Basic[] }> {
  if (kind === "track") {
    const t = await spotifyGet(env, `tracks/${id}`);
    return { name: null, tracks: t?.uri ? [basic(t)] : [] };
  }
  if (kind === "album") {
    const al = await spotifyGet(env, `albums/${id}`);
    const tracks: Basic[] = (al.tracks?.items ?? []).filter((t: any) => t?.uri?.startsWith("spotify:track:")).map(basic);
    let next = al.tracks?.next;
    let off = tracks.length;
    while (next && off < 500) {
      const more = await spotifyGet(env, `albums/${id}/tracks`, { limit: "50", offset: String(off) });
      const items = more.items ?? [];
      for (const t of items) if (t?.uri?.startsWith("spotify:track:")) tracks.push(basic(t));
      off += items.length; next = more.next;
      if (!items.length) break;
    }
    return { name: al.name ?? null, tracks };
  }
  // playlist
  const meta = await spotifyGet(env, `playlists/${id}`, { fields: "name" });
  const tracks: Basic[] = [];
  let off = 0;
  while (off < 2000) {
    const r = await spotifyGet(env, `playlists/${id}/tracks`, {
      limit: "100", offset: String(off),
      fields: "items(track(uri,name,artists(name))),next",
    });
    const items = r.items ?? [];
    for (const it of items) {
      const t = it.track;
      if (t?.uri?.startsWith("spotify:track:")) tracks.push(basic(t));
    }
    if (!items.length || !r.next) break;
    off += 100;
  }
  return { name: meta.name ?? null, tracks };
}

// ---- cold-start embed (mirrors bake_projector.py metas + contentDoc) ------
function contentDoc(m: any): string {
  const parts: string[] = [];
  if (m.track_name) parts.push(`${m.track_name}.`);
  if (m.artist) {
    let artist = String(m.artist);
    const n = m.artist_count || 1;
    if (n > 1) artist += ` (with ${(n | 0) - 1} other artist(s))`;
    parts.push(`Artist: ${artist}.`);
  }
  if (m.album) {
    const bits = [m.album_type, m.release_year ? String(m.release_year) : ""].filter(Boolean);
    parts.push(`Album: ${m.album}${bits.length ? ` (${bits.join(", ")})` : ""}.`);
  }
  if (m.genre_primary) parts.push(`Genre: ${m.genre_primary}.`);
  return parts.join(" ") || "Unknown track.";
}

// Build one track's meta from already-fetched Spotify track + artists map
// (mirrors bake_projector.py map_meta). No network — batch fetch happens upstream.
function buildMeta(track: any, artistsById: Map<string, any>): any {
  const arts = (track.artists ?? []).filter((a: any) => a?.id);
  const first = artistsById.get(arts[0]?.id) ?? {};
  const genres = [...new Set(arts.flatMap((a: any) => artistsById.get(a.id)?.genres ?? []) as string[])].sort();
  const album = track.album ?? {};
  const releaseDate: string = album.release_date ?? "";
  return {
    track_uri: `spotify:track:${track.id}`,
    track_name: track.name ?? null,
    album: album.name ?? null,
    artist: track.artists?.[0]?.name ?? null,
    artist_count: (track.artists ?? []).length,
    track_popularity: track.popularity ?? null,
    artist_popularity: first.popularity ?? null,
    artist_followers: first.followers?.total ?? null,
    album_type: album.album_type ?? null,
    release_year: /^\d{4}/.test(releaseDate) ? parseInt(releaseDate.slice(0, 4), 10) : null,
    sp_duration_ms: track.duration_ms ?? null,
    sp_explicit: track.explicit ? 1 : 0,
    sp_track_number: track.track_number ?? null,
    sp_n_markets: (track.available_markets ?? []).length,
    sp_genres: genres,
    sp_genre_count: genres.length,
    genre_primary: genres[0] ?? "unknown",
  };
}

// ---- batched external fetches (50 Spotify ids / 40 ReccoBeats ids per call) --
async function spTracksBatch(env: Env, sids: string[]): Promise<Map<string, any>> {
  const out = new Map<string, any>();
  for (let i = 0; i < sids.length; i += 50) {
    const r = await spotifyGet(env, "tracks", { ids: sids.slice(i, i + 50).join(",") });
    for (const t of r.tracks ?? []) if (t?.id) out.set(t.id, t);
  }
  return out;
}
async function spArtistsBatch(env: Env, ids: string[]): Promise<Map<string, any>> {
  const out = new Map<string, any>();
  const uniq = [...new Set(ids)];
  for (let i = 0; i < uniq.length; i += 50) {
    const r = await spotifyGet(env, "artists", { ids: uniq.slice(i, i + 50).join(",") });
    for (const a of r.artists ?? []) if (a?.id) out.set(a.id, a);
  }
  return out;
}
async function rbBatch(sids: string[]): Promise<Map<string, Record<string, any>>> {
  const out = new Map<string, Record<string, any>>();
  const sidUuid = new Map<string, string>();
  for (let i = 0; i < sids.length; i += 40) {
    try {
      const r = await fetchRetry(`${RECCOBEATS_API}/track?ids=${sids.slice(i, i + 40).join(",")}`, { headers: UA });
      if (!r.ok) continue;
      for (const t of ((await r.json()) as any).content ?? []) {
        const sid = (t.href || "").replace(/\/+$/, "").split("/").pop();
        if (sid && t.id) sidUuid.set(sid, t.id);
      }
    } catch { /* best-effort acoustics */ }
  }
  const pairs = [...sidUuid.entries()];
  const uuidSid = new Map(pairs.map(([s, u]) => [u, s]));
  const uuids = pairs.map(([, u]) => u);
  for (let i = 0; i < uuids.length; i += 40) {
    try {
      const r = await fetchRetry(`${RECCOBEATS_API}/audio-features?ids=${uuids.slice(i, i + 40).join(",")}`, { headers: UA });
      if (!r.ok) continue;
      for (const f of ((await r.json()) as any).content ?? []) {
        const sid = ((f.href || "").replace(/\/+$/, "").split("/").pop()) || uuidSid.get(f.id);
        if (!sid) continue;
        const vals: Record<string, any> = {};
        for (const k of AF_KEYS) if (f[k] != null) vals[`af_${k}`] = f[k];
        out.set(sid, vals);
      }
    } catch { /* best-effort acoustics */ }
  }
  return out;
}
async function embedBatch(env: Env, docs: string[]): Promise<Float32Array[]> {
  const model = env.EMBED_MODEL || DEFAULT_EMBED_MODEL;
  const vecs: Float32Array[] = [];
  for (let i = 0; i < docs.length; i += 50) {
    const res: any = await env.AI.run(model, { text: docs.slice(i, i + 50) });
    const data = res?.data;
    if (!Array.isArray(data)) throw new ApiError(502, "embedding model returned no vectors");
    for (const v of data) vecs.push(Float32Array.from(v));
  }
  return vecs;
}

// Cold-start every unknown seed in a handful of batched calls (was ~5 calls
// PER track → blew the Worker subrequest budget after ~9 tracks). uri → latent.
async function coldStartBatch(env: Env, seeds: Basic[]): Promise<Map<string, { latent: number[]; genre: string }>> {
  const embeds = new Map<string, { latent: number[]; genre: string }>();
  if (!has_projector() || !seeds.length) return embeds;
  const sids = seeds.map((s) => s.uri.split(":").pop()!);
  const trackMap = await spTracksBatch(env, sids);
  const artistIds: string[] = [];
  for (const t of trackMap.values()) for (const a of t.artists ?? []) if (a?.id) artistIds.push(a.id);
  const artistMap = await spArtistsBatch(env, artistIds);
  const rb = await rbBatch(sids);

  const metas: any[] = [];
  for (const sid of sids) {
    const t = trackMap.get(sid);
    if (!t) continue;
    const m = buildMeta(t, artistMap);
    Object.assign(m, rb.get(sid) ?? {});
    metas.push(m);
  }
  const vecs = await embedBatch(env, metas.map(contentDoc));
  metas.forEach((m, j) => {
    if (!vecs[j]) return;
    const vec = project(vecs[j], JSON.stringify(m));
    embeds.set(m.track_uri, { latent: Array.from(vec, (v) => Math.round(v * 1e6) / 1e6), genre: m.genre_primary });
  });
  return embeds;
}

// ---- /api/resolve ----------------------------------------------------------
async function resolve(env: Env, url: string): Promise<any> {
  const link = parseLink(url);
  if (!link) throw new ApiError(400, "not a Spotify playlist/album/track link");
  const { name, tracks } = await expand(env, link.kind, link.id);
  const known = KNOWN ?? new Set<string>();

  const unknown = tracks.filter((t) => t.uri && !known.has(t.uri)).slice(0, MAX_EMBED);
  let embeds = new Map<string, { latent: number[]; genre: string }>();
  try {
    embeds = await coldStartBatch(env, unknown);
  } catch (e) {
    console.warn("batch cold-start failed:", String(e));
  }

  const out: any[] = [];
  let embedded = 0;
  for (const t of tracks) {
    if (!t.uri) continue;
    if (known.has(t.uri)) { out.push({ uri: t.uri, name: t.name, artist: t.artist }); continue; }
    const e = embeds.get(t.uri);
    if (e) { out.push({ uri: t.uri, name: t.name, artist: t.artist, latent: e.latent, genre: e.genre }); embedded += 1; }
  }
  const dropped = tracks.length - out.length;
  return { name, kind: link.kind, tracks: out, counts: { total: tracks.length, embedded, dropped } };
}

// ---- /api/embed ------------------------------------------------------------
// Cold-start a bare list of track uris (the library browser already knows the
// names — it fetched them with the user's own token). Known uris are dropped:
// the browser maps those to baked latents itself.
async function embedUris(env: Env, uris: unknown): Promise<any> {
  if (!Array.isArray(uris)) throw new ApiError(400, "expected { uris: string[] }");
  const known = KNOWN ?? new Set<string>();
  const seen = new Set<string>();
  const wanted: Basic[] = [];
  for (const u of uris) {
    if (typeof u !== "string" || !u.startsWith("spotify:track:")) continue;
    if (known.has(u) || seen.has(u)) continue;
    seen.add(u);
    if (wanted.length < MAX_EMBED) wanted.push({ uri: u, name: "", artist: "" });
  }
  let embeds = new Map<string, { latent: number[]; genre: string }>();
  try {
    embeds = await coldStartBatch(env, wanted);
  } catch (e) {
    console.warn("embed cold-start failed:", String(e));
  }
  const tracks = [...embeds.entries()].map(([uri, e]) => ({ uri, latent: e.latent, genre: e.genre }));
  return { tracks, counts: { requested: uris.length, unknown: wanted.length, embedded: tracks.length } };
}

// ---- /api/art --------------------------------------------------------------
// The baked catalog is a latent index — uri, name, artist, genre — with no
// imagery, so the player has nothing to show. Covers come from the Spotify
// catalog under CLIENT CREDENTIALS, not the user's token: artwork therefore
// works in preview mode, before anyone signs in, which is the mode most
// visitors stay in. Answers are immutable enough to cache for a week.
const MAX_ART = 50;   // one Spotify /tracks call — the journey renders 8 at a time
const ART_TTL = 604800;

async function artFor(env: Env, ids: string[]): Promise<Record<string, { url: string }>> {
  const out: Record<string, { url: string }> = {};
  if (!ids.length) return out;
  const r = await spotifyGet(env, "tracks", { ids: ids.join(",") });
  for (const t of r.tracks ?? []) {
    const imgs = t?.album?.images ?? [];
    if (!t?.id || !imgs.length) continue;
    // Spotify returns 640 / 300 / 64 px. The middle one is crisp at every size
    // this UI draws (42-96 px, retina included) and keeps the journey cheap.
    out[t.id] = { url: (imgs[1] ?? imgs[0]).url };
  }
  return out;
}

// ---- entry -----------------------------------------------------------------
function json(body: unknown, status = 200, cacheSeconds = 0): Response {
  const headers: Record<string, string> = {
    "content-type": "application/json",
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET, POST, OPTIONS",
    "access-control-allow-headers": "content-type",
  };
  if (cacheSeconds) headers["cache-control"] = `public, max-age=${cacheSeconds}`;
  return new Response(JSON.stringify(body), { status, headers });
}

export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(req.url);
    const { pathname } = url;
    // Public config for the browser PKCE save-to-playlist flow. The client_id is
    // public (safe to expose); null hides the Save button when auth isn't set up.
    if (pathname === "/api/config") {
      return json({ spotify_client_id: env.SPOTIFY_CLIENT_ID ?? null });
    }
    if (pathname === "/api/art") {
      if (req.method === "OPTIONS") return json({}, 204);
      // Sort + dedupe BEFORE building the cache key, so the same eight tracks
      // requested in a different order are one cache entry, not two.
      const ids = [...new Set((url.searchParams.get("ids") ?? "").split(",")
        .filter((s) => /^[A-Za-z0-9]{22}$/.test(s)))].sort().slice(0, MAX_ART);
      const key = new Request(`${url.origin}/api/art?ids=${ids.join(",")}`);
      const hit = await caches.default.match(key);
      if (hit) return hit;
      try {
        const res = json({ art: await artFor(env, ids) }, 200, ART_TTL);
        ctx.waitUntil(caches.default.put(key, res.clone()));
        return res;
      } catch (e) {
        // No credentials, or Spotify is down: the player falls back to its
        // mood-tinted placeholders, so this is a soft failure, never cached.
        console.warn("art lookup failed:", String(e));
        return json({ art: {}, error: String((e as Error).message ?? e) });
      }
    }
    if (pathname === "/api/resolve" || pathname === "/api/embed") {
      if (req.method === "OPTIONS") return json({}, 204);
      if (req.method !== "POST") return json({ error: "POST only" }, 405);
      try {
        await ensureReady(env);
        const body = (await req.json().catch(() => ({}))) as { url?: string; uris?: string[] };
        const data = pathname === "/api/embed"
          ? await embedUris(env, body.uris)
          : await resolve(env, body.url ?? "");
        return json(data);
      } catch (e) {
        const status = e instanceof ApiError ? e.status : 500;
        return json({ error: String((e as Error).message ?? e) }, status);
      }
    }
    // everything else → the static app
    return env.ASSETS.fetch(req);
  },
};

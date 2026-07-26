# Docker-hosted deployment

A self-contained stack for the **spotify-next-track** lensing instance —
`lensing-server` (HTTP API + the built React UI, and it spawns the Python
predictors) alongside its **Postgres** and **Qdrant**.

Follows the `snappler` tailnet convention (`/srv/shared/docker/<service>/`):
one published port, reverse-proxied by the Caddy `router` at
`http://nexttrack.snappler`.

## Layout

| file | role |
|---|---|
| `Dockerfile` | multi-stage image: Rust workspace → Vite UI → slim Python runtime carrying `predictors/.venv`. Build context = **repo root**. |
| `docker-compose.yml` | the stack: `lensing-server` + `postgres` + `qdrant`, named volumes, healthchecks. |
| `entrypoint.sh` | idempotent `migrate-data` (file→Postgres backfill), then `serve`. |
| `.env.example` | copy to `.env` (DB creds, port, Spotify/OpenAI keys). |

## Build & run (locally or on the host)

From the repo root, the `zig build` shortcuts wrap the compose commands below:

```sh
cp deploy/.env.example deploy/.env   # fill in DB password + any Spotify/OpenAI keys
zig build docker-serve               # build image + run server + postgres + qdrant (foreground)
zig build docker-build               # build the image only
zig build docker-down                # stop the stack (named volumes survive)
```

Or drive compose directly:

```sh
cd deploy
cp .env.example .env            # fill in DB password + any Spotify/OpenAI keys
docker compose build            # builds ghcr.io/egonik-unlp/spotify-nexttrack from ../ (repo root)
docker compose up -d            # starts server + postgres + qdrant (auto-restart on boot)
docker compose logs -f lensing-server
# → UI + API on http://localhost:8096
docker compose push             # (optional) publish the image to ghcr
```

> The containers use port **8096** and the `spotify-nexttrack-*` names — stop the
> dev server (`serve`) and `zig build db-down` first so they don't collide.

The image bakes the code, the built UI, the model **definitions** (`models.toml`)
and config (`registry.toml`, `domain.toml`). Everything stateful lives in volumes:
`pgdata`, `qdrant_storage`, and `lensing_data` (→ `/app/data`: datasets, run dirs,
promoted-model snapshots).

## Restoring instance state (required on a fresh deploy)

`postgres`, `qdrant` and `lensing_data` start **empty** — no corpus, no runs, no
promoted models. Seed them from an instance bundle (`zig build export-instance`
on the source machine → `instance-bundle/`):

1. **data/** — copy the bundle's `data/` into the `lensing_data` volume
   (e.g. `docker cp instance-bundle/data/. spotify-nexttrack-server:/app/data/`,
   or pre-populate the volume before first `up`).
2. **Postgres** — restore the dump into the `postgres` service
   (`docker exec -i spotify-nexttrack-postgres psql -U pg -d lensing < instance-bundle/postgres.sql`).
3. **Qdrant** — restore the collection snapshot into `qdrant`
   (upload via the Qdrant snapshots API against the container, or drop it into
   the `qdrant_storage` volume before first `up`).

`entrypoint.sh` runs `migrate-data` on every boot, so once `data/` + Postgres are
in place the server reconciles definitions/index automatically. After restore,
confirm with `curl http://localhost:8096/api/runs` and the best-models group.

## Publishing on the tailnet (Caddy router)

Add a block to `/srv/shared/docker/router/Caddyfile` and reload the `router`
stack:

```
# spotify next-track (lensing) UI + API
http://nexttrack.snappler {
	reverse_proxy localhost:8096
}
```

```sh
cd /srv/shared/docker/router && docker compose restart caddy
# → reachable at http://nexttrack.snappler from any tailnet device
```

## Notes

- **CPU-only.** The predictor venv installs the PyTorch CPU wheels (same as
  `zig build py-setup`); no GPU needed.
- **Corpus refresh / OpenAI** paths need the Spotify + OpenAI keys in `.env`;
  plain serving/prediction does not.
- The repo's top-level `docker-compose.yml` remains the **dev** infra (Postgres +
  Qdrant only, for `zig build serve` on a workstation). This `deploy/` stack is
  the full containerized server for hosting.

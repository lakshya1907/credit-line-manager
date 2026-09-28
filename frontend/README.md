# Credit Line Manager — Frontend

React + TypeScript + Vite SPA (not Next.js — this is an internal analytics/ops tool with no SEO/SSR need; see the parent repo's `CLAUDE.md` for the reasoning). Replaces `src/dashboard_app.py` (removed), reading from the FastAPI layer in `../src/api/`.

## Setup

```bash
npm install
npm run dev          # http://localhost:5180 (or whatever port is free); proxies /api/* to
                      # http://localhost:8000 (see vite.config.ts) -- no CORS setup needed locally
```

Requires the backend running (`uvicorn src.api.app:app --reload` from the repo root) and, for anything beyond the health check, a synced Postgres (`alembic upgrade head && python sync_run_to_db.py`) plus `models/*.pkl` (`python run_all.py`).

```bash
npm run build         # tsc -b && vite build -> dist/
```

A production build talks to the API directly via `VITE_API_URL` (see `.env.example`) instead of the dev proxy — set it to wherever the FastAPI instance is deployed, and make sure `CORS_ORIGINS` on that instance includes this app's origin (see `src/api/app.py`).

## Structure

```
src/
  api/
    types.ts     TS interfaces mirroring src/api/schemas.py (hand-kept in sync for now --
                  see the note in CLAUDE.md about generating these from the OpenAPI schema)
    client.ts    thin fetch wrapper
    hooks.ts     TanStack Query hooks (one per endpoint, plus job polling)
  components/    Layout (nav + readiness indicator), ActionBadge, StatCard
  pages/
    RunsPage.tsx               run history + list, trigger new run (background job + polling)
    RunDetailPage.tsx          portfolio summary, policy comparison, stress test + segment
                                charts (recharts), fairness check, backtest -- effectively
                                "Portfolio Overview" + a read-only "Policy Simulator"
                                (pre-computed named scenarios from run_analytics.py's
                                policy_compare.py, not a live slider -- see CLAUDE.md)
    ActionQueuePage.tsx        filterable/paginated recommendations table (TanStack Table)
    CustomerDrilldownPage.tsx  history across runs + live what-if scoring
  lib/format.ts  currency/percent/date formatting helpers
```

## Notes

- Tailwind CSS v4 via `@tailwindcss/vite` (no separate `tailwind.config`/PostCSS setup needed).
- The production bundle is ~734KB minified / ~216KB gzipped, mostly recharts + react-table + react-query — fine at this scope; code-splitting is a legitimate later optimization if the bundle grows, not something to preempt now.
- No auth. This talks to an internal API with no login flow of its own yet; adding one is a `src/api/` change (e.g. an API key), not a frontend-only one.

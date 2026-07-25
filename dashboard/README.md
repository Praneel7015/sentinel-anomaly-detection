# SENTINEL SOC Dashboard

Production-quality React + TypeScript + Vite + Tailwind CSS dashboard for the SENTINEL behavioural anomaly detection system.

## Quick start

```bash
cd dashboard
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The app runs entirely in **mock mode** by default — no Python backend required.

## Mock vs live mode

| Mode | How to enable | Notes |
|------|---------------|-------|
| **Mock** (default) | `VITE_USE_MOCK=true` in `.env` | 2000 deterministic scored events, ~120 entities, all attack types. SSE simulated with a timer emitting new alerts every ~1.5s. |
| **Live** | `VITE_USE_MOCK=false` in `.env` | Talks to the FastAPI backend at `http://127.0.0.1:8000`. Vite dev server proxies `/api/*` there. |

Flipping this one variable is the **only** change needed — no code changes required.

To switch to live mode:

```bash
# dashboard/.env
VITE_USE_MOCK=false
```

Then start the Python service:

```bash
sentinel serve   # from the repo root
```

## Build

```bash
npm run build    # tsc -b && vite build → dist/
npm run preview  # serve the production build locally
```

The build outputs a single HTML + JS + CSS bundle to `dist/`.

## Views

### 1. Triage (default)
Split-pane alert queue with live SSE feed.

- **Left**: ranked alert queue. Dense rows showing risk score, animated risk bar, attack-type chip (colour-coded by type), entity ID + type icon, relative timestamp, cold-start (❄) and novel (✦) badges, and triage status marker.
- **Filters**: attack type, entity type, minimum risk slider, free-text search (entity, resource, IP, event ID), sort order, **alert-budget slider** (0.1–5%). The budget slider moves the alert threshold live through the queue.
- **Stream controls**: play/pause SSE, 0.5×/1×/2×/4× speed, reset, and a "new alerts" ticker showing held count when auto-follow is off.
- **Right pane** (Alert Detail — opens when a row is selected): risk gauge, confidence, attack type + warnings for novel/disagreement, triage buttons (True positive / False positive / Escalate + analyst note), entity summary card, detector scores radar, contribution waterfall (hover for description), analyst narrative, counterfactuals, similar alerts, collapsible raw JSON.

Keyboard navigation:
- `j` / `↓` — next alert
- `k` / `↑` — previous alert
- `1` — mark true positive
- `2` — mark false positive
- `3` — escalate
- `Esc` — close detail pane

### 2. Entity view
Click "View full profile →" in the detail pane or navigate to **Entity** in the top bar.

- Entity list with search (left sidebar when no entity selected)
- Profile summary vs cohort medians
- Risk over time (area chart with alert markers and 70-threshold reference line)
- 24-hour activity histogram
- Top resources (flagged with "new" when not in enrolled baseline)
- Peer comparison radar (entity vs cohort median)
- Drift state banner (Page-Hinkley detection + adaptation status)
- Recent alerts for the entity

### 3. Operations / Model view
Click **Operations** in the top bar.

- PR-AUC, ROC-AUC, FP rate (confounders), FP rate (insider drift) KPI tiles
- Latency KPIs (p50/p95/p99/mean)
- Alert-budget curve (precision and recall vs budget %, current budget marked)
- Precision-Recall curve
- Per-attack-type recall bars (colour-coded green/amber/red by recall level)
- Cold-start and post-drift subgroup precision/recall cards
- MTTD (mean time to detect) table per attack type
- Detector ablation table with PR-AUC delta from full stack
- Confusion matrix heatmap (attack-type classifier, row-normalised intensity)

## Tech stack

| Package | Version | Purpose |
|---------|---------|---------|
| React 18 | 18.3 | UI |
| TypeScript | 6.0 | Type safety |
| Vite | 8 | Build tool + dev server |
| Tailwind CSS | 3.4 | Dark-theme utility styling |
| Recharts | 3 | Charts (PR curve, radar, histogram, waterfall) |
| lucide-react | latest | Icons |
| clsx | 2 | Conditional class names |
| date-fns | 4 | Timestamp formatting |

## Project structure

```
src/
  api/
    types.ts          Wire types (load-bearing field names)
    client.ts         Single switch: VITE_USE_MOCK → mockApi | realApi
    mock.ts           Mock API implementation (SSE via timer)
    mockDataset.ts    Deterministic ~2000-event dataset generator
    mockCatalog.ts    Entity cohorts, attack signatures, feature specs
    mockRandom.ts     Seeded PRNG helpers
    real.ts           HTTP/SSE implementation against FastAPI
  components/
    layout/
      TopBar.tsx        SENTINEL branding, nav, health dot, counters
    triage/
      TriageView.tsx    Main split-pane triage container
      AlertQueue.tsx    Ranked list with budget divider
      AlertRow.tsx      Dense row (memo'd)
      AlertFilters.tsx  Filter bar + budget/minRisk sliders
      StreamControls.tsx Play/pause/speed/ticker
    alertDetail/
      AlertDetail.tsx   Right pane: gauge, waterfall, radar, narrative, triage
    entity/
      EntityView.tsx    Entity profile, risk timeline, peer radar
    ops/
      OpsView.tsx       PR curve, budget curve, ablation, confusion matrix
    ui/
      ChartKit.tsx      Shared Recharts theme helpers
      Chip.tsx          AttackChip, generic Chip
      Controls.tsx      Button, IconButton, Select, Slider, TextInput, Field
      EntityIcon.tsx    User/Server/Cpu icon by entity type
      Panel.tsx         Card with header + body
      Risk.tsx          RiskBar, RiskScore, RiskGauge (SVG)
      Skeleton.tsx      Loading shimmer placeholders
      States.tsx        EmptyState, ErrorState
      StatTile.tsx      KPI number tile
  hooks/
    useAlerts.ts      useAlerts(query), useAlertDetail(eventId)
    useAsync.ts       Generic aborting async state
    useEntity.ts      useEntity(entityId)
    useMetrics.ts     useMetrics()
    useStream.tsx     StreamProvider context + useStream()
  lib/
    domain.ts         Attack/entity metadata, risk colour ramp, formatters
    time.ts           relative(), absolute(), shortTime(), dayHour()
  views.ts            ViewId type + VIEWS descriptor array
  App.tsx             Root layout + navigation state
  main.tsx            React root mount
  index.css           Tailwind base + component classes + Recharts overrides
```

## Colour system

The risk ramp is defined once in `tailwind.config.js` and consumed everywhere as Tailwind utilities:

| Class | Hex | Usage |
|-------|-----|-------|
| `text-risk-low` | `#22c55e` | 0–39 |
| `text-risk-medium` | `#eab308` | 40–64 |
| `text-risk-high` | `#f97316` | 65–84 |
| `text-risk-critical` | `#ef4444` | 85–100 |

Surface palette: `#0a0c10` (base) → `#12151c` → `#1a1f29` → `#232936` → `#2d3442`.  
Accent (cyan): `#22d3ee`.

## Mock dataset

The mock generates:
- **46 attack episodes** covering all 6 attack types, clustered in time
- **4 insider-drift episodes** (gradual shift, benign)
- **34 confounder episodes** (legitimate travel, device enrolment, password rotation, vacation return, maintenance burst)
- Background normal traffic filling to **~2000 total events** across **120 entities**
- Live SSE simulation emitting new events every ~1.5s (30% attacks, 15% confounders, 55% normal)

All events are deterministic from a fixed seed (`0x5e17e1`) so the UI looks identical across reloads except for new SSE events.
